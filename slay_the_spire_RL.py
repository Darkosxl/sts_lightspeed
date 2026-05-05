import os
import random
import time
import multiprocessing as mp

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

from stsenv import StsEnv

_WORKER_ENV = None
_WORKER_POLICY = None
_WORKER_VALUE = None
_CPU_DEVICE = torch.device("cpu")

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _get_device():
    requested = os.environ.get("STS_DEVICE")
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("STS_DEVICE=cuda was requested, but torch.cuda.is_available() is false")
        return device
    return torch.device("cpu")


class VPG_value_net(nn.Module):
    def __init__(self, nstate, nhidden, n_cards, emb_dim, hand_size):
        super().__init__()
        self.card_emb = nn.Embedding(n_cards, emb_dim)
        self.hand_size = hand_size
        self.hidden = nn.Linear(nstate + hand_size * emb_dim, nhidden)
        self.value_head = nn.Linear(nhidden, 1)

    def forward(self, obs, hand_ids):
        hand_vecs = self.card_emb(hand_ids).flatten(start_dim=-2)
        x = torch.cat([obs, hand_vecs], dim=-1)
        x = F.relu(self.hidden(x))
        return self.value_head(x).squeeze(-1)


class VPG_policy_net(nn.Module):
    def __init__(self, nstate, nhidden, naction, n_cards, emb_dim, hand_size):
        super().__init__()
        self.card_emb = nn.Embedding(n_cards, emb_dim)
        self.hand_size = hand_size
        self.hidden = nn.Linear(nstate + hand_size * emb_dim, nhidden)
        self.action_head = nn.Linear(nhidden, naction)

    def forward(self, obs, hand_ids):
        hand_vecs = self.card_emb(hand_ids).flatten(start_dim=-2)
        x = torch.cat([obs, hand_vecs], dim=-1)
        x = F.relu(self.hidden(x))
        return F.softmax(self.action_head(x), dim=-1)


def _to_tensors(obs_dict, device):
    obs = torch.as_tensor(obs_dict["obs"], dtype=torch.float32, device=device).unsqueeze(0)
    hand = torch.as_tensor(obs_dict["hand_ids"], dtype=torch.long, device=device).unsqueeze(0)
    return obs, hand


def _masked_probs(probs, mask):
    masked = probs * mask
    return masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _rollout_episode(env, policy, value_function, device):
    obs_buf, hand_buf, mask_buf, act_buf, rew_buf, val_buf = [], [], [], [], [], []

    state = env.reset()
    done = False
    while not done:
        obs_t, hand_t = _to_tensors(state, device)
        mask = env.legal_mask()
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            probs = _masked_probs(policy(obs_t, hand_t), mask_t)
            v_t = value_function(obs_t, hand_t).item()
        action = Categorical(probs).sample()

        next_state, reward, done = env.step(action.item())

        obs_buf.append(state["obs"])
        hand_buf.append(state["hand_ids"])
        mask_buf.append(mask)
        act_buf.append(action.item())
        rew_buf.append(reward)
        val_buf.append(v_t)

        state = next_state

    rewards = np.asarray(rew_buf, dtype=np.float32)
    return {
        "obs": np.stack(obs_buf).astype(np.float32),
        "hand": np.stack(hand_buf).astype(np.int64),
        "mask": np.stack(mask_buf).astype(np.float32),
        "actions": np.asarray(act_buf, dtype=np.int64),
        "rewards": rewards,
        "values": np.asarray(val_buf, dtype=np.float32),
        "return": float(rewards.sum()),
        "length": len(rewards),
    }


def _discount_cumsum(rewards, gamma):
    out = np.zeros_like(rewards, dtype=np.float32)
    acc = 0.0
    for t in reversed(range(len(rewards))):
        acc = rewards[t] + gamma * acc
        out[t] = acc
    return out


def _prepare_batch(episodes, gamma, lam):
    returns, advantages = [], []

    for episode in episodes:
        rewards = episode["rewards"]
        values = np.append(episode["values"], 0.0).astype(np.float32)

        returns.append(_discount_cumsum(rewards, gamma))

        deltas = rewards + gamma * values[1:] - values[:-1]
        advantages.append(_discount_cumsum(deltas, gamma * lam))

    obs_arr = np.concatenate([ep["obs"] for ep in episodes]).astype(np.float32)
    hand_arr = np.concatenate([ep["hand"] for ep in episodes]).astype(np.int64)
    mask_arr = np.concatenate([ep["mask"] for ep in episodes]).astype(np.float32)
    actions = np.concatenate([ep["actions"] for ep in episodes]).astype(np.int64)
    R = np.concatenate(returns).astype(np.float32)
    A = np.concatenate(advantages).astype(np.float32)
    A = (A - A.mean()) / (A.std() + 1e-8)

    return obs_arr, hand_arr, mask_arr, actions, R, A


def _state_dict_cpu(module):
    return {k: v.detach().cpu().numpy().copy() for k, v in module.state_dict().items()}


def _checkpoint_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _checkpoint_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_checkpoint_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_checkpoint_cpu(v) for v in obj)
    return obj


def _save_checkpoint(path, policy, value_function, pi_optim, vf_optim, n_batch, n_episode):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(
        {
            "batch": n_batch,
            "episode": n_episode,
            "policy": _checkpoint_cpu(policy.state_dict()),
            "value_function": _checkpoint_cpu(value_function.state_dict()),
            "pi_optimizer": _checkpoint_cpu(pi_optim.state_dict()),
            "vf_optimizer": _checkpoint_cpu(vf_optim.state_dict()),
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def _init_rollout_worker():
    global _WORKER_ENV, _WORKER_POLICY, _WORKER_VALUE

    seed = (os.getpid() * 9973 + time.time_ns()) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    _WORKER_ENV = StsEnv()
    _WORKER_POLICY = VPG_policy_net(
        nstate=_WORKER_ENV.OBS_DIM, nhidden=64, naction=_WORKER_ENV.N_ACTIONS,
        n_cards=_WORKER_ENV.N_CARDS, emb_dim=16, hand_size=_WORKER_ENV.HAND_SIZE,
    ).to(_CPU_DEVICE)
    _WORKER_VALUE = VPG_value_net(
        nstate=_WORKER_ENV.OBS_DIM, nhidden=64,
        n_cards=_WORKER_ENV.N_CARDS, emb_dim=16, hand_size=_WORKER_ENV.HAND_SIZE,
    ).to(_CPU_DEVICE)
    _WORKER_POLICY.eval()
    _WORKER_VALUE.eval()


def _worker_rollout(args):
    policy_state, value_state = args
    _WORKER_POLICY.load_state_dict({k: torch.from_numpy(v) for k, v in policy_state.items()})
    _WORKER_VALUE.load_state_dict({k: torch.from_numpy(v) for k, v in value_state.items()})
    return _rollout_episode(_WORKER_ENV, _WORKER_POLICY, _WORKER_VALUE, _CPU_DEVICE)


def VPG(env, policy, value_function,
        gamma=0.99, lam=0.95, train_v_iters=80, pi_lr=3e-4, vf_lr=1e-3,
        num_workers=1, checkpoint_interval=50000, checkpoint_dir="checkpoints",
        max_batches=None):
    device = next(policy.parameters()).device
    pi_optim = torch.optim.Adam(policy.parameters(), lr=pi_lr)
    vf_optim = torch.optim.Adam(value_function.parameters(), lr=vf_lr)

    returns_window = deque(maxlen=100)
    n_episode = 1
    n_batch = 0
    pool = None
    checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")

    if num_workers > 1:
        ctx_name = "spawn" if device.type == "cuda" else "fork"
        pool = mp.get_context(ctx_name).Pool(num_workers, initializer=_init_rollout_worker)

    try:
        while True:
            if pool is None:
                episodes = [_rollout_episode(env, policy, value_function, device)]
            else:
                policy_state = _state_dict_cpu(policy)
                value_state = _state_dict_cpu(value_function)
                episodes = pool.map(
                    _worker_rollout,
                    [(policy_state, value_state) for _ in range(num_workers)],
                )

            obs_arr, hand_arr, mask_arr, actions, R, A = _prepare_batch(episodes, gamma, lam)

            obs_t  = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
            hand_t = torch.as_tensor(hand_arr, dtype=torch.long, device=device)
            mask_t = torch.as_tensor(mask_arr, dtype=torch.float32, device=device)
            act_t  = torch.as_tensor(actions, dtype=torch.long, device=device)
            R_t    = torch.as_tensor(R, dtype=torch.float32, device=device)
            A_t    = torch.as_tensor(A, dtype=torch.float32, device=device)

            # policy update (1 step)
            probs = _masked_probs(policy(obs_t, hand_t), mask_t)
            log_probs = Categorical(probs).log_prob(act_t)
            pi_loss = -(log_probs * A_t).mean()

            pi_optim.zero_grad()
            pi_loss.backward()
            pi_optim.step()

            # value update (train_v_iters steps)
            for _ in range(train_v_iters):
                vf_optim.zero_grad()
                v_loss = ((value_function(obs_t, hand_t) - R_t) ** 2).mean()
                v_loss.backward()
                vf_optim.step()

            for episode in episodes:
                returns_window.append(episode["return"])

            n_batch += 1
            start_ep = n_episode
            n_episode += len(episodes)
            print(
                f"batch {n_batch:7d} | eps {start_ep:5d}-{n_episode - 1:<5d} | "
                f"workers {len(episodes):2d} | "
                f"steps {sum(ep['length'] for ep in episodes):4d} | "
                f"return {np.mean([ep['return'] for ep in episodes]):8.2f} | "
                f"avg100 {np.mean(returns_window):8.2f}",
                flush=True,
            )

            if checkpoint_interval > 0 and n_batch % checkpoint_interval == 0:
                _save_checkpoint(
                    checkpoint_path,
                    policy,
                    value_function,
                    pi_optim,
                    vf_optim,
                    n_batch,
                    n_episode,
                )
                print(f"checkpoint saved: {checkpoint_path}", flush=True)

            if max_batches is not None and n_batch >= max_batches:
                if checkpoint_interval > 0 and n_batch % checkpoint_interval != 0:
                    _save_checkpoint(
                        checkpoint_path,
                        policy,
                        value_function,
                        pi_optim,
                        vf_optim,
                        n_batch,
                        n_episode,
                    )
                    print(f"checkpoint saved: {checkpoint_path}", flush=True)
                print(f"reached max_batches={max_batches}", flush=True)
                break
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


def main():
    env = StsEnv()
    device = _get_device()
    num_workers = int(os.environ.get("STS_WORKERS", "16"))
    checkpoint_interval = int(os.environ.get("STS_CHECKPOINT_INTERVAL", "50000"))
    checkpoint_dir = os.environ.get("STS_CHECKPOINT_DIR", "checkpoints")
    max_batches_env = os.environ.get("STS_MAX_BATCHES")
    max_batches = int(max_batches_env) if max_batches_env else None
    print(
        f"device: {device} | workers: {num_workers} | "
        f"checkpoint_interval: {checkpoint_interval} | "
        f"max_batches: {max_batches if max_batches is not None else 'unlimited'}",
        flush=True,
    )
    policy = VPG_policy_net(
        nstate=env.OBS_DIM, nhidden=64, naction=env.N_ACTIONS,
        n_cards=env.N_CARDS, emb_dim=16, hand_size=env.HAND_SIZE,
    ).to(device)
    value_function = VPG_value_net(
        nstate=env.OBS_DIM, nhidden=64,
        n_cards=env.N_CARDS, emb_dim=16, hand_size=env.HAND_SIZE,
    ).to(device)
    VPG(
        env,
        policy,
        value_function,
        num_workers=num_workers,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        max_batches=max_batches,
    )


if __name__ == "__main__":
    main()
