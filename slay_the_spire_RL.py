import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

from stsenv import StsEnv


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


def _to_tensors(obs_dict):
    obs = torch.as_tensor(obs_dict["obs"], dtype=torch.float32).unsqueeze(0)
    hand = torch.as_tensor(obs_dict["hand_ids"], dtype=torch.long).unsqueeze(0)
    return obs, hand


def _masked_probs(probs, mask):
    masked = probs * mask
    return masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def VPG(env, policy, value_function,
        gamma=0.99, lam=0.95, train_v_iters=80, pi_lr=3e-4, vf_lr=1e-3):
    pi_optim = torch.optim.Adam(policy.parameters(), lr=pi_lr)
    vf_optim = torch.optim.Adam(value_function.parameters(), lr=vf_lr)

    returns_window = deque(maxlen=100)
    n_episode = 1

    while True:
        obs_buf, hand_buf, mask_buf, act_buf, rew_buf, val_buf = [], [], [], [], [], []

        state = env.reset()
        done = False
        while not done:
            obs_t, hand_t = _to_tensors(state)
            mask = env.legal_mask()
            mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
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
        val_buf.append(0.0)  # terminal bootstrap

        rewards = np.array(rew_buf, dtype=np.float32)
        values  = np.array(val_buf, dtype=np.float32)
        obs_arr  = np.stack(obs_buf).astype(np.float32)
        hand_arr = np.stack(hand_buf).astype(np.int64)
        mask_arr = np.stack(mask_buf).astype(np.float32)
        actions  = np.array(act_buf, dtype=np.int64)

        # reward-to-go
        R = np.zeros_like(rewards)
        acc = 0.0
        for t in reversed(range(len(rewards))):
            acc = rewards[t] + gamma * acc
            R[t] = acc

        # GAE-lambda
        deltas = rewards + gamma * values[1:] - values[:-1]
        A = np.zeros_like(deltas)
        acc = 0.0
        for t in reversed(range(len(deltas))):
            acc = deltas[t] + gamma * lam * acc
            A[t] = acc
        A = (A - A.mean()) / (A.std() + 1e-8)

        obs_t  = torch.from_numpy(obs_arr).float()
        hand_t = torch.from_numpy(hand_arr).long()
        mask_t = torch.from_numpy(mask_arr).float()
        act_t  = torch.from_numpy(actions).long()
        R_t    = torch.from_numpy(R).float()
        A_t    = torch.from_numpy(A).float()

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

        ep_return = float(rewards.sum())
        returns_window.append(ep_return)
        print(f"ep {n_episode:5d} | len {len(rewards):4d} | return {ep_return:8.2f} | avg100 {np.mean(returns_window):8.2f}")
        n_episode += 1


def main():
    env = StsEnv()
    policy = VPG_policy_net(
        nstate=env.OBS_DIM, nhidden=64, naction=env.N_ACTIONS,
        n_cards=env.N_CARDS, emb_dim=16, hand_size=env.HAND_SIZE,
    )
    value_function = VPG_value_net(
        nstate=env.OBS_DIM, nhidden=64,
        n_cards=env.N_CARDS, emb_dim=16, hand_size=env.HAND_SIZE,
    )
    VPG(env, policy, value_function)


if __name__ == "__main__":
    main()
