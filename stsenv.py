import slaythespire as s
import random
import numpy as np


def decode(idx):
    # encoding (combat-only):
    # 0           : END_TURN
    # 1..50       : play hand[i] on enemy[j],  i=0..9, j=0..4
    # 51..60      : play hand[i] no target,    i=0..9
    # 61..85      : potion[i] on enemy[j],     i=0..4, j=0..4
    # 86..90      : potion[i] no target,       i=0..4
    # 91..100     : single card select[i],     i=0..9
    if idx == 0:
        return ("END_TURN", None, None)
    if idx < 51:
        k = idx - 1
        return ("CARD", k // 5, k % 5)
    if idx < 61:
        return ("CARD", idx - 51, None)
    if idx < 86:
        k = idx - 61
        return ("POTION", k // 5, k % 5)
    if idx < 91:
        return ("POTION", idx - 86, None)
    if idx < 101:
        return ("SINGLE_CARD_SELECT", idx - 91, None)
    raise ValueError(f"action idx {idx} out of range")


def apply_action(bc, type, i, j):
    if type == "END_TURN":
        bc.execute_action(s.ActionType.END_TURN, 0, 0)
    elif type == "CARD":
        bc.execute_action(s.ActionType.CARD, i, 0 if j is None else j)
    elif type == "POTION":
        bc.execute_action(s.ActionType.POTION, i, 0 if j is None else j)
    elif type == "SINGLE_CARD_SELECT":
        bc.execute_action(s.ActionType.SINGLE_CARD_SELECT, i, 0)
    else:
        raise ValueError(f"unknown action type {type}")


class StsEnv:
    GC_DIM = 412
    BC_DIM = 34
    OBS_DIM = GC_DIM + BC_DIM   # 446 flat features
    HAND_SIZE = 10
    N_CARDS = max(int(v) for v in s.CardId.__members__.values()) + 1
    N_ACTIONS = 101

    def __init__(self, character=s.CharacterClass.IRONCLAD, ascension=0):
        self.character = character
        self.ascension = ascension
        self.interface = s.getNNInterface()
        self.state_max = np.array(self.interface.getObservationMaximums(), dtype=np.float32)
        self.state_max[self.state_max == 0] = 1.0
        self.gc = None
        self.bc = None

    def reset(self):
        seed = random.randint(0, 2**15 - 1)
        self.gc = s.GameContext(self.character, seed, self.ascension)
        self.bc = s.BattleContext()
        self.bc.init_with_encounter(self.gc, s.MonsterEncounter.CULTIST)
        return self._obs()

    def step(self, action_idx):
        type, i, j = decode(action_idx)
        prev_hp = self.bc.player_hp
        apply_action(self.bc, type, i, j)
        reward = self._reward(prev_hp)
        done = self.bc.outcome != s.BattleOutcome.UNDECIDED
        return self._obs(), reward, done

    def _obs(self):
        gc_obs = np.array(self.interface.getObservation(self.gc), dtype=np.float32) / self.state_max
        bc_obs = self.encode_combat()
        return {
            "obs": np.concatenate([gc_obs, bc_obs]).astype(np.float32),
            "hand_ids": self.hand_ids(),
        }

    def _reward(self, prev_hp):
        delta = self.bc.player_hp - prev_hp
        if self.bc.outcome == s.BattleOutcome.PLAYER_VICTORY:
            return delta + 5.0
        if self.bc.outcome == s.BattleOutcome.PLAYER_LOSS:
            return delta - 50.0
        return float(delta)
    
    def encode_combat(self):
        bc = self.bc
        features = [
            bc.player_hp / max(bc.player_max_hp, 1),
            bc.player_energy / 3.0,
            bc.turn / 30.0,
            bc.monster_count / 5.0,
        ]
        for i in range(5):
            if i < bc.monster_count and bc.monster_alive(i):
                features.append(bc.monster_hp(i) / max(bc.monster_max_hp(i), 1))
                features.append(1.0)
            else:
                features.extend([0.0, 0.0])
        for i in range(10):
            if i < bc.cards_in_hand:
                features.append(bc.hand_card_cost(i) / 5.0)
                features.append(1.0 if bc.hand_card_upgraded(i) else 0.0)
            else:
                features.extend([0.0, 0.0])
        return np.array(features, dtype=np.float32)

    def hand_ids(self):
        return np.array(self.bc.hand_ids(), dtype=np.int64)

    def _decode_to_args(self, idx):
        # returns (ActionType_enum, i1, i2) ready for is_valid_action
        type, i, j = decode(idx)
        if type == "END_TURN":
            return s.ActionType.END_TURN, 0, 0
        if type == "CARD":
            return s.ActionType.CARD, i, 0 if j is None else j
        if type == "POTION":
            return s.ActionType.POTION, i, 0 if j is None else j
        if type == "SINGLE_CARD_SELECT":
            return s.ActionType.SINGLE_CARD_SELECT, i, 0
        raise ValueError(type)

    def legal_mask(self):
        mask = np.zeros(self.N_ACTIONS, dtype=np.float32)
        for idx in range(self.N_ACTIONS):
            t, i, j = self._decode_to_args(idx)
            if self.bc.is_valid_action(t, i, j):
                mask[idx] = 1.0
        if mask.sum() == 0:
            # safety: never let mask be all-zero (would NaN softmax)
            mask[0] = 1.0  # fallback to END_TURN
        return mask
