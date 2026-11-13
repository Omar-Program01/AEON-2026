"""
All experiment settings in one place. Every run records the full configuration
in its manifest (explog.py), so every reported number can be traced back to
these values.
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple

from .env import EnvConfig


@dataclass
class TrainConfig:
    # training budget
    train_events: int = 250_000           # arrivals in the continuing training stream
    segment_events: int = 2_500           # logging granularity (no network reset)
    train_load: float = 300.0             # Erlang
    curriculum: bool = True
    curriculum_start_load: float = 100.0
    curriculum_frac: float = 0.4          # load ramps linearly over the first 40 % of arrivals
    # exploration: linear decay over the first eps_decay_frac of the arrivals
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.7
    # optimisation (shared by all learners)
    gamma: float = 0.99                   # per arrival event
    lr: float = 1e-4
    batch_size: int = 128
    replay_capacity: int = 50_000
    learn_start: int = 128
    learn_every: int = 4                  # decision epochs between gradient steps (value learners)
    tau: float = 0.005                    # soft target update after every gradient step
    grad_clip: float = 10.0
    hidden: int = 128                     # per-request networks: obs -> 128 -> 128 -> 30
    hidden_central: int = 256
    mixer_embed: int = 32
    hyper_hidden: int = 64
    gnn_layers: int = 2
    gnn_hidden: int = 64
    # actor-critic (rollout-based updates with bootstrap; see actor_critic.py)
    rollout_len: int = 128
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch: int = 32
    torch_threads: int = 1
    # numerical conditioning (identical for every learner; see rollout.learning_reward)
    reward_transform: str = "none"        # "none" | "affine": r' = (r - w_s) / w_s (policy-invariant)
    dueling: bool = False                 # value learners: Q = V + A - mean(A)
    n_step: int = 1                       # value learners: multi-step returns over decision epochs
    a2c_lambda: float = 1.0               # DeepRMSA (A2C) advantage lambda (1.0 = n-step returns)


@dataclass
class EvalConfig:
    loads: Tuple[float, ...] = (300.0,)
    eval_seeds: Tuple[int, ...] = (900_000, 900_001, 900_002, 900_003, 900_004)
    warmup_holding_times: float = 5.0     # warm-up length in mean holding times
    measure_events: int = 10_000          # measured arrivals per evaluation seed
    qot_check_every: int = 250            # arrivals between QoT re-check samples (gn_v2 only)


def train_stream_seed(train_seed: int) -> int:
    """Training traffic seeds (10,000 + s) never overlap evaluation seeds (900,000 + i)."""
    return 10_000 + int(train_seed)


METHODS = ("KSP-FF", "KSP-AM-FF", "KSP-AM-BF", "DeepRMSA", "PPO", "GNN-DRL",
           "Centralized-DQN", "Independent", "QMIX")
HEURISTICS = ("KSP-FF", "KSP-AM-FF", "KSP-AM-BF")

def as_dict(*cfgs):
    out = {}
    for c in cfgs:
        out[type(c).__name__] = asdict(c)
    return out
