"""
Policy interface shared by heuristics and learners.

Decision epochs. An arrival is a DECISION EPOCH if its feasible set is non-empty;
otherwise it is a FORCED BLOCK (no action can serve it). Learners store one
transition per decision epoch t:

    (o_t, s_t, a_t, G_t, Gamma_t, o_t', mask_t', s_t')

where t' is the next decision epoch, G_t is the reward of epoch t plus the
discounted rewards of every forced block between t and t', and Gamma_t is the
bootstrap discount:

    G_t     = r_t + sum_{j=1..k} gamma^j r_{t+j}      (k forced blocks)
    Gamma_t = gamma^{k+1}

so that the one-step target is  y_t = G_t + Gamma_t * V(next decision state).
Discounting is per arrival event. The stream is continuing: there is no terminal
state; the last open transition at the end of training is discarded instead of
being bootstrapped with zero.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StepContext:
    obs: np.ndarray          # per-source observation (used by all per-request learners)
    mask: np.ndarray         # bool[action_dim]
    global_state: np.ndarray # per-link utilization + mean utilization + network fragmentation
    src: int
    dst: int
    bitrate: float


@dataclass
class Transition:
    obs: np.ndarray
    mask: np.ndarray
    global_state: np.ndarray
    action: int
    ret: float               # G_t
    discount: float          # Gamma_t
    next_obs: np.ndarray
    next_mask: np.ndarray
    next_global_state: np.ndarray
    src: int
    dst: int
    bitrate: float
    next_src: int
    next_dst: int
    next_bitrate: float


class RMSAPolicy:
    name = "base"
    trainable = False
    slot_strategy = "first_fit"

    def act(self, ctx: StepContext, greedy: bool = False) -> Optional[int]:
        raise NotImplementedError

    def observe(self, tr: Transition):
        pass

    def learn(self):
        return None

    def set_epsilon(self, eps: float):
        pass

    def n_parameters(self) -> int:
        return 0


def masked_argmax(q: np.ndarray, mask: np.ndarray) -> Optional[int]:
    if not mask.any():
        return None
    q = np.where(mask, q, -np.inf)
    return int(np.argmax(q))
