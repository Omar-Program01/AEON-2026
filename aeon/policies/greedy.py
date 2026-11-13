"""
KSP-AM-greedy: myopic reference heuristic.

Among the view-feasible actions (path k, format m, block j < J) it chooses the one with
the highest IMMEDIATE reward of the environment,

    r = w_s + w_e * b / (12.5 n(b, m)) - w_f * [F_after - F_before]^+ ,

with the network fragmentation F computed on the source node's VIEW (the information
the node has). This is the action a perfect myopic (gamma = 0) learner would converge
to; with reward centering the argmax is the same because a constant is subtracted.
Ties: fewest slots, then shortest path, then lowest j (the KSP-AM-FF order).
"""
from typing import Optional

import numpy as np

from .base import RMSAPolicy
from ..spectrum import free_runs
from ..distributed import jth_block_start


def _frag_row(row: np.ndarray) -> float:
    free = row.size - int(row.sum())
    if free == 0:
        return 0.0
    _, lens = free_runs(row)
    return 1.0 - int(lens.max()) / free


class KSPAMGreedy(RMSAPolicy):
    name = "KSP-AM-greedy"
    slot_strategy = "jblock"

    def __init__(self, env):
        self.env = env

    def act(self, ctx, greedy=True) -> Optional[int]:
        env = self.env
        mask = ctx.mask
        if not mask.any():
            return None
        c = env.cfg
        r = env.req
        info = env._analyse()
        best, best_key = None, None
        rows_cache = {}
        for a in np.flatnonzero(mask):
            k, m, j = env.decode_full(int(a))
            pi = info[k]
            n = pi.slots[m]
            start = jth_block_start(pi.union, n, j)
            if start is None:
                continue
            if k not in rows_cache:
                rows = env.view_rows(r.src, pi.links)
                rows_cache[k] = (rows, [_frag_row(row) for row in rows])
            rows, f_before = rows_cache[k]
            d = 0.0
            for i in range(len(pi.links)):
                row = rows[i].copy()
                row[start:start + n] = True
                d += _frag_row(row) - f_before[i]
            dF = d / env.L
            rew = c.w_success + c.w_efficiency * r.bitrate / (n * 12.5) - c.w_frag * max(dF, 0.0)
            key = (-rew, n, k, j)
            if best_key is None or key < best_key:
                best, best_key = int(a), key
        return best
