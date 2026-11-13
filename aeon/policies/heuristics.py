"""
Heuristic baselines. Each needs only the current request and the spectrum
database (through the environment's per-request analysis), so each can run at
the source node on a replicated database exactly like the learned policies.

* KSP-FF    : shortest feasible candidate path, most robust (lowest-order)
              feasible format, first-fit spectrum.
* KSP-AM-FF : fewest-slot (highest-order feasible) format; ties -> shortest path;
              first-fit spectrum.
* KSP-AM-BF : fewest-slot format; ties -> path whose tightest fitting free block
              is smallest; best-fit spectrum placement.
"""
from typing import Optional

import numpy as np

from .base import RMSAPolicy, StepContext


class _Heuristic(RMSAPolicy):
    trainable = False

    def __init__(self, env):
        self.env = env
        self.M = env.M


class KSPFF(_Heuristic):
    name = "KSP-FF"

    def act(self, ctx: StepContext, greedy=True) -> Optional[int]:
        feas = np.flatnonzero(ctx.mask)
        if feas.size == 0:
            return None
        return int(feas[0])          # action order = (path asc, format asc)


class KSPAMFF(_Heuristic):
    name = "KSP-AM-FF"

    def act(self, ctx: StepContext, greedy=True) -> Optional[int]:
        feas = np.flatnonzero(ctx.mask)
        if feas.size == 0:
            return None
        info = self.env._analyse()
        best, best_key = None, None
        for a in feas:
            k, m = divmod(int(a), self.M)
            key = (info[k].slots[m], k, m)
            if best_key is None or key < best_key:
                best, best_key = int(a), key
        return best


class KSPAMBF(_Heuristic):
    name = "KSP-AM-BF"
    slot_strategy = "best_fit"

    def act(self, ctx: StepContext, greedy=True) -> Optional[int]:
        feas = np.flatnonzero(ctx.mask)
        if feas.size == 0:
            return None
        info = self.env._analyse()
        best, best_key = None, None
        for a in feas:
            k, m = divmod(int(a), self.M)
            pi = info[k]
            n = pi.slots[m]
            fits = pi.lens[pi.lens >= n]
            tight = int(fits.min()) if fits.size else 10 ** 9
            key = (n, tight, k, m)
            if best_key is None or key < best_key:
                best, best_key = int(a), key
        return best
