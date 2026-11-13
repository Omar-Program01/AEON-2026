"""
Spectrum occupancy database with cached per-link statistics.

Boolean occupancy matrix C[link, slot] (True = occupied), with:
  * per-link occupied-slot counts are maintained incrementally;
  * per-link external fragmentation F_l = 1 - B_l / A_l (B_l largest free block,
    A_l free slots; F_l = 0 when A_l = 0) is cached and recomputed only for
    links touched by an allocation or release;
  * free-block extraction is vectorised (numpy run-length) instead of a Python
    loop over 320 slots.
"""
from typing import Iterable, List, Tuple

import numpy as np


def free_runs(occ_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start indices and lengths of maximal free runs in a boolean occupancy row."""
    free = (~occ_row).astype(np.int8)
    padded = np.concatenate(([0], free, [0]))
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends - starts


class SpectrumDatabase:
    def __init__(self, num_links: int, num_slots: int = 320):
        self.num_links = num_links
        self.num_slots = num_slots
        self.C = np.zeros((num_links, num_slots), dtype=np.bool_)
        self.occ = np.zeros(num_links, dtype=np.int64)
        self._frag = np.zeros(num_links, dtype=np.float64)
        self._frag_sum = 0.0

    # -- mutation ----------------------------------------------------------- #
    def reset(self):
        self.C[:] = False
        self.occ[:] = 0
        self._frag[:] = 0.0
        self._frag_sum = 0.0

    def _refresh(self, lid: int):
        row = self.C[lid]
        free = self.num_slots - int(self.occ[lid])
        if free == 0:
            f = 0.0
        else:
            _, lens = free_runs(row)
            f = 1.0 - (int(lens.max()) / free)
        self._frag_sum += f - self._frag[lid]
        self._frag[lid] = f

    def allocate(self, links: Iterable[int], start: int, n: int):
        for lid in links:
            seg = self.C[lid, start:start + n]
            assert not seg.any(), "allocation over occupied slots"
            seg[:] = True
            self.occ[lid] += n
            self._refresh(lid)

    def release(self, links: Iterable[int], start: int, n: int):
        for lid in links:
            seg = self.C[lid, start:start + n]
            assert seg.all(), "release of free slots"
            seg[:] = False
            self.occ[lid] -= n
            self._refresh(lid)

    # -- queries ------------------------------------------------------------ #
    def union(self, links: Iterable[int]) -> np.ndarray:
        links = list(links)
        if len(links) == 1:
            return self.C[links[0]]
        return np.logical_or.reduce(self.C[links], axis=0)

    def is_available(self, links: Iterable[int], start: int, n: int) -> bool:
        if start < 0 or start + n > self.num_slots:
            return False
        return not self.C[list(links), start:start + n].any()

    def fragmentation(self) -> float:
        """Network external fragmentation: mean of F_l over all directed links."""
        return self._frag_sum / self.num_links

    def link_fragmentation(self, lid: int) -> float:
        return float(self._frag[lid])

    def utilization(self) -> np.ndarray:
        return self.occ / self.num_slots

    def recompute_all(self):
        """Full recomputation (used by tests to validate the cache)."""
        self._frag_sum = 0.0
        self._frag[:] = 0.0
        for lid in range(self.num_links):
            self._refresh(lid)


def choose_start(union_occ: np.ndarray, n: int, strategy: str = "first_fit"):
    """Start slot for a contiguous block of n free slots on a path's union
    occupancy, or None. Strategies: first_fit (lowest index), last_fit (highest
    index), best_fit (smallest free run that fits; ties -> lowest index)."""
    starts, lens = free_runs(union_occ)
    ok = lens >= n
    if not ok.any():
        return None
    if strategy == "first_fit":
        return int(starts[ok][0])
    if strategy == "last_fit":
        i = np.flatnonzero(ok)[-1]
        return int(starts[i] + lens[i] - n)
    if strategy == "best_fit":
        cand = np.flatnonzero(ok)
        j = cand[np.argmin(lens[cand])]
        return int(starts[j])
    raise ValueError(strategy)
