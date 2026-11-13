"""
Distributed (source-routed) control with stale replicated state.

Model:

* True state. One spectrum database holds the real occupancy of every link.
* Advertised state. Link-state information is flooded PERIODICALLY: at times
  0, tau, 2 tau, ... every node's traffic-engineering database (TED) is
  refreshed with a snapshot of the true state (periodic link-state update).
  tau = 0 means every decision sees the true state (centralised / fresh state).
* Local knowledge. Between updates a source node also knows the lightpaths it
  set up or tore down itself (it signalled them), so its VIEW is the last
  snapshot plus its own changes since that snapshot.
* Decision. The source node decides path, format and spectrum on its view:
  the feasibility mask, the QoT estimate and the observation all use the view.
* Reservation (signalling model):
    "forward"  -- the source fixes the spectrum (source-selected spectrum) and
                  the reservation succeeds only if those slots are free on the
                  TRUE state of every link and QoT holds on the true state;
    "backward" -- label-set style: the path's true availability is collected
                  during signalling and the spectrum is chosen on the TRUE path
                  occupancy (destination-selected spectrum).
  A failed reservation is a SETUP FAILURE and the request is blocked (no
  crank-back retry, for every method alike).
* Requests are served one at a time; signalling latency is not modelled, so
  contention arises only from decisions taken on the same stale view.

Action space. a = ((k * M) + m) * J + j: candidate path k, format m, and the
j-th free block (in first-fit order) of the view's path occupancy that fits
n(b, m) slots; the lightpath starts at that block's first slot. With J = 1 and
the first-fit strategy the action space and behaviour equal EONEnv's.
Heuristics use J = 1 and a named slot strategy instead of j.
"""
import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .env import EONEnv, EnvConfig, Connection, PathInfo, SLOT_GHZ
from .spectrum import SpectrumDatabase, free_runs, choose_start
from .policies.base import StepContext


@dataclass
class DistConfig:
    tau: float = 0.0               # link-state update interval, in mean holding times (0 = fresh state)
    signaling: str = "forward"     # "forward" (source-selected spectrum) | "backward" (label set)
    blocks: int = 1                # J: candidate free blocks per (path, format)
    retries: int = 0               # crank-back retries after a setup failure (heuristic sensitivity only):
                                   # re-decide on the view with the failed path(s) excluded, AM rule
    policy_seed_offset: int = 7_777_777   # RNG for random-fit, independent of the traffic RNG


def start_by_strategy(union: np.ndarray, n: int, strategy: str, src: int,
                      rng: np.random.Generator) -> Optional[int]:
    """Start slot on a path occupancy for a named heuristic slot strategy."""
    if strategy in ("first_fit", "last_fit", "best_fit"):
        return choose_start(union, n, strategy)
    if strategy == "random_fit":
        # uniformly random among ALL start positions s with [s, s+n) free
        starts, lens = free_runs(union)
        ok = lens >= n
        if not ok.any():
            return None
        counts = lens[ok] - n + 1
        total = int(counts.sum())
        r = int(rng.integers(total))
        cum = np.cumsum(counts)
        i = int(np.searchsorted(cum, r, side="right"))
        before = int(cum[i - 1]) if i > 0 else 0
        return int(starts[ok][i] + (r - before))
    if strategy == "flf_node":
        # first-last fit by source node: even sources first-fit, odd sources last-fit
        return choose_start(union, n, "first_fit" if src % 2 == 0 else "last_fit")
    raise ValueError(strategy)


def jth_block_start(union: np.ndarray, n: int, j: int) -> Optional[int]:
    starts, lens = free_runs(union)
    fit = np.flatnonzero(lens >= n)
    if j >= fit.size:
        return None
    return int(starts[fit[j]])


class DistributedEONEnv(EONEnv):
    FEAT_EXTRA = 1      # age of the view, normalised by tau (0 when tau = 0)

    def __init__(self, cfg: EnvConfig = None, dcfg: DistConfig = None, seed: int = 0, load: float = 300.0):
        self.dcfg = dcfg or DistConfig()
        assert self.dcfg.signaling in ("forward", "backward")
        assert self.dcfg.blocks >= 1
        self.J = int(self.dcfg.blocks)
        cfg = cfg or EnvConfig()
        assert cfg.phy_model == "gn_v2", "the distributed environment uses the GN physical layer"
        # attributes needed by reset() -> _advance() before super().__init__ finishes
        self.adv = None
        self.ops: Dict[int, list] = {}
        self.next_update = math.inf
        self.last_update = 0.0
        self.window_id = 0
        self.policy_rng = np.random.default_rng(0)
        super().__init__(cfg, seed, load)
        self.action_dim = self.K * self.M * self.J

    # ------------------------------------------------------------------ #
    def reset(self, seed: int, load: float):
        L, S = self.topo.num_links, self.cfg.num_slots
        self.adv = SpectrumDatabase(L, S)
        self.ops = {}
        self.last_update = 0.0
        self.next_update = self.dcfg.tau if self.dcfg.tau > 0 else math.inf
        self.window_id = 0
        self.policy_rng = np.random.default_rng(seed + self.dcfg.policy_seed_offset)
        self.counters = {"setup_fail": 0, "view_block": 0, "updates": 0, "retries": 0}
        super().reset(seed, load)

    @property
    def fresh(self) -> bool:
        return self.dcfg.tau <= 0

    def _snapshot(self, t: float):
        self.adv.C[:] = self.db.C
        self.adv.occ[:] = self.db.occ
        self.ops = {}
        self.last_update = t
        self.window_id += 1
        self.counters["updates"] += 1

    def _advance(self):
        nxt = self.stream.next()
        heap = self.dep_heap
        while True:
            t_dep = heap[0][0] if heap else math.inf
            t_upd = self.next_update
            if min(t_dep, t_upd) > nxt.time:
                break
            if t_dep <= t_upd:
                _, cid = heapq.heappop(heap)
                conn = self.active.pop(cid)
                self.db.release(conn.links, conn.start, conn.n)
                if not self.fresh:
                    self.ops.setdefault(conn.src, []).append((False, conn.links, conn.start, conn.n))
            else:
                self._snapshot(t_upd)
                self.next_update = t_upd + self.dcfg.tau
        if self.fresh:
            self.window_id += 1          # one decision per window when the state is fresh
        self.req = nxt
        self._info = None

    # ------------------------------------------------------------------ #
    # the source node's view
    # ------------------------------------------------------------------ #
    def view_rows(self, src: int, links: List[int]) -> np.ndarray:
        """Per-link view occupancy rows (bool[len(links), S]) of node src."""
        if self.fresh:
            return self.db.C[links]
        rows = self.adv.C[links].copy()
        for alloc, ls, st, n in self.ops.get(src, ()):
            for i, l in enumerate(links):
                if l in ls:
                    rows[i, st:st + n] = alloc
        return rows

    def _analyse(self) -> List[Optional[PathInfo]]:
        if self._info is not None:
            return self._info
        r = self.req
        paths = self.topo.k_paths[r.src][r.dst]
        out: List[Optional[PathInfo]] = []
        for k in range(self.K):
            if k >= len(paths) or not paths[k]:
                out.append(None)
                continue
            links = list(paths[k])
            km = [self.link_km[l] for l in links]
            rows = self.view_rows(r.src, links)
            union = np.logical_or.reduce(rows, axis=0) if len(links) > 1 else rows[0].copy()
            occ = rows.sum(axis=1).astype(float)
            starts, lens = free_runs(union)
            pi = PathInfo(links, km, float(sum(km)), union, starts, lens)
            pi.slots = [self.phy.num_slots(r.bitrate, m) for m in range(self.M)]
            pi.nfit = [int((lens >= n).sum()) for n in pi.slots]
            pi.block_ok = [n <= self.S and pi.nfit[m] >= 1 for m, n in enumerate(pi.slots)]
            snr = self.phy.snr_db_formats(self.link_spans[links], occ, np.asarray(pi.slots, dtype=float))
            pi.qot_ok = [bool(n <= self.S and snr[m] >= self.snr_thr[m]) for m, n in enumerate(pi.slots)]
            out.append(pi)
        self._info = out
        return out

    def feasible_mask(self) -> np.ndarray:
        info = self._analyse()
        J = self.J
        mask = np.zeros(self.action_dim, dtype=np.bool_)
        for k, pi in enumerate(info):
            if pi is None:
                continue
            for m in range(self.M):
                if pi.block_ok[m] and pi.qot_ok[m]:
                    base = (k * self.M + m) * J
                    for j in range(min(J, pi.nfit[m])):
                        mask[base + j] = True
        return mask

    def decode(self, a: int) -> Tuple[int, int]:
        km, _ = divmod(int(a), self.J)
        return divmod(km, self.M)

    def decode_full(self, a: int) -> Tuple[int, int, int]:
        km, j = divmod(int(a), self.J)
        k, m = divmod(km, self.M)
        return k, m, j

    def anchor_action(self, mask: np.ndarray) -> Optional[int]:
        """KSP-AM-FF on the view: fewest-slot format, ties -> shortest path, first block (j = 0)."""
        if not mask.any():
            return None
        info = self._analyse()
        best, best_key = None, None
        for a in np.flatnonzero(mask):
            k, m, j = self.decode_full(int(a))
            if j != 0:
                continue
            key = (info[k].slots[m], k, m)
            if best_key is None or key < best_key:
                best, best_key = int(a), key
        return best

    # ------------------------------------------------------------------ #
    def obs_dim(self) -> int:
        return super().obs_dim() + self.FEAT_EXTRA

    def observation(self) -> np.ndarray:
        base = super().observation()
        age = 0.0 if self.fresh else min((self.req.time - self.last_update) / self.dcfg.tau, 1.0)
        return np.concatenate([base, np.array([age], dtype=np.float32)])

    def context(self) -> StepContext:
        ctx = super().context()            # observation and mask on the view; global state = TRUE state
        ctx.anchor = self.anchor_action(ctx.mask) if self.J > 0 else None
        ctx.window = self.window_id
        return ctx

    # ------------------------------------------------------------------ #
    def _true_qot_ok(self, links: List[int], m: int, n: int) -> bool:
        slots = np.full(self.M, float(n))
        snr = self.phy.snr_db_formats(self.link_spans[links], self.db.occ[links].astype(float), slots)
        return bool(snr[m] >= self.snr_thr[m])

    def _am_alternative(self, info, exclude) -> Optional[Tuple[int, int]]:
        """KSP-AM choice on the view among paths not in `exclude` (crank-back retry)."""
        best, best_key = None, None
        for k, pi in enumerate(info):
            if pi is None or k in exclude:
                continue
            for m in range(self.M):
                if pi.block_ok[m] and pi.qot_ok[m]:
                    key = (pi.slots[m], k, m)
                    if best_key is None or key < best_key:
                        best, best_key = (k, m), key
        return best

    def step(self, action: Optional[int], slot_strategy: str = "first_fit"):
        """Serve the current arrival. `slot_strategy` = "jblock" for learners (start of
        the j-th fitting block), or a heuristic strategy name (J must then be 1)."""
        r = self.req
        c = self.cfg
        f_before = self.db.fragmentation()
        allocated, setup_fail = False, False
        n_used, mod = 0, -1
        if action is None:
            self.counters["view_block"] += 1
        else:
            info = self._analyse()
            k, m, j = self.decode_full(action)
            pi = info[k]
            n = pi.slots[m]
            if self.dcfg.signaling == "forward":
                union = pi.union                                  # the view
            else:
                union = self.db.union(pi.links)                    # true path occupancy (label set)
            if slot_strategy == "jblock":
                start = jth_block_start(union, n, j)
            else:
                assert self.J == 1, "named slot strategies are for J = 1"
                start = start_by_strategy(union, n, slot_strategy, r.src, self.policy_rng)
            ok = (start is not None and self.db.is_available(pi.links, start, n)
                  and self._true_qot_ok(pi.links, m, n))
            tried = {k}
            left = self.dcfg.retries if slot_strategy != "jblock" else 0
            while not ok and left > 0:
                left -= 1
                alt = self._am_alternative(info, tried)
                if alt is None:
                    break
                self.counters["retries"] += 1
                k, m = alt
                tried.add(k)
                pi = info[k]
                n = pi.slots[m]
                union = pi.union if self.dcfg.signaling == "forward" else self.db.union(pi.links)
                start = start_by_strategy(union, n, slot_strategy, r.src, self.policy_rng)
                ok = (start is not None and self.db.is_available(pi.links, start, n)
                      and self._true_qot_ok(pi.links, m, n))
            if ok:
                self.db.allocate(pi.links, start, n)
                self.active[r.conn_id] = Connection(r.conn_id, r.src, r.dst, r.bitrate,
                                                    list(pi.links), m, start, n)
                heapq.heappush(self.dep_heap, (r.time + r.holding, r.conn_id))
                if not self.fresh:
                    self.ops.setdefault(r.src, []).append((True, list(pi.links), start, n))
                allocated, n_used, mod = True, n, m
            else:
                setup_fail = True
                self.counters["setup_fail"] += 1
        f_after = self.db.fragmentation()
        if allocated:
            reward = (c.w_success + c.w_efficiency * r.bitrate / (n_used * SLOT_GHZ)
                      - c.w_frag * max(f_after - f_before, 0.0))
        else:
            reward = -c.w_success
        out = {"allocated": allocated, "blocked": not allocated, "invalid_action": setup_fail,
               "setup_fail": setup_fail, "view_block": action is None,
               "bitrate": r.bitrate, "num_slots": n_used, "mod": mod,
               "frag": f_after, "src": r.src, "dst": r.dst, "time": r.time,
               "window": self.window_id}
        self._advance()
        return reward, out
