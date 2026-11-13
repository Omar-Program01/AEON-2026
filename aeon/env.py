"""
Event-driven elastic optical network environment.

One call to ``step`` serves the current connection arrival and advances the
simulation clock to the next arrival, processing every departure in between.
Arrivals come from a TrafficStream, so training can run on a continuing stream
without resetting the network (steady state), while evaluation resets the
network per traffic seed and discards a warm-up period.

Action space ("first_fit" design): a = k * M + m selects candidate
path k (of K) and modulation format m (of M); the start slot is resolved by a
slot strategy (first-fit by default). |A| = K * M = 30.

Feasible action (mask): path k exists, n(b, m) <= S, a contiguous block of n
free slots exists on every link of path k (spectrum contiguity and continuity),
and QoT holds for format m with the candidate added (see phy.py).
"""
import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .phy import PhyGN, PhyGNConfig, NUM_MODS
from .spectrum import SpectrumDatabase, free_runs, choose_start
from .topology import create_topology
from .traffic import TrafficStream, BITRATES, HOTSPOT_HUBS
from .policies.base import StepContext

SLOT_GHZ = 12.5


@dataclass
class EnvConfig:
    topology: str = "NSFNET"
    num_slots: int = 320
    k_paths: int = 5
    path_order: str = "km"           # "km" (default) | "hops" (K shortest by hop count)
    phy_model: str = "gn_v2"          # closed-form GN model (phy.py)
    guard_slots: int = 1              # gn_v2 only
    w_success: float = 10.0
    w_efficiency: float = 0.1
    w_frag: float = 5.0
    traffic_pattern: str = "uniform"  # "uniform" | "hotspot"
    hubs: Optional[Tuple[int, ...]] = None
    p_hot: float = 0.6
    mean_holding: float = 1.0
    bitrates: Tuple[float, ...] = BITRATES


@dataclass
class Connection:
    conn_id: int
    src: int
    dst: int
    bitrate: float
    links: List[int]
    mod: int
    start: int
    n: int


@dataclass
class PathInfo:
    links: List[int]
    km: List[float]
    length: float
    union: np.ndarray
    starts: np.ndarray
    lens: np.ndarray
    slots: List[int] = field(default_factory=list)      # n(b, m) per format
    block_ok: List[bool] = field(default_factory=list)
    qot_ok: List[bool] = field(default_factory=list)


class EONEnv:
    FEAT_PER_PATH = 11

    def __init__(self, cfg: EnvConfig = None, seed: int = 0, load: float = 300.0):
        self.cfg = cfg or EnvConfig()
        c = self.cfg
        self.topo = create_topology(c.topology, k=c.k_paths, order=c.path_order)
        self.N = self.topo.num_nodes
        self.L = self.topo.num_links
        self.S = c.num_slots
        self.K = c.k_paths
        self.M = NUM_MODS
        self.action_dim = self.K * self.M
        if c.phy_model == "gn_v2":
            self.phy = PhyGN(PhyGNConfig(guard_slots=c.guard_slots))
        else:
            raise ValueError(c.phy_model)
        self.link_km = np.array([self.topo.link_map[l][2] for l in range(self.L)], dtype=float)
        if c.phy_model == "gn_v2":
            self.link_spans = np.array([self.phy.spans(k) for k in self.link_km], dtype=float)
            self.snr_thr = np.array(self.phy.snr_thr_db)
        self.db = SpectrumDatabase(self.L, self.S)
        self.reset(seed, load)

    # ------------------------------------------------------------------ #
    # episode control
    # ------------------------------------------------------------------ #
    def reset(self, seed: int, load: float):
        c = self.cfg
        hubs = c.hubs if c.hubs is not None else HOTSPOT_HUBS.get(c.topology.upper())
        self.stream = TrafficStream(self.N, load, seed, c.mean_holding, c.bitrates,
                                    c.traffic_pattern, hubs, c.p_hot)
        self.db.reset()
        self.active: Dict[int, Connection] = {}
        self.dep_heap: List[Tuple[float, int]] = []
        self.req = None
        self._info = None
        self._advance()

    def set_load(self, load: float):
        self.stream.set_load(load)

    def _advance(self):
        nxt = self.stream.next()
        heap = self.dep_heap
        while heap and heap[0][0] <= nxt.time:
            _, cid = heapq.heappop(heap)
            conn = self.active.pop(cid)
            self.db.release(conn.links, conn.start, conn.n)
        self.req = nxt
        self._info = None

    # ------------------------------------------------------------------ #
    # per-request analysis (computed once per arrival)
    # ------------------------------------------------------------------ #
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
            links = paths[k]
            km = [self.link_km[l] for l in links]
            union = self.db.union(links)
            starts, lens = free_runs(union)
            pi = PathInfo(links, km, float(sum(km)), union, starts, lens)
            largest = int(lens.max()) if lens.size else 0
            pi.slots = [self.phy.num_slots(r.bitrate, m) for m in range(self.M)]
            pi.block_ok = [n <= self.S and largest >= n for n in pi.slots]
            snr = self.phy.snr_db_formats(self.link_spans[links], self.db.occ[links].astype(float),
                                          np.asarray(pi.slots, dtype=float))
            pi.qot_ok = [bool(n <= self.S and snr[m] >= self.snr_thr[m])
                         for m, n in enumerate(pi.slots)]
            out.append(pi)
        self._info = out
        return out

    def feasible_mask(self) -> np.ndarray:
        info = self._analyse()
        mask = np.zeros(self.action_dim, dtype=np.bool_)
        for k, pi in enumerate(info):
            if pi is None:
                continue
            for m in range(self.M):
                if pi.block_ok[m] and pi.qot_ok[m]:
                    mask[k * self.M + m] = True
        return mask

    # ------------------------------------------------------------------ #
    # observations
    # ------------------------------------------------------------------ #
    def obs_dim(self) -> int:
        return self.K * self.FEAT_PER_PATH + 1 + 2 * self.N

    def state_dim(self) -> int:
        return self.L + 2

    def observation(self) -> np.ndarray:
        """11 features per candidate path (over the union occupancy of ALL links
        of the path) + [bitrate/800, one-hot(src), one-hot(dst)]."""
        S = self.S
        r = self.req
        info = self._analyse()
        feat = np.zeros(self.K * self.FEAT_PER_PATH, dtype=np.float32)
        for k, pi in enumerate(info):
            if pi is None:
                continue
            b = k * self.FEAT_PER_PATH
            qot_mods = [m for m in range(self.M) if pi.qot_ok[m]]
            if qot_mods:
                req = pi.slots[max(qot_mods)]       # highest-order QoT-feasible format
                qflag = 1.0
            else:
                req = S
                qflag = 0.0
            starts, lens = pi.starts, pi.lens
            nb = lens.size
            total_free = int(lens.sum())
            largest = int(lens.max()) if nb else 0
            fit = np.flatnonzero(lens >= req)
            ff = int(starts[fit[0]]) if fit.size else -1
            feat[b + 0] = min(req, S) / S
            feat[b + 1] = nb / S
            feat[b + 2] = total_free / S
            feat[b + 3] = (total_free / nb / S) if nb else 0.0
            feat[b + 4] = largest / S
            feat[b + 5] = (starts[0] / S) if nb else 0.0
            feat[b + 6] = (lens[0] / S) if nb else 0.0
            feat[b + 7] = (ff / S) if ff >= 0 else 0.0
            feat[b + 8] = 1.0 if ff >= 0 else 0.0
            feat[b + 9] = pi.length / 5000.0
            feat[b + 10] = qflag
        dem = np.zeros(1 + 2 * self.N, dtype=np.float32)
        dem[0] = r.bitrate / 800.0
        dem[1 + r.src] = 1.0
        dem[1 + self.N + r.dst] = 1.0
        return np.concatenate([feat, dem])

    def global_state(self) -> np.ndarray:
        u = self.db.utilization().astype(np.float32)
        return np.concatenate([u, [np.float32(u.mean()), np.float32(self.db.fragmentation())]])

    def context(self) -> StepContext:
        r = self.req
        return StepContext(self.observation(), self.feasible_mask(), self.global_state(),
                           r.src, r.dst, r.bitrate)

    # ------------------------------------------------------------------ #
    # transition
    # ------------------------------------------------------------------ #
    def decode(self, a: int) -> Tuple[int, int]:
        return divmod(int(a), self.M)

    def step(self, action: Optional[int], slot_strategy: str = "first_fit"):
        """Serve the current arrival with `action` (None = no feasible action /
        block) and advance to the next arrival. Returns (reward, info)."""
        r = self.req
        c = self.cfg
        f_before = self.db.fragmentation()
        allocated, invalid = False, False
        n_used, mod = 0, -1
        if action is not None:
            info = self._analyse()
            k, m = self.decode(action)
            pi = info[k] if k < len(info) else None
            if pi is not None and pi.block_ok[m] and pi.qot_ok[m]:
                n = pi.slots[m]
                start = choose_start(pi.union, n, slot_strategy)
                if start is not None:
                    self.db.allocate(pi.links, start, n)
                    self.active[r.conn_id] = Connection(r.conn_id, r.src, r.dst, r.bitrate,
                                                        list(pi.links), m, start, n)
                    heapq.heappush(self.dep_heap, (r.time + r.holding, r.conn_id))
                    allocated, n_used, mod = True, n, m
            if not allocated:
                invalid = True
        f_after = self.db.fragmentation()
        if allocated:
            reward = (c.w_success + c.w_efficiency * r.bitrate / (n_used * SLOT_GHZ)
                      - c.w_frag * max(f_after - f_before, 0.0))
        else:
            reward = -c.w_success
        out = {"allocated": allocated, "blocked": not allocated, "invalid_action": invalid,
               "bitrate": r.bitrate, "num_slots": n_used, "mod": mod,
               "frag": f_after, "src": r.src, "dst": r.dst, "time": r.time}
        self._advance()
        return reward, out

    # ------------------------------------------------------------------ #
    # QoT re-check of established lightpaths (metric only)
    # ------------------------------------------------------------------ #
    def qot_violation_fraction(self) -> float:
        """Fraction of active lightpaths whose SNR, re-evaluated with the CURRENT
        occupancy, is below their format's threshold (lightpaths are not torn
        down; this measures the effect of later allocations on earlier ones)."""
        if self.cfg.phy_model != "gn_v2" or not self.active:
            return float("nan") if self.cfg.phy_model != "gn_v2" else 0.0
        bad = 0
        for conn in self.active.values():
            km = [self.link_km[l] for l in conn.links]
            occ = [int(self.db.occ[l]) for l in conn.links]
            if not self.phy.qot_ok(self.phy.snr_db(km, occ, 0), conn.mod):
                bad += 1
        return bad / len(self.active)
