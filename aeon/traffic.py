"""
Streaming dynamic traffic.

Poisson arrivals with rate lambda = load / mean_holding (load in Erlang),
exponential holding times (mean 1 time unit), uniform source-destination pairs
(or a hotspot pattern), and bit-rates drawn uniformly from BITRATES.

Arrivals are produced on demand from a dedicated RNG, and every arrival draws
the same random numbers regardless of how the previous request was served, so
for a given seed every method faces exactly the same request sequence. The
offered load can be changed between arrivals (load curriculum) without
resetting the network.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

BITRATES = (40, 100, 200, 400, 800)   # Gb/s, uniform

HOTSPOT_HUBS = {
    "NSFNET": (0, 5, 12), "USNET": (0, 3, 8, 17), "EUROPEAN": (2, 10, 15, 22),
}


@dataclass
class Arrival:
    time: float
    conn_id: int
    src: int
    dst: int
    bitrate: float
    holding: float


class TrafficStream:
    def __init__(self, num_nodes: int, load_erlang: float, seed: int,
                 mean_holding: float = 1.0, bitrates: Sequence[float] = BITRATES,
                 pattern: str = "uniform", hubs: Optional[Sequence[int]] = None,
                 p_hot: float = 0.6):
        self.N = num_nodes
        self.load = float(load_erlang)
        self.mean_holding = float(mean_holding)
        self.bitrates = tuple(bitrates)
        self.pattern = pattern
        self.hubs = tuple(hubs) if hubs is not None else ()
        self.p_hot = p_hot
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.next_id = 0

    def set_load(self, load_erlang: float):
        self.load = float(load_erlang)

    def _pair(self):
        # Always consume the same number of draws per arrival (fixed stream).
        u_hot, u_side, u_hub, u_a, u_b = self.rng.random(5)
        if self.pattern == "hotspot" and self.hubs and u_hot < self.p_hot:
            hub = self.hubs[int(u_hub * len(self.hubs))]
            other = int(u_a * (self.N - 1))
            if other >= hub:
                other += 1
            return (hub, other) if u_side < 0.5 else (other, hub)
        src = int(u_a * self.N)
        dst = int(u_b * (self.N - 1))
        if dst >= src:
            dst += 1
        return src, dst

    def next(self) -> Arrival:
        iat = self.rng.exponential(self.mean_holding / self.load)
        holding = self.rng.exponential(self.mean_holding)
        src, dst = self._pair()
        b = float(self.bitrates[int(self.rng.integers(len(self.bitrates)))])
        self.t += iat
        a = Arrival(self.t, self.next_id, src, dst, b, holding)
        self.next_id += 1
        return a
