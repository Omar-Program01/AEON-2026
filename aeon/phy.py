"""
Physical layer: quality-of-transmission (QoT) models.

The model is implemented by ``PhyGN``:

* ``PhyGN`` is a standard Gaussian-Noise (GN) model:
    - constant launch power spectral density G_tx (power per Hz) for every
      lightpath, so the signal-to-noise ratio (SNR) in the signal bandwidth does
      not depend on how many slots a lightpath occupies;
    - EDFA amplified spontaneous emission with a standard noise figure (5 dB);
    - nonlinear interference (NLI) from the closed-form incoherent GN model
      (Poggiolini 2012, Eq. for a uniformly filled spectrum), evaluated per span
      with the occupied bandwidth OF THAT LINK (per-link NLI);
    - incoherent accumulation over spans and links;
    - acceptance rule: SNR(path, state) >= SNR_req(m) + margin, where SNR_req(m)
      is the Es/N0 that gives the pre-FEC BER target for format m (closed-form
      BER expressions). There is no separate reach table: reach is a derived
      quantity, reported for information by ``derived_reach_km``.

Slots per lightpath: n(b, m) = ceil(b / (eta_m * 12.5 GHz)) + guard_slots,
with effective spectral efficiency eta_m = log2(M) b/s/Hz (same convention as
DeepRMSA) and one guard slot by default.
"""
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

H_PLANCK = 6.62607015e-34
SLOT_HZ = 12.5e9

# name, bits/symbol (eta), constellation size
FORMATS = [
    ("BPSK", 1, 2),
    ("QPSK", 2, 4),
    ("8-QAM", 3, 8),
    ("16-QAM", 4, 16),
    ("32-QAM", 5, 32),
    ("64-QAM", 6, 64),
]
NUM_MODS = len(FORMATS)


# --------------------------------------------------------------------------- #
# BER <-> SNR (Es/N0) closed forms (Proakis, Digital Communications, 5th ed.)
# --------------------------------------------------------------------------- #
def _qfunc(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def ber_from_snr(snr_lin: float, mod_idx: int) -> float:
    """Pre-FEC bit error rate for Gray-coded format `mod_idx` at Es/N0 = snr_lin.

    BPSK: Q(sqrt(2 Es/N0)); QPSK: Q(sqrt(Es/N0)); M-QAM (M>=8): the standard
    nearest-neighbour approximation (4/log2 M)(1-1/sqrt M) Q(sqrt(3 Es/N0/(M-1))),
    exact in the high-SNR limit for square QAM and a common approximation for the
    cross/rectangular 8- and 32-QAM constellations.
    """
    _, k, M = FORMATS[mod_idx]
    if M == 2:
        return _qfunc(math.sqrt(2.0 * snr_lin))
    if M == 4:
        return _qfunc(math.sqrt(snr_lin))
    return (4.0 / k) * (1.0 - 1.0 / math.sqrt(M)) * _qfunc(math.sqrt(3.0 * snr_lin / (M - 1)))


def required_snr_db(mod_idx: int, ber_target: float) -> float:
    """Smallest Es/N0 (dB) whose BER is <= ber_target (bisection on the closed form)."""
    lo, hi = -10.0, 40.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ber_from_snr(10 ** (mid / 10.0), mod_idx) > ber_target:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------- #
# Standard GN model
# --------------------------------------------------------------------------- #
@dataclass
class PhyGNConfig:
    alpha_db_per_km: float = 0.2          # SSMF attenuation
    beta2_abs_s2_per_km: float = 21.27e-27 * 1e3   # |beta2| = 21.27 ps^2/km
    gamma_per_w_km: float = 1.3           # nonlinear coefficient
    span_km: float = 80.0
    nf_db: float = 5.0                    # EDFA noise figure
    launch_dbm_per_ref: float = -1.0      # launch power per 37.5 GHz (sets the PSD)
    launch_ref_hz: float = 37.5e9
    center_hz: float = 193.4e12
    ber_target: float = 3.8e-3            # hard-decision FEC threshold (7 % OH)
    margin_db: float = 2.0                # system margin
    guard_slots: int = 1
    coherent_eps: float = 0.0             # 0 = incoherent GN accumulation


class PhyGN:
    name = "gn_v2"

    def __init__(self, cfg: PhyGNConfig = None):
        self.cfg = cfg or PhyGNConfig()
        c = self.cfg
        self.alpha = c.alpha_db_per_km / (10.0 * math.log10(math.e))   # power, 1/km
        self.gain = math.exp(self.alpha * c.span_km)                    # span loss compensated
        self.l_eff = (1.0 - math.exp(-self.alpha * c.span_km)) / self.alpha
        self.l_eff_a = 1.0 / self.alpha                                # asymptotic L_eff
        nf = 10 ** (c.nf_db / 10.0)
        self.g_ase_span = nf * H_PLANCK * c.center_hz * (self.gain - 1.0)   # W/Hz per span
        self.g_tx = 10 ** (c.launch_dbm_per_ref / 10.0) * 1e-3 / c.launch_ref_hz  # W/Hz
        self.eta = [f[1] for f in FORMATS]
        self.snr_req_db = [required_snr_db(m, c.ber_target) for m in range(NUM_MODS)]
        self.snr_thr_db = [s + c.margin_db for s in self.snr_req_db]
        self._nli_pref = ((8.0 / 27.0) * (c.gamma_per_w_km ** 2) * (self.g_tx ** 3)
                          * (self.l_eff ** 2) / (math.pi * c.beta2_abs_s2_per_km * self.l_eff_a))
        self._asinh_pref = (math.pi ** 2 / 2.0) * c.beta2_abs_s2_per_km * self.l_eff_a

    # -- geometry ---------------------------------------------------------- #
    def num_slots(self, bitrate_gbps: float, mod_idx: int) -> int:
        """Slots including guard band."""
        return int(math.ceil(bitrate_gbps / (self.eta[mod_idx] * 12.5))) + self.cfg.guard_slots

    def data_slots(self, bitrate_gbps: float, mod_idx: int) -> int:
        return int(math.ceil(bitrate_gbps / (self.eta[mod_idx] * 12.5)))

    def spans(self, km: float) -> int:
        return max(1, int(math.ceil(km / self.cfg.span_km)))

    # -- noise ------------------------------------------------------------- #
    def g_nli_span(self, occupied_hz: float) -> float:
        """NLI PSD (W/Hz) generated in one span carrying a uniformly filled
        spectrum of total width occupied_hz (closed-form incoherent GN)."""
        b = max(occupied_hz, SLOT_HZ)
        return self._nli_pref * math.asinh(self._asinh_pref * b * b)

    def snr_db(self, link_km: Sequence[float], link_occ_slots: Sequence[int],
               new_slots: int = 0) -> float:
        """SNR (dB) of a lightpath over links with lengths link_km whose current
        occupancies are link_occ_slots; new_slots is added to every link's
        occupancy (the candidate lightpath itself; 0 for an existing one)."""
        noise = 0.0
        for km, occ in zip(link_km, link_occ_slots):
            ns = self.spans(km)
            occ_hz = (occ + new_slots) * SLOT_HZ
            noise += (ns ** (1.0 + self.cfg.coherent_eps)) * (self.g_ase_span + self.g_nli_span(occ_hz))
        return 10.0 * math.log10(self.g_tx / noise)

    def snr_db_formats(self, spans: np.ndarray, link_occ_slots: np.ndarray,
                       new_slots: np.ndarray) -> np.ndarray:
        """Vectorised snr_db for one path and several candidate widths at once:
        spans, link_occ_slots have one entry per link; new_slots one per format.
        Identical to calling snr_db for each width (checked in tests)."""
        occ_hz = (link_occ_slots[:, None] + new_slots[None, :]) * SLOT_HZ
        occ_hz = np.maximum(occ_hz, SLOT_HZ)
        g_nli = self._nli_pref * np.arcsinh(self._asinh_pref * occ_hz * occ_hz)
        w = spans[:, None] ** (1.0 + self.cfg.coherent_eps)
        noise = (w * (self.g_ase_span + g_nli)).sum(axis=0)
        return 10.0 * np.log10(self.g_tx / noise)

    def qot_ok(self, snr_db: float, mod_idx: int) -> bool:
        return snr_db >= self.snr_thr_db[mod_idx]

    # -- reporting --------------------------------------------------------- #
    def derived_reach_km(self, mod_idx: int, fill_fraction: float = 0.5,
                         total_slots: int = 320, step_km: float = 10.0) -> float:
        """Longest single-link distance meeting the threshold when every span
        carries fill_fraction of the band (for reporting only)."""
        occ = int(round(fill_fraction * total_slots))
        km = step_km
        last_ok = 0.0
        while km <= 20000:
            if self.qot_ok(self.snr_db([km], [occ]), mod_idx):
                last_ok = km
            else:
                break
            km += step_km
        return last_ok
