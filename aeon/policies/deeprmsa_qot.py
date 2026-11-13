"""
DeepRMSA-QoT (Asiri and Wang, 2025) re-implemented for the distributed environment
as an external DRL baseline.

Elements taken from the original design (section of the original article in brackets):
* State [IV-B, eq. 2]: one-hot source and destination (2N) and, for each of K candidate
  lightpaths, 5 spectrum features (x1 start index of the first available block, x2 its
  size, x3 required number of slots, x4 average size of available blocks, x5 number of
  available blocks) and 6 QoT features (x6 traffic volume, x7 modulation format, x8 total
  path length, x9 number of links, x10 longest link, x11 average link length): 2N + 11K.
* Action [IV-C]: K x J -- a candidate lightpath and the j-th first-fit block on it; the
  modulation format is chosen by the QoT unit (highest-order format whose estimated
  GSNR meets the threshold). No action masking: an infeasible choice is rejected.
* Rewards [IV-D, eqs. 3-4]: basic (+1 / -1); heuristic-based (+1 if the selected
  lightpath has the highest GSNR among the candidates, +0.9 otherwise; -1 if rejected);
  QoT-metric-based (+GSNR if accepted, -GSNR otherwise).
* Agent [IV-E, VI-A/B]: DQN, the best agent of the original study; Q-network with 5 hidden layers
  of 128 units and ELU activations; otherwise the Stable-Baselines3 DQN defaults adopted
  in the original study (learning rate 1e-4, batch 32, learning starts 100, train every 4 steps,
  hard target update every 10,000 steps, epsilon 1.0 -> 0.05 over the first 10 % of
  training, gradient-norm clip 10, Huber loss).

Differences from the original design:
* QoT unit: the original ML regressor approximates GSNR; here the simulator's GN model
  gives the GSNR directly, computed on the source node's VIEW (an error-free estimator
  of what the node knows -- this favours DeepRMSA-QoT).
* Training budget, load curriculum and continuing stream are the same as for every
  other learner (the original study trained 5 M requests in episodes of 1,000).
* Feature normalisation is not specified in the original design; features are divided by fixed
  scales (slots by S, bit-rate by 800 Gb/s, format index by M, km by 5,000, links by 10).
* J is not specified in the original design; J = 3 as for the other learners.
* Replay buffer capped at the training budget (SB3 default 1e6 exceeds it).
"""
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RMSAPolicy
from .value import n_params

DEVICE = torch.device("cpu")


def elu_mlp(d_in, d_out, hidden=128, layers=5):
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ELU()]
        d = hidden
    mods.append(nn.Linear(d, d_out))
    return nn.Sequential(*mods)


class DeepRMSAQoT(RMSAPolicy):
    name = "DeepRMSA-QoT"
    trainable = True
    slot_strategy = "jblock"

    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.K, self.J, self.M, self.N = env.K, env.J, env.M, env.N
        self.obs_dim = 2 * self.N + 11 * self.K
        self.act_dim = self.K * self.J
        self.q = elu_mlp(self.obs_dim, self.act_dim)
        self.qt = elu_mlp(self.obs_dim, self.act_dim)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=1e-4)
        self.reward_kind = getattr(cfg, "dqot_reward", "heuristic")
        self.eps = 1.0
        cap = min(1_000_000, int(cfg.train_events))
        f = np.float32
        self.bo = np.zeros((cap, self.obs_dim), f); self.bn = np.zeros((cap, self.obs_dim), f)
        self.ba = np.zeros(cap, np.int64); self.br = np.zeros(cap, f)
        self.cap, self.bi, self.bn_count = cap, 0, 0
        self.steps = 0

    # ---------------------------------------------------------------- observation & mapping
    def features(self):
        env, S = self.env, self.env.S
        r = env.req
        info = env._analyse()
        x = np.zeros(self.obs_dim, np.float32)
        x[r.src] = 1.0
        x[self.N + r.dst] = 1.0
        b = 2 * self.N
        self._fmt = []
        self._gsnr = []
        for k in range(self.K):
            pi = info[k] if k < len(info) else None
            if pi is None:
                self._fmt.append(None)
                self._gsnr.append(-np.inf)
                b += 11
                continue
            q_ok = [m for m in range(self.M) if pi.qot_ok[m]]
            m = max(q_ok) if q_ok else None        # highest-order QoT-feasible format
            self._fmt.append(m)
            n = pi.slots[m] if m is not None else S
            lens, starts = pi.lens, pi.starts
            nb = lens.size
            km = np.asarray(pi.km, dtype=float)
            x[b + 0] = (starts[0] / S) if nb else 0.0
            x[b + 1] = (lens[0] / S) if nb else 0.0
            x[b + 2] = min(n, S) / S
            x[b + 3] = (lens.mean() / S) if nb else 0.0
            x[b + 4] = nb / S
            x[b + 5] = r.bitrate / 800.0
            x[b + 6] = ((m + 1) / self.M) if m is not None else 0.0
            x[b + 7] = pi.length / 5000.0
            x[b + 8] = len(pi.links) / 10.0
            x[b + 9] = km.max() / 5000.0
            x[b + 10] = km.mean() / 5000.0
            if m is not None:
                rows = env.view_rows(r.src, pi.links)
                occ = rows.sum(axis=1).astype(float)
                snr = env.phy.snr_db_formats(env.link_spans[pi.links], occ, np.full(self.M, float(n)))
                self._gsnr.append(float(snr[m]))
            else:
                self._gsnr.append(-np.inf)
            b += 11
        return x

    def to_env_action(self, a: int, mask: np.ndarray) -> Optional[int]:
        k, j = divmod(int(a), self.J)
        m = self._fmt[k] if k < len(self._fmt) else None
        if m is None:
            return None
        ea = (k * self.M + m) * self.J + j
        return ea if mask[ea] else None

    # ---------------------------------------------------------------- acting
    def set_epsilon(self, eps):
        self.eps = eps

    @torch.no_grad()
    def choose(self, x, greedy):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.act_dim)
        return int(self.q(torch.as_tensor(x).unsqueeze(0))[0].argmax())

    def act(self, ctx, greedy=True) -> Optional[int]:
        x = self.features()
        a = self.choose(x, greedy)
        self._last = (x, a)
        return self.to_env_action(a, ctx.mask)

    # ---------------------------------------------------------------- reward and learning
    def reward(self, accepted: bool, a: int) -> float:
        k = a // self.J
        g = self._gsnr[k] if k < len(self._gsnr) and np.isfinite(self._gsnr[k]) else 0.0
        if self.reward_kind == "basic":
            return 1.0 if accepted else -1.0
        if self.reward_kind == "heuristic":
            if not accepted:
                return -1.0
            best = max(v for v in self._gsnr if np.isfinite(v))
            return 1.0 if g >= best else 0.9
        if self.reward_kind == "qot":
            return g if accepted else -g
        raise ValueError(self.reward_kind)

    def store(self, x, a, r, xn):
        j = self.bi
        self.bo[j] = x; self.ba[j] = a; self.br[j] = r; self.bn[j] = xn
        self.bi = (j + 1) % self.cap
        self.bn_count = min(self.bn_count + 1, self.cap)

    def learn(self, gamma):
        if self.bn_count < max(32, 100):
            return None
        ix = np.random.randint(0, self.bn_count, size=32)
        o = torch.as_tensor(self.bo[ix]); a = torch.as_tensor(self.ba[ix])
        r = torch.as_tensor(self.br[ix]); on = torch.as_tensor(self.bn[ix])
        pred = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            y = r + gamma * self.qt(on).max(1).values      # continuing stream: no terminal state
        loss = F.smooth_l1_loss(pred, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    def n_parameters(self):
        return n_params(self.q)

    def n_training_parameters(self):
        return n_params(self.q)


def train_dqot(pol: DeepRMSAQoT, env, tcfg, train_seed: int):
    """SB3-style DQN loop on the continuing stream with the shared load curriculum."""
    import time
    from ..config import train_stream_seed
    from ..rollout import seed_everything, curriculum_load
    torch.set_num_threads(tcfg.torch_threads)
    seed_everything(train_seed)
    env.reset(train_stream_seed(train_seed), curriculum_load(0.0, tcfg))
    T, g = tcfg.train_events, tcfg.gamma
    logs, seg = [], dict(arrivals=0, blocked=0, invalid=0, losses=[])
    t0 = time.perf_counter()
    prev = None
    for t in range(T):
        if t % tcfg.segment_events == 0:
            if t > 0:
                n = max(seg["arrivals"], 1)
                logs.append({"arrivals_done": t, "load": round(env.stream.load, 3), "eps": round(pol.eps, 4),
                             "train_bp": seg["blocked"] / n, "invalid": seg["invalid"] / n,
                             "loss": float(np.mean(seg["losses"])) if seg["losses"] else float("nan"),
                             "elapsed_s": round(time.perf_counter() - t0, 2)})
                seg = dict(arrivals=0, blocked=0, invalid=0, losses=[])
            env.set_load(curriculum_load(t / T, tcfg))
        pol.set_epsilon(max(0.05, 1.0 - (1.0 - 0.05) * (t / T) / 0.1))
        ctx = env.context()
        x = pol.features()
        if prev is not None:
            pol.store(prev[0], prev[1], prev[2], x)
        a = pol.choose(x, greedy=False)
        ea = pol.to_env_action(a, ctx.mask)
        _, info = env.step(ea, "jblock")
        rew = pol.reward(info["allocated"], a)
        prev = (x, a, rew)
        pol.steps += 1
        if pol.steps % 4 == 0:
            loss = pol.learn(g)
            if loss is not None:
                seg["losses"].append(loss)
        if pol.steps % 10_000 == 0:
            pol.qt.load_state_dict(pol.q.state_dict())
        seg["arrivals"] += 1
        seg["blocked"] += int(info["blocked"])
        seg["invalid"] += int(ea is None)
    n = max(seg["arrivals"], 1)
    logs.append({"arrivals_done": T, "load": round(env.stream.load, 3), "eps": round(pol.eps, 4),
                 "train_bp": seg["blocked"] / n, "invalid": seg["invalid"] / n,
                 "loss": float(np.mean(seg["losses"])) if seg["losses"] else float("nan"),
                 "elapsed_s": round(time.perf_counter() - t0, 2)})
    return logs
