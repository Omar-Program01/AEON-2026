"""
Value-based learners (Double-DQN targets, Huber loss, soft target updates).

* Independent : ONE Q-network shared by all source nodes (the source and
                destination are one-hot inputs). At each decision epoch the
                source node evaluates it on its own observation.
* QMIX        : the same shared Q-network plus a monotonic mixing network whose
                weights are produced by hypernetworks from the global state s_t.
                Because exactly one source node acts per decision epoch,
                Q_tot(s_t, a_t) = f_s(Q_i(o_t, a_t)) with f_s monotone
                increasing, so argmax_a Q_tot = argmax_a Q_i: the mixer cannot
                change any decision; it only reshapes the TD target with
                global-state information during training. It is discarded at
                execution.
* CentralDQN  : one Q-network over [o_t ; s_t] (per-request features plus the
                global state).
"""
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RMSAPolicy, StepContext, Transition, masked_argmax

DEVICE = torch.device("cpu")


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Dueling(nn.Module):
    """obs -> h -> h -> (V, A);  Q = V + A - mean_a A  (Wang et al., 2016)."""
    def __init__(self, sizes):
        super().__init__()
        d_in, h1, h2, d_out = sizes
        self.body = nn.Sequential(nn.Linear(d_in, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        self.v = nn.Linear(h2, 1)
        self.a = nn.Linear(h2, d_out)

    def forward(self, x):
        z = self.body(x)
        adv = self.a(z)
        return self.v(z) + adv - adv.mean(dim=1, keepdim=True)


def qnet(sizes, dueling: bool):
    return Dueling(sizes) if dueling else mlp(sizes)


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def soft_update(target, source, tau):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)


class QMixer(nn.Module):
    """Monotonic mixer for a single active agent: Q_tot = w2(s)^T ELU(w1(s) Q_i + b1(s)) + b2(s),
    with w1, w2 >= 0 (absolute value of hypernetwork outputs)."""
    def __init__(self, state_dim, embed=32, hyper=64):
        super().__init__()
        self.hw1 = nn.Sequential(nn.Linear(state_dim, hyper), nn.ReLU(), nn.Linear(hyper, embed))
        self.hw2 = nn.Sequential(nn.Linear(state_dim, hyper), nn.ReLU(), nn.Linear(hyper, embed))
        self.hb1 = nn.Linear(state_dim, embed)
        self.hb2 = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, q_i, s):
        w1 = torch.abs(self.hw1(s))
        h = F.elu(q_i * w1 + self.hb1(s))
        w2 = torch.abs(self.hw2(s))
        return (h * w2).sum(1, keepdim=True) + self.hb2(s)


class Replay:
    def __init__(self, cap, obs_dim, state_dim, act_dim):
        self.cap, self.i, self.n = cap, 0, 0
        f = np.float32
        self.o = np.zeros((cap, obs_dim), f); self.no = np.zeros((cap, obs_dim), f)
        self.s = np.zeros((cap, state_dim), f); self.ns = np.zeros((cap, state_dim), f)
        self.a = np.zeros(cap, np.int64); self.g = np.zeros(cap, f); self.d = np.zeros(cap, f)
        self.nm = np.zeros((cap, act_dim), np.bool_)

    def add(self, tr: Transition):
        j = self.i
        self.o[j] = tr.obs; self.no[j] = tr.next_obs
        self.s[j] = tr.global_state; self.ns[j] = tr.next_global_state
        self.a[j] = tr.action; self.g[j] = tr.ret; self.d[j] = tr.discount
        self.nm[j] = tr.next_mask
        self.i = (j + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        ix = np.random.randint(0, self.n, size=b)
        t = lambda x: torch.as_tensor(x[ix], device=DEVICE)
        return t(self.o), t(self.s), t(self.a), t(self.g), t(self.d), t(self.no), t(self.ns), t(self.nm)


class _DQN(RMSAPolicy):
    trainable = True
    uses_global_input = False

    def __init__(self, obs_dim, state_dim, act_dim, cfg):
        self.cfg = cfg
        self.obs_dim, self.state_dim, self.act_dim = obs_dim, state_dim, act_dim
        self.eps = 1.0
        self.buf = Replay(cfg.replay_capacity, obs_dim, state_dim, act_dim)
        self._build()
        self.opt = torch.optim.Adam(self._train_params(), lr=cfg.lr)

    def _inp(self, obs, s):
        return obs

    def set_epsilon(self, eps):
        self.eps = eps

    @torch.no_grad()
    def act(self, ctx: StepContext, greedy=False) -> Optional[int]:
        if not ctx.mask.any():
            return None
        if not greedy and random.random() < self.eps:
            return int(np.random.choice(np.flatnonzero(ctx.mask)))
        x = self._inp(torch.as_tensor(ctx.obs, device=DEVICE).unsqueeze(0),
                      torch.as_tensor(ctx.global_state, device=DEVICE).unsqueeze(0))
        return masked_argmax(self.q(x)[0].numpy(), ctx.mask)

    def observe(self, tr: Transition):
        self.buf.add(tr)

    def _next_value(self, no, ns, nm, x_next):
        q_on = self.q(x_next).masked_fill(~nm, -1e9)
        a_star = q_on.argmax(1, keepdim=True)
        return self.qt(x_next).gather(1, a_star)          # (B,1)

    def learn(self):
        if self.buf.n < max(self.cfg.batch_size, self.cfg.learn_start):
            return None
        o, s, a, g, d, no, ns, nm = self.buf.sample(self.cfg.batch_size)
        x, xn = self._inp(o, s), self._inp(no, ns)
        q_sa = self.q(x).gather(1, a.unsqueeze(1))
        pred = self._mix(q_sa, s, target=False).squeeze(1)
        with torch.no_grad():
            nv = self._next_value(no, ns, nm, xn)
            y = g + d * self._mix(nv, ns, target=True).squeeze(1)
        loss = F.smooth_l1_loss(pred, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._train_params(), self.cfg.grad_clip)
        self.opt.step()
        self._soft()
        return float(loss.item())

    # hooks
    def _mix(self, q, s, target):
        return q

    def _soft(self):
        soft_update(self.qt, self.q, self.cfg.tau)

    def _train_params(self):
        return list(self.q.parameters())

    def n_parameters(self):
        return n_params(self.q)            # parameters used at execution

    def n_training_parameters(self):
        return sum(p.numel() for p in self._train_params())


class Independent(_DQN):
    name = "Independent"

    def _build(self):
        h = self.cfg.hidden
        du = getattr(self.cfg, "dueling", False)
        self.q = qnet([self.obs_dim, h, h, self.act_dim], du)
        self.qt = qnet([self.obs_dim, h, h, self.act_dim], du)
        self.qt.load_state_dict(self.q.state_dict())


class QMIX(Independent):
    name = "QMIX"

    def _build(self):
        super()._build()
        c = self.cfg
        self.mixer = QMixer(self.state_dim, c.mixer_embed, c.hyper_hidden)
        self.mixer_t = QMixer(self.state_dim, c.mixer_embed, c.hyper_hidden)
        self.mixer_t.load_state_dict(self.mixer.state_dict())

    def _mix(self, q, s, target):
        return (self.mixer_t if target else self.mixer)(q, s)

    def _soft(self):
        soft_update(self.qt, self.q, self.cfg.tau)
        soft_update(self.mixer_t, self.mixer, self.cfg.tau)

    def _train_params(self):
        return list(self.q.parameters()) + list(self.mixer.parameters())


class CentralDQN(_DQN):
    name = "Centralized-DQN"
    uses_global_input = True

    def _inp(self, obs, s):
        return torch.cat([obs, s], dim=1)

    def _build(self):
        h = self.cfg.hidden_central
        d = self.obs_dim + self.state_dim
        du = getattr(self.cfg, "dueling", False)
        self.q = qnet([d, h, h, self.act_dim], du)
        self.qt = qnet([d, h, h, self.act_dim], du)
        self.qt.load_state_dict(self.q.state_dict())
