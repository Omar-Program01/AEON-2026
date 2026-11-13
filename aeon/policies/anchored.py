"""
Heuristic-anchored value learners for distributed RMSA.

Anchor. In every decision the environment supplies the anchor action a_h:
KSP-AM-FF computed on the source node's view (fewest-slot format, ties ->
shortest path, first fitting block).

Policy (discrete residual / "default action with learned deviation").
Q(o, a) = V(o) + A(o, a) - mean_a A(o, a)  (dueling head, Wang et al. 2016) with
the advantage output layer initialised to ZERO, so at initialisation all
feasible actions have equal value and the tie is broken in favour of a_h: the
untrained policy IS the heuristic. The greedy policy deviates from the anchor
only when the learned advantage exceeds a margin delta >= 0:

    pi(o) = a*  if  Q(o, a*) - Q(o, a_h) > delta,   a* = argmax_{feasible a} Q(o, a)
            a_h otherwise.

This is the discrete analogue of residual policy learning (Silver et al. 2018:
learn a correction on top of a good base controller) with a conservative
deviation constraint relative to the base action (cf. constrained residual RL,
Staessens et al. 2021). The bootstrap target evaluates the same anchored
greedy policy (Double-DQN style: online network selects, target network
evaluates).

* AnchoredIndependent : one shared Q-network evaluated by each source node on its
  own observation; one transition per decision.
* AnchoredQMIX        : the same shared Q-network plus a monotonic mixer (Rashid
  et al. 2018) over the N source nodes. A TRANSITION IS A LINK-STATE UPDATE WINDOW:
  all decisions taken between two updates act on the same stale information, so
  their joint outcome is a cooperative multi-agent problem. Per-node utilities
  are the sums of the Q-values of that node's decisions in the window; the mixer
  (hypernetworks on the true global state at the window start; centralised
  training, decentralised execution) combines the N utilities. With fresh state
  (tau = 0) every window holds one decision.
"""
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RMSAPolicy, StepContext
from .value import soft_update, n_params

DEVICE = torch.device("cpu")


class AnchoredDueling(nn.Module):
    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.v = nn.Linear(hidden, 1)
        self.a = nn.Linear(hidden, d_out)
        nn.init.zeros_(self.a.weight)
        nn.init.zeros_(self.a.bias)

    def forward(self, x):
        z = self.body(x)
        adv = self.a(z)
        return self.v(z) + adv - adv.mean(dim=-1, keepdim=True)


def anchored_greedy_np(q: np.ndarray, mask: np.ndarray, anchor: Optional[int], delta: float) -> Optional[int]:
    if not mask.any():
        return None
    qm = np.where(mask, q, -np.inf)
    a_star = int(np.argmax(qm))
    if anchor is None or not mask[anchor]:
        return a_star
    return a_star if qm[a_star] - qm[anchor] > delta else int(anchor)


def anchored_greedy_t(q: torch.Tensor, mask: torch.Tensor, anchor: torch.Tensor, delta: float) -> torch.Tensor:
    """Batched anchored greedy action. q, mask: (B, A); anchor: (B,) with -1 = no anchor."""
    qm = q.masked_fill(~mask, -1e9)
    a_star = qm.argmax(-1)
    has = anchor >= 0
    anc = anchor.clamp(min=0)
    q_star = qm.gather(-1, a_star.unsqueeze(-1)).squeeze(-1)
    q_anc = qm.gather(-1, anc.unsqueeze(-1)).squeeze(-1)
    keep = has & (q_star - q_anc <= delta)
    return torch.where(keep, anc, a_star)


class _AnchoredBase(RMSAPolicy):
    trainable = True
    slot_strategy = "jblock"
    anchored = True           # False: anchors are ignored (ablation)

    def __init__(self, obs_dim, state_dim, act_dim, n_agents, cfg):
        self.cfg = cfg
        self.obs_dim, self.state_dim, self.act_dim, self.N = obs_dim, state_dim, act_dim, n_agents
        self.delta = float(getattr(cfg, "anchor_margin", 0.0))
        self.eps = 1.0
        self.q = AnchoredDueling(obs_dim, cfg.hidden, act_dim)
        self.qt = AnchoredDueling(obs_dim, cfg.hidden, act_dim)
        self.qt.load_state_dict(self.q.state_dict())
        self._build_extra()
        self.opt = torch.optim.Adam(self._train_params(), lr=cfg.lr)

    def _build_extra(self):
        pass

    def _train_params(self):
        return list(self.q.parameters())

    def set_epsilon(self, eps):
        self.eps = eps

    @torch.no_grad()
    def act(self, ctx: StepContext, greedy=False) -> Optional[int]:
        if not ctx.mask.any():
            return None
        if not greedy and random.random() < self.eps:
            return int(np.random.choice(np.flatnonzero(ctx.mask)))
        q = self.q(torch.as_tensor(ctx.obs, device=DEVICE).unsqueeze(0))[0].numpy()
        anchor = getattr(ctx, "anchor", None) if self.anchored else None
        return anchored_greedy_np(q, ctx.mask, anchor, self.delta)

    def n_parameters(self):
        return n_params(self.q)

    def n_training_parameters(self):
        return sum(p.numel() for p in self._train_params())


# ---------------------------------------------------------------------- #
class DecisionReplay:
    def __init__(self, cap, obs_dim, act_dim):
        f = np.float32
        self.cap, self.i, self.n = cap, 0, 0
        self.o = np.zeros((cap, obs_dim), f); self.no = np.zeros((cap, obs_dim), f)
        self.a = np.zeros(cap, np.int64); self.g = np.zeros(cap, f); self.d = np.zeros(cap, f)
        self.nm = np.zeros((cap, act_dim), np.bool_); self.na = np.zeros(cap, np.int64)

    def add(self, o, a, g, d, no, nm, na):
        j = self.i
        self.o[j] = o; self.a[j] = a; self.g[j] = g; self.d[j] = d
        self.no[j] = no; self.nm[j] = nm; self.na[j] = -1 if na is None else na
        self.i = (j + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        ix = np.random.randint(0, self.n, size=b)
        t = lambda x: torch.as_tensor(x[ix], device=DEVICE)
        return t(self.o), t(self.a), t(self.g), t(self.d), t(self.no), t(self.nm), t(self.na)


class AnchoredIndependent(_AnchoredBase):
    name = "A-Independent"
    window_based = False

    def _build_extra(self):
        self.buf = DecisionReplay(self.cfg.replay_capacity, self.obs_dim, self.act_dim)

    def observe_decision(self, o, a, g, d, no, nm, na):
        self.buf.add(o, a, g, d, no, nm, na)

    def learn(self):
        c = self.cfg
        if self.buf.n < max(c.batch_size, c.learn_start):
            return None
        o, a, g, d, no, nm, na = self.buf.sample(c.batch_size)
        pred = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_next = anchored_greedy_t(self.q(no), nm, na, self.delta)
            y = g + d * self.qt(no).gather(1, a_next.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(pred, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._train_params(), c.grad_clip)
        self.opt.step()
        soft_update(self.qt, self.q, c.tau)
        return float(loss.item())


# ---------------------------------------------------------------------- #
class NMixer(nn.Module):
    """QMIX mixer over N agents (Rashid et al. 2018): Q_tot = w2(s)^T ELU(W1(s)^T q + b1(s)) + b2(s),
    with W1, w2 >= 0 so that dQ_tot/dq_i >= 0 for every agent i."""
    def __init__(self, n_agents, state_dim, embed=32, hyper=64):
        super().__init__()
        self.N, self.E = n_agents, embed
        self.hw1 = nn.Sequential(nn.Linear(state_dim, hyper), nn.ReLU(), nn.Linear(hyper, n_agents * embed))
        self.hb1 = nn.Linear(state_dim, embed)
        self.hw2 = nn.Sequential(nn.Linear(state_dim, hyper), nn.ReLU(), nn.Linear(hyper, embed))
        self.hb2 = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, q_agents, s):                     # q_agents: (B, N), s: (B, S)
        B = q_agents.shape[0]
        w1 = torch.abs(self.hw1(s)).view(B, self.N, self.E)
        h = F.elu(torch.bmm(q_agents.unsqueeze(1), w1).squeeze(1) + self.hb1(s))
        w2 = torch.abs(self.hw2(s))
        return (h * w2).sum(1) + self.hb2(s).squeeze(1)  # (B,)


class WindowReplay:
    """Each entry: one window (<= D decisions) and its successor window."""
    def __init__(self, cap, D, obs_dim, act_dim, state_dim):
        f = np.float32
        self.cap, self.D, self.i, self.n = cap, D, 0, 0
        self.o = np.zeros((cap, D, obs_dim), f); self.a = np.zeros((cap, D), np.int64)
        self.nd = np.zeros((cap, D), np.int64); self.v = np.zeros((cap, D), np.bool_)
        self.s = np.zeros((cap, state_dim), f)
        self.no = np.zeros((cap, D, obs_dim), f); self.nm = np.zeros((cap, D, act_dim), np.bool_)
        self.na = np.zeros((cap, D), np.int64); self.nnd = np.zeros((cap, D), np.int64)
        self.nv = np.zeros((cap, D), np.bool_); self.ns = np.zeros((cap, state_dim), f)
        self.g = np.zeros(cap, f); self.d = np.zeros(cap, f)

    def add(self, w, nw, g, d):
        j = self.i
        for arr in (self.v, self.nv):
            arr[j] = False
        for i, (o, a, node) in enumerate(w["dec"]):
            self.o[j, i] = o; self.a[j, i] = a; self.nd[j, i] = node; self.v[j, i] = True
        self.s[j] = w["s"]
        for i, (o, m, anc, node) in enumerate(nw["next"]):
            self.no[j, i] = o; self.nm[j, i] = m; self.na[j, i] = -1 if anc is None else anc
            self.nnd[j, i] = node; self.nv[j, i] = True
        self.ns[j] = nw["s"]
        self.g[j] = g; self.d[j] = d
        self.i = (j + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        ix = np.random.randint(0, self.n, size=b)
        t = lambda x: torch.as_tensor(x[ix], device=DEVICE)
        return (t(self.o), t(self.a), t(self.nd), t(self.v), t(self.s), t(self.no), t(self.nm),
                t(self.na), t(self.nnd), t(self.nv), t(self.ns), t(self.g), t(self.d))


class AnchoredQMIX(_AnchoredBase):
    name = "A-QMIX"
    window_based = True

    def _build_extra(self):
        c = self.cfg
        self.D = int(getattr(c, "window_max_decisions", 32))
        self.mixer = NMixer(self.N, self.state_dim, c.mixer_embed, c.hyper_hidden)
        self.mixer_t = NMixer(self.N, self.state_dim, c.mixer_embed, c.hyper_hidden)
        self.mixer_t.load_state_dict(self.mixer.state_dict())
        cap = max(1000, c.replay_capacity // 4)
        self.buf = WindowReplay(cap, self.D, self.obs_dim, self.act_dim, self.state_dim)

    def _train_params(self):
        return list(self.q.parameters()) + list(self.mixer.parameters())

    def observe_window(self, w, nw, g, d):
        self.buf.add(w, nw, g, d)

    def _node_sum(self, qd, nodes, valid):          # qd: (B, D) -> (B, N)
        qd = qd * valid.float()
        out = torch.zeros(qd.shape[0], self.N, device=DEVICE)
        return out.scatter_add(1, nodes, qd)

    def learn(self):
        c = self.cfg
        if self.buf.n < max(c.batch_size, c.learn_start):
            return None
        o, a, nd, v, s, no, nm, na, nnd, nv, ns, g, d = self.buf.sample(c.batch_size)
        B, D = a.shape
        q_all = self.q(o.view(B * D, -1)).view(B, D, -1)
        qd = q_all.gather(2, a.unsqueeze(2)).squeeze(2)
        pred = self.mixer(self._node_sum(qd, nd, v), s)
        with torch.no_grad():
            nm_f = nm.view(B * D, -1)
            # rows without a decision get an all-True mask so argmax is defined; they are zeroed by nv
            nm_f = torch.where(nv.view(B * D, 1), nm_f, torch.ones_like(nm_f))
            na_f = na.view(B * D) if self.anchored else torch.full_like(na.view(B * D), -1)
            a_next = anchored_greedy_t(self.q(no.view(B * D, -1)), nm_f, na_f, self.delta)
            qn = self.qt(no.view(B * D, -1)).gather(1, a_next.unsqueeze(1)).view(B, D)
            y = g + d * self.mixer_t(self._node_sum(qn, nnd, nv), ns)
        loss = F.smooth_l1_loss(pred, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._train_params(), c.grad_clip)
        self.opt.step()
        soft_update(self.qt, self.q, c.tau)
        soft_update(self.mixer_t, self.mixer, c.tau)
        return float(loss.item())


class WindowQMIX(AnchoredQMIX):
    """Ablation: the same window-level QMIX learner WITHOUT the anchor (standard
    initialisation of the advantage layer, anchors ignored in acting and in the target)."""
    name = "W-QMIX"
    anchored = False

    def _build_extra(self):
        torch.nn.init.kaiming_uniform_(self.q.a.weight, a=5 ** 0.5)
        bound = 1.0 / (self.q.a.weight.shape[1] ** 0.5)
        torch.nn.init.uniform_(self.q.a.bias, -bound, bound)
        self.qt.load_state_dict(self.q.state_dict())
        super()._build_extra()
