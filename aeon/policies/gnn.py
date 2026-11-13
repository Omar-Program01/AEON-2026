"""
GNN-DRL baseline: a graph convolutional encoder over node features derived from
the global link-utilization state, followed by a Q head over the 30 actions
(Double-DQN target, same replay/target settings as the value learners).

Node features: [mean utilization of the node's outgoing links, is_src, is_dst,
bitrate/800]. Q head input: [z_src ; z_dst ; mean_pool(z) ; bitrate/800].

The replay buffer stores the true (src, dst, bitrate) of both the current and the
next decision epoch, so training and acting use identical features.
"""
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RMSAPolicy, StepContext, Transition, masked_argmax
from .value import mlp, n_params, soft_update


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, layers):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layers)])

    def forward(self, x, A):
        for lin in self.lins:
            x = F.relu(lin(torch.einsum("ij,bjf->bif", A, x)))
        return x


class GNNDRL(RMSAPolicy):
    name = "GNN-DRL"
    trainable = True

    def __init__(self, env, act_dim, cfg):
        self.cfg = cfg
        topo = env.topo
        self.N, self.L = topo.num_nodes, topo.num_links
        inc = np.zeros((self.N, self.L), np.float32)
        for lid, (u, v, _) in topo.link_map.items():
            inc[u, lid] = 1.0
        inc /= np.maximum(inc.sum(1, keepdims=True), 1.0)
        self.inc = torch.as_tensor(inc)
        A = np.zeros((self.N, self.N), np.float32)
        for (u, v, _) in topo.link_map.values():
            A[u, v] = A[v, u] = 1.0
        A += np.eye(self.N, dtype=np.float32)
        dinv = np.diag(1.0 / np.sqrt(A.sum(1)))
        self.A = torch.as_tensor(dinv @ A @ dinv)
        h = cfg.gnn_hidden
        self.enc, self.enc_t = GCN(4, h, cfg.gnn_layers), GCN(4, h, cfg.gnn_layers)
        self.head, self.head_t = mlp([3 * h + 1, h, act_dim]), mlp([3 * h + 1, h, act_dim])
        self.enc_t.load_state_dict(self.enc.state_dict())
        self.head_t.load_state_dict(self.head.state_dict())
        self.params = list(self.enc.parameters()) + list(self.head.parameters())
        self.opt = torch.optim.Adam(self.params, lr=cfg.lr)
        self.eps = 1.0
        cap = cfg.replay_capacity
        self._cap, self._i, self._n = cap, 0, 0
        self.B = {k: np.zeros(shape, dt) for k, shape, dt in [
            ("u", (cap, self.L), np.float32), ("nu", (cap, self.L), np.float32),
            ("sd", (cap, 2), np.int64), ("nsd", (cap, 2), np.int64),
            ("b", (cap,), np.float32), ("nb", (cap,), np.float32),
            ("a", (cap,), np.int64), ("g", (cap,), np.float32), ("d", (cap,), np.float32),
            ("nm", (cap, act_dim), np.bool_)]}

    def set_epsilon(self, eps):
        self.eps = eps

    def _q(self, util, sd, b, enc, head):
        Bn = util.shape[0]
        x = torch.zeros(Bn, self.N, 4)
        x[:, :, 0] = util @ self.inc.T
        idx = torch.arange(Bn)
        x[idx, sd[:, 0], 1] = 1.0
        x[idx, sd[:, 1], 2] = 1.0
        x[:, :, 3] = (b / 800.0).unsqueeze(1)
        z = enc(x, self.A)
        hcat = torch.cat([z[idx, sd[:, 0]], z[idx, sd[:, 1]], z.mean(1), (b / 800.0).unsqueeze(1)], 1)
        return head(hcat)

    @torch.no_grad()
    def act(self, ctx: StepContext, greedy=False) -> Optional[int]:
        if not ctx.mask.any():
            return None
        if not greedy and random.random() < self.eps:
            return int(np.random.choice(np.flatnonzero(ctx.mask)))
        util = torch.as_tensor(ctx.global_state[:self.L]).unsqueeze(0)
        sd = torch.tensor([[ctx.src, ctx.dst]])
        b = torch.tensor([float(ctx.bitrate)])
        return masked_argmax(self._q(util, sd, b, self.enc, self.head)[0].numpy(), ctx.mask)

    def observe(self, tr: Transition):
        j, B = self._i, self.B
        B["u"][j] = tr.global_state[:self.L]; B["nu"][j] = tr.next_global_state[:self.L]
        B["sd"][j] = (tr.src, tr.dst); B["nsd"][j] = (tr.next_src, tr.next_dst)
        B["b"][j] = tr.bitrate; B["nb"][j] = tr.next_bitrate
        B["a"][j] = tr.action; B["g"][j] = tr.ret; B["d"][j] = tr.discount
        B["nm"][j] = tr.next_mask
        self._i = (j + 1) % self._cap
        self._n = min(self._n + 1, self._cap)

    def learn(self):
        if self._n < max(self.cfg.batch_size, self.cfg.learn_start):
            return None
        ix = np.random.randint(0, self._n, size=self.cfg.batch_size)
        t = {k: torch.as_tensor(v[ix]) for k, v in self.B.items()}
        q = self._q(t["u"], t["sd"], t["b"], self.enc, self.head).gather(1, t["a"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            qn = self._q(t["nu"], t["nsd"], t["nb"], self.enc, self.head).masked_fill(~t["nm"], -1e9)
            a_star = qn.argmax(1, keepdim=True)
            nv = self._q(t["nu"], t["nsd"], t["nb"], self.enc_t, self.head_t).gather(1, a_star).squeeze(1)
            y = t["g"] + t["d"] * nv
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
        self.opt.step()
        soft_update(self.enc_t, self.enc, self.cfg.tau)
        soft_update(self.head_t, self.head, self.cfg.tau)
        return float(loss.item())

    def n_parameters(self):
        return n_params(self.enc) + n_params(self.head)

    n_training_parameters = n_parameters
