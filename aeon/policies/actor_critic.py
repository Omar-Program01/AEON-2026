"""
Actor-critic baselines on the per-request observation (same input as the
value-based learners), with masked categorical policies.

Both collect rollouts of `rollout_len` decision epochs and bootstrap from the
critic's value of the next decision state, using the per-transition discount
Gamma_t (see base.py):
    delta_t = G_t + Gamma_t V(o_{t+1}) - V(o_t)
    A_t     = delta_t + Gamma_t * lambda * A_{t+1}
* DeepRMSA (A2C): lambda = 1 (n-step returns), one gradient step per rollout.
* PPO          : GAE lambda = cfg.gae_lambda, clipped surrogate, cfg.ppo_epochs
                 epochs over minibatches of cfg.ppo_minibatch per rollout.
Old log-probabilities are evaluated at the start of each update with the
parameters that generated the rollout (parameters change only at updates).
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .base import RMSAPolicy, StepContext, Transition
from .value import mlp, n_params, DEVICE


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class _AC(RMSAPolicy):
    trainable = True
    lam = 1.0

    def __init__(self, obs_dim, state_dim, act_dim, cfg):
        self.cfg = cfg
        self.net = ActorCritic(obs_dim, act_dim, cfg.hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.roll = []

    @torch.no_grad()
    def act(self, ctx: StepContext, greedy=False) -> Optional[int]:
        if not ctx.mask.any():
            return None
        logits, _ = self.net(torch.as_tensor(ctx.obs).unsqueeze(0))
        logits = logits.masked_fill(~torch.as_tensor(ctx.mask).unsqueeze(0), -1e9)
        if greedy:
            return int(logits.argmax(1).item())
        return int(Categorical(logits=logits).sample().item())

    def observe(self, tr: Transition):
        self.roll.append(tr)
        if len(self.roll) >= self.cfg.rollout_len:
            self._update()
            self.roll = []

    def _batch(self):
        R = self.roll
        o = torch.as_tensor(np.stack([t.obs for t in R]))
        m = torch.as_tensor(np.stack([t.mask for t in R]))
        a = torch.as_tensor([t.action for t in R], dtype=torch.int64)
        g = torch.as_tensor([t.ret for t in R], dtype=torch.float32)
        d = torch.as_tensor([t.discount for t in R], dtype=torch.float32)
        no = torch.as_tensor(np.stack([t.next_obs for t in R]))
        with torch.no_grad():
            logits, v = self.net(o)
            _, nv = self.net(no)
            old_logp = Categorical(logits=logits.masked_fill(~m, -1e9)).log_prob(a)
            delta = g + d * nv - v
            adv = torch.zeros_like(delta)
            run = 0.0
            for t in reversed(range(len(R))):
                run = delta[t] + d[t] * self.lam * run
                adv[t] = run
            ret = adv + v
        return o, m, a, old_logp, adv, ret

    def _loss(self, o, m, a, old_logp, adv, ret, clip):
        logits, v = self.net(o)
        dist = Categorical(logits=logits.masked_fill(~m, -1e9))
        logp = dist.log_prob(a)
        if clip is None:
            pi_loss = -(logp * adv).mean()
        else:
            ratio = torch.exp(logp - old_logp)
            pi_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
        v_loss = F.mse_loss(v, ret)
        return pi_loss + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * dist.entropy().mean()

    def _step(self, loss):
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.opt.step()

    def n_parameters(self):
        return n_params(self.net)

    n_training_parameters = n_parameters


class DeepRMSA(_AC):
    name = "DeepRMSA"

    def __init__(self, *args):
        super().__init__(*args)
        self.lam = getattr(self.cfg, "a2c_lambda", 1.0)

    def _update(self):
        o, m, a, old_logp, adv, ret = self._batch()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self._step(self._loss(o, m, a, old_logp, adv, ret, clip=None))


class PPO(_AC):
    name = "PPO"

    def __init__(self, *args):
        super().__init__(*args)
        self.lam = self.cfg.gae_lambda

    def _update(self):
        o, m, a, old_logp, adv, ret = self._batch()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(self.roll)
        mb = self.cfg.ppo_minibatch
        for _ in range(self.cfg.ppo_epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, mb):
                j = torch.as_tensor(perm[i:i + mb])
                self._step(self._loss(o[j], m[j], a[j], old_logp[j], adv[j], ret[j],
                                      clip=self.cfg.ppo_clip))
