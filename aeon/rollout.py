"""
Training and evaluation procedures (identical for every method).

TRAINING (continuing stream). One traffic stream of `train_events` arrivals from
seed train_stream_seed(s); the network is never reset, so after the first few
holding times training data come from (quasi-)steady state. The offered load
follows the curriculum, exploration decays linearly, and transitions are built
per decision epoch with forced blocks accumulated into the return (base.py).

EVALUATION (steady state). For each evaluation traffic seed: empty network,
greedy policy, warm-up of `warmup_holding_times` mean holding times
(= load x holding-times arrivals), then `measure_events` measured arrivals.
"""
import math
import random
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .config import TrainConfig, EvalConfig, train_stream_seed
from .env import EONEnv
from .policies.base import Transition


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def curriculum_load(frac: float, tcfg: TrainConfig) -> float:
    if not tcfg.curriculum or frac >= tcfg.curriculum_frac:
        return tcfg.train_load
    return tcfg.curriculum_start_load + (tcfg.train_load - tcfg.curriculum_start_load) * frac / tcfg.curriculum_frac


def epsilon(frac: float, tcfg: TrainConfig) -> float:
    return max(tcfg.eps_end, tcfg.eps_start - (tcfg.eps_start - tcfg.eps_end) * frac / tcfg.eps_decay_frac)


def learning_reward(r: float, env: EONEnv, tcfg: TrainConfig) -> float:
    """Reward used for learning. "affine": r' = (r - w_s) / w_s. Every arrival yields
    exactly one reward and discounting is per arrival, so subtracting the same constant
    from every reward shifts every value by w_s/(1-gamma) and dividing by w_s > 0 scales
    them: the optimal policy is unchanged; only the numerical scale of the targets is."""
    if tcfg.reward_transform == "affine":
        w = env.cfg.w_success
        return (r - w) / w
    return r


class NStep:
    """Compose one-step decision-epoch transitions into n-step transitions:
    ret = G_0 + Gamma_0 G_1 + ... + (Gamma_0...Gamma_{n-2}) G_{n-1},
    discount = Gamma_0 ... Gamma_{n-1}, next state = next state of the last one.
    Used only for value learners (actor-critic methods use lambda-returns)."""
    def __init__(self, n):
        self.n, self.q = n, []

    def push(self, tr: Transition):
        self.q.append(tr)
        if len(self.q) < self.n:
            return None
        ret, disc = 0.0, 1.0
        for t in self.q:
            ret += disc * t.ret
            disc *= t.discount
        first, last = self.q[0], self.q[-1]
        out = Transition(first.obs, first.mask, first.global_state, first.action, ret, disc,
                         last.next_obs, last.next_mask, last.next_global_state,
                         first.src, first.dst, first.bitrate,
                         last.next_src, last.next_dst, last.next_bitrate)
        self.q.pop(0)
        return out


def _is_value_learner(policy) -> bool:
    return hasattr(policy, "buf") or hasattr(policy, "B")


def train(policy, env: EONEnv, tcfg: TrainConfig, train_seed: int,
          on_segment: Optional[Callable[[Dict], None]] = None) -> List[Dict]:
    """Train `policy` in place on one continuing stream. Returns segment logs."""
    torch.set_num_threads(tcfg.torch_threads)
    seed_everything(train_seed)
    env.reset(train_stream_seed(train_seed), curriculum_load(0.0, tcfg))
    g = tcfg.gamma
    pending = None
    closed = 0
    nstep = NStep(tcfg.n_step) if (tcfg.n_step > 1 and _is_value_learner(policy)) else None
    logs: List[Dict] = []
    seg = dict(arrivals=0, blocked=0, decisions=0, reward=0.0, losses=[])
    t0 = time.perf_counter()
    for t in range(tcfg.train_events):
        if t % tcfg.segment_events == 0:
            frac = t / tcfg.train_events
            if t > 0:
                row = _seg_row(seg, t, env.stream.load, policy, t0)
                logs.append(row)
                if on_segment:
                    on_segment(row)
                seg = dict(arrivals=0, blocked=0, decisions=0, reward=0.0, losses=[])
            env.set_load(curriculum_load(frac, tcfg))
            policy.set_epsilon(epsilon(frac, tcfg))
        ctx = env.context()
        if ctx.mask.any():
            if pending is not None:
                p = pending
                tr = Transition(
                    p["ctx"].obs, p["ctx"].mask, p["ctx"].global_state, p["a"], p["G"], p["disc"],
                    ctx.obs, ctx.mask, ctx.global_state,
                    p["ctx"].src, p["ctx"].dst, p["ctx"].bitrate, ctx.src, ctx.dst, ctx.bitrate)
                if nstep is not None:
                    tr = nstep.push(tr)
                if tr is not None:
                    policy.observe(tr)
                closed += 1
                if closed % tcfg.learn_every == 0:
                    loss = policy.learn()
                    if loss is not None:
                        seg["losses"].append(loss)
            a = policy.act(ctx, greedy=False)
            r, info = env.step(a, policy.slot_strategy)
            assert info["allocated"], "a masked-feasible action must be served (sequential model)"
            pending = {"ctx": ctx, "a": a, "G": learning_reward(r, env, tcfg), "disc": g}
            seg["decisions"] += 1
        else:
            r, info = env.step(None)
            if pending is not None:
                pending["G"] += pending["disc"] * learning_reward(r, env, tcfg)
                pending["disc"] *= g
        seg["arrivals"] += 1
        seg["blocked"] += int(info["blocked"])
        seg["reward"] += r
    # the last open transition has no successor decision epoch: discarded (no terminal state)
    row = _seg_row(seg, tcfg.train_events, env.stream.load, policy, t0)
    logs.append(row)
    if on_segment:
        on_segment(row)
    return logs


def _seg_row(seg, t, load, policy, t0):
    n = max(seg["arrivals"], 1)
    return {"arrivals_done": t, "load": round(load, 3), "eps": round(getattr(policy, "eps", 0.0), 4),
            "train_bp": seg["blocked"] / n, "reward_per_arrival": seg["reward"] / n,
            "decisions": seg["decisions"],
            "loss": float(np.mean(seg["losses"])) if seg["losses"] else float("nan"),
            "elapsed_s": round(time.perf_counter() - t0, 2)}


@torch.no_grad()
def evaluate(policy, env: EONEnv, load: float, eval_seed: int, ecfg: EvalConfig,
             batch: int = 1000) -> Dict:
    env.reset(eval_seed, load)
    warm = int(math.ceil(ecfg.warmup_holding_times * load / env.cfg.mean_holding))
    n = blocked = decisions = 0
    off_bw = blk_bw = 0.0
    se_sum, se_n = 0.0, 0
    frag_sum = util_sum = 0.0
    qot = []
    batches, b_blk, b_n = [], 0, 0
    for i in range(warm + ecfg.measure_events):
        ctx = env.context()
        a = policy.act(ctx, greedy=True) if ctx.mask.any() else None
        r, info = env.step(a, policy.slot_strategy)
        if i < warm:
            continue
        n += 1
        decisions += int(a is not None)
        off_bw += info["bitrate"]
        if info["blocked"]:
            blocked += 1
            blk_bw += info["bitrate"]
        else:
            se_sum += info["bitrate"] / (info["num_slots"] * 12.5)
            se_n += 1
        frag_sum += info["frag"]
        util_sum += float(env.db.occ.mean()) / env.S
        b_n += 1
        b_blk += int(info["blocked"])
        if b_n == batch:
            batches.append(b_blk / b_n)
            b_n = b_blk = 0
        if ecfg.qot_check_every and (n % ecfg.qot_check_every == 0):
            qot.append(env.qot_violation_fraction())
    return {"load": load, "eval_seed": eval_seed, "n_warmup": warm, "n_measured": n,
            "bp": blocked / n, "bbp": blk_bw / off_bw, "se": se_sum / max(se_n, 1),
            "frag": frag_sum / n, "util": util_sum / n, "decision_frac": decisions / n,
            "qot_violation": float(np.nanmean(qot)) if qot and not all(np.isnan(qot)) else float("nan"),
            "batch_bp": batches}
