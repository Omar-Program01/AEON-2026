"""
Training and evaluation on the distributed environment.

Kept separate from rollout.py (centralised environment). Differences from rollout.py:

* a masked (view-feasible) action may FAIL at reservation (setup failure);
* the anchor action (KSP-AM-FF on the view) is passed to anchored learners and
  stored with each successor state for the bootstrap target;
* AnchoredQMIX learns on link-state update windows (see policies/anchored.py);
* reward transform "center": r' = (r - r_bar) / w_s with r_bar an exponential
  running mean of the observed rewards (reward centering, Naik et al. 2024).

Discounting is per arrival event, as in rollout.py. Forced blocks (no feasible
action in the view) add their discounted reward to the open transition.
"""
import math
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import TrainConfig, EvalConfig, train_stream_seed
from .rollout import seed_everything, curriculum_load, epsilon


@dataclass
class P3TrainConfig(TrainConfig):
    anchor_margin: float = 0.0          # delta in the anchored greedy rule
    window_max_decisions: int = 32      # A-QMIX: a window with more decisions is split
    center_rate: float = 1e-3           # step size of the running mean for reward centering
    dqot_reward: str = "heuristic"      # DeepRMSA-QoT reward: "basic" | "heuristic" | "qot" (Asiri & Wang 2025)


class RewardShaper:
    def __init__(self, tcfg: TrainConfig, w_success: float):
        self.mode, self.w = tcfg.reward_transform, w_success
        self.rate = getattr(tcfg, "center_rate", 1e-3)
        self.rbar = 0.0

    def __call__(self, r: float) -> float:
        if self.mode == "affine":
            return (r - self.w) / self.w
        if self.mode == "center":
            self.rbar += self.rate * (r - self.rbar)
            return (r - self.rbar) / self.w
        return r


def train_dist(policy, env, tcfg: TrainConfig, train_seed: int) -> List[Dict]:
    torch.set_num_threads(tcfg.torch_threads)
    seed_everything(train_seed)
    env.reset(train_stream_seed(train_seed), curriculum_load(0.0, tcfg))
    g = tcfg.gamma
    shape = RewardShaper(tcfg, env.cfg.w_success)
    windowed = getattr(policy, "window_based", False)
    D = getattr(policy, "D", 1)
    logs: List[Dict] = []
    seg = dict(arrivals=0, blocked=0, setup_fail=0, decisions=0, reward=0.0, losses=[])
    t0 = time.perf_counter()
    pending = None            # A-Independent: open decision transition
    win = None                # A-QMIX: window being filled
    prev = None               # A-QMIX: closed window waiting for its successor
    n_dec = 0

    def learn_tick():
        nonlocal n_dec
        n_dec += 1
        if n_dec % tcfg.learn_every == 0:
            loss = policy.learn()
            if loss is not None:
                seg["losses"].append(loss)

    for t in range(tcfg.train_events):
        if t % tcfg.segment_events == 0:
            frac = t / tcfg.train_events
            if t > 0:
                logs.append(_row(seg, t, env, policy, t0))
                seg = dict(arrivals=0, blocked=0, setup_fail=0, decisions=0, reward=0.0, losses=[])
            env.set_load(curriculum_load(frac, tcfg))
            policy.set_epsilon(epsilon(frac, tcfg))
        ctx = env.context()
        decide = bool(ctx.mask.any())
        if windowed and decide:
            new_window = win is None or ctx.window != win["id"] or len(win["dec"]) >= D
            if new_window:
                if win is not None:
                    win["next_stub"] = True
                    if prev is not None:
                        prev_w, prev_g, prev_d = prev
                        policy.observe_window(prev_w, win, prev_g, prev_d)
                    prev = (win, win["g"], win["disc"])
                win = {"id": ctx.window, "dec": [], "next": [], "s": ctx.global_state.copy(),
                       "g": 0.0, "disc": 1.0}
            win["next"].append((ctx.obs, ctx.mask, getattr(ctx, "anchor", None), ctx.src))
        if not windowed and decide and pending is not None:
            p = pending
            policy.observe_decision(p["o"], p["a"], p["g"], p["d"], ctx.obs, ctx.mask,
                                    getattr(ctx, "anchor", None))
            learn_tick()
        if decide:
            a = policy.act(ctx, greedy=False)
            r, info = env.step(a, policy.slot_strategy)
            seg["decisions"] += 1
        else:
            a = None
            r, info = env.step(None)
        lr = shape(r)
        if windowed:
            if win is not None:
                if decide:
                    win["dec"].append((ctx.obs, a, ctx.src))
                    learn_tick()
                win["g"] += win["disc"] * lr
                win["disc"] *= g
        else:
            if decide:
                pending = {"o": ctx.obs, "a": a, "g": lr, "d": g}
            elif pending is not None:
                pending["g"] += pending["d"] * lr
                pending["d"] *= g
        seg["arrivals"] += 1
        seg["blocked"] += int(info["blocked"])
        seg["setup_fail"] += int(info.get("setup_fail", False))
        seg["reward"] += r
    logs.append(_row(seg, tcfg.train_events, env, policy, t0))
    return logs


def _row(seg, t, env, policy, t0):
    n = max(seg["arrivals"], 1)
    return {"arrivals_done": t, "load": round(env.stream.load, 3), "eps": round(getattr(policy, "eps", 0.0), 4),
            "train_bp": seg["blocked"] / n, "train_setup_fail": seg["setup_fail"] / n,
            "reward_per_arrival": seg["reward"] / n, "decisions": seg["decisions"],
            "loss": float(np.mean(seg["losses"])) if seg["losses"] else float("nan"),
            "elapsed_s": round(time.perf_counter() - t0, 2)}


@torch.no_grad()
def evaluate_dist(policy, env, load: float, eval_seed: int, ecfg: EvalConfig,
                  slot_strategy: Optional[str] = None, batch: int = 1000) -> Dict:
    strat = slot_strategy or policy.slot_strategy
    env.reset(eval_seed, load)
    warm = int(math.ceil(ecfg.warmup_holding_times * load / env.cfg.mean_holding))
    n = blocked = setup_fail = view_block = deviations = decisions = 0
    off_bw = blk_bw = 0.0
    se_sum, se_n = 0.0, 0
    frag_sum = util_sum = 0.0
    batches, b_blk, b_n = [], 0, 0
    for i in range(warm + ecfg.measure_events):
        ctx = env.context()
        a = policy.act(ctx, greedy=True) if ctx.mask.any() else None
        anchor = getattr(ctx, "anchor", None)
        r, info = env.step(a, strat)
        if i < warm:
            continue
        n += 1
        if a is not None:
            decisions += 1
            deviations += int(anchor is not None and a != anchor)
        off_bw += info["bitrate"]
        if info["blocked"]:
            blocked += 1
            blk_bw += info["bitrate"]
            setup_fail += int(info.get("setup_fail", False))
            view_block += int(info.get("view_block", False))
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
    return {"load": load, "eval_seed": eval_seed, "n_warmup": warm, "n_measured": n,
            "bp": blocked / n, "bbp": blk_bw / off_bw, "setup_fail": setup_fail / n,
            "view_block": view_block / n, "se": se_sum / max(se_n, 1), "frag": frag_sum / n,
            "util": util_sum / n, "deviation_frac": deviations / max(decisions, 1),
            "updates": env.counters["updates"], "batch_bp": batches}
