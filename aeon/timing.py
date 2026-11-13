"""
Reproducible decision-latency harness, run in the same configuration as the
performance results.

Two components are timed separately at every measured arrival:
  * env_ms    : building the request context (path analysis, QoT checks,
                feasibility mask, observation, global state) -- common to all
                methods that use the same inputs;
  * decide_ms : the policy's decision given the context (policy.act, greedy).
Procedure: fresh network at the given load, warm-up arrivals untimed, then
`n_measure` timed arrivals, repeated `reps` times with different traffic seeds.
Single process; torch threads fixed. Reports mean, median, p95 per component.
"""
import math
import time

import numpy as np
import torch

from .explog import platform_info


@torch.no_grad()
def time_decisions(policy, env, load, seeds=(910_000, 910_001, 910_002), warm_holding=5.0,
                   n_measure=2000, threads=1):
    torch.set_num_threads(threads)
    env_t, dec_t = [], []
    for s in seeds:
        env.reset(s, load)
        warm = int(math.ceil(warm_holding * load / env.cfg.mean_holding))
        for i in range(warm + n_measure):
            t0 = time.perf_counter()
            ctx = env.context()
            t1 = time.perf_counter()
            a = policy.act(ctx, greedy=True) if ctx.mask.any() else None
            t2 = time.perf_counter()
            env.step(a, policy.slot_strategy)
            if i >= warm:
                env_t.append((t1 - t0) * 1e3)
                if a is not None:
                    dec_t.append((t2 - t1) * 1e3)

    def summ(x):
        x = np.asarray(x)
        return {"mean": float(x.mean()), "median": float(np.median(x)),
                "p95": float(np.percentile(x, 95)), "n": int(x.size)}

    return {"env_ms": summ(env_t), "decide_ms": summ(dec_t), "load": load,
            "seeds": list(seeds), "n_measure_per_seed": n_measure,
            "threads": threads, "platform": platform_info()}
