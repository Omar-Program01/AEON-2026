"""Candidate-set sensitivity of the heuristics (exploratory): #hops ordering and larger K
(Doherty et al. 2025).

Settings: KSP-AM-FF with #hops ordering (K = 5, 10); KSP-AM-greedy with K = 10 (km and #hops).
All topologies, tau in {0, 0.05}, the 5 calibrated loads, confirmation traffic 930000-930004,
forward signalling, no retry. The km/K=5 baselines are in the heuristics and greedy state files.
  python scripts/phase3_kcheck.py CALIBRATION_JSON OUTDIR WORKERS   (resumable: kcheck_state.json)
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
CONF_EVAL = [930000, 930001, 930002, 930003, 930004]
SETTINGS = [  # label, policy, K, order
    ("FF-hops-K5", "ff", 5, "hops"),
    ("FF-hops-K10", "ff", 10, "hops"),
    ("greedy-km-K10", "greedy", 10, "km"),
    ("greedy-hops-K10", "greedy", 10, "hops"),
]
COST = {"ff": 1, "greedy": 20}


def job(args):
    label, kind, K, order, topo, tau, loads = args
    from aeon.env import EnvConfig
    from aeon.distributed import DistributedEONEnv, DistConfig
    from aeon.policies.heuristics import KSPAMFF
    from aeon.policies.greedy import KSPAMGreedy
    from aeon.rollout_dist import evaluate_dist
    from aeon.config import EvalConfig
    ec = EvalConfig(measure_events=10_000)
    if kind == "ff":
        env = DistributedEONEnv(EnvConfig(topology=topo, k_paths=K, path_order=order), DistConfig(tau=tau))
        pol, strat = KSPAMFF(env), "first_fit"
    else:
        env = DistributedEONEnv(EnvConfig(topology=topo, k_paths=K, path_order=order), DistConfig(tau=tau, blocks=3))
        pol, strat = KSPAMGreedy(env), "jblock"
    out = []
    for L in loads:
        for s in CONF_EVAL:
            r = evaluate_dist(pol, env, L, s, ec, strat)
            out.append({k: r[k] for k in ("load", "eval_seed", "bp", "setup_fail", "view_block")})
    return f"kcheck|{label}|{topo}|{tau}", out


if __name__ == "__main__":
    cal = json.load(open(sys.argv[1])); outdir = sys.argv[2]; w = int(sys.argv[3])
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "kcheck_state.json")
    state = json.load(open(p)) if os.path.exists(p) else {}
    jobs = [(lab, kind, K, order, t, tau, cal[t]["loads"]) for (lab, kind, K, order) in SETTINGS
            for t in ("NSFNET", "USNET", "EUROPEAN") for tau in (0.0, 0.05)
            if f"kcheck|{lab}|{t}|{tau}" not in state]
    jobs.sort(key=lambda j: -COST[j[1]] * (2 if j[4] == "USNET" else 1))   # longest first
    print(f"{len(jobs)} jobs to run on {w} workers ({len(state)} already done)", flush=True)
    with ProcessPoolExecutor(w) as ex:
        for fu in as_completed([ex.submit(job, j) for j in jobs]):
            k, v = fu.result(); state[k] = v
            json.dump(state, open(p, "w"), indent=1); print("done", k, flush=True)
    print(f"Done. Results written to {p}.", flush=True)
