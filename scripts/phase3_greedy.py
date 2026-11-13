"""KSP-AM-greedy (myopic reference, exploratory) on the confirmation traffic,
all topologies x tau x loads, forward signalling, J = 3. Writes OUT/greedy_state.json."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
CONF_EVAL = [930000, 930001, 930002, 930003, 930004]

def job(args):
    topo, tau, loads = args
    from aeon.env import EnvConfig
    from aeon.distributed import DistributedEONEnv, DistConfig
    from aeon.policies.greedy import KSPAMGreedy
    from aeon.rollout_dist import evaluate_dist
    from aeon.config import EvalConfig
    env = DistributedEONEnv(EnvConfig(topology=topo), DistConfig(tau=tau, blocks=3))
    pol = KSPAMGreedy(env)
    out = []
    for L in loads:
        for s in CONF_EVAL:
            r = evaluate_dist(pol, env, L, s, EvalConfig(measure_events=10000))
            out.append({k: r[k] for k in ("load", "eval_seed", "bp", "setup_fail", "view_block", "deviation_frac")})
    return f"greedy|{topo}|{tau}", out

if __name__ == "__main__":
    cal = json.load(open(sys.argv[1])); outdir = sys.argv[2]; w = int(sys.argv[3])
    p = os.path.join(outdir, "greedy_state.json")
    state = json.load(open(p)) if os.path.exists(p) else {}
    jobs = [(t, tau, cal[t]["loads"]) for t in ("NSFNET", "USNET", "EUROPEAN") for tau in (0.0, 0.01, 0.05)
            if f"greedy|{t}|{tau}" not in state]
    with ProcessPoolExecutor(w) as ex:
        for fu in as_completed([ex.submit(job, j) for j in jobs]):
            k, v = fu.result(); state[k] = v
            json.dump(state, open(p, "w"), indent=1); print("done", k, flush=True)
