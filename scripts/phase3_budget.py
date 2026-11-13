"""Training-budget sensitivity (exploratory).

A-QMIX with its confirmed configuration (reward centering, gamma = 0.99) on NSFNET at tau = 0 and 0.05,
confirmation training seeds 200 and 201, trained with 250k (default) and 1M (4x) arrivals. Everything
else is identical (the curriculum and epsilon schedules are fractions of the budget, so they stretch
with it; replay capacity is unchanged). All four budget pairs run on the SAME machine, so the 250k vs
1M comparison is paired by seed and platform. Evaluation: the 5 calibrated loads x traffic 930000-930004.
  python scripts/phase3_budget.py RESULTS_ROOT OUTDIR WORKERS     (resumable: budget_state.json)
A progress line is printed every 10 minutes.
"""
import json, os, sys, time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import phase3  # noqa: E402  (experiment constants and the learn_cell worker)

SEEDS = [200, 201]
BUDGETS = [1_000_000, 250_000]
TAUS = [0.0, 0.05]
HEARTBEAT = 600


def main():
    root, out, w = sys.argv[1], os.path.abspath(sys.argv[2]), int(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    cal = json.load(open(os.path.join(root, "phase3_calibrate", "calibration.json")))
    final = json.load(open(os.path.join(root, "phase3_tune", "state.json")))["final"]["A-QMIX"]["cfg"]
    p = os.path.join(out, "budget_state.json")
    state = json.load(open(p)) if os.path.exists(p) else {}
    jobs = []
    for b in BUDGETS:
        for tau in TAUS:
            for s in SEEDS:
                key = f"B|A-QMIX|NSFNET|{tau}|s{s}|{b}"
                if key in state:
                    continue
                jobs.append((key, {"kind": "learn", "tag": "phase3_budget", "method": "A-QMIX", "cfg": final,
                                   "topo": "NSFNET", "tau": tau, "train_load": cal["NSFNET"]["primary_load"],
                                   "seed": s, "budget": b, "loads": cal["NSFNET"]["loads"],
                                   "eval_seeds": phase3.CONF_EVAL, "root": out}))
    print(f"A-QMIX config {final}; {len(jobs)} runs to do on {w} workers ({len(state)} already done). "
          f"Longest (1M) runs first; expect about 3-4 hours.", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(w) as ex:
        pend = {ex.submit(phase3.learn_cell, spec): key for key, spec in jobs}
        while pend:
            done, _ = wait(pend, timeout=HEARTBEAT, return_when=FIRST_COMPLETED)
            for fu in done:
                key = pend.pop(fu)
                res = fu.result()
                state[key] = res
                json.dump(state, open(p, "w"), indent=1)
                bp = {}
                for e in res["evals"]:
                    bp.setdefault(e["load"], []).append(e["bp"])
                print(f"  done {key}  bp=" + ",".join(f"{sum(v)/len(v):.4f}" for v in bp.values())
                      + f"  train {res['train_s']/60:.0f} min  elapsed {(time.time()-t0)/60:.0f} min", flush=True)
            if not done:
                print(f"  ... still working: {len(pend)} runs left ({min(w, len(pend))} in progress), "
                      f"elapsed {(time.time()-t0)/60:.0f} min", flush=True)
    print(f"Done. Results written to {p}.", flush=True)


if __name__ == "__main__":
    main()
