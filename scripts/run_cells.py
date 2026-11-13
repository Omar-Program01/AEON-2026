"""
Run (topology, method, train_seed) cells: train (learned methods), evaluate on
held-out traffic seeds at each load, time decisions, and write everything under
results/<exp_id>/ with a manifest.

Example (pilot):
  python scripts/run_cells.py --tag pilot --topologies NSFNET \
      --methods KSP-AM-FF Independent QMIX --seeds 0 1 --workers 2
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# TPU runtimes: do not auto-load out-of-tree device plug-ins such as torch_tpu;
# this study uses CPUs only, and many processes initialising the TPU collide on its lock file.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_cell(args):
    (tag, topo, method, seed, tdict, edict, envdict, root, do_timing) = args
    import torch
    # Seed the CPU generator only (same CPU random stream as torch.manual_seed on a CPU-only
    # build; avoids initialising accelerator plug-ins present on some runtimes).
    torch.manual_seed = torch.random.default_generator.manual_seed
    torch.set_num_threads(1)
    from aeon.env import EONEnv, EnvConfig
    from aeon.config import TrainConfig, EvalConfig, HEURISTICS
    from aeon.factory import build_policy
    from aeon.rollout import train, evaluate, seed_everything
    from aeon.timing import time_decisions
    from aeon.explog import Experiment

    tcfg, ecfg = TrainConfig(**tdict), EvalConfig(**edict)
    ecfg.loads = tuple(ecfg.loads)
    ecfg.eval_seeds = tuple(ecfg.eval_seeds)
    envcfg = EnvConfig(topology=topo, **envdict)
    cfg_record = {"env": vars(envcfg), "train": vars(tcfg), "eval": vars(ecfg)}
    heuristic = method in HEURISTICS
    if heuristic:
        cfg_record.pop("train")
    exp = Experiment(root, tag, method, cfg_record, {"train_seed": None if heuristic else seed})
    env = EONEnv(envcfg, seed=0, load=tcfg.train_load)
    seed_everything(seed)
    pol = build_policy(method, env, tcfg)
    t0 = time.time()
    logs = []
    if not heuristic:
        logs = train(pol, env, tcfg, seed)
        with open(exp.dir / "train_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
            w.writeheader()
            w.writerows(logs)
        sd = {}
        for attr in ("q", "mixer", "net", "enc", "head"):
            mod = getattr(pol, attr, None)
            if mod is not None:
                sd[attr] = mod.state_dict()
        torch.save(sd, exp.dir / "checkpoint.pt")
    train_s = time.time() - t0
    t1 = time.time()
    evals = []
    for load in ecfg.loads:
        for es in ecfg.eval_seeds:
            r = evaluate(pol, env, load, es, ecfg)
            r.update({"topology": topo, "method": method, "train_seed": seed})
            evals.append(r)
    eval_s = time.time() - t1
    exp.write("eval.json", evals)
    timing = time_decisions(pol, env, ecfg.loads[0]) if do_timing else None
    if timing:
        exp.write("timing.json", timing)
    exp_id = exp.finish({"train_s": round(train_s, 1), "eval_s": round(eval_s, 1),
                         "n_parameters": pol.n_parameters(),
                         "n_training_parameters": getattr(pol, "n_training_parameters", pol.n_parameters)()})
    return {"exp_id": exp_id, "topology": topo, "method": method, "seed": seed,
            "train_s": train_s, "eval_s": eval_s,
            "bp": [round(e["bp"], 4) for e in evals],
            "decide_ms": timing["decide_ms"]["median"] if timing else None,
            "env_ms": timing["env_ms"]["median"] if timing else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--topologies", nargs="+", default=["NSFNET"])
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--train-events", type=int, default=None)
    ap.add_argument("--train-load", type=float, default=None)
    ap.add_argument("--loads", nargs="+", type=float, default=None)
    ap.add_argument("--eval-seeds", nargs="+", type=int, default=None)
    ap.add_argument("--measure-events", type=int, default=None)
    ap.add_argument("--phy", default="gn_v2")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--tset", nargs="*", default=[],
                    help="TrainConfig overrides key=value, e.g. reward_transform=affine dueling=true gamma=0.995")
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    from aeon.config import TrainConfig, EvalConfig, HEURISTICS
    t, e = vars(TrainConfig()), vars(EvalConfig())
    if a.train_events: t["train_events"] = a.train_events
    if a.train_load: t["train_load"] = a.train_load
    if a.loads: e["loads"] = a.loads
    if a.eval_seeds: e["eval_seeds"] = a.eval_seeds
    if a.measure_events: e["measure_events"] = a.measure_events
    for kv in a.tset:
        k, v = kv.split("=", 1)
        if k not in t:
            raise SystemExit(f"unknown TrainConfig field {k}")
        d = t[k]
        t[k] = (v.lower() in ("1", "true", "yes")) if isinstance(d, bool) else type(d)(v)
    envd = {"phy_model": a.phy}
    tasks = []
    for topo in a.topologies:
        for m in a.methods:
            seeds = [0] if m in HEURISTICS else a.seeds
            for s in seeds:
                tasks.append((a.tag, topo, m, s, t, e, envd, a.root, not a.no_timing))
    os.makedirs(a.root, exist_ok=True)
    index = os.path.join(a.root, f"index_{a.tag}.jsonl")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_cell, x): x for x in tasks}
        for fu in as_completed(futs):
            r = fu.result()
            with open(index, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"DONE {r['topology']}/{r['method']}/s{r['seed']} exp={r['exp_id']} "
                  f"train={r['train_s']:.0f}s eval={r['eval_s']:.0f}s bp={r['bp']} "
                  f"decide_ms={r['decide_ms']} env_ms={r['env_ms']}", flush=True)


if __name__ == "__main__":
    main()
