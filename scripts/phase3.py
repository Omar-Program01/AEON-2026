"""
Stale-state study (phase3): distributed control under stale state.

Stages (run in this order; every stage is resumable and skips finished cells):
  python scripts/phase3.py calibrate                # loads per topology (KSP-AM-FF, tau = 0)
  python scripts/phase3.py heuristics               # all heuristics, all settings, confirmation traffic
  python scripts/phase3.py tune [--learners ...]     # T1 + T2 on NSFNET, tau = 0.05, primary load
  python scripts/phase3.py confirm [--learners ...]  # fresh seeds, 3 topologies x 3 tau x 5 loads
  (the report and decision are produced by scripts/phase3_report.py)

Common options: --workers N, --root DIR (default results/). Each stage writes its own
state file results/phase3_<stage>[_<shard>]/state.json; finished cells of any state
file of the same stage in the root are imported at start, so shards can run on
different machines and be merged by copying their state folders together.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

if importlib.util.find_spec("torch_tpu") is not None and os.environ.get("AEON_ALLOW_TPU_PLUGIN") != "1":
    sys.exit("STOP: this Python contains the TPU plug-in (torch_tpu). Use a CPU-only Python.")

HEARTBEAT_S = int(os.environ.get("AEON_HEARTBEAT_S", "300"))

# ----------------------------------------------------------------------------- experiment constants
TOPOS = ["NSFNET", "USNET", "EUROPEAN"]
TAUS = [0.0, 0.01, 0.05]
TARGETS = [0.001, 0.005, 0.01, 0.03, 0.10]
PRIMARY = 2                                    # index of the 1 % target
CAL_SEEDS = [800000, 800001, 800002]
TUNE_SEEDS = [0, 1, 2]
TUNE_EVAL = [900000, 900001, 900002, 900003, 900004]
CONF_SEEDS = [200, 201, 202, 203, 204]
CONF_EVAL = [930000, 930001, 930002, 930003, 930004]
TUNE_TAU = 0.05
HEURISTICS = {"KSP-AM-FF": ("KSPAMFF", "first_fit"), "KSP-AM-BF": ("KSPAMBF", "best_fit"),
              "KSP-AM-RF": ("KSPAMFF", "random_fit"), "KSP-AM-FLF": ("KSPAMFF", "flf_node")}
LEARNERS = ["A-QMIX", "A-Independent", "W-QMIX", "DeepRMSA-QoT"]
ANCHORED = {"A-QMIX", "A-Independent"}
T1 = [{"reward_transform": rt, "gamma": g} for rt in ("affine", "center") for g in (0.0, 0.9, 0.99)]
T2_DELTAS = [0.1, 0.3]
T1_DQOT = [{"dqot_reward": rw, "gamma": g} for rw in ("basic", "heuristic", "qot") for g in (0.9, 0.99)]
BUDGET = 250_000


def cfg_key(d):
    return json.dumps(d, sort_keys=True)


# ----------------------------------------------------------------------------- cell workers
def _env(topo, tau, signaling="forward", blocks=1, k=5, retries=0):
    from aeon.env import EnvConfig
    from aeon.distributed import DistributedEONEnv, DistConfig
    return DistributedEONEnv(EnvConfig(topology=topo, k_paths=k),
                             DistConfig(tau=tau, signaling=signaling, blocks=blocks, retries=retries))


def window_params(load, tau):
    """A-QMIX / W-QMIX window size cap and batch size."""
    lam_tau = load * tau
    D = max(1, math.ceil(2.5 * lam_tau))
    batch = int(min(128, max(8, math.floor(128 / max(1.0, lam_tau)))))
    return D, batch


def cal_search(spec):
    """Log-scale bisection for the load at which KSP-AM-FF (tau = 0) blocks `target`."""
    from aeon.config import EvalConfig
    from aeon.policies.heuristics import KSPAMFF
    from aeon.rollout_dist import evaluate_dist
    topo, target = spec["topo"], spec["target"]
    env = _env(topo, 0.0)
    pol = KSPAMFF(env)
    ec = EvalConfig(measure_events=10_000)
    trace = []

    def bp(load):
        v = [evaluate_dist(pol, env, load, s, ec, "first_fit")["bp"] for s in CAL_SEEDS]
        b = sum(v) / len(v)
        trace.append((round(load, 2), round(b, 5)))
        return b

    lo = 20.0
    while lo > 1.0 and bp(lo) >= target:          # lower bracket: blocking below target
        lo /= 2.0
    hi = lo * 2.0
    while hi < 20000 and bp(hi) < target:         # upper bracket: blocking at or above target
        lo, hi = hi, hi * 2.0
    for _ in range(12):
        mid = math.sqrt(lo * hi)
        b = bp(mid)
        if abs(math.log(max(b, 1e-6) / target)) < math.log(1.05) or hi / lo < 1.01:
            lo = hi = mid
            break
        if b < target:
            lo = mid
        else:
            hi = mid
    load = round(math.sqrt(lo * hi))
    return {"topo": topo, "target": target, "load": float(load), "bp_at_load": bp(float(load)),
            "trace": trace}


def heur_group(spec):
    """One heuristic in one setting, evaluated at all loads x confirmation traffic seeds."""
    from aeon.config import EvalConfig
    from aeon.explog import Experiment
    from aeon.policies import heuristics as H
    from aeon.rollout_dist import evaluate_dist
    env = _env(spec["topo"], spec["tau"], spec["signaling"], 1, spec["k"], spec["retries"])
    cls, strat = HEURISTICS[spec["method"]]
    pol = getattr(H, cls)(env)
    ec = EvalConfig(measure_events=10_000)
    exp = Experiment(spec["root"], "phase3_heur", spec["method"],
                     {"spec": {k: v for k, v in spec.items() if k != "root"}, "eval": vars(ec)}, {})
    evals = []
    for load in spec["loads"]:
        for s in CONF_EVAL:
            r = evaluate_dist(pol, env, load, s, ec, strat)
            r.pop("batch_bp", None)
            evals.append(r)
    exp.write("eval.json", evals)
    exp.finish()
    return {"exp_id": exp.exp_id, "evals": [{k: e[k] for k in ("load", "eval_seed", "bp", "setup_fail", "view_block",
                                                                 "bbp", "se")} for e in evals]}


def learn_cell(spec):
    """Train one learner on one (topology, tau) with one seed; evaluate at the given loads."""
    import torch
    torch.manual_seed = torch.random.default_generator.manual_seed
    torch.set_num_threads(1)
    from aeon.config import EvalConfig
    from aeon.explog import Experiment
    from aeon.rollout import seed_everything
    from aeon.rollout_dist import P3TrainConfig, train_dist, evaluate_dist
    from aeon.policies.anchored import AnchoredIndependent, AnchoredQMIX, WindowQMIX
    topo, tau, load, seed = spec["topo"], spec["tau"], spec["train_load"], spec["seed"]
    env = _env(topo, tau, "forward", 3)
    D, wbatch = window_params(load, tau)
    windowed = spec["method"] in ("A-QMIX", "W-QMIX")
    t = dict(train_events=spec.get("budget", BUDGET), segment_events=2500, train_load=load,
             curriculum=True, curriculum_start_load=load / 3.0, curriculum_frac=0.4,
             eps_start=0.2, eps_end=0.01, eps_decay_frac=0.7,
             window_max_decisions=D, batch_size=wbatch if windowed else 128)
    t.update(spec["cfg"])
    tcfg = P3TrainConfig(**t)
    ec = EvalConfig(measure_events=10_000)
    exp = Experiment(spec["root"], spec["tag"], spec["method"],
                     {"env": {"topology": topo, "tau": tau, "signaling": "forward", "blocks": 3},
                      "train": vars(tcfg), "eval": vars(ec), "eval_loads": spec["loads"],
                      "eval_seeds": spec["eval_seeds"]}, {"train_seed": seed})
    seed_everything(seed)
    t0 = time.time()
    if spec["method"] == "DeepRMSA-QoT":
        from aeon.policies.deeprmsa_qot import DeepRMSAQoT, train_dqot
        pol = DeepRMSAQoT(env, tcfg)
        logs = train_dqot(pol, env, tcfg, seed)
    else:
        cls = {"A-QMIX": AnchoredQMIX, "A-Independent": AnchoredIndependent, "W-QMIX": WindowQMIX}[spec["method"]]
        pol = cls(env.obs_dim(), env.state_dim(), env.action_dim, env.N, tcfg)
        logs = train_dist(pol, env, tcfg, seed)
    train_s = time.time() - t0
    with open(exp.dir / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        w.writeheader()
        w.writerows(logs)
    sd = {"q": pol.q.state_dict()}
    if hasattr(pol, "mixer"):
        sd["mixer"] = pol.mixer.state_dict()
    torch.save(sd, exp.dir / "checkpoint.pt")
    evals = []
    for L in spec["loads"]:
        for s in spec["eval_seeds"]:
            r = evaluate_dist(pol, env, L, s, ec)
            r.pop("batch_bp", None)
            evals.append(r)
    exp.write("eval.json", evals)
    exp.finish({"train_s": round(train_s, 1), "n_parameters": pol.n_parameters(),
                "n_training_parameters": pol.n_training_parameters()})
    return {"exp_id": exp.exp_id, "train_s": train_s,
            "evals": [{k: e[k] for k in ("load", "eval_seed", "bp", "setup_fail", "view_block", "bbp", "se",
                                         "deviation_frac")} for e in evals]}


WORKERS = {"cal": cal_search, "heur": heur_group, "learn": learn_cell}


def run_spec(spec):
    return WORKERS[spec["kind"]](spec)


# ----------------------------------------------------------------------------- state and pool
def cpu_quota():
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if q == "max" else int(q) / int(per)
    except (OSError, ValueError):
        return None


class Runner:
    def __init__(self, a, stage, shard=""):
        self.a = a
        self.root = os.path.abspath(a.root)
        self.base = f"phase3_{stage}"
        self.dir = os.path.join(self.root, self.base + (f"_{shard}" if shard else ""))
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "state.json")
        self.state = json.load(open(self.path)) if os.path.exists(self.path) else {"cells": {}}
        self._import_others()
        w = a.workers if a.workers else max(1, (os.cpu_count() or 2) - 1)
        q = cpu_quota()
        if q and w > math.ceil(1.5 * q):
            print(f"Note: CPU quota {q:g} cores; using {math.ceil(1.5 * q)} workers.", flush=True)
            w = math.ceil(1.5 * q)
        self.workers = w

    def _import_others(self):
        added = 0
        for d in sorted(os.listdir(self.root)):
            if not (d == self.base or d.startswith(self.base + "_")):
                continue
            p = os.path.join(self.root, d, "state.json")
            if os.path.join(self.root, d) == self.dir or not os.path.exists(p):
                continue
            try:
                other = json.load(open(p))
            except (OSError, ValueError):
                continue
            for k, v in other.get("cells", {}).items():
                if k not in self.state["cells"]:
                    self.state["cells"][k] = v
                    added += 1
        if added:
            print(f"Imported {added} finished cells from other {self.base} state files.", flush=True)
            self.save()

    def save(self):
        tmp = self.path + ".tmp"
        json.dump(self.state, open(tmp, "w"), indent=1)
        os.replace(tmp, self.path)

    def run(self, jobs, label):
        """jobs: list of (key, spec, cost). Runs the missing ones, most expensive first."""
        todo = [j for j in jobs if j[0] not in self.state["cells"]]
        todo.sort(key=lambda j: -j[2])
        print(f"\n== {label}: {len(jobs)} cells, {len(jobs) - len(todo)} already done, "
              f"{len(todo)} to run on {self.workers} workers ==", flush=True)
        if not todo:
            return
        t0, done = time.time(), 0
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(run_spec, j[1]): j for j in todo}
            pending = set(futs)
            while pending:
                fin, pending = wait(pending, timeout=HEARTBEAT_S, return_when=FIRST_COMPLETED)
                if not fin:
                    print(f"  ... still working: {min(self.workers, len(pending))} runs in progress, "
                          f"{done}/{len(todo)} finished, elapsed {(time.time() - t0) / 60:.0f} min", flush=True)
                    continue
                for fu in fin:
                    key, spec, _ = futs[fu]
                    res = fu.result()
                    self.state["cells"][key] = {"spec": {k: v for k, v in spec.items() if k != "root"},
                                                "result": res}
                    self.save()
                    done += 1
                    print(f"  [{done}/{len(todo)}] {key}  {summary(res)}  elapsed "
                          f"{(time.time() - t0) / 60:.0f} min", flush=True)


def summary(res):
    if "load" in res and "target" in res:
        return f"load={res['load']:.0f} bp={res['bp_at_load']:.4f}"
    ev = res.get("evals", [])
    if not ev:
        return ""
    loads = sorted({e["load"] for e in ev})
    m = [sum(e["bp"] for e in ev if e["load"] == L) / sum(1 for e in ev if e["load"] == L) for L in loads]
    return "bp=" + ",".join(f"{x:.4f}" for x in m)


# ----------------------------------------------------------------------------- stages
def load_calibration(root):
    p = os.path.join(root, "phase3_calibrate", "calibration.json")
    if not os.path.exists(p):
        sys.exit("Run the calibrate stage first (calibration.json missing).")
    return json.load(open(p))


def stage_calibrate(a):
    R = Runner(a, "calibrate")
    jobs = [(f"cal|{t}|{x}", {"kind": "cal", "topo": t, "target": x}, 1.0 / x) for t in TOPOS for x in TARGETS]
    R.run(jobs, "Load calibration (KSP-AM-FF, tau = 0, seeds 800000-800002)")
    cal = {}
    for t in TOPOS:
        loads = [R.state["cells"][f"cal|{t}|{x}"]["result"]["load"] for x in TARGETS]
        bps = [R.state["cells"][f"cal|{t}|{x}"]["result"]["bp_at_load"] for x in TARGETS]
        cal[t] = {"targets": TARGETS, "loads": loads, "bp_at_load": bps, "primary_load": loads[PRIMARY]}
    json.dump(cal, open(os.path.join(R.dir, "calibration.json"), "w"), indent=1)
    print("\nCalibration:", json.dumps(cal, indent=1), flush=True)


def stage_heuristics(a):
    cal = load_calibration(os.path.abspath(a.root))
    R = Runner(a, "heuristics", a.shard)
    jobs = []
    settings = [("forward", 5, 0), ("backward", 5, 0), ("forward", 10, 0), ("forward", 5, 1)]
    for t in TOPOS:
        for tau in TAUS:
            for sig, k, rt in settings:
                for h in HEURISTICS:
                    key = f"heur|{t}|{tau}|{sig}|K{k}|R{rt}|{h}"
                    spec = {"kind": "heur", "topo": t, "tau": tau, "signaling": sig, "k": k, "retries": rt,
                            "method": h, "loads": cal[t]["loads"], "root": os.path.abspath(a.root)}
                    jobs.append((key, spec, sum(cal[t]["loads"])))
    R.run(jobs, "Heuristics (confirmation traffic 930000-930004, all loads)")


def tune_jobs(a, cal, learners, stage, base_cfgs):
    p = cal["NSFNET"]["primary_load"]
    jobs = []
    for m in learners:
        for c in base_cfgs[m]:
            for s in TUNE_SEEDS:
                key = f"{stage}|{m}|{cfg_key(c)}|s{s}"
                spec = {"kind": "learn", "tag": f"phase3_{stage}", "method": m, "cfg": c, "topo": "NSFNET",
                        "tau": TUNE_TAU, "train_load": p, "seed": s, "loads": [p], "eval_seeds": TUNE_EVAL,
                        "root": os.path.abspath(a.root)}
                cost = 5.0 if m in ("A-QMIX", "W-QMIX") else 1.0
                jobs.append((key, spec, cost))
    return jobs


def tuned_mean(R, stage, m, c):
    v = []
    for s in TUNE_SEEDS:
        cell = R.state["cells"].get(f"{stage}|{m}|{cfg_key(c)}|s{s}")
        if cell is None:
            return None
        ev = cell["result"]["evals"]
        v.append(sum(e["bp"] for e in ev) / len(ev))
    return sum(v) / len(v)


def stage_tune(a):
    cal = load_calibration(os.path.abspath(a.root))
    learners = a.learners or LEARNERS
    R = Runner(a, "tune", a.shard)
    grid = {m: (T1_DQOT if m == "DeepRMSA-QoT" else T1) for m in learners}
    R.run(tune_jobs(a, cal, learners, "T1", grid), "Tuning T1 (reward x gamma)")
    best1 = {m: min(grid[m], key=lambda c: tuned_mean(R, "T1", m, c)) for m in learners}
    t2 = {m: [dict(best1[m], anchor_margin=d) for d in T2_DELTAS] for m in learners if m in ANCHORED}
    if t2:
        R.run(tune_jobs(a, cal, list(t2), "T2", t2), "Tuning T2 (deviation margin)")
    final = {}
    for m in learners:
        cands = [("T1", c) for c in grid[m]] + [("T2", c) for c in t2.get(m, [])]
        st, c = min(cands, key=lambda sc: tuned_mean(R, sc[0], m, sc[1]))
        final[m] = {"cfg": c, "tuned_bp": tuned_mean(R, st, m, c)}
    R.state["final"] = {**R.state.get("final", {}), **final}
    R.save()
    print("\nSelected configurations:", json.dumps(final, indent=1), flush=True)


def confirm_jobs(a, cal, finals):
    jobs = []
    prio = {("NSFNET", 0.0): 0, ("NSFNET", 0.01): 0, ("NSFNET", 0.05): 0,
            ("USNET", 0.0): 1, ("USNET", 0.05): 1, ("EUROPEAN", 0.0): 1, ("EUROPEAN", 0.05): 1,
            ("USNET", 0.01): 2, ("EUROPEAN", 0.01): 2}
    for name, (m, c) in finals.items():
        for t in TOPOS:
            for tau in TAUS:
                for s in CONF_SEEDS:
                    key = f"C|{name}|{t}|{tau}|s{s}"
                    spec = {"kind": "learn", "tag": "phase3_confirm", "method": m, "cfg": c, "topo": t,
                            "tau": tau, "train_load": cal[t]["primary_load"], "seed": s,
                            "loads": cal[t]["loads"], "eval_seeds": CONF_EVAL, "root": os.path.abspath(a.root)}
                    # priority level first (level 0 before 1 before 2), then longest runs first
                    cost = (2 - prio[(t, tau)]) * 1000.0 + (5.0 if m in ("A-QMIX", "W-QMIX") else 1.0)
                    jobs.append((key, spec, cost))
    return jobs


def stage_confirm(a):
    root = os.path.abspath(a.root)
    cal = load_calibration(root)
    final = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "state.json")
        if (d == "phase3_tune" or d.startswith("phase3_tune_")) and os.path.exists(p):
            final.update(json.load(open(p)).get("final", {}))
    learners = a.learners or LEARNERS
    missing = [m for m in learners if m not in final]
    if missing:
        sys.exit(f"Tuning not finished for {missing}.")
    finals = {m: (m, final[m]["cfg"]) for m in learners}
    if "A-QMIX" in learners:
        finals["A-QMIX-myopic"] = ("A-QMIX", dict(final["A-QMIX"]["cfg"], gamma=0.0))
    R = Runner(a, "confirm", a.shard)
    R.run(confirm_jobs(a, cal, finals), "Confirmation (seeds 200-204, traffic 930000-930004)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "heuristics", "tune", "confirm"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--learners", nargs="*", default=[])
    ap.add_argument("--shard", default="")
    ap.add_argument("--root", default=os.path.join(HERE, "..", "results"))
    a = ap.parse_args()
    if a.learners and not a.shard:
        a.shard = "-".join(sorted(a.learners))
    {"calibrate": stage_calibrate, "heuristics": stage_heuristics, "tune": stage_tune,
     "confirm": stage_confirm}[a.stage](a)


if __name__ == "__main__":
    main()
