"""
Fresh-state study (phase2a): equal-budget learnability study with a go/no-go decision rule.

Stages (identical for every learner, NSFNET, 300 Erlang, tuning seeds 0 and 1,
evaluation traffic seeds 900000-900004):
  A  gamma in {0, 0.9, 0.99, 0.997} x reward_transform in {none, affine}      (8 configs)
  B  from the best gamma>0 config of A: fragmentation weight w_f in {0, 50, 200} (3 configs)
  C  from the best gamma>0 config of A+B, one factor at a time:
       budget x4 (1,000,000 arrivals) | lr 3e-4 | credit assignment
       (value learners: 5-step returns; DeepRMSA: lambda 0.95; PPO: lambda 1.0)   (3 configs)
  Final config per learner = best gamma>0 config of A+B+C ("best" = lowest mean
  blocking over the two tuning seeds). Its myopic counterpart = same config, gamma 0.
Confirmation (fresh seeds, fresh traffic): the two learners with the lowest tuned
blocking; final and myopic configs; training seeds 100-104; evaluation traffic seeds
920000-920004; KSP-AM-FF evaluated on the same traffic.
GO if, for at least one confirmed learner, BOTH hold (one-sided, alpha = 0.05, t, df 4):
  (i)  noninferior to KSP-AM-FF:  upper bound of mean(BP_learner - BP_KSP-AM-FF) < 0.01
  (ii) better than its own myopic version: lower bound of mean(BP_myopic - BP_final) > 0
PAN-EUROPEAN check (reported, not part of the rule): final and myopic configs of the
confirmed learners, seeds 100-102, same evaluation traffic seeds.

Resumable: completed cells are recorded in results/phase2a/state.json and skipped
on restart. Report: results/phase2a/report.md (rewritten after every stage).

  python scripts/phase2a.py                 # full study, workers = cores - 1
  python scripts/phase2a.py --smoke         # 3-minute pipeline check with tiny budgets

Parallel machines (shards). Stages A-C are independent per learner, so they can be
split across machines, each running a subset of learners into its own state file:
  python scripts/phase2a.py --learners DeepRMSA PPO          # machine 1
  python scripts/phase2a.py --learners GNN-DRL Centralized-DQN   # machine 2
  python scripts/phase2a.py --learners Independent QMIX       # machine 3
A shard stops after Stage C. When all shards have finished (and their
results/phase2a_shard_*/state.json files are in the same results folder), run
  python scripts/phase2a.py
which merges every shard's finished cells (nothing is re-run) and completes the
selection, confirmation and decision. Every run also imports finished cells from
the other state files in the results folder at start-up, so work done by an earlier
run is never repeated. Run the final merge only after the shards have finished.
"""
import argparse
import json
import math
import os
import sys
import time
import importlib.util
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

# Execution safeguard (no effect on configurations or results): on TPU runtimes the system
# PyTorch may load the TPU plug-in into every worker, which stalls the study. Use a CPU-only Python.
if importlib.util.find_spec("torch_tpu") is not None and os.environ.get("AEON_ALLOW_TPU_PLUGIN") != "1":
    sys.exit("STOP: this Python contains the TPU plug-in (torch_tpu), which stalls the study.\n"
             "Use a CPU-only Python environment (or set AEON_ALLOW_TPU_PLUGIN=1).")

HEARTBEAT_S = int(os.environ.get("AEON_HEARTBEAT_S", "300"))   # status line every 5 min while runs are in progress


def cpu_quota():
    """CPU cores this container may use (cgroup quota), or None if unlimited/unknown."""
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if q == "max" else int(q) / int(per)
    except (OSError, ValueError):
        return None


from run_cells import run_cell  # noqa: E402

LEARNERS = ["DeepRMSA", "PPO", "GNN-DRL", "Centralized-DQN", "Independent", "QMIX"]
VALUE_LEARNERS = {"GNN-DRL", "Centralized-DQN", "Independent", "QMIX"}
TUNE_SEEDS = [0, 1]
CONF_SEEDS = [100, 101, 102, 103, 104]
EUR_SEEDS = [100, 101, 102]
TUNE_EVAL = [900000, 900001, 900002, 900003, 900004]
CONF_EVAL = [920000, 920001, 920002, 920003, 920004]
MARGIN = 0.01
LOAD = 300.0
T_CRIT_95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}


def cfg_key(c):
    return json.dumps(c, sort_keys=True)


def stage_a():
    return [{"t": {"gamma": g, "reward_transform": rt}, "e": {"w_frag": 5.0}}
            for g in (0.0, 0.9, 0.99, 0.997) for rt in ("none", "affine")]


def stage_b(base):
    out = []
    for wf in (0.0, 50.0, 200.0):
        c = json.loads(cfg_key(base))
        c["e"]["w_frag"] = wf
        out.append(c)
    return out


def stage_c(base, method, budget4):
    out = []
    c = json.loads(cfg_key(base)); c["t"]["train_events"] = budget4; out.append(c)
    c = json.loads(cfg_key(base)); c["t"]["lr"] = 3e-4; out.append(c)
    c = json.loads(cfg_key(base))
    if method in VALUE_LEARNERS:
        c["t"]["n_step"] = 5
    elif method == "DeepRMSA":
        c["t"]["a2c_lambda"] = 0.95
    else:
        c["t"]["gae_lambda"] = 1.0
    out.append(c)
    return out


def myopic(c):
    m = json.loads(cfg_key(c))
    m["t"]["gamma"] = 0.0
    return m


class Study:
    def __init__(self, a):
        from aeon.config import TrainConfig, EvalConfig
        self.a = a
        self.root = os.path.abspath(a.root)
        base = "phase2a" + ("_smoke" if a.smoke else "")
        all_learners = ["DeepRMSA", "QMIX"] if a.smoke else LEARNERS
        self.shard = bool(a.learners) and set(a.learners) != set(all_learners)
        if a.learners:
            bad = [m for m in a.learners if m not in all_learners]
            if bad:
                raise SystemExit(f"unknown learner(s) {bad}; choose from {all_learners}")
        name = base + ("_shard_" + "-".join(sorted(a.learners)) if self.shard else "")
        self.dir = os.path.join(self.root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "state.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {"cells": {}}
        self.merge_others(base)
        self.T0, self.E0 = vars(TrainConfig()), vars(EvalConfig())
        self.budget = 3000 if a.smoke else self.T0["train_events"]
        self.budget4 = 6000 if a.smoke else 4 * self.T0["train_events"]
        self.measure = 1000 if a.smoke else self.E0["measure_events"]
        self.learners = [m for m in all_learners if m in a.learners] if self.shard else all_learners
        self.tune_seeds = [0] if a.smoke else TUNE_SEEDS
        self.conf_seeds = [100, 101] if a.smoke else CONF_SEEDS
        self.eur_seeds = [100, 101] if a.smoke else EUR_SEEDS
        self.tune_eval = TUNE_EVAL[:1] if a.smoke else TUNE_EVAL
        self.conf_eval = CONF_EVAL[:1] if a.smoke else CONF_EVAL
        self.workers = max(1, (os.cpu_count() or 2) - 1) if a.workers == "auto" else int(a.workers)
        # Never run more workers than the CPU quota sustains (benchmark: 1.5 x quota is the optimum;
        #    24 workers on a 4-core quota made every run about 10x slower).
        q = cpu_quota()
        if q:
            cap = max(1, math.ceil(1.5 * q))
            if self.workers > cap:
                print(f"Note: {self.workers} workers requested but the CPU quota is {q:g} cores; "
                      f"using {cap} workers.", flush=True)
                self.workers = cap

    def merge_others(self, base):
        """Import finished cells from every other state file of the same study
        (main run and shards) in the results folder; own cells are never overwritten."""
        added = 0
        for d in sorted(os.listdir(self.root)):
            same_kind = d == base or d.startswith(base + "_shard_")
            if not same_kind or os.path.join(self.root, d) == self.dir:
                continue
            p = os.path.join(self.root, d, "state.json")
            if not os.path.exists(p):
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
            print(f"Imported {added} finished cells from other state files in {self.root}", flush=True)
            self.save()

    def save(self):
        tmp = self.state_path + ".tmp"
        json.dump(self.state, open(tmp, "w"), indent=1)
        os.replace(tmp, self.state_path)

    def key(self, stage, topo, method, seed, c, eval_seeds):
        return f"{stage}|{topo}|{method}|{seed}|{cfg_key(c)}|{eval_seeds[0]}"

    def task(self, stage, topo, method, seed, c, eval_seeds):
        t = dict(self.T0)
        t["train_events"] = self.budget
        t["train_load"] = LOAD
        t.update(c.get("t", {}))
        if t["train_events"] == 4 * self.T0["train_events"] and self.a.smoke:
            t["train_events"] = self.budget4
        e = dict(self.E0)
        e.update({"loads": [LOAD], "eval_seeds": list(eval_seeds), "measure_events": self.measure,
                  "qot_check_every": 0})
        env = {"phy_model": "gn_v2"}
        env.update(c.get("e", {}))
        tag = "phase2a" + ("_smoke" if self.a.smoke else "") + f"_{stage}"
        return (tag, topo, method, seed, t, e, env, self.root, False)

    def run(self, jobs, label):
        """jobs: list of (stage, topo, method, seed, cfg, eval_seeds). Runs missing ones."""
        todo = [j for j in jobs if self.key(*j) not in self.state["cells"]]
        todo.sort(key=lambda j: -j[4].get("t", {}).get("train_events", 0))   # longest runs first
        print(f"\n== {label}: {len(jobs)} cells, {len(jobs) - len(todo)} already done, "
              f"{len(todo)} to run on {self.workers} workers ==", flush=True)
        t0 = time.time()
        done = 0
        if todo:
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(run_cell, self.task(*j)): j for j in todo}
                pending = set(futs)
                while pending:
                    finished, pending = wait(pending, timeout=HEARTBEAT_S, return_when=FIRST_COMPLETED)
                    if not finished:
                        el = time.time() - t0
                        running = min(self.workers, len(pending))
                        print(f"  ... still working: {running} runs in progress, {done}/{len(todo)} finished, "
                              f"elapsed {el/60:.0f} min (status every {HEARTBEAT_S // 60} min)", flush=True)
                        continue
                    for fu in finished:
                        done += 1
                        self._record(futs[fu], fu, t0, done, len(todo))
        return [self.state["cells"][self.key(*j)] for j in jobs]

    def _record(self, j, fu, t0, done, n):
        r = fu.result()
        self.state["cells"][self.key(*j)] = {
            "stage": j[0], "topology": j[1], "method": j[2], "seed": j[3], "cfg": j[4],
            "exp_id": r["exp_id"], "bp_eval": r["bp"], "bp": sum(r["bp"]) / len(r["bp"]),
            "train_s": r["train_s"]}
        self.save()
        el = time.time() - t0
        eta = el / done * (n - done)
        print(f"  [{done}/{n}] {j[1]}/{j[2]}/s{j[3]} {cfg_key(j[4])} "
              f"bp={self.state['cells'][self.key(*j)]['bp']:.4f}  elapsed {el/60:.0f} min, "
              f"ETA {eta/60:.0f} min (rough)", flush=True)

    # ----------------------------------------------------------------- #
    def mean_bp(self, stage_cells, method, c):
        v = [x["bp"] for x in stage_cells if x["method"] == method and cfg_key(x["cfg"]) == cfg_key(c)]
        return sum(v) / len(v)

    def best_nonmyopic(self, cells, method, configs):
        cand = [c for c in configs if c["t"]["gamma"] > 0]
        return min(cand, key=lambda c: self.mean_bp(cells, method, c))

    def main(self):
        S = self.state.setdefault("selections", {})
        if self.shard:
            print(f"Shard run for learners {self.learners}: stages A-C only.", flush=True)
        else:
            href = self.run([("H", "NSFNET", h, 0, {"t": {}, "e": {}}, self.tune_eval)
                             for h in ("KSP-FF", "KSP-AM-FF", "KSP-AM-BF")], "Heuristics, tuning traffic")
            S["heuristics_tune"] = {x["method"]: x["bp"] for x in href}
        # Stage A
        A = stage_a()
        cA = self.run([("A", "NSFNET", m, s, c, self.tune_eval) for m in self.learners for c in A
                       for s in self.tune_seeds], "Stage A (horizon x reward normalisation)")
        baseA = {m: self.best_nonmyopic(cA, m, A) for m in self.learners}
        S["stageA_best"] = baseA
        self.report()
        # Stage B
        jobsB, cfgB = [], {}
        for m in self.learners:
            cfgB[m] = stage_b(baseA[m])
            jobsB += [("B", "NSFNET", m, s, c, self.tune_eval) for c in cfgB[m] for s in self.tune_seeds]
        cB = self.run(jobsB, "Stage B (fragmentation weight)")
        allAB = cA + cB
        baseB = {m: self.best_nonmyopic(allAB, m, [baseA[m]] + cfgB[m]) for m in self.learners}
        S["stageB_best"] = baseB
        self.report()
        # Stage C
        jobsC, cfgC = [], {}
        for m in self.learners:
            cfgC[m] = stage_c(baseB[m], m, 4 * self.T0["train_events"])
            jobsC += [("C", "NSFNET", m, s, c, self.tune_eval) for c in cfgC[m] for s in self.tune_seeds]
        cC = self.run(jobsC, "Stage C (budget, learning rate, credit assignment)")
        allABC = allAB + cC
        final = {m: self.best_nonmyopic(allABC, m, [baseB[m]] + cfgC[m]) for m in self.learners}
        tuned = {m: self.mean_bp(allABC, m, final[m]) for m in self.learners}
        S["final"] = final
        S["tuned_bp"] = tuned
        self.save()
        self.report()
        if self.shard:
            print("\nShard finished (stages A-C). When every shard has finished, run "
                  "'python scripts/phase2a.py' without --learners to merge and confirm.", flush=True)
            return
        # Confirmation
        top = sorted(self.learners, key=lambda m: tuned[m])[:2]
        S["confirmed"] = top
        jobs = [("CONF", "NSFNET", "KSP-AM-FF", 0, {"t": {}, "e": {}}, self.conf_eval)]
        for m in top:
            for c in (final[m], myopic(final[m])):
                jobs += [("CONF", "NSFNET", m, s, c, self.conf_eval) for s in self.conf_seeds]
        cCONF = self.run(jobs, "Confirmation (fresh seeds, fresh traffic), NSFNET")
        jobs = [("EUR", "EUROPEAN", "KSP-AM-FF", 0, {"t": {}, "e": {}}, self.conf_eval)]
        for m in top:
            for c in (final[m], myopic(final[m])):
                jobs += [("EUR", "EUROPEAN", m, s, c, self.conf_eval) for s in self.eur_seeds]
        cEUR = self.run(jobs, "PAN-EUROPEAN check")
        S["decision"] = self.decide(cCONF, top, final, self.conf_seeds)
        S["european"] = self.decide(cEUR, top, final, self.eur_seeds)
        self.save()
        self.report()
        print("\nFresh-state study complete. Decision:", "GO" if S["decision"]["go"] else "NO-GO", flush=True)

    def decide(self, cells, top, final, seeds):
        h = [x["bp"] for x in cells if x["method"] == "KSP-AM-FF"][0]
        out = {"ksp_am_ff_bp": h, "learners": {}, "go": False}
        for m in top:
            fin = {x["seed"]: x["bp"] for x in cells if x["method"] == m and cfg_key(x["cfg"]) == cfg_key(final[m])}
            myo = {x["seed"]: x["bp"] for x in cells if x["method"] == m and cfg_key(x["cfg"]) == cfg_key(myopic(final[m]))}
            n = len(seeds)
            tc = T_CRIT_95.get(n - 1, 2.0)
            d = [fin[s] - h for s in seeds]
            e = [myo[s] - fin[s] for s in seeds]
            md, sd = _mean_sd(d)
            me, se = _mean_sd(e)
            ub = md + tc * sd / math.sqrt(n)
            lb = me - tc * se / math.sqrt(n)
            noninf, better = ub < MARGIN, lb > 0
            out["learners"][m] = {"bp_final": fin, "bp_myopic": myo,
                                  "diff_vs_heuristic_mean": md, "diff_vs_heuristic_upper95": ub,
                                  "noninferior": noninf, "myopic_minus_final_mean": me,
                                  "myopic_minus_final_lower95": lb, "better_than_myopic": better,
                                  "passes": noninf and better}
            out["go"] = out["go"] or (noninf and better)
        return out

    # ----------------------------------------------------------------- #
    def report(self):
        S = self.state.get("selections", {})
        cells = list(self.state["cells"].values())
        L = ["# Fresh-state learnability study report", "",
             f"NSFNET, {LOAD:.0f} Erlang. Blocking = mean over evaluation "
             "traffic seeds; tuning values are means over tuning seeds. Margin = +0.01 absolute.", ""]
        if "heuristics_tune" in S:
            L += ["## Heuristics (tuning traffic)", "", "| Method | Blocking |", "|---|---|"]
            L += [f"| {k} | {v:.4f} |" for k, v in S["heuristics_tune"].items()] + [""]
        for st, title in (("A", "Stage A"), ("B", "Stage B"), ("C", "Stage C")):
            sc = [x for x in cells if x["stage"] == st]
            if not sc:
                continue
            L += [f"## {title}", "", "| Learner | Config | Blocking per seed | Mean |", "|---|---|---|---|"]
            groups = {}
            for x in sc:
                groups.setdefault((x["method"], cfg_key(x["cfg"])), []).append(x)
            for (m, ck), xs in sorted(groups.items()):
                xs = sorted(xs, key=lambda x: x["seed"])
                per = ", ".join("{:.4f}".format(x["bp"]) for x in xs)
                mean = sum(x["bp"] for x in xs) / len(xs)
                L.append("| {} | `{}` | {} | {:.4f} |".format(m, _short(json.loads(ck)), per, mean))
            L.append("")
        if "final" in S:
            L += ["## Final configuration per learner", "", "| Learner | Final config | Tuned blocking |", "|---|---|---|"]
            for m, c in S["final"].items():
                L.append(f"| {m} | `{_short(c)}` | {S['tuned_bp'][m]:.4f} |")
            L.append("")
        for k, title in (("decision", "Confirmation and decision (NSFNET, fresh seeds and traffic)"),
                         ("european", "PAN-EUROPEAN check (reported, not part of the rule)")):
            if k in S:
                D = S[k]
                L += [f"## {title}", "", f"KSP-AM-FF blocking: {D['ksp_am_ff_bp']:.4f}", "",
                      "| Learner | Final BP per seed | Myopic BP per seed | Final − KSP-AM-FF (mean / upper 95%) | "
                      "Noninferior | Myopic − final (mean / lower 95%) | Beats myopic | Passes |",
                      "|---|---|---|---|---|---|---|---|"]
                for m, r in D["learners"].items():
                    fin = ", ".join("{:.4f}".format(v) for v in r["bp_final"].values())
                    myo = ", ".join("{:.4f}".format(v) for v in r["bp_myopic"].values())
                    L.append(f"| {m} | {fin} | {myo} | "
                             f"{r['diff_vs_heuristic_mean']:+.4f} / {r['diff_vs_heuristic_upper95']:+.4f} | "
                             f"{'yes' if r['noninferior'] else 'no'} | {r['myopic_minus_final_mean']:+.4f} / "
                             f"{r['myopic_minus_final_lower95']:+.4f} | {'yes' if r['better_than_myopic'] else 'no'} | "
                             f"{'**yes**' if r['passes'] else 'no'} |")
                if k == "decision":
                    L += ["", f"**Decision: {'GO' if D['go'] else 'NO-GO'}**"]
                L.append("")
        open(os.path.join(self.dir, "report.md"), "w", encoding="utf-8").write("\n".join(L))


def _mean_sd(x):
    n = len(x)
    m = sum(x) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1)) if n > 1 else float("inf")
    return m, sd


def _short(c):
    d = dict(c.get("t", {}))
    d.update({k: v for k, v in c.get("e", {}).items()})
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--learners", nargs="*", default=[],
                    help="run stages A-C for these learners only (a shard); omit for the full study / merge")
    ap.add_argument("--root", default=os.path.join(HERE, "..", "results"))
    Study(ap.parse_args()).main()
