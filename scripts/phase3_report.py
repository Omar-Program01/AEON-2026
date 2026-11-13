"""
Stale-state study analysis. Reads the state files of the
calibrate, heuristics, tune and confirm stages and writes tables (markdown + json).

  python scripts/phase3_report.py --root RESULTS_ROOT --heur HEURISTICS_STATE_JSON --out OUTDIR
"""
import argparse
import json
import math
import os
import statistics as st
from collections import defaultdict

T1SIDED = {4: 2.132, 3: 2.353, 2: 2.920, 1: 6.314}      # one-sided 95% t quantiles by df
HEURS = ["KSP-AM-FF", "KSP-AM-BF", "KSP-AM-RF", "KSP-AM-FLF"]
LEARN = ["A-QMIX", "A-Independent", "W-QMIX", "A-QMIX-myopic", "DeepRMSA-QoT"]
TOPOS = ["NSFNET", "USNET", "EUROPEAN"]
TAUS = [0.0, 0.01, 0.05]


def load(p):
    return json.load(open(p))


def mean_ci(d):
    n = len(d)
    m = st.mean(d)
    if n < 2:
        return m, float("nan"), float("nan")
    h = T1SIDED[n - 1] * st.stdev(d) / math.sqrt(n)
    return m, m - h, m + h          # one-sided 95% lower / upper bounds


def heur_table(H):
    """H[(topo, tau, sig, K, R, method)] -> {load: {eval_seed: (bp, setup_fail)}}"""
    out = {}
    for key, cell in H.items():
        _, topo, tau, sig, K, R, m = key.split("|")
        d = defaultdict(dict)
        for e in cell["result"]["evals"]:
            d[e["load"]][e["eval_seed"]] = (e["bp"], e["setup_fail"])
        out[(topo, float(tau), sig, K, R, m)] = d
    return out


def learner_table(C):
    """-> L[(name, topo, tau)][seed][load] = (bp mean over traffic seeds, setup_fail, deviation)"""
    out = defaultdict(lambda: defaultdict(dict))
    for key, cell in C.items():
        _, name, topo, tau, s = key.split("|")
        per = defaultdict(list)
        for e in cell["result"]["evals"]:
            per[e["load"]].append(e)
        for L, ev in per.items():
            out[(name, topo, float(tau))][int(s[1:])][L] = (
                st.mean(x["bp"] for x in ev), st.mean(x["setup_fail"] for x in ev),
                st.mean(x["deviation_frac"] for x in ev))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--heur", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cal = load(os.path.join(a.root, "phase3_calibrate", "calibration.json"))
    H = heur_table(load(a.heur)["cells"])
    C = load(os.path.join(a.root, "phase3_confirm", "state.json"))["cells"]
    L = learner_table(C)
    tune = load(os.path.join(a.root, "phase3_tune", "state.json"))
    md, js = [], {"calibration": cal, "final_configs": tune.get("final", {})}

    def hbp(topo, tau, m, load_, sig="forward", K="K5", R="R0"):
        d = H[(topo, tau, sig, K, R, m)][load_]
        return st.mean(v[0] for v in d.values()), st.mean(v[1] for v in d.values())

    md.append("# Stale-state study results (confirmation)\n")
    md.append("Blocking at each topology's calibrated loads; learners: mean over training seeds (each the mean over "
              "5 traffic seeds 930000-930004); heuristics: mean over the same 5 traffic seeds. "
              "Forward signalling, K = 5, no retry unless stated.\n")

    # ---------------- primary endpoint
    topo = "NSFNET"
    Lp = cal[topo]["primary_load"]
    js["primary"] = {}
    md.append(f"\n## Primary endpoint (NSFNET, {Lp:.0f} E, forward signalling)\n")
    ff0 = hbp(topo, 0.0, "KSP-AM-FF", Lp)[0]
    best5 = min(HEURS, key=lambda h: hbp(topo, 0.05, h, Lp)[0])
    bh5 = hbp(topo, 0.05, best5, Lp)[0]
    md.append(f"KSP-AM-FF at tau = 0: {ff0:.4f}. Best stale-aware heuristic at tau = 0.05: {best5} {bh5:.4f}.\n")
    md.append("| Learner | (i) BP ratio vs KSP-AM-FF at tau 0: mean / upper 95% | < 1.10? | (ii) BP_bestH - BP_L at tau 0.05: mean / lower 95% | > 0? | GO |")
    md.append("|---|---|---|---|---|---|")
    for name in ("A-QMIX", "A-Independent"):
        s0 = L.get((name, topo, 0.0), {})
        s5 = L.get((name, topo, 0.05), {})
        if len(s0) < 2 or len(s5) < 2:
            md.append(f"| {name} | incomplete | | | | |")
            continue
        lr = [math.log(s0[s][Lp][0] / ff0) for s in sorted(s0)]
        m, lo, hi = mean_ci(lr)
        diff = [bh5 - s5[s][Lp][0] for s in sorted(s5)]
        dm, dlo, dhi = mean_ci(diff)
        c1, c2 = math.exp(hi) < 1.10, dlo > 0
        js["primary"][name] = {"ratio_mean": math.exp(m), "ratio_upper": math.exp(hi), "cond_i": c1,
                               "diff_mean": dm, "diff_lower": dlo, "cond_ii": c2, "seeds_tau0": len(s0),
                               "seeds_tau05": len(s5)}
        md.append(f"| {name} | {math.exp(m):.3f} / {math.exp(hi):.3f} | {'yes' if c1 else 'no'} | "
                  f"{dm:+.4f} / {dlo:+.4f} | {'yes' if c2 else 'no'} | {'GO' if c1 and c2 else 'no'} |")
    go = any(v["cond_i"] and v["cond_ii"] for v in js["primary"].values())
    md.append(f"\n**Decision (primary rule): {'GO' if go else 'NO-GO'}.** (Holm correction would only make a GO harder.)\n")
    js["decision"] = "GO" if go else "NO-GO"

    # ---------------- full tables per topology
    for topo in TOPOS:
        loads = cal[topo]["loads"]
        md.append(f"\n## {topo}: blocking at loads {', '.join(f'{x:.0f}' for x in loads)} E\n")
        for tau in TAUS:
            rows = []
            for h in HEURS:
                try:
                    v = [hbp(topo, tau, h, x)[0] for x in loads]
                    rows.append((h, v, None))
                except KeyError:
                    pass
            for sig_label, sig, K, R in (("backward", "backward", "K5", "R0"), ("1 retry", "forward", "K5", "R1"),
                                         ("K=10", "forward", "K10", "R0")):
                try:
                    best = min(HEURS, key=lambda h: hbp(topo, tau, h, loads[2], sig, K, R)[0])
                    v = [hbp(topo, tau, best, x, sig, K, R)[0] for x in loads]
                    rows.append((f"best heuristic, {sig_label} ({best})", v, None))
                except KeyError:
                    pass
            for name in LEARN:
                ss = L.get((name, topo, tau), {})
                if not ss:
                    continue
                v = [st.mean(ss[s][x][0] for s in ss) for x in loads]
                sd = [st.stdev([ss[s][x][0] for s in ss]) if len(ss) > 1 else float("nan") for x in loads]
                rows.append((f"{name} (n={len(ss)})", v, sd))
            if not rows:
                continue
            md.append(f"\n**tau = {tau}**\n")
            md.append("| Method | " + " | ".join(f"{x:.0f} E" for x in loads) + " |")
            md.append("|---|" + "---|" * len(loads))
            for nm, v, sd in rows:
                cells = [f"{x:.4f}" + (f" ±{s:.4f}" if sd is not None and not math.isnan(s) else "")
                         for x, s in zip(v, sd or [None] * len(v))] if sd else [f"{x:.4f}" for x in v]
                md.append(f"| {nm} | " + " | ".join(cells) + " |")
                js.setdefault("tables", {}).setdefault(topo, {}).setdefault(str(tau), {})[nm] = v

    # ---------------- mechanism (NSFNET, primary load, paired by training seed)
    topo, Lp = "NSFNET", cal["NSFNET"]["primary_load"]
    md.append(f"\n## Mechanism ablations (NSFNET, {Lp:.0f} E; paired by training seed; difference = other - A-QMIX, "
              f"one-sided 95% bounds)\n")
    md.append("| tau | vs A-Independent (mixer) | vs W-QMIX (anchor) | vs A-QMIX-myopic (horizon) | A-QMIX setup-fail | A-QMIX deviation from anchor |")
    md.append("|---|---|---|---|---|---|")
    for tau in TAUS:
        base = L.get(("A-QMIX", topo, tau), {})
        cells = []
        for other in ("A-Independent", "W-QMIX", "A-QMIX-myopic"):
            o = L.get((other, topo, tau), {})
            common = sorted(set(base) & set(o))
            if len(common) < 2:
                cells.append("n/a")
                continue
            d = [o[s][Lp][0] - base[s][Lp][0] for s in common]
            m, lo, hi = mean_ci(d)
            cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        sf = st.mean(base[s][Lp][1] for s in base) if base else float("nan")
        dv = st.mean(base[s][Lp][2] for s in base) if base else float("nan")
        md.append(f"| {tau} | " + " | ".join(cells) + f" | {sf:.4f} | {dv:.3f} |")
    open(os.path.join(a.out, "phase3_results.md"), "w").write("\n".join(md) + "\n")
    json.dump(js, open(os.path.join(a.out, "phase3_results.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
