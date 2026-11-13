"""Summaries of the secondary and sensitivity analyses, used by phase3_figures_extra.py.

  python scripts/phase3_summaries.py --mech MECHANISM_STATE --budget BUDGET_STATE --horizon HORIZON_STATE \
         --root RESULTS_ROOT --out OUTDIR

Writes OUTDIR/mechanism_summary.json (A-QMIX deviation analysis, NSFNET),
OUTDIR/budget_compare.json and OUTDIR/horizon_compare.json (blocking at the 5 calibrated loads,
mean over traffic seeds, per (tau, seed) pair; plus the paired difference at the primary load).
"""
import argparse, json, math, os, statistics as st
from collections import defaultdict

T1SIDED = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132}


def mech_summary(recs):
    n = len(recs)
    dev = [r for r in recs if r[0] != "none"]
    non = [r for r in recs if r[0] == "none"]
    path = [r for r in recs if r[0] == "path"]
    rate = lambda xs, i: (sum(x[i] for x in xs) / len(xs)) if xs else 0
    mean = lambda xs, i: (st.mean(x[i] for x in xs)) if xs else 0
    out = {"n": n, "dev_frac": len(dev) / n,
           "frac_path": len(path) / n, "frac_format": sum(r[0] == "format" for r in recs) / n,
           "frac_block": sum(r[0] == "block" for r in recs) / n,
           "block_dev": rate(dev, 4), "block_non": rate(non, 4), "sf_dev": rate(dev, 5), "sf_non": rate(non, 5),
           "hops_anchor_dev": mean(dev, 2), "hops_chosen_dev": mean(dev, 3),
           "hops_path_dev_chosen": mean(path, 3), "hops_path_dev_anchor": mean(path, 2)}
    return out


def by_age(recs):
    q = defaultdict(list)
    for r in recs:
        q[min(int(r[1] * 4), 3)].append(r)
    return {str(k): {"dev": sum(r[0] != "none" for r in v) / len(v),
                     "blockdev": sum(r[0] == "block" for r in v) / len(v),
                     "setupfail": sum(r[5] for r in v) / len(v)} for k, v in sorted(q.items())}


def per_load(evals, loads):
    by = defaultdict(list)
    for e in evals:
        by[e["load"]].append(e["bp"])
    return [st.mean(by[L]) for L in loads]


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m = st.mean(d)
    if len(d) < 2:
        return [m, float("nan"), float("nan")]
    h = T1SIDED[len(d) - 1] * st.stdev(d) / math.sqrt(len(d))
    return [m, m - h, m + h]


def main():
    ap = argparse.ArgumentParser()
    for x in ("--mech", "--budget", "--horizon", "--root", "--out"):
        ap.add_argument(x, required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cal = json.load(open(os.path.join(a.root, "phase3_calibrate", "calibration.json")))["NSFNET"]
    loads, prim = cal["loads"], cal["primary_load"]
    ip = loads.index(prim)

    M = json.load(open(a.mech))
    groups = defaultdict(list)
    for k, v in M.items():
        if not k.startswith("dev|"):
            continue
        _, _, _, _, tau, _, load = k.split("|")
        load = float(load)
        if load == prim:
            groups[tau].extend(v["decisions"])
        groups[f"{tau}@{load}"].extend(v["decisions"])
    ms = {}
    for g, recs in groups.items():
        if "@" in g and not g.startswith("0.05"):
            continue
        ms[g] = mech_summary(recs)
        if not g.startswith("0.0@") and g not in ("0.0",):
            ms[g]["by_age_quartile"] = by_age(recs)
    json.dump(ms, open(os.path.join(a.out, "mechanism_summary.json"), "w"), indent=1)

    for name, path in (("budget", a.budget), ("horizon", a.horizon)):
        S = json.load(open(path))
        out = defaultdict(dict)
        for k, cell in S.items():
            parts = k.split("|")
            tau, seed, var = parts[3], parts[4], parts[5]
            lab = {"250000": "250k", "1000000": "1M"}.get(var, var.replace("gamma", "g"))
            out[f"{tau}|{seed}"][lab] = per_load(cell["evals"], loads)
        for key, d in out.items():
            labs = ("250k", "1M") if name == "budget" else ("g0.99", "g0.999")
            if all(l in d for l in labs):
                d[f"primary_{labs[0]}_minus_{labs[1]}"] = d[labs[0]][ip] - d[labs[1]][ip]
        json.dump(out, open(os.path.join(a.out, f"{name}_compare.json"), "w"), indent=1)
    print("written to", a.out)


if __name__ == "__main__":
    main()
