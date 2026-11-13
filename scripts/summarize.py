"""
Summarise a set of cells (by tag): per-method evaluation metrics (training seed
= statistical unit; evaluation seeds averaged within a training seed), learning
curves, timing, parameters and wall-clock cost.

  python scripts/summarize.py --tag pilot
writes results/summary_<tag>.md, results/summary_<tag>.json, results/curves_<tag>.png
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

METHOD_ORDER = ["KSP-FF", "KSP-AM-FF", "KSP-AM-BF", "DeepRMSA", "PPO", "GNN-DRL",
                "Centralized-DQN", "Independent", "QMIX"]


def load(root, tag):
    cells = []
    with open(os.path.join(root, f"index_{tag}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            d = os.path.join(root, r["exp_id"])
            r["manifest"] = json.load(open(os.path.join(d, "manifest.json")))
            r["eval"] = json.load(open(os.path.join(d, "eval.json")))
            tp = os.path.join(d, "timing.json")
            r["timing"] = json.load(open(tp)) if os.path.exists(tp) else None
            lp = os.path.join(d, "train_log.csv")
            r["log"] = list(csv.DictReader(open(lp))) if os.path.exists(lp) else []
            cells.append(r)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    cells = load(a.root, a.tag)
    by = defaultdict(list)
    for c in cells:
        by[(c["topology"], c["method"])].append(c)
    out = {}
    lines = [f"# Summary: {a.tag}\n",
             "Training seed is the unit; each seed's value is the mean over its evaluation seeds.\n",
             "| Topology | Method | Seeds | BP per training seed | BP mean | BBP mean | SE | Frag | QoT viol. | "
             "Decide ms (median) | Context ms (median) | Exec. params | Train min/seed |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = sorted(by, key=lambda k: (k[0], METHOD_ORDER.index(k[1]) if k[1] in METHOD_ORDER else 99))
    for k in keys:
        cs = sorted(by[k], key=lambda c: c["seed"])
        per = []
        for c in cs:
            ev = c["eval"]
            per.append({m: float(np.mean([e[m] for e in ev])) for m in ("bp", "bbp", "se", "frag", "qot_violation")})
        bp = [p["bp"] for p in per]
        row = {m: float(np.mean([p[m] for p in per])) for m in per[0]}
        row["bp_seeds"] = bp
        row["n_seeds"] = len(cs)
        row["decide_ms"] = float(np.median([c["timing"]["decide_ms"]["median"] for c in cs if c["timing"]]))
        row["env_ms"] = float(np.median([c["timing"]["env_ms"]["median"] for c in cs if c["timing"]]))
        row["params"] = cs[0]["manifest"].get("n_parameters")
        row["train_min"] = float(np.mean([c["train_s"] for c in cs])) / 60.0
        row["exp_ids"] = [c["exp_id"] for c in cs]
        out[f"{k[0]}/{k[1]}"] = row
        lines.append(f"| {k[0]} | {k[1]} | {len(cs)} | {', '.join(f'{x:.4f}' for x in bp)} | {row['bp']:.4f} | "
                     f"{row['bbp']:.4f} | {row['se']:.2f} | {row['frag']:.3f} | {row['qot_violation']:.4f} | "
                     f"{row['decide_ms']:.3f} | {row['env_ms']:.3f} | {row['params']} | {row['train_min']:.1f} |")
    md = os.path.join(a.root, f"summary_{a.tag}.md")
    open(md, "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(a.root, f"summary_{a.tag}.json"), "w"), indent=1)
    print("\n".join(lines))

    # learning curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    learned = [k for k in keys if by[k][0]["log"]]
    if learned:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, k in enumerate(learned):
            for c in by[k]:
                x = [int(r["arrivals_done"]) / 1000 for r in c["log"]]
                y = [float(r["train_bp"]) for r in c["log"]]
                z = [float(r["reward_per_arrival"]) for r in c["log"]]
                lab = k[1] if c["seed"] == min(cc["seed"] for cc in by[k]) else None
                axes[0].plot(x, y, color=colors[i % len(colors)], lw=1, alpha=0.85, label=lab)
                axes[1].plot(x, z, color=colors[i % len(colors)], lw=1, alpha=0.85)
        for k in keys:
            if not by[k][0]["log"] and k[1] == "KSP-AM-FF":
                v = out[f"{k[0]}/{k[1]}"]["bp"]
                axes[0].axhline(v, color="k", ls="--", lw=1, label="KSP-AM-FF (eval, 300 E)")
        axes[0].set_xlabel("Training arrivals (thousands)")
        axes[0].set_ylabel("Training blocking (exploring, curriculum load)")
        axes[1].set_xlabel("Training arrivals (thousands)")
        axes[1].set_ylabel("Mean reward per arrival")
        axes[0].legend(fontsize=7, ncol=2)
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.axvline(0.4 * float(by[learned[0]][0]["log"][-1]["arrivals_done"]) / 1000, color="grey", lw=0.8, ls=":")
        fig.suptitle(f"Training curves ({a.tag}); dotted line = end of load curriculum", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(a.root, f"curves_{a.tag}.png"), dpi=150)


if __name__ == "__main__":
    main()
