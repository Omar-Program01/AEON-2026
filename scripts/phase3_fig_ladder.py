"""Baseline ladder: the verdict depends on the baseline. Blocking ratio learner / baseline at each topology's 1% load.
Input: ladder.json (produced by scripts/phase3_ladder.py)."""
import json, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, FixedLocator, FixedFormatter
lad = json.load(open(sys.argv[1])); out = sys.argv[2]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
TOP = [("NSFNET", "NSFNET", "#2a78d6", "o"), ("USNET", "USNET", "#eb6834", "s"), ("EUROPEAN", "Pan-European", "#1baf7a", "D")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 600})
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={"width_ratios": [4, 5]})
for ax, key, title in ((axs[0], "tau0", "Fresh state ($\\tau$ = 0): best learner (A-QMIX-myopic)"),
                       (axs[1], "tau05", "Stale state ($\\tau$ = 0.05): A-QMIX")):
    names = []
    for t, _, _, _ in TOP:
        for n in lad[t][key]["base"]:
            n2 = "best spreading heuristic" if n.startswith("best spreading") else n
            if n2 not in names:
                names.append(n2)
    for ti, (t, lab, col, mk) in enumerate(TOP):
        L = lad[t][key]["L"]
        for n, b in lad[t][key]["base"].items():
            n2 = "best spreading heuristic" if n.startswith("best spreading") else n
            y = names.index(n2) + (ti - 1) * 0.18
            ax.plot(L / b, y, marker=mk, color=col, ms=5, ls="none", markeredgecolor="white", markeredgewidth=0.5,
                    label=lab if (key == "tau0" and n2 == names[0]) else None)
    ax.axvline(1.0, color=MUTED, lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(0.4, 6)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4])); ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2", "4"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=7.5, color=INK, pad=14)
    ax.set_xlabel("Blocking ratio, learner / baseline (log)")
    ax.text(0.97, 1.0, "\u2190 learner better", fontsize=6.3, color=MUTED, va="bottom", ha="right", transform=ax.get_xaxis_transform())
    ax.text(1.03, 1.0, "baseline better \u2192", fontsize=6.3, color=MUTED, va="bottom", ha="left", transform=ax.get_xaxis_transform())
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(w_pad=1.5, rect=(0, 0.07, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"fig6_baseline_ladder.{ext}"), bbox_inches="tight")
print("ok")
