"""Result figures of the stale-state study.

  python scripts/phase3_figures.py --root RESULTS --heur HEUR_STATE --greedy GREEDY_STATE --out DIR

fig3_blocking_vs_load       blocking vs load: 3 topologies x tau in {0, 0.05}; heuristics, greedy (exploratory), learners.
fig4_staleness_signalling   staleness penalty and signalling at each topology's primary load.
fig5_ablations              NSFNET mechanism ablations (paired by training seed, one-sided 95% bounds).
Colours: validated categorical palette (light surface), always paired with marker + line style
so identity never depends on colour (print / colour-vision deficiency).
"""
import argparse, json, math, os, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_report import heur_table, learner_table, mean_ci  # noqa: E402

TOPOS = [("NSFNET", "NSFNET"), ("USNET", "USNET"), ("EUROPEAN", "Pan-European")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
STYLE = {  # label: (colour, marker, linestyle)
    "KSP-AM-FF": ("#2a78d6", "o", "-"),
    "best stale-aware heuristic*": ("#eb6834", "s", "-"),
    "KSP-AM-greedy (expl.)": ("#1baf7a", "D", "--"),
    "A-QMIX": ("#6250d6", "^", "-"),
    "A-QMIX-myopic": ("#e87ba4", "v", ":"),
    "DeepRMSA-QoT": ("#eda100", "P", "-."),
}
HEURS = ["KSP-AM-FF", "KSP-AM-BF", "KSP-AM-RF", "KSP-AM-FLF"]
SHORT = {"KSP-AM-FLF": "FLF", "KSP-AM-BF": "BF", "KSP-AM-RF": "RF", "KSP-AM-FF": "FF"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.3, "lines.markersize": 4,
                     "legend.frameon": False, "savefig.dpi": 600})


def main():
    ap = argparse.ArgumentParser()
    for a in ("--root", "--heur", "--greedy", "--out"):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cal = json.load(open(os.path.join(a.root, "phase3_calibrate", "calibration.json")))
    H = heur_table(json.load(open(a.heur))["cells"])
    L = learner_table(json.load(open(os.path.join(a.root, "phase3_confirm", "state.json")))["cells"])
    G = json.load(open(a.greedy))

    def h(topo, tau, m, ld, sig="forward", K="K5", R="R0"):
        return st.mean(v[0] for v in H[(topo, tau, sig, K, R, m)][ld].values())

    def g(topo, tau, ld):
        return st.mean(e["bp"] for e in G.get(f"greedy|{topo}|{tau}", []) if e["load"] == ld)

    # ---------------- blocking vs load
    bests = {}
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 4.0), sharey="row")
    for c, (topo, title) in enumerate(TOPOS):
        loads = cal[topo]["loads"]
        for r, tau in enumerate((0.0, 0.05)):
            ax = axs[r, c]
            best = min(HEURS, key=lambda m: h(topo, tau, m, loads[2]))
            series = [("KSP-AM-FF", [h(topo, tau, "KSP-AM-FF", x) for x in loads], None)]
            if tau > 0:
                series.append(("best stale-aware heuristic*", [h(topo, tau, best, x) for x in loads], None))
                bests[topo] = SHORT[best]
            if f"greedy|{topo}|{tau}" in G:
                series.append(("KSP-AM-greedy (expl.)", [g(topo, tau, x) for x in loads], None))
            for nm in ("A-QMIX", "A-QMIX-myopic", "DeepRMSA-QoT"):
                ss = L.get((nm, topo, tau), {})
                if len(ss) >= 2:
                    series.append((nm, [st.mean(ss[s][x][0] for s in ss) for x in loads], None))
            for nm, v, tag in series:
                col, mk, ls = STYLE[nm]
                ax.plot(loads, [100 * max(y, 1e-4) for y in v], color=col, marker=mk, ls=ls, label=nm,
                        markeredgecolor="white", markeredgewidth=0.5)
                if tag:
                    ax.annotate(tag, (loads[0], v[0]), xytext=(-2, 4), textcoords="offset points",
                                fontsize=6.5, color=INK, ha="right")
            ax.set_yscale("log"); ax.set_ylim(0.01 if tau == 0 else 2.0, 30)
            tk = [0.01, 0.1, 1, 10] if tau == 0 else [2, 5, 10, 20]
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(tk))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:g}" for t in tk]))
            ax.grid(True, which="major", color=GRID, lw=0.5); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_title(f"{title}, $\\tau$ = {tau:g}", fontsize=7.5, color=INK)
            if r == 1:
                ax.set_xlabel("Offered load (Erlang)")
            if c == 0:
                ax.set_ylabel("Blocking probability (%)")
    hs, ls_ = [], []
    for ax in axs.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in ls_:
                hs.append(hh); ls_.append(ll)
    note = "*best stale-aware heuristic at the primary load: " + ", ".join(
        f"{t} {bests[k]}" for k, t in TOPOS if k in bests) + ". Values below 0.01% are drawn at 0.01%."
    fig.text(0.5, 0.005, note, ha="center", fontsize=6.3, color=MUTED)
    fig.legend(hs, ls_, loc="lower center", ncol=6, fontsize=6.8, bbox_to_anchor=(0.5, 0.025),
               handlelength=2.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig3_blocking_vs_load.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # ---------------- staleness penalty and signalling (primary load)
    bars = [("Fresh state ($\\tau$ = 0)", "#2a78d6"), ("$\\tau$ = 0.05, forward, no retry", "#eb6834"),
            ("$\\tau$ = 0.05, one crank-back retry", "#1baf7a"), ("$\\tau$ = 0.05, backward reservation", "#6250d6")]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    w = 0.19
    tab = {}
    for i, (topo, title) in enumerate(TOPOS):
        p = cal[topo]["primary_load"]
        vals = [h(topo, 0.0, "KSP-AM-FF", p), h(topo, 0.05, "KSP-AM-FF", p),
                min(h(topo, 0.05, m, p, "forward", "K5", "R1") for m in HEURS),
                h(topo, 0.05, "KSP-AM-FF", p, "backward")]
        tab[topo] = vals
        for j, ((lab, col), v) in enumerate(zip(bars, vals)):
            x = i + (j - 1.5) * (w + 0.02)
            ax.bar(x, 100 * v, w, color=col, label=lab if i == 0 else None, edgecolor="white", linewidth=0.8)
            ax.text(x, 100 * v + 0.12, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=5.6, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{t}\n{cal[k]['primary_load']:.0f} E" for k, t in TOPOS])
    ax.set_ylabel("Blocking probability (%)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig4_staleness_signalling.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # ---------------- ablations (NSFNET, primary load)
    topo, p = "NSFNET", cal["NSFNET"]["primary_load"]
    contr = [("A-Independent", "no mixer"), ("W-QMIX", "no anchor"), ("A-QMIX-myopic", "myopic ($\\gamma$ = 0)")]
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    rows, abl = [], {}
    for ti, tau in enumerate((0.0, 0.01, 0.05)):
        base = L.get(("A-QMIX", topo, tau), {})
        for ci, (other, lab) in enumerate(contr):
            o = L.get((other, topo, tau), {})
            common = sorted(set(base) & set(o))
            if len(common) < 2:
                continue
            d = [100 * (o[s][p][0] - base[s][p][0]) for s in common]
            m, lo, hi = mean_ci(d)
            abl[f"{other}|{tau}"] = [m, lo, hi]
            y = ti * 4 + ci
            col, mk = ("#2a78d6", "o") if ci == 0 else (("#eb6834", "s") if ci == 1 else ("#e87ba4", "v"))
            ax.plot([lo, hi], [y, y], color=col, lw=1.4)
            ax.plot(m, y, marker=mk, color=col, markeredgecolor="white", markeredgewidth=0.5,
                    label=lab if ti == 0 else None)
            rows.append(y)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks([1, 5, 9]); ax.set_yticklabels(["$\\tau$ = 0", "$\\tau$ = 0.01", "$\\tau$ = 0.05"])
    ax.invert_yaxis()
    ax.set_xlim(-1.9, 8.1); ax.set_xticks([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_xlabel("Blocking difference vs A-QMIX (percentage points)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig5_ablations.{ext}"), bbox_inches="tight")
    plt.close(fig)
    json.dump({"fig4_primary_load_bp": tab, "fig5_ablations_pp": abl},
              open(os.path.join(a.out, "figures_data.json"), "w"), indent=1)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
