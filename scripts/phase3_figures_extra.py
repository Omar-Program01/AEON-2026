"""Additional result figures:
 fig_tau        blocking vs staleness tau at each topology's 1% load
 fig_decomp     setup failures vs view blocks at tau = 0.05 (1% load)
 fig_deviation  A-QMIX deviation analysis (NSFNET): (a) vs view age, (b) deviation types
 fig_training   training curves (NSFNET; tau = 0 and 0.05; mean and range over 5 seeds)
 fig_robust     budget (4x) and horizon (gamma 0.999) checks, paired by seed
"""
import argparse, csv, glob, json, os, statistics as st, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_report import heur_table, learner_table, mean_ci  # noqa
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#eda100"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "lines.linewidth": 1.3,
                     "lines.markersize": 4, "legend.frameon": False, "savefig.dpi": 600})
TOPOS = [("NSFNET", "NSFNET"), ("USNET", "USNET"), ("EUROPEAN", "Pan-European")]
HEURS = ["KSP-AM-FF", "KSP-AM-BF", "KSP-AM-RF", "KSP-AM-FLF"]
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
def save(fig, out, name):
    for e in ("pdf", "png"): fig.savefig(os.path.join(out, f"{name}.{e}"), bbox_inches="tight")
    plt.close(fig)

ap = argparse.ArgumentParser()
for a in ("--root", "--heur", "--mech", "--budget", "--horizon", "--out"): ap.add_argument(a, required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
cal = json.load(open(os.path.join(a.root, "phase3_calibrate", "calibration.json")))
Hs = json.load(open(a.heur))["cells"]; H = heur_table(Hs)
Cs = json.load(open(os.path.join(a.root, "phase3_confirm", "state.json")))["cells"]; L = learner_table(Cs)
def h(t, tau, m, ld, sig="forward", K="K5", R="R0", idx=0):
    return st.mean(v[idx] for v in H[(t, tau, sig, K, R, m)][ld].values())
def lm(name, t, tau, ld, idx=0):
    ss = L[(name, t, tau)]; return st.mean(ss[s][ld][idx] for s in ss)

# ---------------- fig_tau
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.4), sharey=True)
taus = [0.0, 0.01, 0.05]
for ax, (t, title) in zip(axs, TOPOS):
    p = cal[t]["primary_load"]
    series = [("KSP-AM-FF", [h(t, x, "KSP-AM-FF", p) for x in taus], C[0], "o", "-"),
              ("best stale-aware heuristic", [min(h(t, x, m, p) for m in HEURS) for x in taus], C[1], "s", "-"),
              ("KSP-AM-FF, backward reservation", [h(t, x, "KSP-AM-FF", p, "backward") for x in taus], C[2], "D", "--"),
              ("A-QMIX", [lm("A-QMIX", t, x, p) for x in taus], C[3], "^", "-"),
              ("A-QMIX-myopic", [lm("A-QMIX-myopic", t, x, p) for x in taus], C[4], "v", ":"),
              ("DeepRMSA-QoT", [lm("DeepRMSA-QoT", t, x, p) for x in taus], C[5], "P", "-.")]
    for nm, v, col, mk, ls in series:
        ax.plot([0, 1, 2], [100 * y for y in v], color=col, marker=mk, ls=ls, label=nm, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["0", "0.01", "0.05"]); ax.set_xlabel("Update period τ (holding times)")
    ax.set_title(f"{title}, {p:.0f} E", fontsize=7.5, color=INK); clean(ax)
axs[0].set_ylabel("Blocking probability (%)")
hs, ls_ = axs[0].get_legend_handles_labels()
fig.legend(hs, ls_, loc="lower center", ncol=3, fontsize=6.6, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.12, 1, 1)); save(fig, a.out, "fig_tau")

# ---------------- fig_decomp (tau = 0.05, primary load): stacked setup failure + view block
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.4), sharey=True)
for ax, (t, title) in zip(axs, TOPOS):
    p = cal[t]["primary_load"]
    best = min(HEURS, key=lambda m: h(t, 0.05, m, p))
    rows = [("FF", h(t, 0.05, "KSP-AM-FF", p), h(t, 0.05, "KSP-AM-FF", p, idx=1)),
            (best[7:], h(t, 0.05, best, p), h(t, 0.05, best, p, idx=1)),
            ("FF bwd", h(t, 0.05, "KSP-AM-FF", p, "backward"), h(t, 0.05, "KSP-AM-FF", p, "backward", idx=1)),
            ("A-QMIX", lm("A-QMIX", t, 0.05, p), lm("A-QMIX", t, 0.05, p, 1)),
            ("A-Ind", lm("A-Independent", t, 0.05, p), lm("A-Independent", t, 0.05, p, 1)),
            ("DQoT", lm("DeepRMSA-QoT", t, 0.05, p), lm("DeepRMSA-QoT", t, 0.05, p, 1))]
    x = range(len(rows))
    sf = [100 * r[2] for r in rows]; vb = [100 * (r[1] - r[2]) for r in rows]
    ax.bar(x, sf, 0.62, color=C[1], label="setup failures (collision on the true state)", edgecolor="white", linewidth=0.8)
    ax.bar(x, vb, 0.62, bottom=sf, color=C[0], label="view blocks (no feasible action in the view)", edgecolor="white", linewidth=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right", fontsize=6.5)
    ax.set_title(f"{title}, τ = 0.05, {p:.0f} E", fontsize=7.5, color=INK); clean(ax)
axs[0].set_ylabel("Blocking probability (%)")
hs, ls_ = axs[0].get_legend_handles_labels()
fig.legend(hs, ls_, loc="lower center", ncol=2, fontsize=6.6, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.1, 1, 1)); save(fig, a.out, "fig_decomp")

# ---------------- fig_deviation
M = json.load(open(a.mech))
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axs[0]; qx = [0.125, 0.375, 0.625, 0.875]
for tau, col, mk in (("0.05", C[3], "^"), ("0.01", C[0], "o")):
    q = M[tau]["by_age_quartile"]; ks = sorted(q, key=float)
    ax.plot(qx, [100 * q[k]["dev"] for k in ks], color=col, marker=mk, label=f"deviation rate, τ = {tau}")
    ax.plot(qx, [100 * q[k]["setupfail"] * 10 for k in ks], color=col, marker=mk, ls="--", label=f"setup-failure rate ×10, τ = {tau}")
ax.set_xlabel("Age of the node's view (fraction of τ since the last update)"); ax.set_ylabel("Share of decisions (%)")
ax.set_xticks(qx); ax.set_xticklabels(["0–¼", "¼–½", "½–¾", "¾–1"]); ax.set_ylim(0, 100); clean(ax)
ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28)); ax.set_title("(a) Deviation does not adapt to staleness", fontsize=7.5, color=INK)
ax = axs[1]; cats = [("0.0", "τ = 0"), ("0.01", "τ = 0.01"), ("0.05", "τ = 0.05")]
y = range(len(cats)); left = [0] * len(cats)
for key, lab, col in (("frac_block", "other block", C[0]), ("frac_path", "other path", C[1]), ("frac_format", "other format", C[2])):
    v = [100 * M[c][key] for c, _ in cats]
    ax.barh(list(y), v, 0.6, left=left, color=col, label=lab, edgecolor="white", linewidth=0.8)
    left = [l + w for l, w in zip(left, v)]
ax.barh(list(y), [100 - l for l in left], 0.6, left=left, color="#d9d8d2", label="anchor kept", edgecolor="white", linewidth=0.8)
ax.set_yticks(list(y)); ax.set_yticklabels([c[1] for c in cats]); ax.set_xlabel("Share of decisions (%)"); ax.set_xlim(0, 100)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28)); ax.set_title("(b) How A-QMIX deviates from its anchor", fontsize=7.5, color=INK)
fig.tight_layout(); save(fig, a.out, "fig_deviation")

# ---------------- fig_training (NSFNET, confirmation runs)
def logs(name, tau):
    out = []
    for k, cell in Cs.items():
        _, nm, t, tt, s = k.split("|")
        if nm == name and t == "NSFNET" and float(tt) == tau:
            p = os.path.join(a.root, cell["result"]["exp_id"], "train_log.csv")
            if os.path.exists(p):
                r = list(csv.DictReader(open(p)))
                out.append(([int(x["arrivals_done"]) for x in r], [float(x["train_bp"]) for x in r]))
    return out
def smooth(v, w=8):
    return [st.mean(v[max(0, i - w + 1):i + 1]) for i in range(len(v))]
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.4), sharey=False)
for ax, tau in zip(axs, (0.0, 0.05)):
    for (nm, col) in (("A-QMIX", C[3]), ("A-Independent", C[0]), ("W-QMIX", C[1]), ("DeepRMSA-QoT", C[5])):
        lg = logs(nm, tau)
        if not lg: continue
        xs = lg[0][0]; ys = [smooth(y) for _, y in lg]
        n = min(len(y) for y in ys); m = [st.mean(y[i] for y in ys) for i in range(n)]
        lo = [min(y[i] for y in ys) for i in range(n)]; hi = [max(y[i] for y in ys) for i in range(n)]
        ax.fill_between([x / 1e3 for x in xs[:n]], [100 * v for v in lo], [100 * v for v in hi], color=col, alpha=0.15, lw=0)
        ax.plot([x / 1e3 for x in xs[:n]], [100 * v for v in m], color=col, label=nm, lw=1.1)
    ax.axvline(100, color=MUTED, lw=0.6, ls=":"); ax.text(102, ax.get_ylim()[1] * 0.95, "end of load\ncurriculum", fontsize=5.5, color=MUTED, va="top")
    ax.set_xlabel("Training arrivals (thousands)"); ax.set_ylabel("Training blocking (%)")
    ax.set_title(f"NSFNET, τ = {tau:g} (training stream, ε-greedy)", fontsize=7.5, color=INK); clean(ax)
hs, ls_ = axs[0].get_legend_handles_labels()
fig.legend(hs, ls_, loc="lower center", ncol=4, fontsize=6.6, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, a.out, "fig_training")

# ---------------- fig_robust
B = json.load(open(a.budget)); Hh = json.load(open(a.horizon))
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.2), sharey=True)
for ax, D, k0, k1, l0, l1, title in ((axs[0], B, "250k", "1M", "250k arrivals", "1M arrivals", "(a) Training budget ×4"),
                                     (axs[1], Hh, "g0.99", "g0.999", "γ = 0.99", "γ = 0.999", "(b) Discount horizon ×10")):
    keys = ["0.0|s200", "0.0|s201", "0.05|s200", "0.05|s201"]
    lab = ["τ=0\nseed 200", "τ=0\nseed 201", "τ=0.05\nseed 200", "τ=0.05\nseed 201"]
    x = range(len(keys))
    ax.bar([i - 0.18 for i in x], [100 * D[k][k0][2] for k in keys], 0.34, color=C[0], label=l0, edgecolor="white", linewidth=0.8)
    ax.bar([i + 0.18 for i in x], [100 * D[k][k1][2] for k in keys], 0.34, color=C[1], label=l1, edgecolor="white", linewidth=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(lab, fontsize=6.3); ax.set_title(title + " (A-QMIX, NSFNET, 153 E)", fontsize=7.5, color=INK)
    ax.legend(fontsize=6.5, loc="upper left"); clean(ax)
axs[0].set_ylabel("Blocking probability (%)")
fig.tight_layout(); save(fig, a.out, "fig_robust")
print("ok")
