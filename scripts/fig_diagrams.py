"""Schematic diagrams: (a, b) distributed stale-state model; (a, b) anchored multi-agent learner. Each panel is a separate file."""
import sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Ellipse
out = sys.argv[1]
INK, MUT, BLUE, ORG, GRN, VIO, FILL = "#1f1f1e", "#6b6a64", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#f3f2ee"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "savefig.dpi": 600})
W = 6.6
def canvas(h):
    fig, ax = plt.subplots(figsize=(W, h)); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.03, 1.0); ax.axis("off"); return fig, ax, h
def box(ax, x, y, w, h, text, fc=FILL, ec=MUT, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)
def arrow(ax, x1, y1, x2, y2, c=MUT):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, color=c, lw=0.8))
def strip(ax, x, y, occ, w=0.024, h=0.07, hl=None, hc=ORG):
    for i, o in enumerate(occ):
        fc = "#9b9a92" if o == 1 else ("white" if o == 0 else o)
        ax.add_patch(Rectangle((x + i * w, y), w, h, fc=fc, ec=MUT, lw=0.4))
    if hl:
        ax.add_patch(Rectangle((x + hl[0] * w, y), (hl[1] - hl[0]) * w, h, fc="none", ec=hc, lw=1.4))
def save(fig, name):
    for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"{name}.{e}"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

# ---------------- stale views
fig, ax, H = canvas(2.4)
y0 = 0.86; ax.plot([0.02, 0.80], [y0, y0], color=INK, lw=0.8); arrow(ax, 0.79, y0, 0.81, y0, c=INK)
for k, x in enumerate((0.04, 0.38, 0.72)):
    ax.plot([x, x], [y0 - 0.05, y0 + 0.05], color=BLUE, lw=1.4); ax.text(x, y0 + 0.07, ["0", "τ", "2τ"][k], ha="center", color=BLUE, fontsize=7)
ax.text(0.82, y0, "time", va="center", fontsize=6.5, color=MUT)
for x, lab, c, dy in ((0.17, "request at node A", ORG, 0.07), (0.26, "request at node B", VIO, -0.10)):
    ax.plot(x, y0, marker="v", color=c, ms=5); ax.text(x, y0 + dy, lab, ha="center", fontsize=6, color=c)
ax.text(0.40, y0 - 0.12, "blue ticks: snapshots of the true state broadcast to all nodes", fontsize=5.8, color=BLUE)
rows = (("True state when B reserves", INK, [1, 1, ORG, ORG, 0, 0, 0, 0, 0, 0, 0, 0], (2, 4), VIO, 0.56),
        ("View of node A", ORG, [1, 1] + [0] * 10, (2, 4), ORG, 0.42),
        ("View of node B", VIO, [1, 1] + [0] * 10, (2, 4), VIO, 0.28))
for lab, col, occ, hl, hc, yy in rows:
    ax.text(0.02, yy + 0.035, lab, fontsize=6.6, color=col, va="center"); strip(ax, 0.30, yy, occ, hl=hl, hc=hc)
ax.text(0.62, 0.385, "view = last snapshot\n+ the node's own changes", fontsize=6.2, color=MUT, va="center")
ax.text(0.02, 0.17, "Both nodes choose slots 3–4 (first-fit) on the same stale view. A reserves first (orange); with forward signalling\n"
        "B's reservation collides on the true state: a setup failure, so B is blocked.", fontsize=6.3, color=INK, va="top")
save(fig, "fig2a_stale_views")

# ---------------- signalling
fig, ax, H = canvas(2.3)
for yy, title, desc, fwd in ((0.72, "Forward (source-selected)", "slots chosen on the view; the reservation\nfails if they are already taken on any link", True),
                             (0.36, "Backward (destination-selected)", "free slots are collected along the path;\nthe destination chooses on the true state", False)):
    xs = [0.06, 0.24, 0.42]
    for i, x in enumerate(xs):
        ax.add_patch(Ellipse((x, yy), 0.045, 0.045 * W / H, fc="white", ec=INK, lw=0.8)); ax.text(x, yy, "SID"[i], ha="center", va="center", fontsize=6.5)
    ax.plot([xs[0] + 0.023, xs[2] - 0.023], [yy, yy], color=MUT, lw=0.8, zorder=0)
    if fwd:
        arrow(ax, 0.08, yy + 0.11, 0.40, yy + 0.11, c=ORG); ax.text(0.24, yy + 0.15, "PATH / reserve the chosen slots", ha="center", fontsize=6, color=ORG)
    else:
        arrow(ax, 0.08, yy + 0.11, 0.40, yy + 0.11, c=MUT); ax.text(0.24, yy + 0.15, "PATH (collect the free slots)", ha="center", fontsize=6, color=MUT)
        arrow(ax, 0.40, yy - 0.11, 0.08, yy - 0.11, c=GRN); ax.text(0.24, yy - 0.20, "RESV with the chosen slots", ha="center", fontsize=6, color=GRN)
    ax.text(0.50, yy + 0.04, title, fontsize=7, color=INK, weight="bold", va="center")
    ax.text(0.50, yy - 0.08, desc, fontsize=6.3, color=INK, va="center")
ax.text(0.02, 0.0, "Blocking = setup failures + view blocks. With τ = 0 every view equals the true state\nand the model reduces exactly to the centralised model.", fontsize=6.0, color=MUT, va="bottom")
save(fig, "fig2b_signalling")

# ---------------- decision of a source node
fig, ax, H = canvas(2.3)
box(ax, 0.00, 0.52, 0.19, 0.40, "Observation o\nview features of the\nK candidate paths,\nrequest, node id,\nage of the view", fs=5.8)
box(ax, 0.22, 0.52, 0.20, 0.40, "Shared Q-network\n(dueling head)\nQ = V + A − mean(A)\nadvantage layer\ninitialised to 0", fc="#e8f1fc", ec=BLUE, fs=5.8)
box(ax, 0.22, 0.06, 0.20, 0.30, "Anchor a_h:\nKSP-AM-FF\non the view", fc="#fdeee6", ec=ORG, fs=5.8)
box(ax, 0.45, 0.40, 0.30, 0.52, "Anchored greedy rule\n\nπ(o) = a*  if  Q(o,a*) − Q(o,a_h) > δ,\nelse a_h\n\na* = best view-feasible action", fc="#eaf7f1", ec=GRN, fs=5.8)
box(ax, 0.78, 0.52, 0.21, 0.40, "Action (k, m, j):\ncandidate path k,\nformat m,\nj-th free block (j ≤ J)", fs=5.8)
arrow(ax, 0.19, 0.72, 0.22, 0.72); arrow(ax, 0.42, 0.72, 0.45, 0.72); arrow(ax, 0.42, 0.21, 0.45, 0.46); arrow(ax, 0.75, 0.72, 0.78, 0.72)
ax.text(0.46, 0.30, "Untrained: all advantages are equal, so π = a_h (the heuristic).\nLearning moves π away only where an advantage has been learned.\nExecution uses only the node's view.", fontsize=6.2, color=MUT, va="top")
save(fig, "fig3a_decision")

# ---------------- window-level training
fig, ax, H = canvas(2.5)
ax.text(0.01, 0.95, "One transition = one link-state update window (all decisions between two snapshots)", fontsize=6.6, color=INK, va="top")
rows = (("node 1: U₁ = Σ Q(o, a) of its decisions", 0.74), ("node 2: U₂", 0.60), (None, 0.49), ("node N: U_N", 0.38))
for lab, yy in rows:
    if lab is None:
        ax.text(0.15, yy, "⋮", ha="center", va="center", fontsize=9, color=MUT); continue
    box(ax, 0.00, yy - 0.05, 0.33, 0.10, lab, fs=5.6); arrow(ax, 0.33, yy, 0.42, 0.56)
box(ax, 0.42, 0.40, 0.21, 0.32, "QMIX mixer\nQ_tot = f(U₁ … U_N; s)\nweights ≥ 0\n(monotonic)", fc="#f0eefc", ec=VIO, fs=5.6)
box(ax, 0.38, 0.05, 0.29, 0.18, "hypernetworks ← true global state s\nat the window start", fs=5.6)
arrow(ax, 0.525, 0.23, 0.525, 0.40, c=VIO)
arrow(ax, 0.63, 0.56, 0.67, 0.56, c=VIO)
box(ax, 0.67, 0.40, 0.33, 0.32, "Target: y = G + γᵏ Q_tot⁻(next window)\nG = discounted rewards in the window\nloss = Huber(Q_tot − y)\nDouble-DQN with the anchored rule", fs=5.4)
ax.text(0.69, 0.30, "Centralised training only: the mixer and the\nglobal state are not used when acting.", fontsize=6.0, color=MUT, va="top")
save(fig, "fig3b_training")
print("ok")
