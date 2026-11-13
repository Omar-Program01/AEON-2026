"""Baseline ladder: blocking of the best learner and of successively stronger baselines at each topology's
primary (1%) load. Fresh state: A-QMIX-myopic; tau = 0.05: A-QMIX. Writes ladder.json for phase3_fig_ladder.py.

  python scripts/phase3_ladder.py --root RESULTS_ROOT --heur HEURISTICS_STATE_JSON --kcheck KCHECK_STATE_JSON --out ladder.json
"""
import argparse, json, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_report import heur_table, learner_table  # noqa: E402

TOPOS = ["NSFNET", "USNET", "EUROPEAN"]
SPREAD = ["KSP-AM-BF", "KSP-AM-RF", "KSP-AM-FLF"]


def main():
    ap = argparse.ArgumentParser()
    for a in ("--root", "--heur", "--kcheck", "--out"):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    cal = json.load(open(os.path.join(a.root, "phase3_calibrate", "calibration.json")))
    H = heur_table(json.load(open(a.heur))["cells"])
    L = learner_table(json.load(open(os.path.join(a.root, "phase3_confirm", "state.json")))["cells"])
    K = json.load(open(a.kcheck))

    def h(t, tau, m, ld, sig="forward", k="K5", r="R0"):
        return st.mean(v[0] for v in H[(t, tau, sig, k, r, m)][ld].values())

    def kc(label, t, tau, ld):
        return st.mean(e["bp"] for e in K[f"kcheck|{label}|{t}|{tau}"] if e["load"] == ld)

    def lm(name, t, tau, ld):
        ss = L[(name, t, tau)]
        return st.mean(ss[s][ld][0] for s in ss)

    out = {}
    for t in TOPOS:
        p = cal[t]["primary_load"]
        best = min(SPREAD, key=lambda m: h(t, 0.05, m, p))
        out[t] = {
            "tau0": {"learner": "A-QMIX-myopic", "L": lm("A-QMIX-myopic", t, 0.0, p),
                     "base": {"KSP-AM-FF (length, K = 5)": h(t, 0.0, "KSP-AM-FF", p),
                              "KSP-AM-FF (length, K = 10)": h(t, 0.0, "KSP-AM-FF", p, k="K10"),
                              "KSP-AM-FF (hop count, K = 5)": kc("FF-hops-K5", t, 0.0, p),
                              "KSP-AM-FF (hop count, K = 10)": kc("FF-hops-K10", t, 0.0, p)}},
            "tau05": {"learner": "A-QMIX", "L": lm("A-QMIX", t, 0.05, p),
                      "base": {"KSP-AM-FF (length, K = 5)": h(t, 0.05, "KSP-AM-FF", p),
                               "KSP-AM-FF (hop count, K = 10)": kc("FF-hops-K10", t, 0.05, p),
                               f"best spreading ({best[7:]})": h(t, 0.05, best, p),
                               "KSP-AM-FF + 1 crank-back retry": h(t, 0.05, "KSP-AM-FF", p, r="R1"),
                               "KSP-AM-FF, backward reservation": h(t, 0.05, "KSP-AM-FF", p, sig="backward")}}}
    json.dump(out, open(a.out, "w"), indent=1)
    print("written", a.out)


if __name__ == "__main__":
    main()
