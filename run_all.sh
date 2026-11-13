#!/usr/bin/env bash
# Runs the complete experiment pipeline. Every stage is resumable: re-running a command
# skips the cells that are already finished. Set WORKERS to the number of CPU processes.
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-4}"
R=results
F=figures
mkdir -p "$R" "$F"

# 0) unit tests
python -m pytest -q tests

# 1) fresh-state learnability study (centralised control, NSFNET)
python scripts/phase2a.py --workers "$WORKERS" --root "$R"

# 2) stale-state study (distributed control)
python scripts/phase3.py calibrate  --workers "$WORKERS" --root "$R"
python scripts/phase3.py heuristics --workers "$WORKERS" --root "$R"
python scripts/phase3.py tune       --workers "$WORKERS" --root "$R"
python scripts/phase3.py confirm    --workers "$WORKERS" --root "$R"
python scripts/phase3_report.py --root "$R" --heur "$R/phase3_heuristics/state.json" --out "$R/report"

# 3) reference rule and sensitivity analyses
python scripts/phase3_greedy.py  "$R/phase3_calibrate/calibration.json" "$R/phase3_greedy" "$WORKERS"
python scripts/phase3_kcheck.py  "$R/phase3_calibrate/calibration.json" "$R/phase3_kcheck" "$WORKERS"
python scripts/phase3_budget.py  "$R" "$R/phase3_budget"  "$WORKERS"
python scripts/phase3_horizon.py "$R" "$R/phase3_horizon" "$WORKERS"
python -m zipfile -c "$R/phase3_results.zip" "$R/phase3_calibrate" "$R/phase3_confirm" $(python - <<'PY'
import json
cells = json.load(open("results/phase3_confirm/state.json"))["cells"]
print(" ".join(sorted({"results/" + c["result"]["exp_id"] for c in cells.values()})))
PY
)
python scripts/phase3_mechanism.py "$R/phase3_results.zip" "$R/phase3_mechanism" "$WORKERS"
python scripts/phase3_summaries.py --mech "$R/phase3_mechanism/mechanism_state.json" \
       --budget "$R/phase3_budget/budget_state.json" --horizon "$R/phase3_horizon/horizon_state.json" \
       --root "$R" --out "$R/summaries"

# 4) figures
python scripts/fig_topologies.py "$F"
python scripts/fig_diagrams.py   "$F"
python scripts/phase3_figures.py --root "$R" --heur "$R/phase3_heuristics/state.json" \
       --greedy "$R/phase3_greedy/greedy_state.json" --out "$F"
python scripts/phase3_ladder.py  --root "$R" --heur "$R/phase3_heuristics/state.json" \
       --kcheck "$R/phase3_kcheck/kcheck_state.json" --out "$R/summaries/ladder.json"
python scripts/phase3_fig_ladder.py "$R/summaries/ladder.json" "$F"
python scripts/phase3_figures_extra.py --root "$R" --heur "$R/phase3_heuristics/state.json" \
       --mech "$R/summaries/mechanism_summary.json" --budget "$R/summaries/budget_compare.json" \
       --horizon "$R/summaries/horizon_compare.json" --out "$F"
