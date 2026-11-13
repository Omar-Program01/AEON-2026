# AEON: distributed RMSA under stale network state

AEON is an event-driven simulator for dynamic routing, modulation and spectrum assignment (RMSA) in elastic optical networks (EONs). It models distributed control, in which every source node decides on a periodically synchronised and therefore possibly outdated replica of the network state. The repository contains heuristic and deep-reinforcement-learning (DRL) methods, heuristic-anchored multi-agent learners, and the scripts that run the experiments, analyses and figures.

## Features

- **Network and traffic model.** NSFNET (14 nodes, 21 links), USNET (24 nodes, 43 links) and PAN-EUROPEAN (28 nodes, 41 links). Each link has two directed fibres with 320 slots of 12.5 GHz. Requests arrive as a Poisson process with exponential holding times, uniform source–destination pairs and bit rates in {40, 100, 200, 400, 800} Gb/s.
- **Physical layer.** Closed-form Gaussian-noise (GN) model with ASE and nonlinear interference, evaluated per link and accumulated over spans. Six formats, BPSK to 64-QAM, are accepted if the SNR exceeds the requirement at a pre-FEC BER of 3.8×10⁻³ plus a 2 dB margin.
- **Distributed control with stale state.**
  - Periodic link-state updates every τ mean holding times.
  - Each node's view is the last snapshot plus its own changes.
  - Forward (source-selected) or backward (destination-selected) reservation, with an optional crank-back retry.
  - Blocking is decomposed into setup failures and view blocks.
  - With τ = 0 the model reduces exactly to centralised control.
- **Heuristics.** KSP-FF, KSP-AM-FF, KSP-AM-BF (best fit), KSP-AM-RF (random fit), KSP-AM-FLF (node-based first–last fit) and the myopic reference rule KSP-AM-greedy. Candidate paths can be ordered by length or by hop count.
- **Learners.**
  - DeepRMSA (A2C), PPO, a graph-neural-network DQN, a centralised DQN, independent DQN and QMIX on the centralised environment.
  - Heuristic-anchored learners on the distributed environment:
    - A-Independent;
    - A-QMIX, which uses window-level QMIX over link-state update windows;
    - W-QMIX, the same learner without the anchor;
    - A-QMIX with γ = 0.
  - A re-implementation of DeepRMSA-QoT.
- **Reproducibility.** Every run is identified by a hash of the code version and its full configuration. Its manifest, evaluation results, training log and model checkpoint are written to `results/<exp_id>/`.

## Repository layout

```
aeon/                  simulator package
  topology.py          topologies and K-shortest-path routing (length or hop ordering)
  traffic.py           Poisson request stream with fixed per-seed random draws
  spectrum.py          spectrum database, free blocks, fragmentation, slot-selection rules
  phy.py               GN-model QoT, BER/SNR conversion, slots per lightpath
  env.py               centralised event-driven environment
  distributed.py       distributed environment with stale replicated state and signalling
  rollout.py           training and evaluation on the centralised environment
  rollout_dist.py      training and evaluation on the distributed environment
  timing.py            decision-latency measurement
  explog.py            experiment identifiers and manifests
  config.py, factory.py
  policies/            heuristics, value learners, actor-critic, GNN, anchored learners,
                       DeepRMSA-QoT, KSP-AM-greedy
scripts/               experiment drivers, analyses and figures
tests/                 unit tests
run_all.sh             complete pipeline
```

## Installation

Python 3.10 or newer is required. The learners run on CPU.

```
pip install -r requirements.txt
```

## Tests

```
python -m pytest -q tests
```

The tests cover the following:

- the stale-view semantics and the exact reduction to centralised control at τ = 0;
- the GN-model implementation;
- action masking and the slot-selection rules;
- candidate-path ordering;
- the transition and return construction;
- the identity of an untrained anchored learner with its heuristic.

## Running the experiments

`run_all.sh` runs the complete pipeline. Set `WORKERS` to the number of CPU processes, for example `WORKERS=8 bash run_all.sh`. Each stage is resumable, and the individual stages are:

| Step | Command | Output |
|---|---|---|
| Fresh-state study | `python scripts/phase2a.py --workers N` | `results/phase2a/` |
| Load calibration | `python scripts/phase3.py calibrate` | `results/phase3_calibrate/calibration.json` |
| Heuristics (all settings) | `python scripts/phase3.py heuristics` | `results/phase3_heuristics/state.json` |
| Learner tuning | `python scripts/phase3.py tune` | `results/phase3_tune/state.json` |
| Learner confirmation | `python scripts/phase3.py confirm` | `results/phase3_confirm/state.json` |
| Result tables | `python scripts/phase3_report.py --root results --heur results/phase3_heuristics/state.json --out results/report` | Markdown and JSON tables |
| KSP-AM-greedy | `python scripts/phase3_greedy.py CALIBRATION OUTDIR N` | `greedy_state.json` |
| Candidate-set sensitivity | `python scripts/phase3_kcheck.py CALIBRATION OUTDIR N` | `kcheck_state.json` |
| Training-budget sensitivity | `python scripts/phase3_budget.py RESULTS OUTDIR N` | `budget_state.json` |
| Discount-horizon sensitivity | `python scripts/phase3_horizon.py RESULTS OUTDIR N` | `horizon_state.json` |
| Deviation analysis and decision time | `python scripts/phase3_mechanism.py RESULTS_ZIP OUTDIR N` | `mechanism_state.json` |
| Summaries | `python scripts/phase3_summaries.py ...` | `mechanism_summary.json`, `budget_compare.json`, `horizon_compare.json` |
| Baseline ladder | `python scripts/phase3_ladder.py ...` | `ladder.json` |

Long runs print a progress line at regular intervals. Stages that support sharding (`--learners`, `--shard`) can be split across machines; their state files are merged automatically when placed in the same results folder.

## Figures

| Script | Figures |
|---|---|
| `fig_topologies.py` | the three topologies with link lengths |
| `fig_diagrams.py` | stale views, signalling, anchored decision, window-level training |
| `phase3_figures.py` | blocking versus load, staleness and signalling, ablations |
| `phase3_fig_ladder.py` | blocking ratio of the best learner to successively stronger baselines |
| `phase3_figures_extra.py` | blocking versus τ, blocking decomposition, deviation analysis, training curves, budget and horizon checks |

All figures are written as PNG (600 dpi) and PDF.

## Main settings

| Setting | Value |
|---|---|
| Spectrum | 320 slots × 12.5 GHz per fibre |
| Candidate paths | K = 5 (length-ordered); K = 10 and hop ordering as options |
| Update period τ | 0, 0.01, 0.05 mean holding times |
| Loads | calibrated per topology so that KSP-AM-FF with fresh state blocks 0.1, 0.5, 1, 3 and 10% |
| Training | 250,000 arrivals in one continuing stream with a load curriculum |
| Evaluation | 5 unseen traffic seeds, 10,000 measured arrivals after a warm-up of 5 mean holding times |
| Anchored learners | shared dueling Q-network (2 × 128 ReLU), QMIX mixer (embedding 32, hypernetworks 64), Adam 10⁻⁴, Huber loss, soft target update 0.005 |

The full configuration of every run is stored in its manifest.
