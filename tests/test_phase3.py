"""Tests: distributed stale-state environment, stale-aware heuristics, anchored learners."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aeon.env import EONEnv, EnvConfig                                    # noqa: E402
from aeon.distributed import DistributedEONEnv, DistConfig, start_by_strategy, jth_block_start  # noqa: E402
from aeon.policies.heuristics import KSPAMFF, KSPAMBF                     # noqa: E402
from aeon.policies.anchored import AnchoredIndependent, AnchoredQMIX, anchored_greedy_np  # noqa: E402
from aeon.rollout_dist import P3TrainConfig, train_dist, evaluate_dist     # noqa: E402
from aeon.config import EvalConfig                                          # noqa: E402
from aeon.spectrum import free_runs                                         # noqa: E402


class AnchorPolicy:
    slot_strategy = "jblock"

    def act(self, ctx, greedy=True):
        return getattr(ctx, "anchor", None)

    def set_epsilon(self, e):
        pass


def _run(env, pol, strat, n=3000, seed=900000, load=300.0):
    env.reset(seed, load)
    out = []
    for _ in range(n):
        ctx = env.context()
        a = pol.act(ctx) if ctx.mask.any() else None
        r, info = env.step(a, strat)
        out.append((info["allocated"], info["num_slots"], round(r, 9)))
    return out


@pytest.mark.parametrize("topo", ["NSFNET", "USNET"])
def test_tau0_reproduces_base_env(topo):
    for P, strat in ((KSPAMFF, "first_fit"), (KSPAMBF, "best_fit")):
        e0 = EONEnv(EnvConfig(topology=topo))
        e1 = DistributedEONEnv(EnvConfig(topology=topo), DistConfig(tau=0.0))
        assert _run(e0, P(e0), strat) == _run(e1, P(e1), strat)


def test_anchor_with_blocks_equals_kspamff_at_tau0():
    e0 = EONEnv(EnvConfig())
    e3 = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.0, blocks=3))
    assert _run(e0, KSPAMFF(e0), "first_fit") == _run(e3, AnchorPolicy(), "jblock")


def test_tau0_has_no_setup_failures_and_backward_equals_forward():
    a = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.0, signaling="forward"))
    b = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.0, signaling="backward"))
    ra, rb = _run(a, KSPAMFF(a), "first_fit"), _run(b, KSPAMFF(b), "first_fit")
    assert ra == rb
    assert a.counters["setup_fail"] == 0


def test_stale_view_semantics():
    """Own changes are visible at once; a node without own changes sees exactly the last snapshot."""
    env = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.5))
    env.reset(900000, 300.0)
    pol = KSPAMFF(env)
    n_checked = 0
    for _ in range(400):
        r0 = env.req
        ctx = env.context()
        a = pol.act(ctx) if ctx.mask.any() else None
        env.step(a, "first_fit")
        conn = env.active.get(r0.conn_id)
        if conn is None:
            continue
        own = env.view_rows(conn.src, conn.links)
        assert own[:, conn.start:conn.start + conn.n].all(), "own allocation must be in own view"
        for other in range(env.N):
            if other != conn.src and not env.ops.get(other):
                assert np.array_equal(env.view_rows(other, conn.links), env.adv.C[conn.links])
                n_checked += 1
                break
    assert n_checked > 50


def test_view_equals_true_state_right_after_update():
    env = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.05))
    env.reset(900001, 300.0)
    pol = KSPAMFF(env)
    for _ in range(800):
        ctx = env.context()
        a = pol.act(ctx) if ctx.mask.any() else None
        env.step(a, "first_fit")
    env._snapshot(env.req.time)
    links = list(range(env.L))
    for node in range(env.N):
        assert np.array_equal(env.view_rows(node, links), env.db.C)
    assert env.counters["updates"] > 10


def test_setup_failures_accounting():
    env = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.05, signaling="forward"))
    ev = evaluate_dist(KSPAMFF(env), env, 300.0, 900000, EvalConfig(measure_events=3000), "first_fit")
    assert ev["setup_fail"] > 0
    assert abs(ev["bp"] - (ev["setup_fail"] + ev["view_block"])) < 1e-12


def test_random_fit_and_jblock_starts_are_valid():
    rng = np.random.default_rng(0)
    for _ in range(300):
        row = rng.random(64) < 0.5
        n = int(rng.integers(1, 6))
        s = start_by_strategy(row, n, "random_fit", 0, rng)
        _, lens = free_runs(row)
        if not (lens >= n).any():
            assert s is None
        else:
            assert s is not None and not row[s:s + n].any() and s + n <= 64
        for j in range(3):
            sj = jth_block_start(row, n, j)
            if sj is not None:
                assert not row[sj:sj + n].any()


def test_anchored_greedy_rule():
    q = np.array([1.0, 2.0, 1.5])
    m = np.array([True, True, True])
    assert anchored_greedy_np(q, m, 0, 0.0) == 1
    assert anchored_greedy_np(q, m, 0, 1.5) == 0
    assert anchored_greedy_np(np.zeros(3), m, 2, 0.0) == 2      # ties -> anchor


@pytest.mark.parametrize("cls", [AnchoredIndependent, AnchoredQMIX])
def test_untrained_anchored_learner_is_the_heuristic(cls):
    env = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.05, blocks=3))
    cfg = P3TrainConfig()
    pol = cls(env.obs_dim(), env.state_dim(), env.action_dim, env.N, cfg)
    env2 = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.05, blocks=3))
    greedy = type("G", (), {"slot_strategy": "jblock",
                            "act": lambda self, ctx, greedy=True: pol.act(ctx, greedy=True)})()
    assert _run(env, greedy, "jblock", n=1500) == _run(env2, AnchorPolicy(), "jblock", n=1500)


@pytest.mark.parametrize("cls", [AnchoredIndependent, AnchoredQMIX])
def test_anchored_training_smoke(cls):
    env = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.05, blocks=3))
    cfg = P3TrainConfig(train_events=3000, segment_events=1000, learn_start=64, batch_size=32,
                        reward_transform="center")
    pol = cls(env.obs_dim(), env.state_dim(), env.action_dim, env.N, cfg)
    logs = train_dist(pol, env, cfg, 0)
    assert len(logs) == 3
    assert pol.buf.n > 0
    assert any(np.isfinite(r["loss"]) for r in logs)
    ev = evaluate_dist(pol, env, 300.0, 900000, EvalConfig(measure_events=1000))
    assert 0.0 <= ev["bp"] <= 1.0


def test_window_qmix_ablation_ignores_anchor_and_trains():
    from aeon.policies.anchored import WindowQMIX
    env = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.05, blocks=3))
    cfg = P3TrainConfig(train_events=2000, segment_events=1000, learn_start=32, batch_size=16)
    pol = WindowQMIX(env.obs_dim(), env.state_dim(), env.action_dim, env.N, cfg)
    assert not pol.anchored and pol.q.a.weight.abs().sum() > 0
    train_dist(pol, env, cfg, 0)
    assert pol.buf.n > 0


def test_crankback_retry_reduces_setup_failures_and_k10_extends_paths():
    base = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.05))
    retry = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.05, retries=1))
    ec = EvalConfig(measure_events=3000)
    e0 = evaluate_dist(KSPAMFF(base), base, 300.0, 900000, ec, "first_fit")
    e1 = evaluate_dist(KSPAMFF(retry), retry, 300.0, 900000, ec, "first_fit")
    assert retry.counters["retries"] > 0
    assert e1["setup_fail"] < e0["setup_fail"]
    k10 = DistributedEONEnv(EnvConfig(k_paths=10), DistConfig(tau=0.0))
    assert k10.K == 10 and k10.action_dim == 60
    e2 = evaluate_dist(KSPAMFF(k10), k10, 300.0, 900000, ec, "first_fit")
    assert 0.0 <= e2["bp"] <= 1.0


def test_deeprmsa_qot_shapes_rewards_and_training():
    from aeon.policies.deeprmsa_qot import DeepRMSAQoT, train_dqot
    env = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.05, blocks=3))
    for kind in ("basic", "heuristic", "qot"):
        cfg = P3TrainConfig(train_events=1500, segment_events=500, dqot_reward=kind)
        pol = DeepRMSAQoT(env, cfg)
        env.reset(900000, 300.0)
        x = pol.features()
        assert x.shape == (2 * env.N + 11 * env.K,)
        assert pol.act_dim == env.K * env.J
        assert sum(1 for p in pol.q if isinstance(p, torch.nn.Linear)) == 6
        logs = train_dqot(pol, env, cfg, 0)
        assert len(logs) == 3
    ev = evaluate_dist(pol, env, 300.0, 900000, EvalConfig(measure_events=800))
    assert 0.0 <= ev["bp"] <= 1.0


def test_greedy_oracle_picks_max_immediate_reward_action():
    from aeon.policies.greedy import KSPAMGreedy
    env = DistributedEONEnv(EnvConfig(), DistConfig(tau=0.0, blocks=3))
    env.reset(900000, 300.0)
    pol = KSPAMGreedy(env)
    for _ in range(300):
        ctx = env.context()
        a = pol.act(ctx) if ctx.mask.any() else None
        if a is not None:
            assert ctx.mask[a]
        r, info = env.step(a, "jblock")
        if a is not None:
            assert info["allocated"]          # fresh state: feasible actions always succeed


def test_hops_path_order_is_lexicographic_and_km_default_unchanged():
    """'hops' = K shortest by (number of hops, km) (Doherty et al. 2025)."""
    import itertools
    from aeon.topology import create_topology
    for name in ("NSFNET", "USNET"):
        km5 = create_topology(name, k=5)
        assert km5.order == "km" and km5._hop_w == 0.0
        h5 = create_topology(name, k=5, order="hops")
        h10 = create_topology(name, k=10, order="hops")
        for s, d in itertools.permutations(range(h5.num_nodes), 2):
            ps = h10.k_paths[s][d]
            keys = [(len(p), h10.get_path_length(p)) for p in ps]
            assert keys == sorted(keys)                              # ordered by hops, then km
            assert h5.k_paths[s][d] == ps[:5]                        # K=5 is a prefix of K=10
            assert len(ps[0]) == min(len(p) for p in km5.k_paths[s][d])   # min-hop path first
            assert len({tuple(p) for p in ps}) == len(ps)            # loop-free, no duplicates
            assert all(len(set(h10.get_path_nodes(p))) == len(p) + 1 for p in ps)
    env = EONEnv(EnvConfig(path_order="hops"))
    assert env.topo.order == "hops"
    with pytest.raises(ValueError):
        create_topology("NSFNET", order="foo")
