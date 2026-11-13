"""Unit tests for the simulator and learners.  Run:  python -m pytest -q tests"""
import math
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aeon.env import EONEnv, EnvConfig
from aeon.config import TrainConfig, EvalConfig
from aeon.factory import build_policy
from aeon.rollout import train, evaluate
from aeon.phy import PhyGN, required_snr_db, ber_from_snr
from aeon.policies.base import RMSAPolicy
from aeon.spectrum import SpectrumDatabase, choose_start, free_runs

def test_snr_vectorised_equals_scalar():
    p = PhyGN()
    km = [80.0, 450.0, 1210.0]
    occ = np.array([0.0, 37.0, 250.0])
    spans = np.array([p.spans(k) for k in km], float)
    widths = np.array([2.0, 5.0, 17.0, 65.0])
    v = p.snr_db_formats(spans, occ, widths)
    s = [p.snr_db(km, occ, int(w)) for w in widths]
    assert np.allclose(v, s, atol=1e-9)


def test_required_snr_meets_ber_target_and_is_monotone():
    p = PhyGN()
    for m in range(6):
        snr = required_snr_db(m, p.cfg.ber_target)
        assert ber_from_snr(10 ** (snr / 10), m) <= p.cfg.ber_target * 1.0001
        assert ber_from_snr(10 ** ((snr - 0.05) / 10), m) > p.cfg.ber_target
    assert all(np.diff(p.snr_req_db) > 0)


def test_snr_decreases_with_distance_and_load():
    p = PhyGN()
    assert p.snr_db([800], [10], 4) > p.snr_db([1600], [10], 4)
    assert p.snr_db([800], [10], 4) > p.snr_db([800], [300], 4)


def test_free_runs_and_slot_strategies():
    row = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0], bool)
    s, l = free_runs(row)
    assert list(s) == [1, 4, 9] and list(l) == [2, 4, 3]
    assert choose_start(row, 2, "first_fit") == 1
    assert choose_start(row, 3, "first_fit") == 4
    assert choose_start(row, 3, "best_fit") == 9
    assert choose_start(row, 2, "last_fit") == 10
    assert choose_start(row, 5, "first_fit") is None


def test_fragmentation_cache_matches_full_recompute():
    rng = random.Random(1)
    db = SpectrumDatabase(6, 40)
    live = []
    for _ in range(400):
        if live and rng.random() < 0.45:
            links, st, n = live.pop(rng.randrange(len(live)))
            db.release(links, st, n)
        else:
            links = rng.sample(range(6), rng.randint(1, 3))
            n = rng.randint(1, 6)
            st = choose_start(db.union(links), n, rng.choice(["first_fit", "best_fit", "last_fit"]))
            if st is not None:
                db.allocate(links, st, n)
                live.append((links, st, n))
        cached = db.fragmentation()
        db2 = SpectrumDatabase(6, 40)
        db2.C[:] = db.C
        db2.occ[:] = db.C.sum(1)
        db2.recompute_all()
        assert math.isclose(cached, db2.fragmentation(), abs_tol=1e-12)
        assert (db.occ == db.C.sum(1)).all()


@pytest.mark.parametrize("phy", ["gn_v2"])
def test_masked_actions_always_served_and_infeasible_blocked(phy):
    env = EONEnv(EnvConfig("NSFNET", phy_model=phy), seed=3, load=350)
    rng = np.random.default_rng(0)
    served = 0
    for _ in range(3000):
        ctx = env.context()
        feas = np.flatnonzero(ctx.mask)
        infeas = np.flatnonzero(~ctx.mask)
        if feas.size and rng.random() < 0.85:
            r, info = env.step(int(rng.choice(feas)), rng.choice(["first_fit", "best_fit", "last_fit"]))
            assert info["allocated"]
            served += 1
        elif infeas.size:
            r, info = env.step(int(rng.choice(infeas)))
            assert info["blocked"] and info["invalid_action"] and r == -env.cfg.w_success
        else:
            env.step(None)
    assert served > 1000


def test_reward_equation():
    cfg = EnvConfig("NSFNET")
    env = EONEnv(cfg, seed=5, load=200)
    for _ in range(500):
        ctx = env.context()
        if not ctx.mask.any():
            r, info = env.step(None)
            assert r == -cfg.w_success
            continue
        f0 = env.db.fragmentation()
        a = int(np.flatnonzero(ctx.mask)[0])
        k, m = env.decode(a)
        n = env._analyse()[k].slots[m]
        b = env.req.bitrate
        # fragmentation right after allocation (before departures of the next advance)
        st = choose_start(env._analyse()[k].union, n, "first_fit")
        db = SpectrumDatabase(env.L, env.S)
        db.C[:] = env.db.C
        db.occ[:] = env.db.occ
        db.recompute_all()
        db.allocate(env._analyse()[k].links, st, n)
        f1 = db.fragmentation()
        r, info = env.step(a)
        expect = cfg.w_success + cfg.w_efficiency * b / (n * 12.5) - cfg.w_frag * max(f1 - f0, 0.0)
        assert math.isclose(r, expect, rel_tol=1e-9, abs_tol=1e-12)


class _Recorder(RMSAPolicy):
    """Trainable stub: first feasible action, records transitions."""
    trainable = True

    def __init__(self):
        self.trs = []

    def act(self, ctx, greedy=False):
        f = np.flatnonzero(ctx.mask)
        return int(f[0]) if f.size else None

    def observe(self, tr):
        self.trs.append(tr)


def test_transitions_accumulate_forced_blocks():
    """Replay the same stream and rebuild G_t and Gamma_t from the event log."""
    cfg = EnvConfig("NSFNET")
    tc = TrainConfig(train_events=4000, segment_events=1000, curriculum=False, train_load=900.0, gamma=0.9)
    env = EONEnv(cfg, seed=0, load=900.0)
    pol = _Recorder()
    train(pol, env, tc, train_seed=7)
    # rebuild from an identical replay
    env2 = EONEnv(cfg, seed=0, load=900.0)
    from aeon.config import train_stream_seed
    env2.reset(train_stream_seed(7), 900.0)
    events = []
    for _ in range(tc.train_events):
        ctx = env2.context()
        a = pol.act(ctx)
        r, info = env2.step(a)
        events.append((a is not None, r))
    dec_idx = [i for i, (d, _) in enumerate(events) if d]
    assert any(not d for d, _ in events), "test needs forced blocks; raise the load"
    assert len(pol.trs) == len(dec_idx) - 1          # last open transition discarded
    for j, tr in enumerate(pol.trs):
        i0, i1 = dec_idx[j], dec_idx[j + 1]
        G = sum((tc.gamma ** (i - i0)) * events[i][1] for i in range(i0, i1))
        assert math.isclose(tr.ret, G, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(tr.discount, tc.gamma ** (i1 - i0), rel_tol=1e-12)
        assert tr.next_mask.any()


def test_parameter_counts_match_architecture():
    env = EONEnv(EnvConfig("NSFNET"), seed=0, load=300)
    tc = TrainConfig()
    o, s, a, h, hc = env.obs_dim(), env.state_dim(), env.action_dim, tc.hidden, tc.hidden_central
    mlp = lambda d: d[0] * d[1] + d[1] + d[1] * d[2] + d[2] + d[2] * d[3] + d[3]
    assert (o, s, a) == (84, 44, 30)
    assert build_policy("Independent", env, tc).n_parameters() == mlp((o, h, h, a)) == 31262
    q = build_policy("QMIX", env, tc)
    assert q.n_parameters() == 31262                 # executed network only
    assert q.n_training_parameters() > 31262          # + mixer (training only)
    assert build_policy("Centralized-DQN", env, tc).n_parameters() == mlp((o + s, hc, hc, a))


def test_all_methods_face_identical_traffic():
    seqs = []
    for name in ["KSP-FF", "KSP-AM-BF", "QMIX"]:
        env = EONEnv(EnvConfig("USNET"), seed=0, load=300)
        pol = build_policy(name, env, TrainConfig())
        env.reset(42, 300)
        seq = []
        for _ in range(500):
            ctx = env.context()
            r = env.req
            seq.append((r.conn_id, r.src, r.dst, r.bitrate, round(r.time, 12)))
            env.step(pol.act(ctx, greedy=True) if ctx.mask.any() else None, pol.slot_strategy)
        seqs.append(seq)
    assert seqs[0] == seqs[1] == seqs[2]


def test_steady_state_after_warmup():
    env = EONEnv(EnvConfig("NSFNET"), seed=0, load=300)
    pol = build_policy("KSP-AM-FF", env, TrainConfig())
    env.reset(900_000, 300)
    util = []
    for _ in range(1500 + 6000):
        ctx = env.context()
        env.step(pol.act(ctx) if ctx.mask.any() else None)
        util.append(env.db.occ.mean())
    early = np.mean(util[:250])             # first 250 arrivals
    w1, w2 = np.mean(util[1500:4500]), np.mean(util[4500:])
    assert abs(w1 - w2) / w2 < 0.05          # stationary after 5 holding times
    assert early < 0.8 * w2                  # the first 250 arrivals are far from steady state


class _ValueRecorder(_Recorder):
    buf = None          # marks it as a value learner so n-step composition applies


def test_nstep_composition_matches_one_step_chain():
    cfg = EnvConfig("NSFNET")
    base = dict(train_events=3000, segment_events=1000, curriculum=False, train_load=900.0, gamma=0.9)
    one = _ValueRecorder()
    train(one, EONEnv(cfg, seed=0, load=900.0), TrainConfig(**base), train_seed=3)
    n = 4
    multi = _ValueRecorder()
    train(multi, EONEnv(cfg, seed=0, load=900.0), TrainConfig(n_step=n, **base), train_seed=3)
    assert len(multi.trs) == len(one.trs) - (n - 1)
    for j, tr in enumerate(multi.trs):
        chain = one.trs[j:j + n]
        ret, disc = 0.0, 1.0
        for t in chain:
            ret += disc * t.ret
            disc *= t.discount
        assert math.isclose(tr.ret, ret, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(tr.discount, disc, rel_tol=1e-12)
        assert tr.action == chain[0].action
        assert (tr.next_obs == chain[-1].next_obs).all()
