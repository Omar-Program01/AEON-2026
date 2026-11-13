"""Secondary analyses (no training): deviation analysis and decision time.

(1) Deviation analysis: the confirmed A-QMIX models (NSFNET, tau in {0, 0.01, 0.05}, seeds 200-204) are
    re-evaluated with per-decision logging: does the agent deviate from its anchor (KSP-AM-FF), how
    (other path / other format on the anchor path / other block only), and when (age of the node's view
    in units of tau, anchor-path hop count, load); outcome = blocking and setup failure of the decision.
    Loads: the primary load (all tau) and the 0.1% and 3% loads at tau = 0.05; traffic seed 930000.
    The stored confirmation blocking of the same (model, load, traffic) is reported as a reproduction check.
(2) Decision time: context (feature) and act time per decision for every method, NSFNET, tau = 0.05,
    primary load, traffic 930000, one process, one thread, after warm-up.
  python scripts/phase3_mechanism.py PHASE3_RESULTS_ZIP OUTDIR WORKERS     (resumable: mechanism_state.json)
"""
import io, json, math, os, statistics as st, sys, time, zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
HEARTBEAT = 600
SEEDS = [200, 201, 202, 203, 204]
TRAFFIC = 930000


def _zip_member(z, suffix):
    names = [n for n in z.namelist() if n.replace("\\", "/").endswith(suffix)]
    if not names:
        raise KeyError(suffix)
    return z.read(names[0])


def load_policy(zpath, key):
    """Rebuild a confirmed learner from the results zip; returns (policy, env, manifest, stored evals)."""
    import torch
    torch.set_num_threads(1)
    import phase3
    from aeon.rollout_dist import P3TrainConfig
    from aeon.policies.anchored import AnchoredIndependent, AnchoredQMIX, WindowQMIX
    from aeon.policies.deeprmsa_qot import DeepRMSAQoT
    z = zipfile.ZipFile(zpath)
    cells = json.loads(_zip_member(z, "phase3_confirm/state.json"))["cells"]
    cell = cells[key]["result"]
    eid = cell["exp_id"]
    man = json.loads(_zip_member(z, f"{eid}/manifest.json"))
    sd = torch.load(io.BytesIO(_zip_member(z, f"{eid}/checkpoint.pt")), map_location="cpu")
    tr = dict(man["config"]["train"]); tr["train_events"] = 1000     # small replay allocation only
    tcfg = P3TrainConfig(**tr)
    e = man["config"]["env"]
    env = phase3._env(e["topology"], e["tau"], e.get("signaling", "forward"), e.get("blocks", 3))
    m = man["method"]
    if m == "DeepRMSA-QoT":
        pol = DeepRMSAQoT(env, tcfg)
    else:
        cls = {"A-QMIX": AnchoredQMIX, "A-Independent": AnchoredIndependent, "W-QMIX": WindowQMIX}[m]
        pol = cls(env.obs_dim(), env.state_dim(), env.action_dim, env.N, tcfg)
    pol.q.load_state_dict(sd["q"])
    if "mixer" in sd and hasattr(pol, "mixer"):
        pol.mixer.load_state_dict(sd["mixer"])
    return pol, env, man, cell["evals"]


def deviation_job(args):
    zpath, key, load = args
    from aeon.config import EvalConfig
    pol, env, man, stored = load_policy(zpath, key)
    ec = EvalConfig(measure_events=10_000)
    tau = env.dcfg.tau
    env.reset(TRAFFIC, load)
    warm = int(math.ceil(ec.warmup_holding_times * load / env.cfg.mean_holding))
    recs, n, blocked = [], 0, 0
    for i in range(warm + ec.measure_events):
        ctx = env.context()
        a = pol.act(ctx, greedy=True) if ctx.mask.any() else None
        anchor = getattr(ctx, "anchor", None)
        rec = None
        if i >= warm and a is not None and anchor is not None:
            info = env._analyse()
            ka, ma, ja = env.decode_full(int(anchor)); k, m, j = env.decode_full(int(a))
            kind = "none" if a == anchor else ("path" if k != ka else ("format" if m != ma else "block"))
            age = 0.0 if tau == 0 else (env.req.time - env.last_update) / tau
            rec = [kind, round(age, 4), len(info[ka].links), len(info[k].links)]
        r, inf = env.step(a, "jblock")
        if i < warm:
            continue
        n += 1; blocked += int(inf["blocked"])
        if rec is not None:
            recs.append(rec + [int(inf["blocked"]), int(inf.get("setup_fail", False))])
    ref = [e["bp"] for e in stored if e["load"] == load and e["eval_seed"] == TRAFFIC]
    return f"dev|{key}|{load}", {"bp": blocked / n, "stored_bp": ref[0] if ref else None, "n": n, "decisions": recs}


def latency(zpath, cal):
    import torch
    torch.set_num_threads(1)
    import phase3
    from aeon.policies import heuristics as H
    from aeon.policies.greedy import KSPAMGreedy
    load = cal["NSFNET"]["primary_load"]
    out = {}
    methods = [("KSP-AM-FF", "heur", "first_fit"), ("KSP-AM-FLF", "heur", "flf_node"), ("KSP-AM-greedy", "greedy", "jblock"),
               ("A-QMIX", "C|A-QMIX|NSFNET|0.05|s200", None), ("A-Independent", "C|A-Independent|NSFNET|0.05|s200", None),
               ("W-QMIX", "C|W-QMIX|NSFNET|0.05|s200", None), ("DeepRMSA-QoT", "C|DeepRMSA-QoT|NSFNET|0.05|s200", None)]
    for name, src, strat in methods:
        if src == "heur":
            env = phase3._env("NSFNET", 0.05, "forward", 1); pol = H.KSPAMFF(env); npar = 0
        elif src == "greedy":
            env = phase3._env("NSFNET", 0.05, "forward", 3); pol = KSPAMGreedy(env); strat = "jblock"; npar = 0
        else:
            pol, env, _, _ = load_policy(zpath, src); strat = pol.slot_strategy
            npar = int(pol.n_parameters())
        env.reset(TRAFFIC, load)
        tc, ta = [], []
        for i in range(3000 + 2000):
            t0 = time.perf_counter(); ctx = env.context(); t1 = time.perf_counter()
            a = pol.act(ctx, greedy=True) if ctx.mask.any() else None
            t2 = time.perf_counter()
            env.step(a, strat)
            if i >= 3000:
                tc.append((t1 - t0) * 1e6); ta.append((t2 - t1) * 1e6)
        q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
        out[name] = {"context_us_median": q(tc, 0.5), "act_us_median": q(ta, 0.5), "act_us_p95": q(ta, 0.95),
                     "total_us_median": q([x + y for x, y in zip(tc, ta)], 0.5), "n_parameters": npar}
        print(f"  latency {name}: act median {out[name]['act_us_median']:.0f} us, total {out[name]['total_us_median']:.0f} us", flush=True)
    return out


def main():
    zpath, outdir, w = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3])
    os.makedirs(outdir, exist_ok=True)
    z = zipfile.ZipFile(zpath)
    cal = json.loads(_zip_member(z, "phase3_calibrate/calibration.json"))
    cells = json.loads(_zip_member(z, "phase3_confirm/state.json"))["cells"]
    p = os.path.join(outdir, "mechanism_state.json")
    state = json.load(open(p)) if os.path.exists(p) else {}
    L = cal["NSFNET"]["loads"]; Lp = cal["NSFNET"]["primary_load"]
    jobs = []
    for tau in (0.0, 0.01, 0.05):
        for s in SEEDS:
            key = f"C|A-QMIX|NSFNET|{tau}|s{s}"
            if key not in cells:
                continue
            for load in ([L[0], Lp, L[3]] if tau == 0.05 else [Lp]):
                if f"dev|{key}|{load}" not in state:
                    jobs.append((zpath, key, load))
    print(f"Deviation analysis: {len(jobs)} evaluations on {w} workers ({len(state)} already done); then decision times.", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(w) as ex:
        pend = {ex.submit(deviation_job, j): j for j in jobs}
        while pend:
            done, _ = wait(pend, timeout=HEARTBEAT, return_when=FIRST_COMPLETED)
            for fu in done:
                pend.pop(fu)
                k, v = fu.result(); state[k] = v
                json.dump(state, open(p, "w"))
                print(f"  done {k}  bp={v['bp']:.4f} (stored {v['stored_bp']})  elapsed {(time.time()-t0)/60:.0f} min", flush=True)
            if not done:
                print(f"  ... still working: {len(pend)} left, elapsed {(time.time()-t0)/60:.0f} min", flush=True)
    if "latency" not in state:
        print("Measuring decision times (one process; please leave the PC idle for ~10 minutes)...", flush=True)
        state["latency"] = latency(zpath, cal)
        json.dump(state, open(p, "w"))
    print(f"Done. Results written to {p}.", flush=True)


if __name__ == "__main__":
    main()
