"""Heuristic-only staleness pilot: NSFNET, 300 E, traffic seed 900000."""
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, time
from aeon.env import EnvConfig
from aeon.distributed import DistributedEONEnv, DistConfig
from aeon.policies.heuristics import KSPAMFF, KSPAMBF
def bp(topo, load, tau, sig, P, strat, n=8000, warm=2000):
    e = DistributedEONEnv(EnvConfig(topology=topo), DistConfig(tau=tau, signaling=sig)); e.reset(900000, load)
    pol = P(e); b=sf=vb=0
    for i in range(warm+n):
        ctx=e.context(); a=pol.act(ctx) if ctx.mask.any() else None
        r,info=e.step(a,strat)
        if i>=warm: b+=info["blocked"]; sf+=info["setup_fail"]; vb+=info["view_block"]
    return b/n, sf/n, vb/n
t0=time.time()
for tau in (0.0, 0.01, 0.05):
    for sig in ("forward","backward"):
        row=[]
        for P,s in ((KSPAMFF,"first_fit"),(KSPAMBF,"best_fit"),(KSPAMFF,"random_fit"),(KSPAMFF,"flf_node")):
            x=bp("NSFNET",300,tau,sig,P,s); row.append(f"{s[:5]} {x[0]:.3f}({x[1]:.3f})")
        print(f"tau={tau} {sig:8s}", " | ".join(row))
print("s", round(time.time()-t0))
