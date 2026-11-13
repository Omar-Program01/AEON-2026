"""Build any method for a given environment and training configuration."""
from .policies.heuristics import KSPFF, KSPAMFF, KSPAMBF
from .policies.value import Independent, QMIX, CentralDQN
from .policies.actor_critic import DeepRMSA, PPO
from .policies.gnn import GNNDRL


def build_policy(name, env, tcfg):
    o, s, a = env.obs_dim(), env.state_dim(), env.action_dim
    if name == "KSP-FF":
        return KSPFF(env)
    if name == "KSP-AM-FF":
        return KSPAMFF(env)
    if name == "KSP-AM-BF":
        return KSPAMBF(env)
    if name == "DeepRMSA":
        return DeepRMSA(o, s, a, tcfg)
    if name == "PPO":
        return PPO(o, s, a, tcfg)
    if name == "GNN-DRL":
        return GNNDRL(env, a, tcfg)
    if name == "Centralized-DQN":
        return CentralDQN(o, s, a, tcfg)
    if name == "Independent":
        return Independent(o, s, a, tcfg)
    if name == "QMIX":
        return QMIX(o, s, a, tcfg)
    raise ValueError(name)
