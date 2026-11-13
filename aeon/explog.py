"""
Experiment identifiers and manifests.

exp_id = first 12 hex digits of SHA-256 over (code version, full configuration,
method, seeds, experiment tag). Every reported number points to an exp_id; results are written to results/<exp_id>/ with a manifest.json that
records the configuration, code version, platform and wall-clock times.
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent


def code_version() -> str:
    """SHA-256 over the source of every module in the aeon package."""
    h = hashlib.sha256()
    for p in sorted(PKG_DIR.rglob("*.py")):
        h.update(p.relative_to(PKG_DIR).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def platform_info() -> dict:
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import torch
        tv = torch.__version__
        th = torch.get_num_threads()
    except Exception:
        tv, th = None, None
    import numpy
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu": cpu,
            "cpu_count": os.cpu_count(), "torch": tv, "torch_threads": th, "numpy": numpy.__version__}


def make_exp_id(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


class Experiment:
    def __init__(self, results_root, tag: str, method: str, config: dict, seeds: dict):
        self.payload = {"code_version": code_version(), "tag": tag, "method": method,
                        "config": config, "seeds": seeds}
        self.exp_id = make_exp_id(self.payload)
        self.dir = Path(results_root) / self.exp_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()

    def write(self, name: str, obj):
        with open(self.dir / name, "w") as f:
            json.dump(obj, f, indent=1, default=str)

    def finish(self, extra: dict = None):
        m = dict(self.payload)
        m.update({"exp_id": self.exp_id, "platform": platform_info(),
                  "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.t0)),
                  "wall_s": round(time.time() - self.t0, 2)})
        if extra:
            m.update(extra)
        self.write("manifest.json", m)
        return self.exp_id
