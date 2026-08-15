#!/usr/bin/env python
"""Time one flow-matching member over the eval grid, to size the full job."""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.uncertainty_maps.common import EXP, load_marginal  # noqa: E402
from analysis.uncertainty_maps.compute_members import build_system, load_cfg  # noqa: E402

RUN = "fm_high_total"
EPOCH = 10

print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = load_cfg(RUN)
system = build_system(cfg)
radius = float(cfg.attractor_radius)
from hydra.utils import get_class  # noqa: E402
cls = get_class(cfg.flow_matcher._target_)

states = load_marginal("fm", "high", "total", EPOCH)["states"]
print("N =", len(states))

md = sorted((EXP / RUN / f"epoch_{EPOCH:03d}").glob("member_*"))[0]
ck = sorted(glob.glob(str(md / "**" / "best*.ckpt"), recursive=True))[0]
t = time.time()
model = cls.load_from_checkpoint(ck, device=device).eval().to(device)
print(f"load {time.time()-t:.1f}s")

x = torch.as_tensor(states, dtype=torch.float32, device=device)
for batch in (4096, 16384):
    torch.cuda.synchronize() if device == "cuda" else None
    t = time.time()
    with torch.no_grad():
        hits = torch.zeros(len(states), dtype=torch.float64, device=device)
        for i in range(0, len(states), batch):
            pred = model.predict_endpoint(x[i:i + batch])
            lab = system.classify_attractor(pred, radius)
            hits[i:i + batch] += (lab == 1).double()
    torch.cuda.synchronize() if device == "cuda" else None
    dt = time.time() - t
    print(f"batch={batch:6d}  ONE full sweep (K=1, 1 member): {dt:.2f}s  "
          f"-> K=20 x 5 members = {dt*20*5/60:.1f} min per (arm,epoch)")
    print("   p_hit mean", float(hits.mean()))
