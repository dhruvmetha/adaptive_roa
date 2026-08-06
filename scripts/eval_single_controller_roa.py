"""Single-controller ROA predictions for visualization/metrics.

Computes MC p_success for:
  - rl / rl-weak cal states (RL-FM, success = predicted endpoint in G1)
  - lqr cal + test states   (LQR-FM, success = predicted endpoint upright)

The rl / rl-weak TEST p_success is not recomputed: the G1 adaptive runs'
epoch_018 full_roa_per_point.npz already holds it (same model, same G1
criterion via PendulumG1System, K=10).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

from adaptive_roa.flow_matching.pendulum.latent_conditional.flow_matcher import (
    PendulumLatentConditionalFlowMatcher,
)
from adaptive_roa.systems.pendulum import PendulumG1System, PendulumSystem
from adaptive_roa.utils.env_config import get_data_dir

DEVICE = "cuda:0"
K = 10
BATCH = 4096
PEND = Path(get_data_dir()) / "deterministic" / "pendulum"
OUT = Path(__file__).resolve().parent.parent / "docs" / "pendulum_roa_eval_2026-07-24"

RL_RUNS = {
    "rl": "/common/users/shared/pracsys/adaptive_roa_experiments/adaptive_pendulum_rl_dhruv/outputs/training_index_0_warm_start_False_adapt_iter_19/2026-07-23_14-49-19",
    "rl-weak": "/common/users/shared/pracsys/adaptive_roa_experiments/adaptive_pendulum_rl-weak_dhruv/outputs/training_index_0_warm_start_False_adapt_iter_19/2026-07-23_14-49-19",
}
LQR_RUN = "/common/users/shared/pracsys/adaptive_roa_experiments/dhruv/adaptive_pendulum_dhruv/outputs/training_index_0_d2_ratio_1.0_warm_start_False_threshold_mode_dynamic_adapt_iter_20_alpha_0.1_sampling_mode_direct/2026-02-20_01-56-07"


def load_fm(run_dir: str, epoch: int, dataset_dir: str | None = None):
    ckpts = glob.glob(f"{run_dir}/epoch_{epoch:03d}/checkpoints/best*.ckpt") or \
            glob.glob(f"{run_dir}/epoch_{epoch}/checkpoints/best*.ckpt")
    return PendulumLatentConditionalFlowMatcher.load_from_checkpoint(
        ckpts[0], device=DEVICE, dataset_dir=dataset_dir
    )


@torch.no_grad()
def p_success(fm, X: np.ndarray, judge) -> np.ndarray:
    counts = np.zeros(len(X), dtype=np.int64)
    for s in range(0, len(X), BATCH):
        xb = torch.tensor(X[s:s + BATCH], dtype=torch.float32, device=DEVICE)
        for _ in range(K):
            e = fm.predict_endpoint(xb)
            counts[s:s + BATCH] += judge(e).cpu().numpy()
        print(f"  {min(s + BATCH, len(X))}/{len(X)}", flush=True)
    return counts / K


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # RL controllers: cal states only (test p_success comes from the run npz)
    for name, run in RL_RUNS.items():
        g1 = PendulumG1System(str(PEND / name))
        fm = load_fm(run, 18)
        fm.eval()
        cal = np.loadtxt(PEND / f"{name}_to_lqr" / "cal_set.txt", delimiter=",")[:, :2].astype(np.float32)
        p = p_success(fm, cal, lambda e: g1.in_g1(e))
        np.savez(OUT / f"{name}_cal_p_success.npz", X=cal, p_success=p)
        print(f"{name} cal done")
        del fm; torch.cuda.empty_cache()

    # LQR controller: cal + test with upright criterion
    lqr_sys = PendulumSystem(str(PEND / "lqr"))
    fm = load_fm(LQR_RUN, 19, dataset_dir=str(PEND / "lqr"))
    fm.eval()
    judge = lambda e: lqr_sys.classify_attractor(e, radius=0.1) == 1
    for split in ["cal_set", "test_set"]:
        data = np.loadtxt(PEND / "lqr" / f"{split}.txt", delimiter=",")
        X, y = data[:, :2].astype(np.float32), data[:, -1].astype(int)
        p = p_success(fm, X, judge)
        np.savez(OUT / f"lqr_{split}_p_success.npz", X=X, p_success=p, y=y)
        print(f"lqr {split} done")


if __name__ == "__main__":
    main()
