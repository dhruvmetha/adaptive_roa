"""Chained composition evaluation: RL swing-up FM -> LQR FM -> upright check.

For each start state x0, MC-sample the composed rollout with two learned models:
  1. h = RL_FM.predict_endpoint(x0)      (predicted handoff state)
  2. gate: h must lie inside G1 (no handoff -> composition failure)
  3. f = LQR_FM.predict_endpoint(h)      (predicted stabilization endpoint)
  4. success iff f is inside the upright attractor (classify_attractor == 1)

p_success(x0) = fraction of K MC samples that succeed. A decision threshold
lambda* is chosen on the calibration set (max F1) and applied to the test set.

Usage:
  python scripts/evaluate_composition.py \
      --rl-run  <adaptive_pendulum_rl run dir> [--rl-epoch 18] \
      --lqr-run <lqr FM run dir>               [--lqr-epoch 19] \
      --dataset rl --device cuda:0 --mc-samples 10
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from adaptive_roa.flow_matching.pendulum.latent_conditional.flow_matcher import (
    PendulumLatentConditionalFlowMatcher,
)
from adaptive_roa.systems.pendulum import PendulumG1System, PendulumSystem
from adaptive_roa.utils.env_config import get_data_dir


def load_fm(run_dir: Path, epoch: int, device: str, dataset_dir: str | None = None):
    ckpts = glob.glob(str(run_dir / f"epoch_{epoch:03d}" / "checkpoints" / "best*.ckpt"))
    if not ckpts:  # older runs use unpadded epoch dirs
        ckpts = glob.glob(str(run_dir / f"epoch_{epoch}" / "checkpoints" / "best*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No best*.ckpt under {run_dir} epoch {epoch}")
    print(f"Loading {ckpts[0]}")
    return PendulumLatentConditionalFlowMatcher.load_from_checkpoint(
        ckpts[0], device=device, dataset_dir=dataset_dir
    )


def load_split(path: Path):
    data = np.loadtxt(path, delimiter=",")
    return data[:, :2].astype(np.float32), data[:, -1].astype(int)  # states, {0,1}


@torch.no_grad()
def composed_p_success(rl_fm, lqr_fm, g1_sys, upright_sys, X: np.ndarray,
                       K: int, batch_size: int, device: str) -> np.ndarray:
    N = len(X)
    success_counts = np.zeros(N, dtype=np.int64)
    for start in range(0, N, batch_size):
        xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        for _ in range(K):
            h = rl_fm.predict_endpoint(xb)                       # [B, 2] handoff
            gate = g1_sys.in_g1(h)                               # handoff fired?
            succ = torch.zeros(len(xb), dtype=torch.bool, device=device)
            if gate.any():
                f = lqr_fm.predict_endpoint(h[gate])             # [Bg, 2] final
                labels = upright_sys.classify_attractor(f, radius=0.1)
                succ[gate] = labels == 1
            success_counts[start:start + batch_size] += succ.cpu().numpy()
        print(f"  {min(start + batch_size, N)}/{N} states", flush=True)
    return success_counts / K


def pick_lambda(p_cal: np.ndarray, y_cal: np.ndarray) -> float:
    """Grid-search decision threshold maximizing F1 on the calibration set."""
    best_lam, best_f1 = 0.5, -1.0
    for lam in np.linspace(0.0, 1.0, 201):
        pred = p_cal >= lam
        tp = (pred & (y_cal == 1)).sum()
        fp = (pred & (y_cal == 0)).sum()
        fn = (~pred & (y_cal == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_f1, best_lam = f1, lam
    return best_lam


def metrics(p: np.ndarray, y: np.ndarray, lam: float) -> dict:
    pred = (p >= lam).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    # AUC via rank statistic
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) if n_pos and n_neg else None
    return {"lambda": lam, "accuracy": acc, "f1": f1, "auc": auc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-run", required=True, type=Path)
    ap.add_argument("--rl-epoch", type=int, default=18)
    ap.add_argument("--lqr-run", required=True, type=Path)
    ap.add_argument("--lqr-epoch", type=int, default=19)
    ap.add_argument("--dataset", choices=["rl", "rl-weak"], required=True)
    ap.add_argument("--mc-samples", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    pend_root = Path(get_data_dir()) / "deterministic" / "pendulum"
    comp_root = pend_root / f"{args.dataset}_to_lqr"

    g1_sys = PendulumG1System(str(pend_root / args.dataset))
    upright_sys = PendulumSystem(str(pend_root / "lqr"))

    rl_fm = load_fm(args.rl_run, args.rl_epoch, args.device)
    # Old checkpoints may store the pre-restructure dataset path; point the
    # system at the current lqr dataset dir (same data, new location).
    lqr_fm = load_fm(args.lqr_run, args.lqr_epoch, args.device,
                     dataset_dir=str(pend_root / "lqr"))
    rl_fm.eval(); lqr_fm.eval()

    X_cal, y_cal = load_split(comp_root / "cal_set.txt")
    X_test, y_test = load_split(comp_root / "test_set.txt")
    print(f"cal: {len(X_cal)}  test: {len(X_test)}  (composition labels)")

    print("Estimating p_success on calibration set...")
    p_cal = composed_p_success(rl_fm, lqr_fm, g1_sys, upright_sys, X_cal,
                               args.mc_samples, args.batch_size, args.device)
    lam = pick_lambda(p_cal, y_cal)
    print(f"lambda* (max-F1 on cal): {lam:.3f}")

    print("Estimating p_success on test set...")
    p_test = composed_p_success(rl_fm, lqr_fm, g1_sys, upright_sys, X_test,
                                args.mc_samples, args.batch_size, args.device)

    results = {
        "dataset": args.dataset,
        "rl_run": str(args.rl_run), "rl_epoch": args.rl_epoch,
        "lqr_run": str(args.lqr_run), "lqr_epoch": args.lqr_epoch,
        "mc_samples": args.mc_samples,
        "cal": metrics(p_cal, y_cal, lam),
        "test": metrics(p_test, y_test, lam),
        "test_at_0.5": metrics(p_test, y_test, 0.5),
    }
    out = args.output_dir or (args.rl_run / "composition_eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    np.savez(out / "per_point.npz", X_test=X_test, p_success=p_test, y_test=y_test,
             X_cal=X_cal, p_cal=p_cal, y_cal=y_cal, lambda_star=lam)
    print(json.dumps({k: results[k] for k in ("cal", "test", "test_at_0.5")}, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
