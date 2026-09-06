#!/usr/bin/env python
"""Score one ensemble arm's last epoch under BOTH success criteria in one GPU pass.

WHY THIS EXISTS
---------------
`scripts/reevaluate.py` cannot run these arms: its `load_checkpoint` reads
`epoch_dir/checkpoints/best*.ckpt`, but every ensemble arm stores weights under
`epoch_dir/member_M/checkpoints/`. It also takes a single `--attractor_radius`,
so getting two criteria out of it would mean two GPU passes with independent MC
draws -- an unpaired comparison at twice the cost.

This driver instead samples the ensemble ONCE and classifies each batch under
both criteria before discarding it, so:

  * the comparison is PAIRED -- identical sampled endpoints, only the classifier
    differs, which removes MC noise from the delta entirely;
  * nothing is cached or materialised. Peak memory is one batch, not the
    [N, K, state_dim] endpoint array (5.15 GB on quad3D);
  * the pendulum criterion is applied as the BOX it actually is. Approximating
    it with a ball of radius 0.05*sqrt(2) agrees to within 20/100,000 on dataset
    endpoints, but errs exactly in the corner region a PREDICTED endpoint can
    occupy -- the regime under test.

The ensemble is reproduced exactly: `EnsembleFlowMatcherHandle.predict_endpoint`
round-robins members through a stateful cursor rather than sampling one at
random, so K=100 over 5 members is exactly 20 draws each. The loop order here
mirrors `compute_mc_predictions` (batch outer, sample inner) so that cursor
walks the same sequence it did during the campaign.

Usage:
    python scripts/eval_success_criteria.py --run <run_dir> [--epoch N] --out r.json
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from adaptive_roa.adaptive.data_source import load_eval_states          # noqa: E402
from adaptive_roa.adaptive_v2.trainers.ensemble_flow_matching_trainer import (  # noqa: E402
    EnsembleFlowMatcherHandle,
)
import stoch_prob_metrics as spm                                        # noqa: E402
from reevaluate import _resolve_system_classes, load_hydra_config       # noqa: E402

DATA = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories")

# The criterion each dataset was actually LABELLED with, from the deterministic
# dataset_description.json. Verified against the shipped labels: each reproduces
# them at 1.0000 agreement on every surviving level.
DATASET_CRITERION = {
    "PendulumSystem":    {"kind": "box",    "half": 0.05},   # |th|<.05 and |thd|<.05
    "CartPoleSystem":    {"kind": "radius", "value": 0.05},
    "Quadrotor2DSystem": {"kind": "radius", "value": 0.20},
    "Quadrotor3DSystem": {"kind": "radius", "value": 0.05},
}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def last_epoch_dir(run: Path, epoch: int | None) -> Path:
    eps = sorted(d for d in run.glob("epoch_*") if (d / "artifacts_v2.json").exists())
    if not eps:
        raise SystemExit(f"{run}: no epoch with artifacts_v2.json")
    if epoch is None:
        return eps[-1]
    for d in eps:
        if int(re.search(r"\d+", d.name).group()) == epoch:
            return d
    raise SystemExit(f"{run}: epoch {epoch} not found")


def load_ensemble(epoch_dir: Path, FlowMatcherClass, device: str):
    """Reload members with the same glob EnsembleFlowMatchingTrainer._load_member uses."""
    members = []
    for m in range(64):
        md = epoch_dir / f"member_{m}"
        if not md.is_dir():
            break
        ck = sorted(glob.glob(str(md / "**" / "best*.ckpt"), recursive=True))
        if not ck:
            raise SystemExit(f"{md}: no best*.ckpt -- member did not finish training")
        members.append(FlowMatcherClass.load_from_checkpoint(ck[0], device=device))
    if not members:                       # non-ensemble fallback
        ck = sorted(glob.glob(str(epoch_dir / "checkpoints" / "best*.ckpt")))
        if not ck:
            raise SystemExit(f"{epoch_dir}: neither member_*/ nor checkpoints/")
        return FlowMatcherClass.load_from_checkpoint(ck[0], device=device), 1
    if len(members) == 1:
        return members[0], 1
    return EnsembleFlowMatcherHandle(members), len(members)


def success_mask(pred: torch.Tensor, system, crit: dict) -> torch.Tensor:
    """Boolean success under one criterion. Radius uses the production
    classify_attractor path so the FM-side number is bit-identical to a fresh
    eval; the box is applied directly because no radius can express it."""
    if crit["kind"] == "radius":
        return system.classify_attractor(pred, crit["value"]) == 1
    h = crit["half"]
    th = torch.remainder(pred[:, 0] + torch.pi, 2 * torch.pi) - torch.pi
    return (th.abs() < h) & (pred[:, 1].abs() < h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_mc_samples", type=int, default=None)
    a = ap.parse_args()

    t0 = time.time()
    cfg = load_hydra_config(a.run)
    SystemClass, FlowMatcherClass, sys_name = _resolve_system_classes(cfg)
    system = SystemClass()

    ep_dir = last_epoch_dir(a.run, a.epoch)
    ep_num = int(re.search(r"\d+", ep_dir.name).group())
    test_file = cfg["data_source"]["test_set_file"]
    # dataset_root is unset in these configs; the ground truth sits beside the
    # test set, so derive it rather than trusting an absent key.
    ds_rel = Path(test_file).parent

    # conformal.attractor_radius / num_mc_samples_eval are also unset here. The
    # values the campaign ACTUALLY evaluated with are recorded per epoch, so read
    # them from the artifacts instead of re-deriving a default that might differ.
    with np.load(ep_dir / "full_roa_per_point.npz") as z:
        r_fm = float(z["attractor_radius"])
    K = a.num_mc_samples
    if K is None:
        try:
            K = int(json.loads((ep_dir / "artifacts_v2.json").read_text())
                    ["eval_metrics"]["num_mc_samples"])
        except (KeyError, ValueError):
            K = 100
    bs = a.batch_size or int(cfg.get("val_batch_size") or 2048)
    crit_ds = DATASET_CRITERION[sys_name]
    crit_fm = {"kind": "radius", "value": r_fm}

    print(f"  run={a.run.name} epoch={ep_num} system={sys_name}", flush=True)
    print(f"  criteria: dataset={crit_ds}  fm_eval=radius {r_fm}", flush=True)
    print(f"  K={K} batch={bs} device={a.device}", flush=True)

    handle, n_mem = load_ensemble(ep_dir, FlowMatcherClass, a.device)
    handle.eval()
    handle.to(a.device)
    print(f"  loaded {n_mem} member(s)", flush=True)

    # state_dim is LOAD-BEARING: without it a 5-column stochastic file is read
    # as "2 start + 2 end + label" when it is really "4 state + p_success",
    # which silently truncates cartpole to 2-D (full_roa.py documents the same
    # trap). Pendulum is 2-D so it survives the guess; nothing else does.
    X, _, _ = load_eval_states(test_file, state_dim=getattr(system, "state_dim", None))
    N = len(X)
    Xt = torch.from_numpy(np.ascontiguousarray(X)).float().to(a.device)
    n_ds = np.zeros(N, dtype=np.int32)
    n_fm = np.zeros(N, dtype=np.int32)

    nb = (N + bs - 1) // bs
    with torch.no_grad():
        for bi, s in enumerate(range(0, N, bs)):
            e = min(s + bs, N)
            xb = Xt[s:e]
            acc_ds = torch.zeros(e - s, dtype=torch.int32, device=a.device)
            acc_fm = torch.zeros(e - s, dtype=torch.int32, device=a.device)
            for _ in range(K):
                pred = handle.predict_endpoint(xb)
                acc_ds += success_mask(pred, system, crit_ds).to(torch.int32)
                acc_fm += success_mask(pred, system, crit_fm).to(torch.int32)
            n_ds[s:e] = acc_ds.cpu().numpy()
            n_fm[s:e] = acc_fm.cpu().numpy()
            if bi % 25 == 0 or bi == nb - 1:
                print(f"    batch {bi+1}/{nb}  {time.time()-t0:.0f}s", flush=True)

    p_ds = n_ds.astype(np.float64) / K
    p_fm = n_fm.astype(np.float64) / K

    gt_starts, gt_p, gt_s, gt_t = spm.load_ground_truth(Path(ds_rel))
    idx = spm.match_to_truth(X, gt_starts)
    res = {
        "run": a.run.name, "epoch": ep_num, "system": sys_name,
        "dataset_root": str(ds_rel), "n_members": n_mem, "K": K, "n_eval": N,
        "criterion_dataset": crit_ds, "criterion_fm_eval": crit_fm,
        "mean_p_dataset": float(p_ds.mean()), "mean_p_fm": float(p_fm.mean()),
        "frac_cells_changed": float(np.mean(p_ds != p_fm)),
        "mean_abs_dp": float(np.mean(np.abs(p_ds - p_fm))),
        "metrics_dataset": spm.all_metrics(p_ds, gt_p[idx], gt_s[idx], gt_t[idx], K),
        "metrics_fm_eval": spm.all_metrics(p_fm, gt_p[idx], gt_s[idx], gt_t[idx], K),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    md, mf = res["metrics_dataset"], res["metrics_fm_eval"]
    print(f"  DATASET  KL={md['KL']:.4f} brier_deb={md['brier_debiased']:.4f} sAUROC={md['sAUROC']:.4f}")
    print(f"  FM_EVAL  KL={mf['KL']:.4f} brier_deb={mf['brier_debiased']:.4f} sAUROC={mf['sAUROC']:.4f}")
    print(f"  cells changed: {res['frac_cells_changed']:.4%}   wrote {a.out}")


if __name__ == "__main__":
    main()
