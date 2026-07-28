#!/usr/bin/env python3
"""
Experiment A: does mode separation find the decision boundary where dispersion did not?

Offline. No training. Scores frozen checkpoints from the completed dispersion
campaign with three label-free functionals computed from ONE shared endpoint
cloud, then grades them against ground truth the scorer never saw.

  dispersion  -- mean pairwise distance (the incumbent that lost)
  mode_sep    -- 4p(1-p)*G/(G+W), gap across a 2-way split of the cloud
  idem        -- mean displacement when endpoints are fed back through the model
  gated       -- mode_sep * exp(-idem), the F3 combination

Ground truth comes from each epoch's full_roa_per_point.npz (start_states,
true_labels, model p_success). The criterion is used only to GRADE, never to
score -- which is exactly the contract this research direction is trying to keep.

Metrics per (system, checkpoint, score):
  spearman_u      Spearman vs label-based uncertainty u = -|p_success - 0.5|,
                  the signal the incumbent `ranked` acquisition actually uses
  top_succ_frac   fraction of the top-N that are TRUE successes. Dispersion
                  starved this class (campaign FNR .22-.27 on quad2d); a score
                  that fixes the failure must not.
  top_bdry_frac   fraction of the top-N sitting on the true decision boundary,
                  defined label-free-at-scoring-time as "the k nearest eval
                  states contain both classes". This is the metric that matters:
                  boundary points are what a ROA classifier needs.
  base_bdry_frac  the same fraction over the whole candidate set (random
                  baseline). top_bdry_frac / base_bdry_frac is the lift.

Falsifiers, fixed before running (see the run report for adjudication):
  F1  mode_sep must beat dispersion on quad2d mid-training: spearman_u >= +0.3
      AND top_bdry_frac >= 2x dispersion's. Otherwise cloud geometry does not
      encode basin ambiguity and the direction retreats to a hybrid.
  F2  if spearman(mode_sep, dispersion) > 0.9 everywhere, the clouds never
      become multimodal -- the conditional FM averages instead of mode-splitting,
      and the fix is model expressiveness, not the acquisition score.
  F3  if top_succ_frac still collapses on quad2d, the balance term is not enough
      and the `gated` score is the fallback.

Usage:
  python scripts/experiment_mode_separation_bakeoff.py --systems pendulum cartpole quad2d
"""
from __future__ import annotations

import argparse, glob, importlib, json, os, sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_roa.adaptive_v2.strategy.dispersion_score import (
    mean_pairwise_dispersion, mode_separation, idempotence_defect,
)
from adaptive_roa.conformal.config import ConformalConfig
from adaptive_roa.conformal.probability_estimator import ProbabilityEstimator

EXP = "/common/users/shared/pracsys/adaptive_roa_experiments"
# The campaign trained on Amarel, so checkpoints and configs carry Amarel's data
# root. That path does not exist on iLab; translate it rather than re-staging.
AMAREL_DATA = "/scratch/st1122/genMoPlan-exp/data_trajectories"
ILAB_DATA = "/common/users/shared/pracsys/genMoPlan/data_trajectories"


def to_ilab(path: str) -> str:
    return path.replace(AMAREL_DATA, ILAB_DATA) if path else path


def dataset_dir_of(epoch_dir: Path) -> str:
    """Resolved dataset dir from the checkpoint hparams, remapped to iLab.

    The Hydra config stores dataset_root unresolved (${data_dir}/...), so the
    checkpoint's own `system_dataset_dir` is the only resolved copy on disk.
    """
    ckpt = glob.glob(str(epoch_dir / "checkpoints" / "best*.ckpt"))[0]
    hp = torch.load(ckpt, map_location="cpu", weights_only=False).get("hyper_parameters", {})
    ds = hp.get("system_dataset_dir")
    if not ds:
        raise KeyError(f"no system_dataset_dir in {ckpt}")
    return to_ilab(ds)

# system key -> (experiment dir, system module/class, flow-matcher module/class)
REGISTRY = {
    "pendulum": ("adaptive_pendulum_lqr",
                 "adaptive_roa.systems.pendulum", "PendulumSystem",
                 "adaptive_roa.flow_matching.pendulum.latent_conditional.flow_matcher",
                 "PendulumLatentConditionalFlowMatcher"),
    "cartpole": ("adaptive_cartpole_pybullet",
                 "adaptive_roa.systems.cartpole", "CartPoleSystem",
                 "adaptive_roa.flow_matching.cartpole.latent_conditional.flow_matcher",
                 "CartPoleLatentConditionalFlowMatcher"),
    "quad2d":   ("adaptive_quadrotor2d",
                 "adaptive_roa.systems.quadrotor2d", "Quadrotor2DSystem",
                 "adaptive_roa.flow_matching.quadrotor_2d.latent_conditional.flow_matcher",
                 "Quadrotor2DLatentConditionalFlowMatcher"),
    "quad3d":   ("adaptive_quadrotor3d",
                 "adaptive_roa.systems.quadrotor3d", "Quadrotor3DSystem",
                 "adaptive_roa.flow_matching.quadrotor_3d.latent_conditional.flow_matcher",
                 "Quadrotor3DLatentConditionalFlowMatcher"),
}


def pick_epochs(run_dir: Path, n=3):
    """Early / mid / late epoch dirs that have both a checkpoint and per-point eval."""
    eps = []
    for e in sorted(run_dir.glob("epoch_*")):
        if glob.glob(str(e / "checkpoints" / "best*.ckpt")) and (e / "full_roa_per_point.npz").exists():
            eps.append(e)
    if len(eps) <= n:
        return eps
    idx = [0, len(eps) // 2, len(eps) - 1][:n]
    return [eps[i] for i in idx]


def boundary_mask(states, labels, scales, circ, k=10, chunk=2048, device="cpu"):
    """True where a state's k nearest eval states are not all one class.

    A label-mixed neighbourhood is the operational definition of "on the decision
    boundary". Computed in the same normalized, wrapped metric the scores use.
    """
    S = torch.as_tensor(states, dtype=torch.float32, device=device)
    L = torch.as_tensor(labels, dtype=torch.float32, device=device)
    sc = torch.as_tensor(scales, dtype=torch.float32, device=device)
    cm = torch.as_tensor(circ, dtype=torch.bool, device=device)
    out = np.empty(len(states), dtype=bool)
    for s in range(0, len(states), chunk):
        e = min(s + chunk, len(states))
        diff = S[s:e].unsqueeze(1) - S.unsqueeze(0)            # [m, N, D]
        if bool(cm.any()):
            c = diff[..., cm]
            diff[..., cm] = torch.atan2(torch.sin(c), torch.cos(c))
        d = torch.linalg.vector_norm(diff.div_(sc), dim=-1)     # [m, N]
        nn = d.topk(k + 1, largest=False).indices               # includes self
        lab = L[nn]
        out[s:e] = (lab.max(dim=1).values != lab.min(dim=1).values).cpu().numpy()
    return out


def run_system(key, n_cand, K, top_n, device, seed=0):
    exp_dir, smod, scls, fmod, fcls = REGISTRY[key]
    runs = sorted(glob.glob(f"{EXP}/{exp_dir}/outputs/*sampling_mode_dispersion*/*/"))
    if not runs:
        print(f"  [{key}] no dispersion runs found, skipping"); return []
    run_dir = Path(runs[0])
    System = getattr(importlib.import_module(smod), scls)
    FM = getattr(importlib.import_module(fmod), fcls)
    epochs = pick_epochs(run_dir)
    if not epochs:
        print(f"  [{key}] no epoch has both a checkpoint and per-point eval, skipping"); return []
    ds_dir = dataset_dir_of(epochs[0])
    system = System(dataset_dir=ds_dir)
    scales = system.get_normalization_scales().cpu().numpy().astype(np.float64)
    circ = np.zeros(len(scales), dtype=bool)
    ci = system.get_circular_indices()
    if ci: circ[np.asarray(ci, dtype=int)] = True

    rows = []
    for epoch_dir in epochs:
        npz = np.load(epoch_dir / "full_roa_per_point.npz")
        states_all, true_all, ps_all = npz["start_states"], npz["true_labels"], npz["p_success"]
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(states_all), size=min(n_cand, len(states_all)), replace=False)
        X, y_true, p_succ = states_all[sel].astype(np.float32), true_all[sel], ps_all[sel]

        ckpt = glob.glob(str(epoch_dir / "checkpoints" / "best*.ckpt"))[0]
        fm = FM.load_from_checkpoint(ckpt, device=device, dataset_dir=ds_dir)
        est = ProbabilityEstimator(fm, system, ConformalConfig(num_mc_samples=K), device)

        cloud = est.sample_endpoints(X, num_samples=K, verbose=False)          # [M,K,D]
        M, _, D = cloud.shape
        remapped = est.sample_endpoints(cloud.reshape(-1, D), num_samples=1, verbose=False)
        remapped = remapped.reshape(M, K, D)

        scores = {
            "dispersion": mean_pairwise_dispersion(cloud, scales, circ, device=device),
            "mode_sep":   mode_separation(cloud, scales, circ, device=device),
            "idem":       idempotence_defect(cloud, remapped, scales, circ),
        }
        scores["gated"] = scores["mode_sep"] * np.exp(-scores["idem"])

        u = -np.abs(p_succ - 0.5)
        bdry = boundary_mask(X, y_true, scales, circ, device=device)
        base_b, base_s = bdry.mean(), (y_true == 1).mean()

        for name, sc in scores.items():
            ok = np.isfinite(sc)
            top = np.flatnonzero(ok)[np.argsort(-sc[ok])[:top_n]]
            rho = spearmanr(sc[ok], u[ok]).statistic if ok.sum() > 2 and np.ptp(sc[ok]) > 0 else np.nan
            rows.append(dict(system=key, epoch=int(epoch_dir.name.split("_")[1]), score=name,
                             spearman_u=float(rho),
                             top_succ_frac=float((y_true[top] == 1).mean()),
                             top_bdry_frac=float(bdry[top].mean()),
                             base_succ_frac=float(base_s), base_bdry_frac=float(base_b),
                             n_cand=int(M)))
        r_ms = spearmanr(scores["mode_sep"], scores["dispersion"]).statistic
        rows.append(dict(system=key, epoch=int(epoch_dir.name.split("_")[1]),
                         score="_corr_modesep_vs_dispersion", spearman_u=float(r_ms),
                         top_succ_frac=np.nan, top_bdry_frac=np.nan,
                         base_succ_frac=np.nan, base_bdry_frac=np.nan, n_cand=int(M)))
        print(f"  [{key}] epoch {epoch_dir.name}: scored {M} candidates "
              f"(base boundary {base_b:.3f}, base success {base_s:.3f})")
        del fm, est
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["pendulum", "cartpole", "quad2d"])
    ap.add_argument("--n-candidates", type=int, default=5000)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="docs/plots/mode_separation_bakeoff.csv")
    a = ap.parse_args()

    rows = []
    for key in a.systems:
        print(f"[{key}]")
        rows += run_system(key, a.n_candidates, a.k, a.top_n, a.device)

    import csv
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out} ({len(rows)} rows)\n")

    print("=" * 96)
    print("%-10s %-6s %-12s %8s %10s %10s %8s" %
          ("system", "epoch", "score", "rho_u", "top_succ", "top_bdry", "lift"))
    print("=" * 96)
    for r in rows:
        if r["score"].startswith("_"):
            print("%-10s %-6s %-12s %8.3f   (mode_sep vs dispersion rank corr -- F2 check)" %
                  (r["system"], r["epoch"], "CORR", r["spearman_u"])); continue
        lift = r["top_bdry_frac"] / r["base_bdry_frac"] if r["base_bdry_frac"] else np.nan
        print("%-10s %-6s %-12s %8.3f %10.3f %10.3f %8.2fx" %
              (r["system"], r["epoch"], r["score"], r["spearman_u"],
               r["top_succ_frac"], r["top_bdry_frac"], lift))


if __name__ == "__main__":
    main()
