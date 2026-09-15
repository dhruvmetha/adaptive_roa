"""Separate Part-X's SAMPLING from its READOUT.

partx_faithful scores an eval state with the GP of the leaf containing it, while
every competing arm scores from one global model. So a low score could mean either
"Part-X collected a worse dataset" or "the piecewise readout is worse", and the
campaign numbers cannot tell those apart.

This fits ONE global GP classifier on the points Part-X actually evaluated, and the
SAME model on the points the uniform control evaluated, then scores both on the same
eval states. Model class, capacity and eval set are held fixed, so the only thing
that differs is which points the acquisition chose:

  partx piecewise  vs  partx global   -> how much the readout costs
  partx global     vs  uniform global -> whether the sampling is actually worse

Both global fits use the same `--cap` subsample, so neither gets a capacity edge.

Usage:
  python scripts/partx_global_rescore.py --cell cp_base [--cap 2000] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from adaptive_roa.adaptive.data_source import TrajectoryDataSourceConfig
from adaptive_roa.adaptive.npz_data_source import NpzTrajectoryDataSource
from adaptive_roa.partx_faithful.gp_laplace import LaplaceGPClassifier

EXP = "/common/users/shared/pracsys/adaptive_roa_experiments/timeout_fix"

# cell -> (hydra overrides needed to locate the pool, last epoch dir)
CELLS = {
    "cp_base":      ["system=cartpole_stoch", "controller=safe_explorer_ppo", "noise_level=baseline"],
    "cp_low":       ["system=cartpole_stoch", "controller=safe_explorer_ppo", "noise_level=low"],
    "cp_med":       ["system=cartpole_stoch", "controller=safe_explorer_ppo", "noise_level=med"],
    "cp_high":      ["system=cartpole_stoch", "controller=safe_explorer_ppo", "noise_level=high"],
    "pen_base":     ["system=pendulum_stoch", "noise_level=baseline"],
    "pen_low":      ["system=pendulum_stoch", "noise_level=low"],
    "pen_med":      ["system=pendulum_stoch", "noise_level=med"],
    "pen_high":     ["system=pendulum_stoch", "noise_level=high"],
    "q2d_base":     ["system=quadrotor2d_stoch", "noise_family=corridor_sine_ambient", "noise_level=baseline"],
    "q2d_smooth":   ["system=quadrotor2d_stoch", "noise_family=corridor_sine_ambient", "noise_level=smooth"],
    "q2d_loud":     ["system=quadrotor2d_stoch", "noise_family=corridor_sine_ambient", "noise_level=loud"],
    "q3d_f000":     ["system=quadrotor3d_ppo_stoch", "noise_family=corridor_sine_ambient",
                     "+controller=ppo_1500K", "noise_level=f_0.00"],
    "q3d_f012a03":  ["system=quadrotor3d_ppo_stoch", "noise_family=corridor_sine_ambient",
                     "+controller=ppo_1500K", "noise_level=f_0.12_a0.03"],
    "q3d_f020a035": ["system=quadrotor3d_ppo_stoch", "noise_family=corridor_sine_ambient",
                     "+controller=ppo_1500K", "noise_level=f_0.20_a0.035"],
    "q3d_f040a04":  ["system=quadrotor3d_ppo_stoch", "noise_family=corridor_sine_ambient",
                     "+controller=ppo_1500K", "noise_level=f_0.40_a0.04"],
}


def _compose(overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from adaptive_roa.utils.env_config import get_data_dir
    if not OmegaConf.has_resolver("data_dir"):
        OmegaConf.register_new_resolver("data_dir", lambda: get_data_dir())
    cfgdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "configs", "adaptive_v2")
    with initialize_config_dir(config_dir=cfgdir, version_base=None):
        return compose(config_name="partx_faithful",
                       overrides=overrides + ["output_dir=/tmp/unused"])


def last_epoch(run_dir):
    eps = sorted(d for d in os.listdir(run_dir) if d.startswith("epoch_"))
    return os.path.join(run_dir, eps[-1])


def partx_points(cell):
    """The (X, y) Part-X actually evaluated, from its checkpoint."""
    with open(os.path.join(EXP, f"tf_{cell}_partx_faithful", "partx_state.pkl"), "rb") as f:
        px = pickle.load(f)["px"]
    store = px.store
    ids = np.arange(store.n)
    return store.X(ids), store.y(ids)


def uniform_points(cell, overrides):
    """The (X, y) the uniform control evaluated, via its pool indices."""
    run = os.path.join(EXP, f"tf_{cell}_dir00_s42")
    if not os.path.isdir(run):
        return None, None
    idx = []
    for name in ("train_trajectories.txt", "val_trajectories.txt"):
        p = os.path.join(run, "datasets", name)
        if os.path.exists(p):
            with open(p) as f:
                idx += [int(line) for line in f if line.strip()]
    if not idx:
        return None, None
    cfg = _compose(overrides)
    ds = NpzTrajectoryDataSource(TrajectoryDataSourceConfig(
        trajectories_dir=cfg.data_source.trajectories_dir,
        shuffled_indices_file=cfg.data_source.shuffled_indices_file,
        shuffled_labels_file=cfg.data_source.get("shuffled_labels_file", None)))
    starts = np.asarray(ds.start_states, dtype=float)
    labels = np.asarray(ds.labels, dtype=float)
    idx = np.asarray(sorted(set(idx)), dtype=int)
    idx = idx[idx < len(starts)]
    return starts[idx], labels[idx]


def fit_global(X, y, cap, seed=42):
    gp = LaplaceGPClassifier(seed=seed, n_restarts=5, max_points=cap,
                             lengthscale_bounds=(0.01, 100.0), sigma_f2_bounds=(0.01, 25.0))
    gp.fit(X, y)
    return gp


def scores(p, y_true):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "auprc": float("nan"),
                "brier": float(np.mean((p - y) ** 2)), "n": int(len(y))}
    return {"auc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "brier": float(np.mean((p - y) ** 2)), "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--cap", type=int, default=2000,
                    help="subsample size for BOTH global fits, so neither gets a capacity edge")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    overrides = CELLS[args.cell]
    run = os.path.join(EXP, f"tf_{args.cell}_partx_faithful")

    ev = np.load(os.path.join(last_epoch(run), "full_roa_per_point.npz"), allow_pickle=True)
    eval_X = np.asarray(ev["start_states"], dtype=float)
    raw = np.asarray(ev["true_labels"]).astype(int)
    # true_labels are {-1 failure, 1 success}; anything else is invalid/unlabelled.
    # Never cast {-1,1} with astype(bool) -- -1 is truthy and every point reads success.
    keep = np.isin(raw, (-1, 1))
    eval_X = eval_X[keep]
    eval_y = (raw[keep] == 1).astype(int)
    piecewise_p = np.asarray(ev["p_success"], dtype=float)[keep]

    out = {"cell": args.cell, "cap": args.cap, "n_eval": int(len(eval_y)),
           "base_rate": float(eval_y.mean()),
           "partx_piecewise": scores(piecewise_p, eval_y)}

    pX, py = partx_points(args.cell)
    out["n_partx_points"] = int(len(pX))
    out["partx_global"] = scores(fit_global(pX, py, args.cap).predict_proba(eval_X), eval_y)

    uX, uy = uniform_points(args.cell, overrides)
    if uX is None:
        out["uniform_global"] = None
        out["n_uniform_points"] = 0
    else:
        out["n_uniform_points"] = int(len(uX))
        out["uniform_global"] = scores(fit_global(uX, uy, args.cap).predict_proba(eval_X), eval_y)

    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
