"""Shared plumbing for the ensemble uncertainty-decomposition maps.

The campaign we are visualising is `ensemble_epistemic` on the noisy pendulum:
2 predictors (clf, fm) x 5 noise levels (det, low, med, high, xhigh) x 5
acquisition arms (dir00, total, aleat, epi_var, epi_bald) x ~19 adaptive epochs.

WHAT CAN BE RECOVERED FROM DISK, AND WHAT CANNOT
------------------------------------------------
`full_roa_per_point.npz` is written every epoch for every run and holds the
ensemble MARGINAL p_success over the 39,770-point evaluation set. From the
marginal alone the *total* uncertainty is exact -- both flavours:

    H_total   = H(p_bar)                 (binary entropy, nats)
    Var_total = p_bar (1 - p_bar)        (Bernoulli variance)

The aleatoric/epistemic SPLIT is not a function of the marginal. It needs the
per-member probabilities p_1..p_M, which were never serialised. They can only be
recomputed by re-running the members, so the split exists exactly where the
per-epoch checkpoints survived: `clf_high`, `fm_high` and `fm_xhigh`. Every
other predictor x level cell had its checkpoints deleted, so for those the split
is permanently unrecoverable and only the total is shown.

DECOMPOSITION CONVENTIONS
-------------------------
Entropy (matches adaptive_roa/adaptive_v2/strategy/uncertainty_scores.py, which
is what the acquisition arms actually scored):

    H(p_bar) = E_m[H(p_m)] + I(y; m)
    total      aleatoric     epistemic (BALD)

Variance, via the law of total variance:

    p_bar(1-p_bar) = E_m[p_m(1-p_m)] + Var_m(p_m)
    total            aleatoric         epistemic

`Var_m` uses ddof=0 here so the identity holds EXACTLY and the three panels are
guaranteed to add up on screen. The acquisition score `epistemic_var` instead
uses ddof=1 minus an MC debias term; that debiased version is carried alongside
as `epi_var_debiased` for the flow-matching arms, where each p_m is itself a
K-sample estimate and the raw between-member variance is inflated by
mean_m[p_m(1-p_m)]/(K-1).

BINARY, NOT TERNARY. Flow-matching eval assigns three outcomes
(success/failure/invalid) and p_invalid is genuinely non-zero -- up to 1.0 at
`high`. The acquisition backend, however, scores success-vs-not-success:
`EnsembleEndpointMCProbabilityBackend.estimate_members` counts only
`classify_attractor == 1`, so failure and invalid are pooled. Every uncertainty
quantity here follows that binary convention so the maps show what acquisition
actually ranked on. p_invalid is plotted as its own panel rather than
renormalised away.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import xlogy

EXP = Path("/common/users/shared/pracsys/adaptive_roa_experiments/ensemble_epistemic")
POOL = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum/lqr")

LEVELS = ("det", "low", "med", "high", "xhigh")
ARMS = ("dir00", "total", "aleat", "epi_var", "epi_bald")
PREDICTORS = ("clf", "fm")

# Arms whose per-epoch member checkpoints survived, so the split is recoverable.
DECOMPOSABLE = ("clf_high", "fm_high", "fm_xhigh")

ARM_LABEL = {
    "dir00": "dir00 (random control)",
    "total": "total entropy",
    "aleat": "aleatoric",
    "epi_var": "epistemic var",
    "epi_bald": "epistemic BALD",
}


def run_dir(pred: str, level: str, arm: str) -> Path:
    return EXP / f"{pred}_{level}_{arm}"


def epochs_with_eval(pred: str, level: str, arm: str) -> list[int]:
    """Adaptive epochs that produced a per-point eval file, epoch 0 included.

    Epoch 0 is kept here (unlike the earlier fig1) because for a state-space map
    it is the meaningful common baseline: every arm holds identical
    pre-acquisition data, so epoch 0 shows the shared starting point the arms
    then diverge from.
    """
    d = run_dir(pred, level, arm)
    return sorted(int(f.parent.name.split("_")[1])
                  for f in d.glob("epoch_*/full_roa_per_point.npz"))


def load_marginal(pred: str, level: str, arm: str, epoch: int) -> dict:
    """Stored ensemble marginal for one (arm, epoch)."""
    f = run_dir(pred, level, arm) / f"epoch_{epoch:03d}" / "full_roa_per_point.npz"
    with np.load(f) as z:
        return {
            "states": z["start_states"].astype(np.float64),
            "p_success": z["p_success"].astype(np.float64),
            "p_invalid": z["p_invalid"].astype(np.float64),
            "true_labels": z["true_labels"].astype(np.int64),
            "lambda_star": float(z["lambda_star"]),
            "delta": float(z["delta"]),
        }


# --------------------------------------------------------------------- grid
_grid_cache: dict = {}


def grid_axes(level: str) -> tuple[np.ndarray, np.ndarray]:
    """The (theta, theta_dot) rollout grid the eval set is a subset of."""
    f = POOL / level / "eval_success_prob.npz"
    if not f.exists():
        return None, None
    if level not in _grid_cache:
        with np.load(f) as z:
            _grid_cache[level] = (z["grid_theta"].astype(np.float64),
                                  z["grid_theta_dot"].astype(np.float64),
                                  z["starts"].astype(np.float64),
                                  z["p_success"].astype(np.float64))
    th, td, _, _ = _grid_cache[level]
    return th, td


def _nearest_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Index of the closest axis tick for each value.

    The eval states are stored float32 while the rollout grid is float64, so an
    equality or searchsorted-based lookup misses on the theta_dot axis. Nearest
    is exact here because the spacing (~0.04) dwarfs float32 resolution.
    """
    return np.abs(values[:, None] - axis[None, :]).argmin(axis=1)


def rasteriser(level: str, states: np.ndarray):
    """Return (shape, flat_index, extent) to paint eval values onto the grid.

    `det` has no rollout grid file, so its states get their own axes derived
    from the unique coordinates actually present.
    """
    th, td = grid_axes(level)
    if th is None:
        th = np.unique(states[:, 0])
        td = np.unique(states[:, 1])
    i = _nearest_index(th, states[:, 0])
    j = _nearest_index(td, states[:, 1])
    shape = (len(td), len(th))                      # rows = theta_dot
    flat = j * len(th) + i
    extent = (th[0], th[-1], td[0], td[-1])
    return shape, flat, extent


def to_grid(values: np.ndarray, shape, flat) -> np.ndarray:
    """Scatter per-point values onto the grid; unvisited cells stay NaN."""
    out = np.full(int(np.prod(shape)), np.nan, dtype=np.float64)
    out[flat] = values
    return out.reshape(shape)


# ------------------------------------------------------------------- oracle
def oracle_full_image(level: str):
    """Oracle p(success) painted on the COMPLETE rollout grid.

    The eval set covers 39,770 of the 49,770 cells, so painting ground truth on
    the eval points alone speckles it with holes that read as structure. Ground
    truth is known everywhere, so it is drawn everywhere; only the model outputs
    are restricted to cells that were actually evaluated.
    """
    if not (POOL / level / "eval_success_prob.npz").exists():
        return None, None
    grid_axes(level)
    th, td, starts, p = _grid_cache[level]
    i = _nearest_index(th, starts[:, 0])
    j = _nearest_index(td, starts[:, 1])
    img = np.full((len(td), len(th)), np.nan)
    img[j, i] = p
    return img, (th[0], th[-1], td[0], td[-1])


def oracle_prob(level: str, states: np.ndarray) -> tuple[np.ndarray, str]:
    """Oracle p(success) at `states`, plus a label describing its provenance.

    Noisy levels have 90 rollouts per grid cell. `det` is deterministic, so the
    oracle probability is degenerate in {0,1} and equals the binary label.
    """
    f = POOL / level / "eval_success_prob.npz"
    if not f.exists():
        return None, "deterministic: p in {0,1} = label"
    grid_axes(level)
    _, _, starts, p = _grid_cache[level]
    from scipy.spatial import cKDTree
    key = f"_tree_{level}"
    if key not in _grid_cache:
        _grid_cache[key] = cKDTree(starts)
    return p[_grid_cache[key].query(states, k=1)[1]], "oracle, 90 rollouts/cell"


# ------------------------------------------------------------ decomposition
def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Bernoulli entropy in nats, exactly 0 at p=0 and p=1 (xlogy(0,0)=0)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return -(xlogy(p, p) + xlogy(1.0 - p, 1.0 - p))


def decompose(p_members: np.ndarray, k: float | None = None) -> dict:
    """Full entropy and variance decomposition from per-member probabilities.

    p_members is [M, N]. Both identities hold exactly on the returned arrays:
        h_total   == h_aleatoric + h_epistemic
        var_total == var_aleatoric + var_epistemic
    """
    p = np.clip(np.asarray(p_members, dtype=np.float64), 0.0, 1.0)
    if p.ndim != 2 or p.shape[0] < 2:
        raise ValueError(f"p_members must be [M>=2, N]; got {p.shape}")
    pbar = p.mean(axis=0)
    out = {
        "p_bar": pbar,
        "h_total": binary_entropy(pbar),
        "h_aleatoric": binary_entropy(p).mean(axis=0),
        "var_total": pbar * (1.0 - pbar),
        "var_aleatoric": (p * (1.0 - p)).mean(axis=0),
        "var_epistemic": p.var(axis=0),              # ddof=0 -> identity exact
    }
    out["h_epistemic"] = out["h_total"] - out["h_aleatoric"]
    if k is not None and np.isfinite(k) and k > 1:
        # The acquisition score: ddof=1 between-member variance minus each
        # member's own K-sample binomial noise. Can go negative where members
        # genuinely agree, which is informative and is NOT clipped here.
        out["var_epistemic_debiased"] = (
            p.var(axis=0, ddof=1) - (p * (1.0 - p) / (float(k) - 1.0)).mean(axis=0)
        )
    return out


def member_file(pred: str, level: str, arm: str, epoch: int, root: Path) -> Path:
    return Path(root) / f"{pred}_{level}_{arm}" / f"members_epoch_{epoch:03d}.npz"
