#!/usr/bin/env python
"""Stage-0 probe: how much irreducible randomness does each tau level actually carry?

WHY THIS RUNS BEFORE ANY TRAINING
---------------------------------
The classifier half of the ensemble-epistemic campaign found that the five
acquisition arms are indistinguishable until the aleatoric mass in the candidate
pool clears roughly 0.1 nats (docs/experiments/ensemble_epistemic/FINDINGS.md,
"Separation is a monotone function of noise, with a threshold"). Below that, all
five arms are the same experiment run five times. An FM ensemble arm costs ~2.5 h
per epoch x 19 epochs on two GPUs, so choosing levels by guesswork is expensive
in a way this probe is not: it reads ground truth off disk and needs no GPU.

WHAT IT MEASURES, AND WHAT IT DOES NOT
--------------------------------------
The campaign's "aleatoric pool mass" is MODEL-side: mean_m H(p_m) averaged over
candidates, which needs a trained ensemble. This probe reports the GROUND-TRUTH
quantities instead — mean H(p_true), and the fraction of the pool that is
genuinely ambiguous — because those need no model and cannot be distorted by one.

The two are NOT interchangeable, and the direction of the error is not what you
would guess: a MISCALIBRATED ensemble reports more aleatoric mass than the data
holds, not less. That is the whole finding of the classifier half — its p_bar was
distorted enough at high noise that every acquisition score landed on states whose
true p was ~0.02 (FINDINGS.md section 3). So the model-side number is not an upper
bound on anything, and a level cannot be qualified or disqualified by it.

The verdict below therefore rests on `frac_ambiguous`, which IS directly
comparable across campaigns: scripts/acquisition_diagnostics.py:65 computes the
same 0.2 < p_true < 0.8 share, on the same kind of ground-truth grid, and its
random/non-adaptive split is the same "what the pool looks like with no selection"
reference. Those published numbers are hard-coded below as REFERENCE_LEVELS.

Read `acquisition.diagnostics.aleatoric_mean` from any arm's
`epoch_000/artifacts_v2.json` afterwards to see what the ensemble *thinks* the
aleatoric mass is. A large gap over the ground-truth value here is a calibration
warning about that arm, not a property of the dataset.

Two populations are reported because they answer different questions:

  grid  — every one of the 49,770 eval cells, uniform over the state space.
          This is what the model is SCORED on.
  pool  — the same ground truth looked up at the nearest cell to each training
          pool start. This is what acquisition CHOOSES FROM. The pool is sampled
          uniformly at random over the same box, so the two should agree closely;
          a gap means the pool is not covering the space the metric rewards.

Usage:
    python scripts/ensemble/probe_tau_aleatoric.py
    python scripts/ensemble/probe_tau_aleatoric.py --taus tau_0.00 tau_0.30 --pool-sample 50000
    python scripts/ensemble/probe_tau_aleatoric.py --out docs/experiments/ensemble_epistemic/TAU_PROBE.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import xlogy

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from adaptive_roa.utils.env_config import get_data_dir  # noqa: E402

TAU_SUBDIR = "stochastic/pendulum/noisy_torque/lqr"
DEFAULT_TAUS = ("tau_0.00", "tau_0.10", "tau_0.15", "tau_0.30", "tau_0.50")

# "Ambiguous" has no canonical width, so both are printed. The NARROW band is the
# load-bearing one: scripts/acquisition_diagnostics.py:65 uses exactly
# 0.2 < p_true < 0.8, which is what makes REFERENCE_LEVELS below comparable. The
# wide band is reported only because it moves first as tau grows.
BANDS = ((0.2, 0.8), (0.1, 0.9))
PRIMARY_BAND = "frac_ambiguous_0.2_0.8"

# The four levels of the OLDER stochastic pendulum family
# ({DATA_DIR}/noisy/pendulum/lqr/*, reached by system=pendulum_stoch), as measured
# by acquisition_diagnostics.py and published in docs/stoch_compare/acquisition.md.
# Values are the non-adaptive (d2=0) D1(random) rows — the pool with no selection
# applied, which is the same population this probe measures.
#
# These are the anchor because BOTH prior campaigns' verdicts are indexed to them:
#   low   — no arm separation; adaptive gives a small consistent help
#   med   — no arm separation; benefit real in direction, transient in magnitude
#   high  — arms separate; all adaptive arms harmful [CLF]
#   xhigh — arms separate strongly; the epistemic split cuts the harm ~10x [CLF]
# A tau level is worth spending 50 GPU-hours on only if it reaches `high` or above.
REFERENCE_LEVELS = {
    "noisy/low": {"frac_ambiguous": 0.0280, "mean_p": 0.3911, "separates": False},
    "noisy/med": {"frac_ambiguous": 0.0519, "mean_p": 0.3940, "separates": False},
    "noisy/high": {"frac_ambiguous": 0.1323, "mean_p": 0.4068, "separates": True},
    "noisy/xhigh": {"frac_ambiguous": 0.2682, "mean_p": 0.4827, "separates": True},
}

# The lowest reference level at which arms actually separated.
SEPARATION_FLOOR = REFERENCE_LEVELS["noisy/high"]["frac_ambiguous"]

THETA_PERIOD = 2.0 * np.pi


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Bernoulli entropy in nats, exactly 0 at p = 0 and p = 1.

    Mirrors adaptive_roa/adaptive_v2/strategy/uncertainty_scores.py:binary_entropy
    so this probe and the live acquisition score agree to float precision. xlogy
    defines 0*log(0) = 0, which keeps the deterministic control (tau_0.00, where
    every cell is 0 or 1) at exactly 0 rather than at a clipped epsilon.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return -(xlogy(p, p) + xlogy(1.0 - p, 1.0 - p))


def load_grid(root: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Ground-truth (states, p_success, n_batches) for one tau level."""
    with np.load(root / "eval_success_prob.npz") as z:
        return (z["starts"].astype(np.float64),
                z["p_success"].astype(np.float64),
                int(z["n_batches"]))


def load_pool_starts(root: Path, shuffle_variant: int = 0) -> np.ndarray:
    """Training-pool start states, in the order acquisition sees them.

    Acquisition records POOL indices k, not npz rows: the loader maps k through
    the shuffle as npz_row = rollout_ids[k]. Indexing the npz directly with k
    returns unrelated states, and because the shuffle is a random permutation the
    result looks exactly like random selection — a very convincing wrong answer.
    Same trap documented in scripts/acquisition_diagnostics.py:pool_starts.

    Only `starts` is read, so the 590 MB `states` array is never touched.
    """
    with np.load(root / "train.npz") as z:
        starts = z["starts"].astype(np.float64)
    ids_file = root / "train_test_splits" / f"shuffled_indices_{shuffle_variant}.txt"
    rollout_ids = np.loadtxt(ids_file, dtype=np.int64, ndmin=1)
    return starts[rollout_ids]


def nearest_grid_p(pool: np.ndarray, grid: np.ndarray, p_grid: np.ndarray) -> np.ndarray:
    """Ground-truth p for each pool start, via its nearest eval cell.

    theta is periodic on S^1, so the grid is replicated at theta +/- 2*pi before
    the query. Without that, a pool state just past +pi matches a cell at the far
    end of theta instead of its true neighbour — one cell wide out of 158, but it
    lands exactly on the wrap seam where the pendulum's two failure basins meet,
    which is the most ambiguous region on the map.
    """
    shifted = [grid]
    for shift in (-THETA_PERIOD, THETA_PERIOD):
        g = grid.copy()
        g[:, 0] += shift
        shifted.append(g)
    tiled = np.vstack(shifted)
    tiled_p = np.tile(p_grid, len(shifted))
    _, idx = cKDTree(tiled).query(pool, k=1)
    return tiled_p[idx]


def summarize(p: np.ndarray, label: str) -> dict:
    h = binary_entropy(p)
    row = {
        "population": label,
        "n": int(p.size),
        "mean_entropy_nats": float(h.mean()),
        "mean_p": float(p.mean()),
        "frac_decided": float(np.mean((p <= 0.0) | (p >= 1.0))),
    }
    for lo, hi in BANDS:
        row[f"frac_ambiguous_{lo}_{hi}"] = float(np.mean((p > lo) & (p < hi)))
    return row


def probe_level(root: Path, pool_sample: int, seed: int) -> list[dict]:
    grid_states, p_grid, n_batches = load_grid(root)
    rows = [summarize(p_grid, "grid")]
    rows[0]["n_batches"] = n_batches

    pool = load_pool_starts(root)
    if 0 < pool_sample < len(pool):
        # A uniform subsample of a uniformly-sampled pool is still uniform, and
        # the KD-tree query is the only slow step here.
        rng = np.random.default_rng(seed)
        pool = pool[rng.choice(len(pool), size=pool_sample, replace=False)]
    rows.append(summarize(nearest_grid_p(pool, grid_states, p_grid), "pool"))
    rows[1]["n_batches"] = n_batches
    return rows


def render(results: dict[str, list[dict]]) -> str:
    out = [
        "# Stage-0 probe — ground-truth ambiguity by tau",
        "",
        "Generated by `scripts/ensemble/probe_tau_aleatoric.py`. No model involved: every",
        "number is read from each level's ground-truth eval grid.",
        "",
        "`frac ambiguous` is the share of the pool with `0.2 < p_true < 0.8`, matching",
        "`scripts/acquisition_diagnostics.py:65`, which is what makes the reference rows",
        "below comparable. Arms separated at `noisy/high` and above; they did not at",
        f"`noisy/med` and below. Separation floor: **{SEPARATION_FLOOR:.3f}**.",
        "",
        "| level | population | n | frac ambiguous | ambig 0.1-0.9 | mean p | mean H (nats) | decided | reaches separation? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tau, rows in results.items():
        for r in rows:
            verdict = ""
            if r["population"] == "pool":
                verdict = ("**yes**" if r[PRIMARY_BAND] >= SEPARATION_FLOOR
                           else f"no ({r[PRIMARY_BAND] / SEPARATION_FLOOR:.2f}x floor)")
            out.append(
                f"| `{tau}` | {r['population']} | {r['n']:,} | "
                f"{r[PRIMARY_BAND]:.4f} | {r['frac_ambiguous_0.1_0.9']:.4f} | "
                f"{r['mean_p']:.4f} | {r['mean_entropy_nats']:.4f} | "
                f"{r['frac_decided']:.3f} | {verdict} |"
            )
    out.append("")
    out.append("Reference — the older `noisy/pendulum/lqr/*` family, non-adaptive D1(random)")
    out.append("split, from `docs/stoch_compare/acquisition.md`:")
    out.append("")
    out.append("| level | frac ambiguous | mean p | arms separated? |")
    out.append("|---|---|---|---|")
    for name, ref in REFERENCE_LEVELS.items():
        out.append(f"| `{name}` | {ref['frac_ambiguous']:.4f} | {ref['mean_p']:.4f} | "
                   f"{'**yes**' if ref['separates'] else 'no'} |")
    out += [
        "",
        "**Reading it.** `decided` is the share at exactly p = 0 or p = 1. At `tau_0.00` it",
        "is 1.000 and mean H is exactly 0 — the check that the deterministic control really",
        "is deterministic.",
        "",
        "**Grid vs pool agreement is a data check, not a result.** Both are uniform over the",
        "same box, so they should match. A gap would mean acquisition chooses from a",
        "population the metric does not reward.",
        "",
        "**The model-side number is not a substitute for this one.** Read",
        "`acquisition.diagnostics.aleatoric_mean` from an arm's",
        "`epoch_000/artifacts_v2.json` to see what the ensemble believes. A miscalibrated",
        "model reports MORE aleatoric mass than the data holds — that is the classifier",
        "half's entire finding — so a large gap over the ground-truth value here is a",
        "calibration warning about that arm, not evidence about the dataset.",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taus", nargs="+", default=list(DEFAULT_TAUS))
    ap.add_argument("--data-root", default=None,
                    help="defaults to {DATA_DIR}/" + TAU_SUBDIR)
    ap.add_argument("--pool-sample", type=int, default=50000,
                    help="pool starts to subsample; 0 uses all 100k. The default "
                         "matches acquisition.n_candidates so the pool row describes "
                         "the same population an arm scores each epoch.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="write markdown here instead of stdout")
    a = ap.parse_args()

    root = Path(a.data_root) if a.data_root else Path(get_data_dir()) / TAU_SUBDIR

    results: dict[str, list[dict]] = {}
    for tau in a.taus:
        d = root / tau
        if not d.is_dir():
            print(f"SKIP {tau}: {d} not readable", file=sys.stderr)
            continue
        print(f"probing {tau} ...", file=sys.stderr)
        results[tau] = probe_level(d, a.pool_sample, a.seed)

    if not results:
        print("no readable tau levels found", file=sys.stderr)
        return 1

    text = render(results)
    if a.out:
        Path(a.out).write_text(text)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
