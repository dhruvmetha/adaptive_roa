"""Condition metrics on proximity to the basin boundary.

Aggregate accuracy is dominated by basin interiors, where every arm is
correct. Arms differ where the outcome flips -- which is precisely the
regime adaptive sampling exists to resolve -- so a benchmark reported only
in aggregate can rank arms identically while they behave very differently
at the boundary. ``test_boundary_errors_are_invisible_in_the_aggregate`` in
the test module is the concrete demonstration: an arm wrong on EVERY
boundary point still scores > 0.85 overall.

The band is empirical (a point whose k nearest neighbours do not all share
its label), so no analytic separatrix is needed and it applies to all four
systems in this benchmark.

Three design decisions worth stating explicitly, because getting any of
them wrong produces a band that looks plausible and is meaningless:

1. **Normalization is enforced, not trusted.** Distances mix dimensions of
   different physical scale (e.g. an angular velocity in rad/s next to a
   position in metres); the large-scale dimension dominates the k-NN
   computation and the resulting band reflects that dimension alone,
   nothing else. Rather than relying on a docstring telling callers to
   normalize first, ``separatrix_band``/``conditioned_metrics`` accept an
   optional ``system`` argument exposing ``normalize_state`` (the same
   interface every ``adaptive_roa.systems.*`` class already implements) and
   normalize internally when it is given. When it is not given, a
   best-effort heuristic (`_guard_unnormalized_scale`) rejects inputs whose
   per-dimension spread ratio is implausibly large -- it cannot prove the
   input is normalized, but it catches the concrete failure mode (a raw
   rad/s column next to a [-1, 1] column) without needing per-system
   knowledge.

2. **k-NN, not a dense pairwise matrix.** A naive implementation builds an
   n x n distance matrix. Real eval grids in this project are not small:
   ``eval_states.txt`` row counts observed on disk are ~50k (pendulum),
   ~116k (cartpole), ~490k (quadrotor2d), and 1,000,000 (quadrotor3d); the
   humanoid eval set has 650k rows at 30 state dimensions. A dense float64
   distance matrix is 8*n^2 bytes: ~19.8GB already at the pendulum's 49,770
   rows, ~108GB at cartpole's 116,242, and it never becomes viable from
   there (quadrotor3d would need ~8TB). So this module uses
   ``scipy.spatial.cKDTree`` (already an ``install_requires`` dependency of
   this package, not a new one -- see ``setup.py``) for k-NN queries: O(n)
   memory instead of O(n^2), and ``workers=-1`` for parallel queries.

3. **Ties and degenerate inputs**, see the per-function docstrings for the
   exact contract: a single shared label short-circuits to an empty band
   (no boundary is well-defined with one class); ``k`` must be a positive
   integer strictly smaller than the population; duplicate/identical states
   are handled correctly via exclusion by ARRAY INDEX rather than by
   position in the neighbour list (so a true self-match is never confused
   with a duplicate that merely has distance zero); a population of exactly
   2 works for the only k that is legal there (k=1).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

# Reused directly, not reimplemented: this exact subsystem already shipped
# a "confident wrong number" once (adaptive_roa/benchmark/fidelity.py,
# commit 64d3ef6) by treating an unbounded raw output as a probability.
# Works on plain numpy arrays via duck typing (`.min()`/`.max()`) despite
# its torch.Tensor type hint.
from adaptive_roa.predictors.hmc.diagnostics import _require_probabilities

__all__ = ["separatrix_band", "conditioned_metrics"]

# Best-effort guard against un-normalized states when no `system` is given
# to normalize internally (see module docstring, point 1). Chosen loosely:
# legitimately normalized states across this project's systems keep an
# unclamped angle in [-pi, pi] (span ~6.28) next to velocity/position
# dimensions normalized to [-1, 1] (span 2) -- a ~3x ratio -- so the
# threshold sits an order of magnitude above that to avoid false positives,
# while still catching the concrete failure mode this module exists to
# prevent (e.g. a raw angular velocity spanning tens of rad/s next to a
# [-1, 1] column, which produces ratios in the hundreds).
_MAX_DIMENSION_SPAN_RATIO = 50.0


def _validate_states(states, n_expected: int) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    if states.ndim != 2:
        raise ValueError(
            f"states must be 2D (n_points, n_dims); got shape {states.shape}. "
            "A single scalar feature must be reshaped explicitly, e.g. "
            "states.reshape(-1, 1) -- this module does not guess."
        )
    if states.shape[0] != n_expected:
        raise ValueError(
            f"states has {states.shape[0]} rows but labels has {n_expected}"
        )
    if not np.all(np.isfinite(states)):
        raise ValueError("states contains NaN/inf; cannot compute neighbours")
    return states


def _guard_unnormalized_scale(states: np.ndarray) -> None:
    """Best-effort rejection of states nobody normalized.

    Cannot prove the input is normalized without a system to check against
    (see the module docstring); this instead rejects the concrete failure
    mode of mismatched physical units by comparing per-dimension spread.
    Skipped for 1D states, where there is nothing to compare across.
    """
    if states.shape[1] < 2:
        return
    spans = states.max(axis=0) - states.min(axis=0)
    nonzero = spans[spans > 1e-12]
    if nonzero.size < 2:
        return
    ratio = float(nonzero.max() / nonzero.min())
    if ratio > _MAX_DIMENSION_SPAN_RATIO:
        raise ValueError(
            f"states span ratio {ratio:.1f}x across dimensions (max span "
            f"{nonzero.max():.4g}, min span {nonzero.min():.4g}) -- this "
            "looks unnormalized. Distances in mismatched physical units "
            "(e.g. rad/s next to metres) are dominated by the largest-scale "
            "dimension, which makes the k-NN boundary meaningless even "
            "though it will still run and return an answer. Pass "
            "normalized states, or pass `system=` (exposing normalize_state) "
            "so this function normalizes internally."
        )


def _normalize_with_system(states: np.ndarray, system) -> np.ndarray:
    """Normalize via the system's OWN normalize_state, enforced here.

    Always applied when `system` is given -- not conditional on whether the
    input "looks" already normalized -- so a caller with a system object
    never has to get this right on their own.
    """
    if not hasattr(system, "normalize_state"):
        raise ValueError(
            f"system must expose normalize_state(state) -> state (see "
            f"adaptive_roa.systems.base.DynamicalSystem); got "
            f"{type(system).__name__} without it"
        )
    tensor = torch.as_tensor(states, dtype=torch.float32)
    normalized = system.normalize_state(tensor)
    if hasattr(normalized, "detach"):
        normalized = normalized.detach().cpu().numpy()
    normalized = np.asarray(normalized, dtype=float)
    if normalized.shape != states.shape:
        raise ValueError(
            f"system.normalize_state changed shape {states.shape} -> "
            f"{normalized.shape}; expected an elementwise normalization"
        )
    return normalized


def separatrix_band(states, labels, k: int = 5, system=None) -> np.ndarray:
    """Flag points whose k nearest neighbours do not all share their label.

    Args:
        states: [n, d] array. Must already be normalized coordinates unless
            `system` is given (see module docstring, point 1) -- distances
            across dimensions of different physical scale are meaningless.
        labels: [n] array of (any) class labels; only equality is used, so
            any label convention works here (e.g. this codebase's usual
            {-1, 1} or {0, 1}).
        k: number of nearest OTHER points examined per point. Must satisfy
            1 <= k < n -- k=0 has no defined neighbourhood, and k >= n asks
            for more neighbours than exist.
        system: optional object exposing `normalize_state(state) -> state`
            (torch.Tensor in, torch.Tensor/array-like out). When given,
            `states` is normalized internally before distances are
            computed, regardless of whether it already looked normalized.

    Returns:
        [n] boolean array; True where the point is "near boundary".

    Degenerate inputs (see module docstring, point 3):
        - A single shared label across all of `labels`: returns an
          all-False band without computing any distances -- no boundary is
          defined with one class, and dimension scale does not matter
          for a result that is fixed regardless of it.
        - Duplicate/identical states: handled correctly. A point's OWN
          entry is excluded from its own neighbour list by matching its
          array INDEX (not by "the first neighbour returned"), so a
          genuine duplicate (a different point at distance exactly 0) is
          never mistaken for self, and vice versa.
        - Exactly 2 points: only k=1 is legal (k < n); works like any other
          population via the same code path.
    """
    labels = np.asarray(labels)
    n = len(labels)
    states = _validate_states(states, n)

    if k < 1:
        raise ValueError(f"k must be a positive integer; got k={k}")
    if k >= n:
        raise ValueError(f"k={k} must be smaller than the population n={n}")

    if len(np.unique(labels)) < 2:
        return np.zeros(n, dtype=bool)

    if system is not None:
        states = _normalize_with_system(states, system)
    else:
        _guard_unnormalized_scale(states)

    tree = cKDTree(states)
    _, idx = tree.query(states, k=k + 1, workers=-1)

    # Exclude each point's own entry by INDEX, not by neighbour-list
    # position: with duplicate states, several other points sit at distance
    # 0 too, so "self" is not reliably the first (or any fixed) position in
    # the tie-broken order scipy returns. A stable sort on "is this column
    # equal to my own row index" pushes the (at most one) true self-match
    # to the end without disturbing the ascending-distance order of every
    # other column, then the first k columns are the k nearest OTHER points.
    #
    # In the rare case where more than k+1 points are exact duplicates of a
    # given point, self may not appear among the k+1 nearest returned by
    # the tree at all (ties beyond k+1 are broken arbitrarily). This is not
    # an error: nothing is excluded for that row, so it simply uses the
    # first k of the k+1 already-returned duplicates as its neighbours.
    row_ids = np.arange(n)[:, None]
    is_self = idx == row_ids
    order = np.argsort(is_self, axis=1, kind="stable")
    neighbor_idx = np.take_along_axis(idx, order, axis=1)[:, :k]

    neighbor_labels = labels[neighbor_idx]
    return np.any(neighbor_labels != labels[:, None], axis=1)


def _labels_as_success_bool(labels: np.ndarray) -> np.ndarray:
    """Interpret `labels` as a success(1)/failure(0) boolean array.

    Refuses any value outside {0, 1} rather than coercing it. Ground-truth
    labels elsewhere in this codebase use the {-1, 1} (failure, success)
    convention (e.g. full_roa_per_point.npz's `true_labels`,
    adaptive_roa/adaptive/data_source.py's `load_eval_states` docstring).
    `labels.astype(bool)` maps -1 to True exactly like it maps 1 to True,
    silently merging both classes into "success" for the accuracy
    computation below -- a band and a set of counts that both look
    completely plausible while being computed over a corrupted definition
    of "correct". That is exactly the "looks plausible, means nothing"
    failure this module exists to prevent, so an unexpected label value is
    refused here, not coerced.
    """
    unique = np.unique(labels)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            f"labels must use the 0=failure/1=success convention; got "
            f"values {unique.tolist()}. If these are the -1=failure/1=success "
            "labels used elsewhere in this codebase, remap them first: "
            "(labels == 1)."
        )
    return labels.astype(bool)


def conditioned_metrics(states, labels, probs, k: int = 5, system=None) -> dict:
    """Accuracy overall, and split by proximity to the empirical boundary.

    Args:
        states, labels, k, system: see `separatrix_band`.
        probs: [n] array of P(success) in [0, 1] (validated via
            `_require_probabilities`, the same guard `fidelity.py` uses --
            see that module's history for why an unbounded score is refused
            rather than silently thresholded).

    Returns:
        dict with:
            overall: accuracy over the full population.
            near_boundary: accuracy restricted to `separatrix_band(...)`,
                or NaN if that set is empty (all one label, or k >= n was
                never reached because of it -- see `separatrix_band`).
            interior: accuracy restricted to the complement, or NaN if
                empty (e.g. every point is near the boundary).
            n_near, n_interior: sizes of the two groups; always sum to
                `len(labels)` by construction (`band` and `~band` are
                complements of the same boolean array).
    """
    labels = np.asarray(labels)
    n = len(labels)
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (n,):
        raise ValueError(
            f"probs must be a 1D array of length {n} (len(labels)); got "
            f"shape {probs.shape}"
        )
    _require_probabilities(probs)

    preds = probs >= 0.5
    correct = preds == _labels_as_success_bool(labels)
    band = separatrix_band(states, labels, k=k, system=system)

    def _acc(mask: np.ndarray) -> float:
        return float(correct[mask].mean()) if mask.any() else float("nan")

    return {
        "overall": float(correct.mean()),
        "near_boundary": _acc(band),
        "interior": _acc(~band),
        "n_near": int(band.sum()),
        "n_interior": int((~band).sum()),
    }
