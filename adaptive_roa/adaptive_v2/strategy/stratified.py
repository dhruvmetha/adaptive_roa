"""Acquisition that scores freely but cannot change the training marginal.

Why this exists (docs/experiments/ensemble_epistemic/CARTPOLE.md sections 6-12
and the diagnosis in section 13): on stochastic cartpole every uncertainty score
-- BALD, debiased variance, total entropy -- made the model measurably WORSE than
uniform sampling, sign-stable across 11 shared epochs and replicated across two
selection rules and two dataset generations.

The mechanism is not that the scores pick badly among comparable points. It is
that the flow matcher under-predicts success in the minority basin (base rate
0.236), so its uncertain band sits on states whose TRUE p_success averages
0.79-0.85. "Model-uncertain" is therefore a synonym for "true success region",
and every score buys that region: measured mean true p of acquired points is
0.62-0.82, with only 1-9% genuinely ambiguous. Basin training mass triples while
the failure bulk -- ~70% of the eval grid, and where most of the ranking signal
lives -- is starved 3-4x. Resolution collapses even though calibration can
improve.

The fix separates the two things a scored batch does. Selecting WHERE to sample
is what broke; selecting WHICH points within a region is the part worth keeping.
Stratifying candidates by the ensemble mean ``p_bar`` and giving each stratum a
budget proportional to its share of candidates makes the acquired batch inherit
the candidate marginal by construction, whatever the score does. A score can no
longer flood a region, because its budget there is fixed before it is consulted.

This is a NEW module rather than a ``selection_rule`` inside decomposition.py on
purpose: six runs were in flight when it was written, and `mp.spawn` children
re-import strategy modules from disk every adaptive epoch, so editing the module
they use would have changed their behaviour mid-run.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.adaptive_v2.strategy.uncertainty_scores import (
    SCORE_MODES, aleatoric_uncertainty, epistemic_bald, score_by_mode,
)
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState


def stratified_top_score(
    score: np.ndarray,
    p_bar: np.ndarray,
    target_count: int,
    n_strata: int = 10,
) -> np.ndarray:
    """Top-scoring candidates within each ``p_bar`` stratum, budgets ∝ stratum size.

    Returns POSITIONS into ``score``/``p_bar``.

    NaN scores sort last rather than raising: ``score_by_mode`` can emit NaN and
    the unstratified path already tolerates it. Budget shortfalls in thin strata
    are redistributed rather than dropped -- returning fewer than ``target_count``
    would quietly change the per-epoch acquisition budget and break matched-budget
    comparison against the control.
    """
    score = np.asarray(score, dtype=np.float64)
    p_bar = np.asarray(p_bar, dtype=np.float64)
    n = score.size
    if n == 0 or target_count <= 0:
        return np.empty(0, dtype=np.int64)
    if target_count >= n:
        return np.arange(n, dtype=np.int64)

    # NaN last under descending sort.
    key = np.where(np.isfinite(score), score, -np.inf)

    edges = np.linspace(0.0, 1.0, int(n_strata) + 1)
    # clip so p_bar == 1.0 lands in the last stratum, not out of range
    stratum = np.clip(np.digitize(p_bar, edges[1:-1], right=False), 0, int(n_strata) - 1)

    # Largest-remainder apportionment: proportional budgets that sum exactly.
    counts = np.bincount(stratum, minlength=int(n_strata)).astype(np.float64)
    exact = counts / counts.sum() * target_count
    budget = np.floor(exact).astype(np.int64)
    for s in np.argsort(-(exact - budget)):
        if budget.sum() >= target_count:
            break
        budget[s] += 1

    order_within = {s: np.flatnonzero(stratum == s)[np.argsort(-key[stratum == s], kind="stable")]
                    for s in range(int(n_strata))}

    picked: list[int] = []
    # Take each stratum's budget, tracking how much a thin stratum could not fill.
    leftover = 0
    for s in range(int(n_strata)):
        avail = order_within[s]
        take = min(int(budget[s]), avail.size)
        picked.extend(avail[:take].tolist())
        leftover += int(budget[s]) - take

    if leftover:
        # Redistribute by global score among everything not already taken, so the
        # batch still reaches target_count without a stratum being over-drawn
        # before its neighbours have had their share.
        taken = np.zeros(n, dtype=bool)
        taken[picked] = True
        rest = np.flatnonzero(~taken)
        rest = rest[np.argsort(-key[rest], kind="stable")]
        picked.extend(rest[:leftover].tolist())

    return np.asarray(picked[:target_count], dtype=np.int64)


class StratifiedDecompositionAcquisitionStrategy:
    """DecompositionAcquisitionStrategy with the marginal pinned to the pool's.

    Identical scoring; only the selection step differs. ``n_strata=1`` reduces
    exactly to plain greedy, which makes the comparison against the greedy arms
    a single-knob change.
    """

    mode = "decomposition"

    def __init__(self, cfg: Any):
        self.score_mode = str(cfg.score)
        if self.score_mode not in SCORE_MODES:
            raise ValueError(
                f"unknown score mode {self.score_mode!r}; expected one of {SCORE_MODES}")
        self.d2_ratio = float(cfg.d2_ratio)
        self.n_candidates = int(cfg.n_candidates)
        self.n_strata = int(cfg.n_strata)
        if self.n_strata < 1:
            raise ValueError(f"n_strata must be >= 1, got {self.n_strata}")
        self.verbose = bool(cfg.verbose)

    def select(
        self,
        pool: Any,
        probability_backend: Any,
        threshold_backend: Any,
        threshold_state: ThresholdState,
        target_count: int,
        exclude: set[int] | None = None,
    ) -> AcquisitionResult:
        # threshold_state is deliberately unread, as in DecompositionAcquisitionStrategy.
        if target_count <= 0:
            return self._skip("target_count_zero")
        if not hasattr(probability_backend, "estimate_members"):
            raise RuntimeError(
                "StratifiedDecompositionAcquisitionStrategy needs a backend exposing "
                f"estimate_members(); got {type(probability_backend).__name__}.")

        states, indices = pool.sample_candidates_without_marking(
            self.n_candidates, exclude=exclude)
        n_actual = len(indices)
        if n_actual == 0:
            return self._skip("pool_exhausted")

        p_members = probability_backend.estimate_members(states, verbose=self.verbose)
        k = getattr(probability_backend, "member_sample_size", None)
        score = np.asarray(score_by_mode(self.score_mode, p_members, k), dtype=np.float64)
        p_bar = np.asarray(p_members, dtype=np.float64).mean(axis=0)

        positions = stratified_top_score(score, p_bar, target_count, self.n_strata)
        selected = [indices[int(p)] for p in positions]

        diagnostics = self._diagnostics(p_members, score, p_bar, positions, n_actual, k)
        if self.verbose:
            print(f"    [Stratified/{self.score_mode}] {n_actual} candidates, "
                  f"{self.n_strata} strata, selected {len(selected)}; "
                  f"p_bar mean cand={diagnostics['p_bar_mean_candidates']:.4f} "
                  f"sel={diagnostics['p_bar_mean_selected']:.4f}")

        return AcquisitionResult(
            d1_indices=[], d2_indices=selected,
            n_candidates_evaluated=n_actual,
            n_certain_discarded=n_actual - len(selected),
            n_invalid_added=0, diagnostics=diagnostics,
        )

    def _diagnostics(self, p_members, score, p_bar, positions, n_actual, k) -> dict[str, Any]:
        finite = score[np.isfinite(score)]
        epi = epistemic_bald(p_members)
        ale = aleatoric_uncertainty(p_members)
        edges = np.linspace(0.0, 1.0, self.n_strata + 1)
        cand_share = (np.histogram(p_bar, bins=edges)[0] / max(len(p_bar), 1)).tolist()
        sel_share = (np.histogram(p_bar[positions], bins=edges)[0]
                     / max(len(positions), 1)).tolist() if len(positions) else []
        return {
            "score_mode": self.score_mode,
            "n_members": int(np.asarray(p_members).shape[0]),
            "member_sample_size": k,
            "score_mean": float(finite.mean()) if len(finite) else None,
            "score_mean_selected": float(np.nanmean(score[positions])) if len(positions) else None,
            "epistemic_mean": float(np.nanmean(epi)),
            "aleatoric_mean": float(np.nanmean(ale)),
            "epistemic_mean_selected": float(np.nanmean(epi[positions])) if len(positions) else None,
            "aleatoric_mean_selected": float(np.nanmean(ale[positions])) if len(positions) else None,
            # The quantities the fix is judged on: if these two agree, the batch
            # inherited the pool marginal and the diagnosed failure cannot recur.
            "p_bar_mean_candidates": float(np.nanmean(p_bar)),
            "p_bar_mean_selected": float(np.nanmean(p_bar[positions])) if len(positions) else None,
            "p_bar_share_candidates": cand_share,
            "p_bar_share_selected": sel_share,
            "n_candidates_evaluated": int(n_actual),
            "selection_rule": f"stratified_{self.n_strata}",
        }

    @staticmethod
    def _skip(reason: str) -> AcquisitionResult:
        return AcquisitionResult(
            d1_indices=[], d2_indices=[], n_candidates_evaluated=0,
            n_certain_discarded=0, n_invalid_added=0,
            diagnostics={"skipped_reason": reason},
        )
