"""Acquisition that prices uncertainty per unit of training data purchased.

Why this exists (pendulum i100 sweep, 2026-08-15):

The budget is denominated in TRAJECTORIES (samples_per_epoch=100) but the model
trains on PAIRS: a trajectory of length L contributes L-1 endpoint pairs. On
``noisy/pendulum/lqr/high`` that conversion rate is wildly outcome-dependent --
success rollouts terminate on reaching the goal (mean length 136.5) while
failures run to the 1001-step timeout cap. So the pool is 40.3% success by
trajectory but only 8.4% success by pair.

Every uncertainty score we ran bought the model's uncertain band, which on this
system is the true-success region (measured mean true p of acquired points
0.90-0.99), i.e. the SHORT trajectories. The arms therefore accumulated pairs at
~0.4x the control's rate: at 20 epochs epi_var had 358k pairs against the
control's 652k at epoch 9. Scored at matched PAIR count the arms actually beat
uniform on KL by 1.6-2.6x; scored at matched TRAJECTORY count -- the budget the
campaign is denominated in -- they lose. The selection is not what fails; the
yield per selection is.

This strategy multiplies the uncertainty score by the expected number of pairs a
candidate will yield, so a candidate is ranked by informativeness PER UNIT OF
BUDGET rather than per point:

    score'(x) = score(x) * E[L(x)] ** alpha
    E[L(x)]   = p_bar(x) * L_success + (1 - p_bar(x)) * L_failure

``L_success``/``L_failure`` are estimated online from the lengths of
trajectories ALREADY acquired -- never from the candidate itself, whose length
is unknown until it is rolled out. That keeps the strategy honest: it uses only
the ensemble's own p_bar plus statistics of data already paid for.

Because the length coupling is read from data rather than assumed, this also
transfers to systems where the coupling inverts: on stochastic cartpole the
successes are the LONG rollouts (~600 steps vs pool mean 195), and there the
same expression steers the opposite way without any code change.

alpha is a single tempering knob. alpha=0 reduces EXACTLY to the unweighted
strategy (a useful null); alpha=1 is value-per-trajectory.

A NEW module rather than a selection_rule inside decomposition.py on purpose:
sibling runs are in flight and `mp.spawn` children re-import strategy modules
from disk every adaptive epoch, so editing a module they use would change their
behaviour mid-run.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_roa.adaptive_v2.strategy.uncertainty_scores import (
    SCORE_MODES, aleatoric_uncertainty, epistemic_bald, score_by_mode,
)
from adaptive_roa.adaptive_v2.types import AcquisitionResult, ThresholdState


def expected_length(p_bar: np.ndarray, len_success: float, len_failure: float) -> np.ndarray:
    """Pairs a candidate is expected to yield, given the ensemble's p_bar."""
    p = np.clip(np.asarray(p_bar, dtype=np.float64), 0.0, 1.0)
    return p * float(len_success) + (1.0 - p) * float(len_failure)


def yield_weighted_score(
    score: np.ndarray,
    p_bar: np.ndarray,
    len_success: float,
    len_failure: float,
    alpha: float = 1.0,
) -> np.ndarray:
    """``score * E[L] ** alpha``, with NaN preserved so it still ranks last.

    The score may be negative (``epistemic_var`` is MC-debiased and can go below
    zero for genuinely-agreeing members). Multiplying a negative score by a
    larger weight must make it rank LOWER, not higher, so the weight is applied
    to the magnitude and the sign is restored afterwards.
    """
    score = np.asarray(score, dtype=np.float64)
    w = expected_length(p_bar, len_success, len_failure) ** float(alpha)
    return np.sign(score) * np.abs(score) * w


class YieldAwareDecompositionAcquisitionStrategy:
    """DecompositionAcquisitionStrategy, ranked by informativeness per trajectory.

    Identical scoring and identical greedy top-N selection; the only difference
    is the expected-yield weight. ``alpha=0`` reproduces the unweighted arm
    exactly, which makes this a single-knob comparison.
    """

    mode = "decomposition"

    def __init__(self, cfg: Any):
        self.score_mode = str(cfg.score)
        if self.score_mode not in SCORE_MODES:
            raise ValueError(
                f"unknown score mode {self.score_mode!r}; expected one of {SCORE_MODES}")
        self.d2_ratio = float(cfg.d2_ratio)
        self.n_candidates = int(cfg.n_candidates)
        self.alpha = float(cfg.get("alpha", 1.0))
        # Priors used only until enough of each class has been acquired. Taken
        # from the pendulum-high pool; they are replaced by measured values from
        # epoch 0 onward, so a wrong prior costs at most the first batch.
        self.prior_len_success = float(cfg.get("prior_len_success", 150.0))
        self.prior_len_failure = float(cfg.get("prior_len_failure", 1000.0))
        self.min_per_class = int(cfg.get("min_per_class", 5))
        self.verbose = bool(cfg.verbose)

    # ------------------------------------------------------------ length model
    def _length_stats(self, pool: Any) -> tuple[float, float, int, int]:
        """Mean length of acquired success / failure trajectories.

        Falls back to the configured prior for whichever class has fewer than
        ``min_per_class`` examples, so epoch 0 (100 trajectories, possibly
        lopsided) cannot produce a degenerate weight.
        """
        builder = getattr(pool, "dataset_builder", None)
        source = getattr(builder, "data_source", None)
        split = getattr(builder, "train_split", None)
        if source is None or split is None:
            return self.prior_len_success, self.prior_len_failure, 0, 0

        lens_s: list[int] = []
        lens_f: list[int] = []
        for idx in list(getattr(split, "indices", []) or []):
            try:
                L = int(source.get_trajectory_length(int(idx)))
                y = int(source.get_label(int(idx)))
            except Exception:
                continue
            (lens_s if y == 1 else lens_f).append(L)

        ls = float(np.mean(lens_s)) if len(lens_s) >= self.min_per_class else self.prior_len_success
        lf = float(np.mean(lens_f)) if len(lens_f) >= self.min_per_class else self.prior_len_failure
        return ls, lf, len(lens_s), len(lens_f)

    # ----------------------------------------------------------------- select
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
                "YieldAwareDecompositionAcquisitionStrategy needs a backend exposing "
                f"estimate_members(); got {type(probability_backend).__name__}.")

        states, indices = pool.sample_candidates_without_marking(
            self.n_candidates, exclude=exclude)
        n_actual = len(indices)
        if n_actual == 0:
            return self._skip("pool_exhausted")

        p_members = probability_backend.estimate_members(states, verbose=self.verbose)
        k = getattr(probability_backend, "member_sample_size", None)
        base = np.asarray(score_by_mode(self.score_mode, p_members, k), dtype=np.float64)
        p_bar = np.asarray(p_members, dtype=np.float64).mean(axis=0)

        len_s, len_f, n_s, n_f = self._length_stats(pool)
        score = yield_weighted_score(base, p_bar, len_s, len_f, self.alpha)

        key = np.where(np.isfinite(score), score, -np.inf)
        positions = np.argsort(-key, kind="stable")[:target_count]
        selected = [indices[int(p)] for p in positions]

        diagnostics = self._diagnostics(
            p_members, base, score, p_bar, positions, n_actual, k, len_s, len_f, n_s, n_f)
        if self.verbose:
            print(f"    [YieldAware/{self.score_mode}] {n_actual} candidates, alpha={self.alpha}, "
                  f"L_succ={len_s:.0f} L_fail={len_f:.0f}; selected {len(selected)}; "
                  f"p_bar mean cand={diagnostics['p_bar_mean_candidates']:.4f} "
                  f"sel={diagnostics['p_bar_mean_selected']:.4f}; "
                  f"E[pairs]={diagnostics['expected_pairs_selected']:.0f}")

        return AcquisitionResult(
            d1_indices=[], d2_indices=selected,
            n_candidates_evaluated=n_actual,
            n_certain_discarded=n_actual - len(selected),
            n_invalid_added=0, diagnostics=diagnostics,
        )

    def _diagnostics(self, p_members, base, score, p_bar, positions,
                     n_actual, k, len_s, len_f, n_s, n_f) -> dict[str, Any]:
        finite = score[np.isfinite(score)]
        epi = epistemic_bald(p_members)
        ale = aleatoric_uncertainty(p_members)
        exp_len_sel = expected_length(p_bar[positions], len_s, len_f) if len(positions) else np.array([])
        return {
            "score_mode": self.score_mode,
            "alpha": self.alpha,
            "n_members": int(np.asarray(p_members).shape[0]),
            "member_sample_size": k,
            "score_mean": float(finite.mean()) if len(finite) else None,
            "score_mean_selected": float(np.nanmean(score[positions])) if len(positions) else None,
            # the UNWEIGHTED score of the selected set, so this arm stays
            # comparable with the unweighted arms epoch by epoch
            "base_score_mean_selected": float(np.nanmean(base[positions])) if len(positions) else None,
            "epistemic_mean": float(np.nanmean(epi)),
            "aleatoric_mean": float(np.nanmean(ale)),
            "epistemic_mean_selected": float(np.nanmean(epi[positions])) if len(positions) else None,
            "aleatoric_mean_selected": float(np.nanmean(ale[positions])) if len(positions) else None,
            "p_bar_mean_candidates": float(np.nanmean(p_bar)),
            "p_bar_mean_selected": float(np.nanmean(p_bar[positions])) if len(positions) else None,
            # the quantities this arm is judged on: it is meant to raise pair
            # yield without the batch collapsing onto the failure bulk
            "len_success_est": float(len_s),
            "len_failure_est": float(len_f),
            "n_acquired_success": int(n_s),
            "n_acquired_failure": int(n_f),
            "expected_pairs_selected": float(exp_len_sel.sum()) if len(positions) else 0.0,
            "n_candidates_evaluated": int(n_actual),
            "selection_rule": f"yield_weighted_alpha{self.alpha:g}",
        }

    @staticmethod
    def _skip(reason: str) -> AcquisitionResult:
        return AcquisitionResult(
            d1_indices=[], d2_indices=[], n_candidates_evaluated=0,
            n_certain_discarded=0, n_invalid_added=0,
            diagnostics={"skipped_reason": reason},
        )
