# Final-State Dispersion Acquisition

**Date:** 2026-07-26
**Status:** Approved design, ready for implementation planning

## Problem

Acquisition currently scores uncertainty from label counts. `ProbabilityEstimator.estimate()`
draws K endpoints per candidate state (fresh latent `z` each pass), then immediately collapses
each endpoint to a discrete label via `system.classify_attractor()`
(`adaptive_roa/conformal/probability_estimator.py:144`). The K endpoints become three scalars —
`p_success`, `p_failure`, `p_invalid` — and every acquisition strategy ranks candidates from
those scalars.

Collapsing the endpoint cloud to a label histogram loses information that matters:

1. **Coarse, tied scores.** With K=10, `p_success` takes 11 distinct values. Large numbers of
   candidates tie, and ranking cannot discriminate among them.
2. **No structure inside "invalid".** Endpoints far from every attractor all map to label 0,
   whether they form one tight cluster (model is confident, just off-attractor) or scatter
   across the state space (model has no idea). These are epistemically opposite.
3. **Dependence on the success criteria.** Uncertainty is a function of `attractor_radius` and
   the hand-enumerated attractor list, so it inherits every arbitrary choice in those
   definitions and transfers poorly to systems whose attractors are not cleanly enumerable.
4. **Multimodality is invisible.** Two well-separated modes that both land in the success basin
   read as `p_success = 1.0`, i.e. maximally certain, even though the model is undecided about
   where the system ends up.

## Goal

Score acquisition uncertainty from the **geometry of the predicted final-state distribution**,
and confine the success criteria (`classify_attractor`, `attractor_radius`, λ\*, δ\*, q_hat) to
threshold optimization and evaluation. Acquisition should not consult them.

## Approach

Add a new acquisition strategy, `dispersion`, that scores each candidate by the **mean pairwise
distance among its K predicted endpoints**, under a range-normalized, circular-aware metric.

The existing `ranked` / `direct` / `conformal` / `partx` strategies are untouched and remain
baselines, so a dispersion run is directly comparable against them on the same pool and
evaluator.

### Why mean pairwise distance

Considered and rejected:

- **kNN differential entropy (Kozachenko–Leonenko).** A literal entropy estimate, but at K=20 it
  is driven by the single closest pair, and its bias grows with dimension — unusable for
  quadrotor3d (D=12) and humanoid (D=20+).
- **Trace of covariance.** Cheapest, but a pure second-moment statistic: it cannot distinguish
  one wide blob from two tight, well-separated modes, and it is sensitive to a single outlier
  endpoint. (`det Σ` is worse — exactly singular whenever K ≤ D.)
- **Flow-matching log-density entropy.** The principled quantity, `-1/K Σ log p(e_k|x)` via the
  ODE's instantaneous change of variables, but it needs a Hutchinson divergence trace at every
  ODE step for every sample — roughly an order of magnitude more compute than the forward passes
  it accompanies, across 50k candidates per epoch.

Mean pairwise distance uses K(K−1)/2 = 190 pairs at K=20, so it is low-variance even at small K;
needs no density estimate; is dimension-agnostic; is continuous (no ties); and rises for both
wide unimodal spread and well-separated multimodality.

## Architecture

### New components

| Component | Location | Responsibility |
|---|---|---|
| `get_normalization_scales()` | `adaptive_roa/systems/base.py` (new method) | Flat `[state_dim]` scale vector: circular dims → π, real dims → `hi−lo` (the full range, so a maximal disagreement normalizes to 1.0 in both dimension types). Uses the same manifold-component loop as `get_loss_weights()`, so all systems inherit it from the base class. |
| `sample_endpoints()` | `adaptive_roa/conformal/probability_estimator.py` (new method) | K-inner-loop over `predict_endpoint`, returns the raw cloud `[N, K, D]`. No classification, no refinement, no `attractor_radius`. |
| `sample_endpoints()` | `adaptive_roa/adaptive_v2/probability/endpoint_mc.py` (new method) | Thin delegate so the strategy talks only to the backend. |
| `dispersion_score.py` | `adaptive_roa/adaptive_v2/strategy/` (new) | Pure functions: the metric and the three selection rules. No pipeline or model dependencies. |
| `DispersionAcquisitionStrategy` | `adaptive_roa/adaptive_v2/strategy/dispersion.py` (new) | Orchestration only: pull candidates, get cloud, score, select, assemble diagnostics. |
| `dispersion.yaml` | `configs/adaptive_v2/acquisition/` (new) | Selected via `acquisition=dispersion`. |

`get_normalization_scales()` is a new method rather than a reuse of `get_loss_weights()`: the
latter's weights are *proportional* to each dimension's range (pendulum: θ→1.0, θ̇→8.0), which is
the wrong sign for a distance metric — it would make velocity spread count 8× per unit.

### Data flow (per epoch)

```
engine.run()
  └─ acquisition.select(pool, probability_backend, threshold_backend, threshold_state, target_count)
       ├─ pool.sample_candidates_without_marking(n_dispersion_candidates)  → X [M,D], indices
       ├─ probability_backend.sample_endpoints(X, K)                       → E [M,K,D]  (raw cloud)
       ├─ dispersion_score(E, scales, circular_idx)                        → s [M]
       ├─ selection_rule(s, X, N)                                          → d2_indices
       └─ (diagnostics) classify_attractor(E) → p_success → Spearman vs s
```

`threshold_state` is accepted for `AcquisitionStrategy` Protocol conformance and **ignored**.
No success-criteria call influences selection.

### Relationship to `estimate()`

`sample_endpoints()` is a second MC path, deliberately **not** a refactor of `estimate()`.
`estimate()` interleaves per-pass refinement of invalid endpoints
(`probability_estimator.py:148`), which is label-driven and meaningless here; folding the two
together would reintroduce the success criteria into the acquisition path.

### What stays unchanged

`threshold_backend.optimize(X_val, y_val)` still runs every epoch to produce λ\*/δ\* for the
evaluator, and the engine still draws random D1 points. Dispersion itself needs neither, but
evaluation does. `d2_ratio` therefore behaves exactly as under `ranked`, defaulting to 0.5.

## Scoring

### Metric

```
Δ_ijd = e_id − e_jd                                    real dimensions
      = atan2(sin(e_id − e_jd), cos(e_id − e_jd))      circular dimensions
s_d   = π                  if dimension d is circular
      = hi_d − lo_d        otherwise (the full range, not half of it)
d_ij  = ‖ Δ_ij / s ‖₂
score(x) = 2 / (K(K−1)) · Σ_{i<j} d_ij
```

Circular wrapping via `atan2(sin, cos)` matches the convention already used by
`is_in_attractor()` and `classify_attractor()` (`adaptive_roa/systems/pendulum.py:130`).

Both dimension types are scaled so that the maximum possible difference normalizes to 1.0:
circular differences wrap to at most π (hence `s_d = π`, not 2π), and real differences reach at
most the full declared range `hi_d − lo_d` (hence dividing by the full range, not half of it).
Range normalization puts every dimension on equal footing, so a full-width spread costs the same
in θ as in θ̇. Without it, the widest-range dimension dominates the score — which would be
severe for quadrotor3d and humanoid. What makes the `proportional` rule's `temperature`
meaningful across systems and epochs is not a fixed bound on the raw score (per-candidate scores
can reach up to `√D` in the worst case, and there is no tighter closed-form bound in general) —
it's that `select_proportional` min-max normalizes the score to `[0, 1]` within each batch before
applying `temperature`, so the same `temperature` value has the same effect regardless of the
raw score's scale.

### Computation

Computed in torch on-device, **chunked over candidates**. The full broadcast is `[M,K,K,D]` —
960 MB at M=50k, K=20, D=12 — so it is processed in blocks of `chunk_size` candidates (2048 by
default, ≈39 MB). There is no algebraic shortcut: unlike variance, mean pairwise distance
genuinely requires all K² pairs. At O(K²D) per candidate the cost is negligible beside the K
forward passes.

### MC budget

`num_mc_samples_dispersion` is independent of the backend's `num_mc_samples` (K=10), defaulting
to **20**. Dispersion is a second-moment statistic and needs more samples than a mean does: K=20
gives 190 pairs versus 45. This doubles acquisition cost — 1.0M forward passes per epoch at
50k candidates, versus 500k for `ranked` — and is tunable per system.

## Selection rules

Set via `selection_rule`. All three operate on the score vector `s [M]` and the candidate initial
states `X [M,D]`.

- **`greedy`** (default) — `argsort(-s)[:N]`. Identical selection mechanics to `sample_ranked()`,
  so any difference against the `ranked` baseline is attributable purely to the score.
- **`greedy_diverse`** — take the top `M' = diversity_pool_multiplier × N` by score (default 5),
  then farthest-point-sample N of them, seeded at the highest-scoring candidate. Distances are in
  **initial-state** space, using the same normalized, wrapped metric. Prevents the batch
  collapsing onto one high-uncertainty pocket.
- **`proportional`** — min-max normalize `s` to [0,1] within the batch, then Gumbel-top-k sampling
  with `temperature`. Gumbel-top-k gives exact sampling-without-replacement in one vectorized
  pass. Reproducible under `seed`.

Because `greedy` is rank-invariant, the metric's absolute scale is irrelevant there; the
within-batch normalization exists so that `temperature` has consistent meaning under
`proportional`.

## Edge cases

- **Pool exhausted (M = 0):** return an empty `AcquisitionResult` with
  `diagnostics={"skipped_reason": ...}`, mirroring `RankedAcquisitionStrategy`.
- **M < N:** select all available and log the shortfall.
- **Non-finite endpoints:** a NaN/inf endpoint poisons the score. Affected candidates are
  **excluded** from selection and counted in `n_nonfinite_excluded`. Treating them as maximally
  uncertain was rejected: divergence is a real pathology, but scoring it +∞ would let numerical
  garbage flood the batch. The count is logged so the decision can be revisited with evidence.

## Diagnostics

Written into the epoch record alongside the existing acquisition fields:

- `dispersion_score_threshold` — lowest score among the selected set (parallel to
  `ranked_score_threshold`)
- `dispersion_score_min` / `_max` / `_mean` / `_median`
- `n_candidates_evaluated`
- `n_nonfinite_excluded`
- `selection_rule`
- `dispersion_label_uncertainty_spearman` — see below

### Score-correlation diagnostic

`log_score_correlation: true` (default) logs
`Spearman(dispersion, u)` over the evaluated candidates, where `u = −|p_success − 0.5|` is the
label-based uncertainty: `u` is maximal (0) at `p_success = 0.5` and minimal (−0.5) when the
label counts are unanimous.

This costs **zero extra forward passes**: the strategy already holds `E [M,K,D]`, so
`system.classify_attractor()` on those same endpoints yields `p_success` directly. No second MC
pass, no `estimate()` call. It also uses K=20 rather than `estimate()`'s K=10, so the correlation
is measured against a better `p_success` than the `ranked` baseline itself consumes.

Selection remains criteria-free; only this logged number touches the success criteria. It is the
fast read on whether dispersion is selecting different points than the NC score. Both quantities
increase with uncertainty, so a correlation near **+1.0** means the two scores are redundant;
well short of that means dispersion sees something the label counts cannot.

## Configuration

`configs/adaptive_v2/acquisition/dispersion.yaml`:

```yaml
# @package _global_
sampling_mode: dispersion          # → output dir ..._sampling_mode_dispersion/
acquisition:
  _target_: adaptive_roa.adaptive_v2.strategy.dispersion.DispersionAcquisitionStrategy
  d2_ratio: 0.5
  n_dispersion_candidates: 50000
  num_mc_samples_dispersion: 20
  selection_rule: greedy           # greedy | greedy_diverse | proportional
  diversity_pool_multiplier: 5     # greedy_diverse only
  temperature: 0.1                 # proportional only
  seed: null                       # proportional only
  chunk_size: 2048
  log_score_correlation: true
  verbose: true
```

Three fields present on the other acquisition configs are deliberately absent:

- `decision_rule` — the engine reads only `.d2_ratio` and `.mode` off the strategy
  (`engine.py:109,111`), and dispersion has no use for a decision rule.
- `batch_size_sampling`, `max_samples_per_epoch` — these are consumed only by `UncertainSampler`
  (`strategy/ranked.py:40`), which dispersion does not use. It samples candidates directly via
  `pool.sample_candidates_without_marking(n_dispersion_candidates)`.

Nothing outside the individual strategies validates these keys, so their absence is safe.
`sampling_mode: dispersion`
follows the pattern used by the other four acquisition configs, so these runs land in a distinct
output directory and do not collide with `ranked` baselines.

## Testing

Following the existing `tests/adaptive_v2/` style.

**`test_dispersion_score.py`** — pure math, no model:

- identical endpoints → score 0
- θ = π−ε versus θ = −π+ε wraps to ≈2ε, not ≈2π
- a full-range spread in θ̇ scores the same as a full-range spread in θ
- a two-mode cloud outscores a single tight blob at equal K
- `greedy` selects the argmax set
- `greedy_diverse` achieves greater min-pairwise separation of the selected set than `greedy`
- `proportional` is reproducible under a fixed seed
- candidates with non-finite endpoints are excluded and counted

**`test_dispersion_strategy.py`** — fake pool and fake backend returning canned clouds:

- correct indices selected for a known score ordering
- all diagnostic keys present in `AcquisitionResult.diagnostics`
- empty-pool path returns `skipped_reason`
- M < N path selects all available

**`test_normalization_scales.py`** — for every registered system: shape equals `state_dim`,
circular entries equal π, all entries finite and positive.

## Validation

Pendulum first: 2D, cheapest, and `ranked` / `direct` baselines already exist. Run
`acquisition=dispersion` against `acquisition=ranked` with the same pool, seed, and epoch count,
and compare full-ROA evaluation metrics. Read `dispersion_label_uncertainty_spearman` early — it
indicates whether the two scores are selecting meaningfully different points before the full
comparison finishes.

## Out of scope

- Changing `ranked`, `direct`, `conformal`, or `partx`.
- Changing threshold optimization, calibration, or evaluation.
- Blending dispersion with the NC score. This was considered and rejected: it reintroduces the
  success-criteria dependence during acquisition and adds a hyperparameter.
- Extending dispersion to the classifier predictor path. This design covers the flow-matching
  endpoint predictor only.
