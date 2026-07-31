# Two-way outcome classification (success vs. everything else)

**Date:** 2026-07-31
**Systems in scope:** cartpole, quad2d, quad3d
**Status:** design approved, implementation pending

## Problem

Cartpole, quad2d and quad3d classify a predicted endpoint three ways:

| label | meaning |
|---|---|
| `1` | inside the goal ball (`dist < attractor_radius`) |
| `-1` | past a divergence threshold (out of bounds / too fast) |
| `0` | neither — "separatrix" / invalid |

Label `0` propagates into `p_invalid`, and any eval point with `p_invalid >= threshold`
is predicted `-2` (INVALID) and **abstained** — dropped from tp/tn/fp/fn, and so from
accuracy, precision, recall and F1.

This inflates the reported metrics. At the final epoch of the `ranked` arm:

| system | reported F1 (λ±δ) | abstain % | F1 at full coverage (`conservative_lambda_delta`) |
|---|---|---|---|
| quad3d | .9686 | 26.8 % | **.7828** |
| quad2d | .9528 | 9.2 % | **.7433** |

On quad3d the abstention splits **22.7 % `p_invalid` vs 4.1 % genuinely uncertain**, so
the invalid class — not the λ±δ band — is what removes coverage. Cartpole is the mirror
image at convergence (0.5 % invalid, 1.2 % uncertain) but abstains ~36 % at epoch 0.

The third class has no basis in the data. Ground truth is already binary everywhere:
`adaptive_roa/adaptive/data_source.py:51` maps external labels with
`{0: -1, 1: 1}`, and the cartpole / quad2d / quad3d dataset descriptions all report
`timeout_trajectories: 0`. "Invalid" exists **only** on the prediction side, when an
FM-predicted endpoint lands in the gap between the goal ball and the divergence bounds.
It is a model artifact being scored as an abstention.

## Goal

Add a two-way mode in which **success is the only criterion and everything else is
failure**, so predictions have full coverage and the reported metrics are honest.

The mode must serve both use cases with one flag:
- end-to-end retraining, where acquisition, threshold optimization and eval are all 2-way;
- re-evaluation of existing frozen 3-way checkpoints under 2-way scoring.

Which of those actually gets run is decided later, at experiment-launch time. This spec
covers code and config only.

## Design

### 1. Collapse at the single source of truth

Every downstream notion of invalid (`p_invalid`, the `-2` eval label, `n_invalid_added`,
`refine_invalids`) derives from `system.classify_attractor()` returning `0`. Collapsing
there covers all of it.

`DynamicalSystem.__init__` (`adaptive_roa/systems/base.py:43`) gains
`binary_outcomes: bool = False`, stored as `self.binary_outcomes`. The three systems
change only their fill value:

```python
# adaptive_roa/systems/cartpole.py:220
# identical shape at quadrotor2d.py:235 and quadrotor3d.py:289
fill = -1 if self.binary_outcomes else 0
labels = torch.full_like(in_attractor, fill, dtype=torch.long)
labels[in_attractor] = 1
labels[exceeded_thresholds] = -1
```

The divergence-threshold check is retained, not deleted: it is now redundant for the
label value but keeps the two branches structurally parallel and keeps three-way mode
working from the same code.

`adaptive_roa/adaptive_v2/engine.py:68` instantiates the system exactly once
(`hydra.utils.instantiate(cfg.system)`) and passes it to the probability, threshold,
calibration and eval backends, so this one flag reaches every consumer. `p_invalid`
becomes identically `0` and `p_success + p_failure == 1` exactly.

Consequence worth recording: `apply_two_sided_rule` reduces to `apply_one_sided_rule`
when `p_success + p_failure == 1`, since its failure test `(1 - p_failure) < λ - δ`
becomes `p_success < λ - δ`. `decision_rule` is therefore left untouched in the configs.

### 2. Explicitly gate the invalid gates on the flag

`p_invalid ≡ 0` does **not** by itself disable the invalid gates. All of them test
`p_invalid >= threshold`, which fires when the threshold is `<= 0`:

- `adaptive_roa/conformal/lambda_optimizer.py:49` and `:84` — `p_invalid >= (λ - δ)`
- `adaptive_roa/adaptive_v2/eval/full_roa.py:281, 314, 340, 402, 492` — five `_predict_*`
  functions, four defaulting to `λ - δ` and `_predict_fixed_threshold` hardcoding `0.5`
- `adaptive_roa/adaptive_v2/filters/confidence_filter.py:128, 136` — passes `p_invalid`
  unconditionally, bypassing `use_p_invalid_veto`

`optimize_lambda_delta` searches `lambda_range=(0.3, 0.7)` × `delta_range=(0.01, 0.3)`,
so `λ - δ = 0.0` is a reachable grid point (λ=0.3, δ=0.3) at which every point would be
flagged invalid.

Each gate is disabled by an **explicit flag**, never by inferring the mode from an
all-zero `p_invalid`. Every default is the current value, so all three-way paths —
including `predictor: classifier`, which already runs with `p_invalid = zeros`
(`adaptive_roa/conformal/classifier_probability_estimator.py:75`) — evaluate exactly the
same expressions they do today and stay bit-identical.

**2a. `lambda_optimizer.py` — no change.** `two_way.yaml` sets
`use_p_invalid_veto: false`; `adaptive_roa/conformal/predictor.py:204` already turns that
into `p_invalid_for_opt = None`, and the existing `p_invalid is None` branch
(`lambda_optimizer.py:50-51`) maps `None` to an all-False mask. The switch that this
design needs is already wired.

**2b. `full_roa.py` — one helper, gated on the flag.**

```python
def _invalid_mask(p_invalid: np.ndarray, threshold: float, n: int,
                  binary_outcomes: bool = False) -> np.ndarray:
    """Points whose predicted-endpoint cloud is mostly invalid.

    Two-way runs have no invalid class, so nothing is ever masked.
    """
    if binary_outcomes:
        return np.zeros(n, dtype=bool)
    return p_invalid >= threshold
```

`evaluate_full_roa_fast(flow_matcher, system, ...)` already receives `system`
(`full_roa.py:624-626`), so the flag is read once as
`binary_outcomes = getattr(system, "binary_outcomes", False)` and threaded into the five
`_predict_*` functions and `optimize_lambda_delta` — no new argument at any external call
site, which also satisfies the project convention that new `evaluate_full_roa_fast`
parameters stay backward-compatible.

**2c. `confidence_filter.py` — new constructor argument.**
`ConfidencePairFilter.__init__` gains `binary_outcomes: bool = False`; when set, it passes
`p_invalid=None` to `apply_one_sided_rule` / `apply_two_sided_rule` instead of
`probs.p_invalid`. `engine.py:279` supplies `binary_outcomes=self.system.binary_outcomes`.
(The filter is already skipped for `predictor_type == "classifier"` at `engine.py:277`, so
the classifier path never reaches it either way.)

**Known degeneracy, left untouched by design.** In three-way mode a backend that reports
`p_invalid = zeros` — the classifier predictor — still marks every point invalid at grid
points where `λ - δ <= 0`. Those candidates score F1 = 0 and never win the optimization,
so the behaviour is inert. Fixing it would change three-way classifier results, which this
design explicitly declines to do; it is recorded here as a separate known issue.

Deliberately **not** changed: the conformal candidate label set
`for label in [-1, 0, 1]` (`adaptive_roa/conformal/calibrator.py:443`) and the
`unknown_mode` variants in `_predict_qhat_multi_class`. The `0` there is the conformal
UNKNOWN label, not `p_invalid`; leaving it means those points read as uncertain, which is
correct. `conformal/refinement.py` becomes a no-op on its own once no label is `0`.

### 3. Config, with run isolation

`docs/plots/plot_clf_vs_fm.py` resolves runs by wildcard glob inside a per-system root —
e.g. `resolve_fm_dispersion` (line 165) globs
`{EXP}/{DISPERSION_ROOT[sys]}/outputs/*d2_ratio_{d2}_*sampling_mode_...`. A 2-way run
sharing that root risks being silently matched by a three-way resolver. 2-way runs
therefore get their own root directory.

- New config group `configs/adaptive_v2/outcome/` with `three_way.yaml` and `two_way.yaml`.
- `outcome: three_way` is appended to the `defaults:` list in
  `configs/adaptive_v2/default.yaml` **after** `system`, so the group's keys win over the
  system group's.
- `three_way.yaml` is a no-op: `outcome_mode: three_way`, `name_suffix: ""`,
  `system.binary_outcomes: false`.
- `two_way.yaml` sets `outcome_mode: two_way`, `name_suffix: _2way`,
  `system.binary_outcomes: true`, `threshold.use_p_invalid_veto: false`.
- `configs/adaptive_v2/system/_base.yaml` declares `name_suffix: ""` as the default.
- The three system configs change `name:` to interpolate it, e.g.
  `name: adaptive_cartpole_pybullet${name_suffix}`.

Three-way runs keep byte-identical output paths (the suffix is empty), so no existing
resolver or recorded path in `docs/plots/EXPERIMENT_DIRS.md` is affected. Two-way runs
land under `adaptive_cartpole_pybullet_2way/`, `adaptive_quadrotor2d_2way/`,
`adaptive_quadrotor3d_2way/` and cannot collide with any existing glob.

`use_p_invalid_veto: false` in `two_way.yaml` is load-bearing, not cosmetic: per §2a it is
the entire mechanism by which the `lambda_optimizer` gates are disabled, which is why that
module needs no code change. It is an existing switch
(`adaptive_roa/conformal/config.py:66`, threaded at `conformal/predictor.py:204`); the
two-way config only flips it.

### 4. MC cache invalidation

`adaptive_roa/adaptive_v2/eval/mc_cache.py` persists both `endpoints` and `labels`, with
metadata keyed on `num_mc_samples, attractor_radius, state_dim, n_states` — not on
outcome mode. Fresh training runs are safe because the new `name` yields a fresh run
directory and therefore a fresh `mc_cache/`. Re-evaluation of an existing checkpoint is
not: a cached three-way `labels` array reused under a two-way flag would silently produce
three-way results.

Fix: record `binary_outcomes` in the cache metadata, and on load, if it disagrees with
the current system, call the existing `MCCache.reclassify(system, radius)` on the stored
endpoints. Endpoints are mode-independent, so this reuses the expensive ODE integration
and redoes only the labelling.

### 5. Re-evaluation entry point

`scripts/reevaluate.py:556` constructs the system with no arguments
(`system = SystemClass()`). Add a `--binary-outcomes` CLI flag that forwards
`binary_outcomes=True`, so the same implementation serves both retraining and
re-evaluation of frozen checkpoints.

## Testing

Unit tests, no training required:

1. For each of cartpole / quad2d / quad3d: with `binary_outcomes=True`,
   `classify_attractor` never returns `0`; points previously labelled `1` and `-1` keep
   their labels; points previously labelled `0` become `-1`.
2. `binary_outcomes=False` reproduces current behaviour exactly (regression guard).
3. In binary mode the probability backend yields `p_invalid == 0` and
   `p_success + p_failure == 1`.
4. `_invalid_mask` returns all-False when `binary_outcomes=True`, even at
   `threshold=0.0`; with `binary_outcomes=False` (the default) it returns exactly
   `p_invalid >= threshold`, including for an identically-zero `p_invalid` at
   `threshold=0.0`, where it must still return all-True.
5. Regression guard for the classifier path: with an all-zero `p_invalid` and
   `binary_outcomes=False`, the five `_predict_*` functions produce byte-identical output
   to the pre-change implementation.
6. `ConfidencePairFilter` with `binary_outcomes=True` passes `p_invalid=None` to the
   decision rule; with the default it passes `probs.p_invalid` unchanged.
5. Loading an `MCCache` whose metadata `binary_outcomes` disagrees with the system
   triggers reclassification and yields labels with no `0`.
6. Hydra composition: `outcome=two_way` sets `system.binary_outcomes: true` and resolves
   `name` to the `_2way` root; the default composition leaves both untouched.

## Interpreting the results

Two-way F1 will read **lower** than the current headline numbers, because those are
computed over only the confident 73–91 % of eval points. The apples-to-apples baseline is
the `conservative_*` block already present in the existing runs' `results.json`
(`_conservative_metrics`, `full_roa.py:62`), which scores them at full coverage: quad3d
.7828, quad2d .7433.

Comparing a retrained two-way run against those isolates whether the gain comes from
better acquisition. Comparing a two-way *re-evaluation* of the same frozen checkpoints
against them isolates the scoring effect alone. Running both separates the two.

## Out of scope

- Pendulum and humanoid. Pendulum also emits label `0` (outside the radius of all three
  attractors) and abstains ~31 % at epoch 0, but is already at F1 .997 with little
  headroom; humanoid's `classify_attractor` never returns `0`, so the flag would be a
  no-op there.
- Choosing the experiment grid (which arms, how many epochs, retrain vs. re-eval). That
  is decided at launch time.
- Regenerating any dataset. Ground truth is already binary.
