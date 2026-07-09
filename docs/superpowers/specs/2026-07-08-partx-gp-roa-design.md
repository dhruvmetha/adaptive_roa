# Part-X-style GP + Level-Set BO + Partitioning for RoA Characterization

**Date:** 2026-07-08
**Status:** Design (awaiting review)
**Scope:** Pendulum (2D) and CartPole (4D), pool-restricted acquisition.

---

## 1. Goal

Add a new **`partx` acquisition mode** to the adaptive-v2 RoA pipeline that adapts the core
ideas of [Part-X](https://github.com/cpslab-asu/part-x) (arXiv:2110.10729) to region-of-attraction
characterization. It replaces the NN/flow-matching surrogate with a **Gaussian-Process classifier**
and replaces global pool-ranking with **branch-and-bound spatial partitioning + level-set (straddle)
Bayesian optimization**. It produces **two guarantees**:

1. **Bayesian region/volume bounds** — Part-X's native certificate: an estimate (with a credible
   interval) of the RoA volume fraction, plus per-region `+ / − / remaining` classification.
2. **Conformal coverage** — the pipeline's existing frequentist per-point guarantee, reused verbatim.

The build is intentionally faithful to Part-X's *structure* (GP posterior mean+variance, region tree,
Monte-Carlo quantile classification, budget-driven refinement) while dropping onto the existing
binary-label + conformal machinery instead of importing GP regression on a hand-defined STL robustness.

---

## 2. Background: why this is an *adaptation*, not a port

Part-X and RoA characterization share machinery but optimize different losses. The distinction drives
two concrete deviations from stock Part-X.

| | Part-X (falsification) | This work (RoA characterization) |
|---|---|---|
| Estimand | Volume/measure `μ(F) = P(ρ<0)` (a scalar) + localization | The RoA indicator/boundary everywhere (a function) |
| Acquisition | **Minimization** BO (find argmin ρ) | **Level-set / straddle** BO (find the ρ=0 boundary) |
| Effort | Failure-biased; abandon confidently-positive regions coarse | Boundary-uniform; class-symmetric |
| Guarantee | Bayesian credible interval on the scalar volume | + Frequentist conformal coverage per point |
| Surrogate target | Real-valued STL robustness `ρ(x)` from simulator | **Binary success/failure** label from the pool |

**Robustness via GP classification.** We do *not* hand-define a continuous `ρ`. We use a **GP
classifier** (Bernoulli likelihood) on binary success/failure labels. Its **latent function `f(x)` is
the robustness**: `p_success(x) = link(f(x))`, and the RoA boundary is `{f=0} = {p_success=0.5}`. The GP
posterior gives latent **mean `m(x)` and variance `s²(x)`**, which are exactly what the region
classification, straddle acquisition, and Bayesian volume bounds consume — and `p_success = link(m)` is
what the existing `Calibrator` consumes for conformal coverage.

**Deviation 1 — level-set, not minimization.** Because we map the whole separatrix, the BO acquisition
is a straddle rule that samples where the posterior mean is near zero *and* variance is high, rather
than driving `f` to its minimum. (A minimization variant is retained behind a config flag for parity
experiments.)

**Deviation 2 — pool-restricted "BO".** There is no live simulator. BO becomes GP-guided *selection*:
the straddle acquisition is scored over the already-labeled-able candidates in the trajectory pool, and
we take the argmax within each unresolved region. No arbitrary continuous query points.

---

## 3. Locked decisions

- **Scope:** pendulum + cartpole first; higher-dim (quad) explicitly out of scope for now.
- **Surrogate:** GP classifier, latent `f` = robustness, `p_success = link(f)`. No hand-defined `ρ`.
- **Query path:** pool-restricted acquisition (score/select from `TrajectoryPool`; no simulator).
- **GP library:** GPyTorch (variational sparse GP + `BernoulliLikelihood`), to be added to `env/`.
  Exposes latent posterior mean+variance directly. `botorch` optional (not required for straddle).
- **Integration:** Option B — implement the existing v2 Protocols and add `acquisition_mode="partx"`;
  reuse `AdaptiveEngine`, D1/q_hat calibration, config, warm-start, and `evaluate_full_roa_fast`.
- **GP training target:** success (`+1`) → 1, failure (`−1`) → 0; separatrix (`0`) and invalid points
  are excluded from the GP fit (still marked used in the pool). Conformal still uses the full
  ternary NC machinery with `p_success = link(m)`, `p_failure = 1 − p_success`, `p_invalid = 0`.

---

## 4. Architecture

### 4.1 How the engine already wires components (verified)

`AdaptiveEngine.__init__` builds every component from a Hydra `_target_`:

```
trainer_cls           = get_class(cfg.predictor.trainer_target); trainer_cls(cfg, system, system_name)
probability_backend   = _instantiate(cfg.probability, system, device)
threshold_backend     = _instantiate(cfg.threshold,   system, device)
calibration_backend   = _instantiate(cfg.calibration, system, device)
acquisition           = _instantiate(cfg.acquisition)
evaluator             = _instantiate(cfg.eval,        system, device)
```

Per-epoch loop (relevant contract):
- `trainer.fit(dataset_files=..., output_dir=..., resume_checkpoint=...) -> model_handle`
  (`model_handle` must support `.eval()` and `.to(device)`).
- `probability_backend.bind_model(model_handle)` then `.estimate(states) -> OutcomeProbabilities`.
- `threshold_backend.optimize(X_val, y_val) -> ThresholdState`; exposes `.predictor`, `.optimize_mode`.
- `calibration_backend.calibrate(states, labels, threshold_state) -> q_hat` (+ `.calibrate_eval`,
  `.decision_rule`, `.verbose`); **runs only when `acquisition_mode == "conformal"`** (line 182).
- `acquisition.select(pool, probability_backend, threshold_backend, threshold_state, target_count,
  exclude) -> AcquisitionResult`; exposes `.d2_ratio`, `.mode`.
- `evaluator.evaluate_epoch(model_handle, threshold_state, ctx) -> dict`; exposes `.max_eval_rows`.

`dataset_kind = "classification" if predictor.type == "classifier" else "endpoint"`. Endpoint-error and
confidence-pair-filter are already skipped for `predictor.type == "classifier"`.

### 4.2 New package `adaptive_roa/partx/`

```
adaptive_roa/partx/
  __init__.py
  gp_classifier.py    # GPyTorch variational Bernoulli GP; latent mean/var + p_success; model_handle
  trainer.py          # GPPredictorTrainer  (PredictorTrainer): fit GP from dataset_files
  backend.py          # GPProbabilityBackend (ProbabilityBackend): estimate() + latent_posterior()
  region.py           # Region: box (raw coords, circular-aware), membership, subdivision
  tree.py             # PartitionTree: build/refine, classify leaves, allocate budget
  classify.py         # region classification rule (quantile of LCB/UCB over MC points)
  bounds.py           # Bayesian RoA-volume estimate + credible interval (R×M posterior sampling)
  acquisition.py      # straddle score + pool-restricted per-region selection
  strategy.py         # PartXAcquisitionStrategy (AcquisitionStrategy): holds persistent tree
  eval.py             # PartXEvaluator: wraps base RoA eval, adds region bounds + viz
  viz.py              # 2D region-tree / boundary / p_success plots
```

Reused as-is: `systems/*` (embed_state, bounds, classify_attractor, circular indices),
`adaptive_v2/pool/*`, `conformal/calibrator.py` (`Calibrator`), the base evaluator
(`evaluate_full_roa_fast`), and existing `threshold`/`calibration` backends (they operate on
probabilities and are surrogate-agnostic — reuse unless optimization proves otherwise).

### 4.3 Engine change (single, guarded)

Extend the q_hat-calibration branch so conformal coverage also runs in partx mode:
`if acquisition_mode == "conformal"` → `if acquisition_mode in ("conformal", "partx")`
(lines 182 and 191). No other engine edits anticipated. If reuse of the existing threshold/calibration
backends surfaces a genuine incompatibility, prefer a new `_target_` class over further engine edits.

---

## 5. Component specifications

### 5.1 `GPClassifier` (`gp_classifier.py`)
- GPyTorch `ApproximateGP` (variational) + `BernoulliLikelihood`; ARD Matérn-5/2 (or RBF) kernel over
  `system.embed_state(x)` features (angles → sin/cos; real dims standardized with stored mean/std).
- Inducing points: k-means (or random subset) of training features, `n_inducing` (default 128; ≤ N).
- Trained by variational ELBO (Adam, `n_iters`, `lr`). Deterministic under the engine seed.
- API:
  - `fit(X_raw, y01)` — `y01 ∈ {0,1}`.
  - `latent_posterior(X_raw) -> (m, s2)` — latent mean & variance (numpy).
  - `p_success(X_raw) -> np.ndarray` — `link(m / sqrt(1 + s2))` (probit-style calibrated squashing).
  - Wrapped in a `model_handle` exposing `.eval()`, `.to(device)` for engine compatibility.

### 5.2 `GPPredictorTrainer` (`trainer.py`) — `PredictorTrainer`
- `__init__(cfg, system, system_name)`.
- `fit(dataset_files, output_dir, resume_checkpoint=None) -> model_handle`: load
  `dataset_files["train"]` (states + labels), map to `{0,1}` (drop separatrix/invalid), fit
  `GPClassifier`, persist state dict + scalers to `output_dir`, return `model_handle`.
- `resume_checkpoint` accepted but GP refit-from-scratch each epoch is the default (cheap at this scale);
  optional warm start of kernel hyperparameters from the previous epoch.

### 5.3 `GPProbabilityBackend` (`backend.py`) — `ProbabilityBackend`
- `bind_model(model_handle)`.
- `estimate(states) -> OutcomeProbabilities(p_success=link(m/sqrt(1+s2)), p_failure=1-p_success,
  p_invalid=0)` — the integrated predictive probability, consistent with §5.1.
- `latent_posterior(states) -> (m, s2)` — extra method the strategy uses for straddle + classification.

### 5.4 `Region` (`region.py`)
- Axis-aligned box in **raw** state coords: `low[d], high[d]`, plus `region_class` in
  `{"+","-","r","min"}`, `id`, `parent_id`, `depth`.
- `contains(X) -> mask`. For circular dims (θ ∈ [−π,π]), membership uses the raw coordinate; the initial
  support spans exactly one period so no wraparound is needed for pendulum (see §11 for cartpole).
- `subdivide(branching_factor, dim=None)` → children split along the longest *normalized* dimension
  (normalized by `system` bounds) at the median of contained labeled points (fallback: midpoint).
- `volume()` — normalized volume (product of normalized side lengths); the measure used for bounds/budget.

### 5.5 `PartitionTree` (`tree.py`)
- Root = full state-space support box (from `system.state_bounds`, θ ∈ [−π,π]).
- `refine(gp_posterior, labeled_points, cfg)`:
  1. For each leaf, draw `M_class` MC points (LHS in the box) and compute `(m, s2)`.
  2. Classify the leaf via `classify.py`.
  3. Subdivide any `"r"` leaf whose normalized volume ≥ `min_volume` (= `delta`-derived) into
     `branching_factor` children; leaves below `min_volume` become `"min"` (terminal remaining).
- Holds persistent state across epochs (lives on the strategy).
- `remaining_leaves()`, `all_leaves()`, `classified_volume()` helpers.

### 5.6 Region classification (`classify.py`)
Given MC-point posteriors `(m_i, s2_i)` in a region and `c = z_alpha` (from `alpha`):
- `LCB_i = m_i − c·sqrt(s2_i)`, `UCB_i = m_i + c·sqrt(s2_i)`.
- Region is `"+"` (in-RoA) if `quantile_alpha(LCB) > 0` (even pessimistically, `f>0` across the region).
- Region is `"-"` (out-RoA) if `quantile_{1-alpha}(UCB) < 0`.
- Else `"r"` (remaining → subdivide). `alpha` is Part-X's region-classification percentile.

### 5.7 Bayesian RoA-volume bound (`bounds.py`)
- Point-estimate: `V_hat = Σ_leaf mean_i[ link(m_i/sqrt(1+s2_i)) ] · volume(leaf)` over MC points
  (RoA volume fraction; falsification fraction = `1 − V_hat`).
- Credible interval (Part-X R×M): draw `R` posterior samples of `f` at the MC points
  (`f ~ N(m, s2)`), threshold at 0 to get an RoA indicator, integrate per draw → `R` volume samples;
  report `[quantile_{α/2}, quantile_{1-α/2}]`. `R`, `M` configurable.
- Returns per-region contributions + totals + CI; consumed by `eval.py`.

### 5.8 Level-set acquisition (`acquisition.py`)
- Straddle score: `a(x) = c·sqrt(s2(x)) − |m(x)|` (Bryan et al. 2005). Config `c` (default 1.96).
  Minimization variant `a_min(x) = −LCB(x)` behind `acquisition.objective ∈ {"straddle","min"}`.
- Pool-restricted selection given `target_count = n_d2`:
  1. Collect available pool candidates (via `pool.sample_candidates_without_marking`, `exclude=`) and
     their `(m, s2)` from `backend.latent_posterior`.
  2. Assign each candidate to its leaf region; keep those in `"r"`/`"min"` (unresolved) leaves.
  3. Allocate `n_d2` across unresolved leaves proportional to `volume(leaf)` (with a floor of 1);
     within each leaf pick the top-`k` by straddle score. Config toggle
     `acquisition.allocation ∈ {"per_region_volume","global_top"}` (global = top-`n_d2` straddle among
     all unresolved-leaf candidates; simpler baseline).
  4. Optional per-region seeding: a newly created leaf with `< init_per_region` labeled points first
     draws its nearest available pool points (the "init sampling" analog) before straddle scoring.
- Returns selected pool indices → `AcquisitionResult.d2_indices`, with rich `diagnostics`
  (per-region counts, tree size, volume estimate).

### 5.9 `PartXAcquisitionStrategy` (`strategy.py`) — `AcquisitionStrategy`
- `__init__(cfg_acquisition)`: reads `d2_ratio`, `mode="partx"`, tree/gp/acq/bounds sub-configs;
  lazily constructs the persistent `PartitionTree` on first `select`.
- `select(pool, probability_backend, threshold_backend, threshold_state, target_count, exclude)`:
  1. `tree.refine(backend.latent_posterior, labeled_points, cfg)` (reclassify + subdivide).
  2. `acquisition.select(...)` → `d2_indices`.
  3. Compute region bounds (`bounds.py`), stash tree snapshot + bounds in `diagnostics`.
  4. Return `AcquisitionResult(d2_indices=..., diagnostics=...)` (n_invalid_added=0,
     n_certain_discarded from the resolved-region candidates that were skipped).

### 5.10 `PartXEvaluator` (`eval.py`) — `Evaluator`
- Wraps the existing evaluator: calls base `evaluate_full_roa_fast` for the comparable ROA metrics
  (coverage/F1/unknown/etc.), then appends Part-X extras from the strategy diagnostics:
  region-volume estimate + CI, per-class region counts, and (2D) saves `viz.py` plots
  (color-coded region tree, level-set boundary, `p_success` heatmap) to `output_dir`.
- `max_eval_rows` passthrough.

---

## 6. Configuration

New files under `configs/adaptive_v2/`:

```
acquisition/partx.yaml       # mode: partx; d2_ratio; objective; allocation; c; init_per_region
predictor/gp.yaml            # type: classifier; trainer_target: adaptive_roa.partx.trainer.GPPredictorTrainer
probability/gp.yaml          # _target_: adaptive_roa.partx.backend.GPProbabilityBackend
eval/partx.yaml              # _target_: adaptive_roa.partx.eval.PartXEvaluator (+ base eval cfg)
partx/gp.yaml                # kernel, n_inducing, n_iters, lr
partx/tree.yaml              # branching_factor, delta (→ min_volume), alpha, M_class, uniform_partitioning
partx/bounds.yaml            # R, M
```

`threshold` and `calibration` groups reuse existing configs. A top-level experiment config
(`configs/adaptive_v2/experiment/partx_pendulum.yaml`) composes these + system + conformal (alpha, delta,
decision_rule=one_sided for pendulum / two_sided for cartpole).

---

## 7. Per-epoch data flow (partx mode)

```
fit GP on accumulated train set (GPPredictorTrainer)         # surrogate refit
bind GPProbabilityBackend / threshold / calibration
optimize thresholds → ThresholdState (λ*≈0.5, δ*)            # reused
sample D1 (random) + mark used + add to train                # reused
calibrate q_hat on D1 (partx now included)                   # conformal coverage
select D2 via PartXAcquisitionStrategy:                      # tree refine + straddle
  tree.refine → reclassify leaves, subdivide 'r' leaves
  straddle-score available pool candidates in unresolved leaves
  allocate n_d2 across leaves → pool indices
mark D2 used + add to train
eval: base RoA metrics + region bounds/CI + conformal coverage + 2D viz
rebuild datasets; persist tree/pool state
```

---

## 8. Testing plan (TDD)

Unit:
- `GPClassifier` recovers a known boundary (circle / two-moons): boundary points get `p≈0.5`,
  interior `p→{0,1}`; latent variance shrinks with data.
- `Region`: `contains`, `subdivide` (longest-normalized-dim, median split), `volume`; circular box.
- `classify`: synthetic `(m,s2)` fields → correct `+/−/r`; percentile behavior vs `alpha`.
- `acquisition`: straddle picks near-boundary points; per-region volume allocation sums to `n_d2`.
- `bounds`: on an analytic RoA (e.g., a disk), `V_hat` → true volume as data grows; CI covers truth at
  nominal rate over repeated seeds.
- Conformal: empirical coverage on held-out ≥ `1−α` using `Calibrator` with GP `p_success`.

Integration:
- End-to-end pendulum smoke test (tiny budget, few epochs): runs, produces `artifacts_v2.json`,
  region-bounds fields, and viz files; no engine regressions for existing modes.

Conventions: pytest under `tests/partx/`, mirror existing test style, deterministic seeds.

---

## 9. Deliverables & comparison

- `acquisition_mode=partx` runnable via the existing adaptive-v2 entrypoint on pendulum & cartpole.
- Per-epoch: base RoA metrics (comparable to conformal/ranked/direct), RoA-volume estimate + credible
  interval, per-region classification, conformal test coverage.
- 2D visualizations of the region tree, level-set boundary, and `p_success` for qualitative validation.

---

## 10. Non-goals

- No live simulator / continuous-space BO (pool-restricted only).
- No quadrotor / humanoid / >4D systems; no sparse/deep-kernel scaling engineering.
- No GP regression on a hand-defined STL robustness; no S-TaLiRo integration.
- No change to existing conformal/ranked/direct behavior.

---

## 11. Risks & open considerations

- **CartPole circular seam:** the upright success attractor sits at θ=±π, so the RoA straddles the
  wrap seam under θ∈[−π,π]. The GP is seam-safe (sin/cos features), but the *tree box* is not.
  Planned fix in the cartpole phase: recenter cartpole's θ domain to [0,2π] (upright at π, interior),
  or add circular-box wraparound to `Region`. Does not block pendulum.
- **GP scaling:** variational sparse GP is fine at pendulum/cartpole scale (thousands of points,
  ≤128–256 inducing). If accumulated data grows large, cap inducing points / subsample.
- **Threshold/calibration reuse:** assumed surrogate-agnostic; validate `λ*` optimization behaves with
  GP `p_success` (expected `λ*≈0.5`). Fall back to a fixed `λ*=0.5` if optimization is unstable.
- **Budget vs epochs:** Part-X is budget-driven; we map it onto the fixed-epoch loop
  (`samples_per_epoch`, `n_epochs`). Incremental tree refinement across epochs preserves the
  budget-driven spirit.

---

## 12. Milestones

1. **GP core:** `GPClassifier` + trainer + backend + config; unit tests (boundary recovery, posterior).
2. **Tree + classification:** `Region`, `PartitionTree`, `classify`; unit tests.
3. **Acquisition + strategy:** straddle + pool-restricted selection + `PartXAcquisitionStrategy`;
   engine guarded tweak; unit tests.
4. **Bounds + eval + viz:** Bayesian volume CI, `PartXEvaluator`, plots; unit tests.
5. **Integration:** end-to-end pendulum smoke + first real pendulum run; then cartpole (seam fix).
