# Ensemble-Based Epistemic Acquisition — Design

**Date:** 2026-08-04
**Status:** Approved

## Goal

Split the label-entropy acquisition score into its aleatoric and epistemic parts using a
deep ensemble, and acquire on the epistemic part alone.

## Motivation

The stochastic-pendulum campaign (`docs/stoch_compare/FINDINGS.md`) established that entropy
acquisition *actively harms* the classifier as noise grows, and measured why:

| level | frac ambiguous (random → scored) | mean true p (random → scored) |
|---|---|---|
| low | 0.027 → 0.296 | 0.388 → 0.361 |
| med | 0.053 → 0.328 | 0.389 → 0.179–0.367 |
| high | 0.133 → **0.010** | 0.406 → **0.046** |
| xhigh | 0.269 → **0.080** | 0.481 → **0.138** |

At low/med the score finds genuinely ambiguous states. At high/xhigh it scores *worse than
random* on ambiguity and instead harvests near-deterministic failure states, dragging the
training marginal from ~41–48% success to ~5–14%. Final cost: +0.157 debiased Brier at high
and +0.088 at xhigh, both ~5× their run-to-run floor, with a monotone dose-response in the
adaptive fraction d2.

The diagnosis is that binary entropy of p(success) cannot distinguish "the outcome is
genuinely random here" (aleatoric, irreducible, worthless to sample) from "the model does not
know" (epistemic, reducible, worth sampling). Under heavy process noise the first term
dominates and the score degenerates. An ensemble can separate them.

Flow matching was never harmed by entropy acquisition and is helped at med and high, so it is
the contrast case rather than the problem case.

## The decomposition

For M members with per-member success probabilities p_1..p_M:

```
H(p̄)              =  E_m[H(p_m)]        +  I(y ; m | x)
total                 aleatoric             epistemic
```

**Classifier.** Each p_m is one forward pass. `EnsemblePosterior.predictive_logit_samples()`
already returns exact per-member logits (`forward_all_members`, M atoms, no sampling), so all
three terms are exact and textbook BALD is unbiased. No correction.

**Flow matching.** Each p_m must be estimated from K endpoint samples, so p̂_m is
binomial-noisy and the naive estimator inflates apparent disagreement. Measured with members
forced to agree exactly (true epistemic = 0, M=5, K=20, 40k trials):

| true p | naive BALD | debiased variance |
|---|---|---|
| 0.50 | +0.0206 | −0.00003 |
| 0.30 | +0.0209 | +0.00004 |
| 0.10 | +0.0234 | −0.00000 |
| 0.02 | +0.0181 | −0.00000 |
| 0.00 | 0.0000 | +0.00000 |

The bias is ~0.021 nats — matching the analytic (1/2K)(1−1/M) — and is roughly **flat across
the interior**, vanishing only at p = 0 and p = 1. It does not shrink as M grows. Genuine
disagreement (members at 0.2 vs 0.8) scores 0.209, so signal:bias is ~10:1 early on.

The consequence is specific. A near-constant offset does not reorder points *within* the
interior, but it lifts every non-deterministic state ~0.02 above every confidently-decided
one regardless of whether members actually disagree. That is an aleatoric-correlated
preference — precisely what this work exists to remove — and it matters most at high noise
(most of the grid has intermediate p) and late in training (genuine disagreement shrinks
toward the same 0.02 scale).

Note the boundary is *not* a problem: at p̂ = 0 or 1 the entropy estimate is exact, and those
states score low and are never selected.

Both estimators therefore run as separate arms, so the question is settled by data:

- **`epi_var`** = `Var_m[p̂_m] − mean_m[p̂_m(1−p̂_m)/(K−1)]` — unbiased at any K; the
  subtracted term is exactly the per-member MC sampling variance. Monotone in mutual
  information for small disagreement, so ranking is near-identical to BALD without the bias.
- **`epi_bald`** = `H(p̄) − mean_m H(p̂_m)` — textbook, uncorrected.

## Arms

Five arms, all sharing one predictor class so no comparison is confounded by ensembling:

| arm | score |
|---|---|
| `dir00` | non-adaptive (random prefix) |
| `total` | H(p̄) — the current method |
| `epi_var` | debiased between-member variance |
| `epi_bald` | naive mutual information |
| `aleat` | mean_m H(p̂_m) — **negative control** |

`aleat` should be the *worst* arm at high noise, since it deliberately targets irreducible
randomness. Its own finite-K bias runs downward (the plug-in entropy estimator underestimates),
roughly uniformly across the interior, so it shifts the score by a near-constant and does not
threaten the arm's role as a control — but it means `aleat` values are not directly comparable
in absolute terms to `total`. If `epi_*` beats `total` and `aleat` loses to `total`, that is a causal
demonstration the split is real rather than a relabelling.

An ensemble non-adaptive and an ensemble total-entropy arm are both required even though
single-model versions already exist: an ensemble marginal is better calibrated than a single
model purely from averaging, so reusing the old runs as controls would conflate the
acquisition change with the ensembling change. As a bonus, ensemble-`total` vs the existing
single-`total` measures the pure ensembling effect.

d2_ratio = 1.0 (fully adaptive), 19 epochs, seed 42 — matching where the previous campaign's
effects were largest and cleanest. M = 5. K = 20 at acquisition and K = 100 at eval (the
existing defaults); the bias figures above are for the acquisition K.

## Architecture

Four additions and one gated modification. Single-model paths are untouched throughout.

**1. `adaptive_roa/adaptive_v2/probability/ensemble_prob.py`** — two backends sharing one
interface: existing `estimate(states) → OutcomeProbabilities` returning the ensemble
*marginal* p̄ (so all threshold, calibration and eval code is unchanged), plus new
`estimate_members(states) → ndarray [M, N]`.

- `EnsembleClassifierProbabilityBackend` — wraps `predictive_logit_samples()`, exact
- `EnsembleEndpointMCProbabilityBackend` — holds M flow matchers, K endpoint samples each

**2. `adaptive_roa/adaptive_v2/strategy/decomposition.py`** — one strategy with a `score`
mode (`total | epistemic_var | epistemic_bald | aleatoric`), requiring a backend exposing
`estimate_members`. Reuses the existing selection machinery unchanged (`select_greedy_diverse`,
diversity pool, chunking). Deliberately a new file: `entropy.py` already carries a cloud path
and an `estimate` fallback, and four more score modes would make it unsafe to edit.

**3. `EnsembleFlowMatchingTrainer`** — `torch.multiprocessing.spawn` of M processes, member m
pinned to device m, each running `Trainer(devices=[m])` and writing `checkpoints/member_{m}/`.
Zero communication; wall-clock ≈ one member. Follows `bayesian_mlp_trainer` conventions
(per-member seed `base+m`, per-member warm start) but parallel rather than sequential, which
is the whole point for FM at ~50h per member.

**4. Ensemble handle** exposing `n_members` and per-member `predict_endpoint`.

**Configs.** New predictor configs `configs/adaptive_v2/predictor/{fm_ensemble,clf_ensemble}.yaml`
(the latter can largely reuse `bnn_ensemble.yaml`, which is already `posterior: ensemble`,
`n_members: 5`), and one acquisition config per score mode under
`configs/adaptive_v2/acquisition/` selecting `decomposition.py` with the appropriate `score`.

**5. `full_roa.py`** — when the handle reports `n_members`, split the K eval samples **evenly
across members** instead of sampling members randomly. This follows a precedent already in
the codebase: `EnsemblePosterior.predictive_logit_samples` refuses to sample its members,
documenting that seeded sampling produced weights `[.125, .281, .109, .234, .250]` against an
exact `.2`, "enough to flip decisions near lambda*". K=100, M=5 gives exactly 20 per member.
~10 lines, gated on `n_members`.

## Experiment matrix and compute

Phase 1: 5 arms × {stoch-high, stoch-xhigh, deterministic-pendulum}, both predictors.

| | per instance | phase 1 |
|---|---|---|
| FM (M=5, one member/GPU, parallel) | 5 arms × 5 GPUs = 25, ~50h | 3 instances |
| CLF (M=5, members sequential — MLPs are cheap) | 5 arms × 1 GPU = 5, ~30h | 5 instances |

The classifier extends to all four noise levels plus the deterministic pendulum, giving the
full noise dose-response cheaply and testing whether the epistemic advantage grows
monotonically with aleatoric content.

Capacity: Amarel ~40 (L40S, ~3× an iLab a4000) + iLab 12 (currently held by the previous
campaign) + arrakis 4. Westeros contributes **0** — GPU5 is faulted and poisons NVML
box-wide, stranding 7 healthy cards; worth a sysadmin ticket independently of this work.
Two FM instances run concurrently, the third follows.

All phase-1 data is already staged on Amarel: `noisy/pendulum/lqr/{low,med,high,xhigh}` and
`deterministic/pendulum`.

Phase 2 (after phase 1 reads out): remaining noise levels and further deterministic systems.

## Experiment log

`docs/experiments/ensemble_epistemic/`

- `runs.jsonl` — one record per run: `run_id`, `launched_at`, system, level, predictor, arm,
  `score_mode`, seed, M, K, `d2_ratio`, cluster, `job_id`, `output_dir`, `code_hash`, status,
  notes
- `LOG.md` — chronological narrative: launches, failures, decisions and their reasons
- `scripts/exp_log.py` — `append` / `update-status` / `report`, called by the launcher and the
  poller so status cannot drift from reality

`code_hash` is a content hash over the relevant source files. The previous campaign rsynced
*uncommitted* working-tree changes to Amarel, and there is now no way to reconstruct which
code a given run used.

## Validation

**Estimator unit tests** (extending the `stoch_prob_metrics.py` selftest pattern):

- members forced to agree → `epi_var` ≈ 0 at every p and K (verified: |error| < 4e-5)
- same setup → naive BALD ≈ (1/2K)(1−1/M) (verified: 0.021 at K=20, M=5), asserting the bias
  rather than assuming it
- members split 0.2/0.8 → both estimators large and positive
- total = aleatoric + epistemic exactly, for the classifier

**Integration:**

- the marginal `full_roa.py` computes equals the backend's `estimate()` — catches the
  member-weighting bug the `EnsemblePosterior` docstring warns about
- **epoch-0 identity**: all five arms score identically before the first acquisition

**Pre-launch smoke tests**: all 5 arms × both predictors on a tiny config before committing
GPUs. This caught the entropy strategy raising outright on a classifier last campaign.

**Scientific validation**: on the deterministic pendulum aleatoric ≈ 0, so `epi_*` and `total`
should select nearly the same points and land on nearly the same metrics. Divergence there
means the decomposition measures something other than what it claims.

**Significance**: reuse the validated machinery — debiased metrics, epoch-0 run-to-run floor,
seed replicates, and no verdict from a single epoch (require two consecutive). That rule
reversed three conclusions during the previous campaign.

## Out of scope

- Non-ensemble epistemic estimators (Laplace, MFVI, GP) — those arms exist and could be
  compared later, but this work is about the ensemble decomposition.
- Tuning d2_ratio, M, or K. Fixed at 1.0, 5, and the existing defaults so the arms differ
  only in score.
- Retiring or replacing the existing entropy strategy.
