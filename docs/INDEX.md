# Docs index

Routes by **what you are trying to do**, not by filename. Retired documents are deliberately
**not listed here** — an index that surfaces stale pages is worse than no index. If you need to
know why something was retired, see `archive/README.md`.

**Rules of the road:** the code is the source of truth. These pages are routing hints and *why*
explanations — never evidence when the code is inspectable. If a page and the code disagree, the
code wins and the page gets fixed or archived.

## Orient (start here)

| I need to… | Read | Status |
|---|---|---|
| understand what problem this repo solves, and why it's hard | **`PROBLEM.md`** | current |
| know **what object** we estimate (set / field / volume / reachability), or **what a label value means** | **`TARGET.md`** | current |
| **run or track an experiment** — the pre-registered loop, verdict rules | **`experiments/WORKFLOW.md`** | current |
| run this repo on **Amarel** instead of iLab — separate filesystem, glibc split, staging | `COMPUTE.md` | current |
| the general Amarel how-to underneath it — login, `salloc`, batch scripts, `squeue` | `AMAREL_GUIDE.md` | current — generic; concrete values live in `COMPUTE.md` |

> If you are about to touch labels: `0` means **separatrix** to `system.classify_attractor` but
> **failure** to `evaluate_roa`'s predictions, and `-1` flips too. `TARGET.md` has the table.
> This has bitten people.

## The method

| I need to… | Read | Status |
|---|---|---|
| **the method end-to-end** — threshold optimisation, D1/D2 adaptive sampling, reevaluation | **`METHODS.md`** | current |
| optimize λ*/δ*, calibrate q̂, understand the decision band | **`THRESHOLD_OPTIMIZATION.md`** | current — *strongest doc in the repo* |
| trace the q̂ calibration chain specifically | `QHAT_CALIBRATION_PIPELINE.md` | current; overlaps THRESHOLD_OPTIMIZATION |
| **split predictive uncertainty for acquisition** — aleatoric vs epistemic, and why BALD is *not* "epistemic focus" | `experiments/ensemble_epistemic/FINDINGS.md` §2 | current — campaign in progress |
| score p(success\|x) **without** committing to an operating point — Murphy decomposition, debiased Brier | `superpowers/specs/2026-07-31-two-way-outcome-classification-design.md` | current |

> The epoch loop itself: read `adaptive_roa/adaptive_v2/engine.py`. The prose walkthrough that used
> to live here was retired — its CLI examples had all rotted, and the code is shorter than the doc.

## Live campaigns — in progress, do not cite as settled

`experiments/ensemble_epistemic/` — acquiring on the epistemic half of the entropy decomposition.
`FINDINGS.md` holds only what is currently defensible; `LOG.md` is the chronological record
*including superseded claims and their corrections*, so read the pair, not either alone.

> The classifier half is a **negative result about miscalibration**, not a result about BALD. That
> classifier's p̄ is distorted enough that every score — including total entropy, which targets
> p = 0.5 by construction — lands on states whose true p is ≈0.02. So "adaptive is worse than random
> at high noise" is true *of this classifier* and is **not** evidence about the method. The
> flow-matching arms are the fair test, and they were unfinished as of 2026-08-05.

> This campaign predates the card system and does not use it. `scripts/docs_lint.py` walks only
> `experiments/log/` and `experiments/archive/`, so nothing under `ensemble_epistemic/` is linted,
> and it will never appear on `experiments/DASHBOARD.md`.

## Evidence — frozen, never edit

Point-in-time records. They are not stale; they are what happened. Do not "fix" them.

| I need to… | Read |
|---|---|
| **what's broken / what's already been cleared** (verified audit) | **`research_journal/2026-07-15-audit-findings.md`** |
| what actually won: classifier vs flow matching, 4 systems | `research_journal/2026-06-21-classifier-vs-flowmatching.md` |
| **stochastic pendulum, adaptive vs non-adaptive × FM vs classifier** (FINAL — 5 arms × 4 noise levels × 2 predictors) | `stoch_compare/FINDINGS.md`; tables in `stoch_compare/report.md` |
| the run-to-run seed floor those verdicts are judged against | `stoch_compare/seed_variance.md` |
| threshold-free scoring of the matched-budget arms | `experiment_analysis_report_2026-08-01_1030.md` (supersedes the `07-31_1950` cut) |
| adaptive vs random on classification ROA, the 2026-06-23 campaign | `classification_adaptive_vs_random_report_2026-06-23.md` ⚠ its CLI examples predate 2026-06-29: top-level `d2_ratio=` was removed, and `sampling_mode=` is cosmetic |
| Part-X first real run, 2-D pendulum (volume stabilises, tree tracks the separatrix) | `partx_pendulum_run.md` |
| Part-X first real run, 4-D cartpole (**honest negative**: 0/64 leaves resolve) | `partx_cartpole_run.md` |
| per-region MC sample errors | `mc_sample_errors_report.md` |
| which SLURM jobs produced what | `research_journal/job_registry.tsv` ⚠ rows 16–19 corrupted; some jobs missing |

## Specs & plans

`superpowers/specs/` and `superpowers/plans/`, dated `YYYY-MM-DD-<slug>.md`, paired by slug
(design → plan). Point-in-time intent, archived when landed. This is the healthiest convention in
the repo — a spec here is often **more current than the docs above**. Notably
`2026-06-29-adaptive-v2-module-decoupling-design.md` (429 lines) beats anything that claimed to be
the adaptive_v2 architecture reference.

Current context-engineering plan: `superpowers/plans/2026-07-15-docs-restructure-and-maintenance.md`

## Not documented here — on purpose

Architecture tours, per-function APIs, config-key tables, file listings. **Read the code**;
`ls configs/adaptive_v2/*/` is the method-axis list and it cannot go stale. Prose that restates
code only rots and then lies.

## Retired

`archive/` — 11 documents, retired 2026-07-15. Not current, not true, kept for history.
`archive/README.md` says what each one was and why it went.
