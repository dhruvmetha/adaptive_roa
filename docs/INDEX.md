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

> If you are about to touch labels: `0` means **separatrix** to `system.classify_attractor` but
> **failure** to `evaluate_roa`'s predictions, and `-1` flips too. `TARGET.md` has the table.
> This has bitten people.

## The method

| I need to… | Read | Status |
|---|---|---|
| the method end-to-end, faithful to the implementation | **`ADAPTIVE_METHOD_THRESHOLD_SAMPLING_REEVALUATION.md`** | current — *best method reference; the filename undersells it* |
| optimize λ*/δ*, calibrate q̂, understand the decision band | **`THRESHOLD_OPTIMIZATION.md`** | current — *strongest doc in the repo* |
| trace the q̂ calibration chain | `QHAT_CALIBRATION_PIPELINE.md` | current, narrow |
| the RoA method's math (system-agnostic) | `ADAPTIVE_ROA_METHOD.md` | ⚠ math durable, prose partly stale |
| the adaptive epoch loop, step by step | `ADAPTIVE_CARTPOLE_PIPELINE.md` | ⚠ loop map unique & useful; **every CLI example misfires** |

## Evidence — frozen, never edit

Point-in-time records. They are not stale; they are what happened. Do not "fix" them.

| I need to… | Read |
|---|---|
| **what's broken / what's already been cleared** (verified audit) | **`research_journal/2026-07-15-audit-findings.md`** |
| what actually won: classifier vs flow matching, 4 systems | `research_journal/2026-06-21-classifier-vs-flowmatching.md` |
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
