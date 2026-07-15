# Plan: context engineering for this repo

**Date:** 2026-07-15 · **Branch:** `context-engineering`
**Supersedes:** the earlier draft of this file (six-doc suite + generated INDEX + doc-class taxonomy).
That draft was wrong: it optimised for *adding* pages. This one optimises for *subtracting* them.

## The one-line thesis

**Context engineering here is ~80% subtraction.** The only clearly valuable change so far was
CLAUDE.md 381 → 75 lines. The next most valuable is getting nine actively-lying documents out of
the reading path. Adding pages is the least valuable part.

**Nothing is ever deleted — only archived.** `git mv` to an `archive/` directory: reversible,
keeps history, and removes the file from the default reading path. Unmapping is what matters, not
destruction. The repo already has this convention (`configs/archive/legacy_adaptive_v1/`).

**Wrong is worse than missing.** A missing doc sends a reader to the code (slow, safe). A wrong doc
sends them confidently the wrong way (fast, dangerous). Every failure observed while writing this
plan was the repo *asserting something false*, never being silent.

## Target structure

Follows the ontology: **Problem → Benchmark (system × type) → Data (full/partial) → Methods**,
plus a frozen evidence layer.

| Artifact | Role | How it's maintained |
|---|---|---|
| `CLAUDE.md` | **schema** — env, guardrails, thesis, task protocol, pointers | hand, ~75 lines, ruthlessly |
| `docs/PROBLEM.md` | why RoA, why hard, why budget binds | hand — durable, science-speed |
| `docs/SYSTEMS.md` | benchmark suite: system × type, dims, manifolds, attractors | **GENERATED** |
| `docs/DATA.md` | the regime × form 2×2, what's on disk, the empty cell | **GENERATED** |
| `docs/METHODS.md` | the axes, the bundles, the comparability rule | hand + generated axis table |
| `docs/EVIDENCE/` | journal, run notes, results | **FROZEN**, append-only |
| `docs/archive/` | the lying docs | `git mv`, unmapped |
| code + tests | source of truth | never duplicated upward |

**No `INDEX.md`.** With ~4 pages, CLAUDE.md lists them. An index is for a corpus, not a shelf.

## Migration from the existing 20 docs

### HARVEST — real content, recycle it
| Doc | Verdict | Destination |
|---|---|---|
| `ADAPTIVE_METHOD_THRESHOLD_SAMPLING_REEVALUATION.md` (440L) | **CURRENT** — best method reference, badly named | becomes the spine of `METHODS.md` |
| `THRESHOLD_OPTIMIZATION.md` (504L) | **CURRENT** — strongest doc in the repo | keep as-is; link from METHODS; fix 1 broken ref |
| `QHAT_CALIBRATION_PIPELINE.md` (224L) | CURRENT but ~60L unique | merge into THRESHOLD_OPTIMIZATION |
| `ADAPTIVE_ROA_METHOD.md` (296L) | math durable, prose stale | harvest into PROBLEM + METHODS |
| `ADAPTIVE_CARTPOLE_PIPELINE.md` (513L) | unique epoch-loop map; every CLI misfires | harvest the loop map; drop the commands |

### GENERATE — never hand-write an inventory
| Doc | Generated from |
|---|---|
| `SYSTEMS.md` | `dataset_description.json` (state_dim, manifold, goal, success criteria) + `configs/system/` |
| `DATA.md` | `ls {DATA_DIR}/{regime}/{dataset}` + dataset descriptions → the 2×2 |
| METHODS axis table | `ls configs/adaptive_v2/{predictor,probability,threshold,calibration,acquisition,eval}/` |

Dimensions, manifolds, support matrices and empty cells are exactly what humans forget to update.

**The trick that makes generation stick:** a test (`tests/test_docs_generated.py`) regenerates and
diffs against the committed file, failing on drift. That puts doc maintenance in **tests** — the
medium the co-author actually uses (he authored 71 of 80 test commits vs 28 of 63 doc commits).

### FREEZE — evidence, already the right shape
`research_journal/` (+ `job_registry.tsv`), `partx_pendulum_run.md`, `partx_cartpole_run.md`,
`mc_sample_errors_report.md`. Never updated; exempt from all checks. **Fix `partx_*_run.md`'s
coverage attribution** — it credits `d1_eval_metrics.coverage`, which `engine.py:326` cannot
produce in partx mode.

### ARCHIVE — actively lying (`git mv docs/archive/`)
| Doc | Evidence |
|---|---|
| `TRAINING_GUIDE.md` | 17 dead `src/` refs; `rm1838`'s home; covers 2 of 6 systems |
| `ROA_ANALYSIS_GUIDE.md` | every class/entry point it documents is gone |
| `ENDPOINT_DATASET_GENERATION.md` | 23 dead `src/` refs |
| `SETUP_GUIDE.md` | teaches the deleted `src/` layout *as the fix* |
| `AMAREL_GUIDE.md` | generic placeholder tutorial; this repo runs on iLab |
| `CONFORMAL_ADAPTIVE_SAMPLING.md` | documents the calibrator *comment* — **disproved 2026-07-15** |
| `ADAPTIVE_V2_ARCHITECTURE.md` | 41-line table of contents; predates `predictor=` |
| `ADAPTIVE_V2_MIGRATION.md` | historical, misfiled → fold into `archive/LEGACY_ADAPTIVE_V1.md` |

### REWRITE
`README.md` → ~20 honest lines. Cannot be archived (GitHub front page). Currently claims the repo
is a flow-matching library — the single most expensive lie in the tree.

### TRIM
`TARGET.md` → durable ontology only. Its label codebook becomes
`tests/test_label_codebook.py`; its line numbers and status flags are dropped.

## Phases

- **Phase 1 — Subtract (hours).** `git mv` the 8; rewrite README; repoint CLAUDE.md.
  *Zero new prose. Biggest single win.*
- **Phase 2 — Schema (30 min).** Add the task protocol to CLAUDE.md:
  - state the exact question before reading broadly
  - verify against code before acting on a name or a doc
  - cite the source file for every behavioural claim
  - docs are routing hints, never evidence when code is inspectable
  - when code and prose conflict, archive or demote the prose
- **Phase 3 — Generate (half day).** `scripts/gen_docs.py` → SYSTEMS/DATA/axes + the drift test.
- **Phase 4 — Harvest (a day).** METHODS.md from the three CURRENT docs. Trim TARGET.
- **Phase 5 — Durability.** Untrack `results/`; stop discarding `artifacts_v2.json`;
  `job_registry.tsv` written by the submit path, not by hand (it already has space-vs-tab
  corruption and missing rows).
- **Phase 6 — Repeatable.** A lint command (contradictions / stale claims / orphans) — the
  operation that produced this audit, made repeatable. Add CI when it's worth 20 lines of YAML.

## Conventions

- Every behavioural claim in prose carries a **`file@SHA:lines` pin**, not a date. A date is a
  claim; a pin is checkable.
- Prose explains **why** a contract exists. **Tests enforce it.**
- Renames beat documentation: `adaptive/` (load-bearing, *not* legacy — holds the only DATA_DIR
  guard) and `partial_trajs` (a T-step dynamics surrogate, not a classifier) mislead by name. No
  doc fixes a lying name.

## Dead code — archive, don't document, don't delete

`interfaces.py` (imported by nothing, already drifted from reality) and the ~1,100 LOC with zero
importers (`visualization/`, `evaluation/`, `callbacks/`, `manifold_integration/`,
`flow_matching/utils/`, `flow_matching/base/inference.py`, `systems/pendulum_universal.py`) move to
an `archive/` package, mirroring `configs/archive/`. Documenting them would canonicalise fiction;
deleting them is not allowed. Archiving unmaps them from the reading path, which is the goal.

## Deliberately not doing

`INDEX.md`; a doc-class taxonomy; hand-written inventories; per-function API docs; config key
tables. No deletions of any kind.

## Open — needs the user

1. `results/` and two `*.txt` guides are gitignored — untrack and commit? (Evidence is the one
   irreplaceable layer and it's currently invisible to collaborators; the partx run artifacts are
   already gone.)
2. Tier 0 renames touch sumanth's code (`adaptive/`, `partial_trajs`). In or out?
3. Is the partx credible interval (marginal variances only, ignores GP covariance → too narrow,
   worsening as the tree refines) a known approximation or a real issue?
