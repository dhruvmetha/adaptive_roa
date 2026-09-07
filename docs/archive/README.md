# Archive — retired documents

**Nothing here is current. Do not use these to answer a question.** They are kept because the
history is worth preserving, not because the content is true. Git preserves the full history of
each file across the move (`git log --follow <file>`).

Archived on **2026-07-15**, after an audit that verified every referenced path, config name, and
command against the tree. The dominant failure was the `src/` → `adaptive_roa/` package rename:
someone ran a sed over dotted imports (`src.` → `adaptive_roa.`) and missed slash paths (`src/`),
so these files contain correct YAML sitting next to broken commands. `grep "_target_: src\." configs/`
returns **zero** — the rot was docs-only.

| File | Why it was retired |
|---|---|
| `TRAINING_GUIDE.md` | 17 dead `src/` refs; hardcodes another user's home (`rm1838`) so its quick-start runs for nobody; covers 2 of 6+ systems |
| `ENDPOINT_DATASET_GENERATION.md` | 23 dead `src/` refs. The workflow is still real — `adaptive_roa/build_shuffled_endpoint_dataset.py` exists — so this is the cheapest one to resurrect (prefix fix only) |
| `ROA_ANALYSIS_GUIDE.md` | Unsalvageable. `demo_roa_analysis.py`, `AttractorBasinAnalyzer`, `FlowMatchingEvaluator`, `*_lcfm.py` are all gone from the tree |
| `SETUP_GUIDE.md` (+`.pdf`) | Teaches the deleted `src/` layout **as the fix**. Actively harmful |
| ~~`AMAREL_GUIDE.md` (+`.pdf`)~~ | **Un-retired 2026-07-28 — the reason for archiving it was wrong.** It was retired on the grounds that "this repo's evidence points at iLab, not Amarel"; that inference was drawn from dhruv's checkout alone. Amarel is st1122's primary compute, and `main` now carries `scripts/sbatch_amarel.sh`, `scripts/stage_dataset_amarel.sh`, and `amarel_*` defaults to prove it. Restored to `docs/AMAREL_GUIDE.md`, with its `/common/`-path and partition errors corrected against `COMPUTE.md`. Kept in this table as a record of the misjudgement |
| `CONFORMAL_ADAPTIVE_SAMPLING.md` | **Actively wrong.** It documents the *comment* in `conformal/calibrator.py` rather than the code. On 2026-07-15 the code was proven correct (0 disagreements over 2e6 random probability triples) and the comment's fuller formula shown to be a provably-dominated no-op. This doc taught the disproved version. `THRESHOLD_OPTIMIZATION.md` and `QHAT_CALIBRATION_PIPELINE.md` document the code and remain live |
| `ADAPTIVE_V2_ARCHITECTURE.md` | A 41-line table of contents, not an architecture. Predates the `predictor=generative\|classifier\|gp` abstraction entirely and never mentions it; says "4 systems" when there are more; asserts `artifacts_v2.json` contains `legacy_epoch_metrics`, a string that appears only in docs and never in code |
| `ADAPTIVE_V2_MIGRATION.md` | Correct, but a one-time v1→v2 cutover changelog — a historical record misfiled as a living doc. Belongs beside `LEGACY_ADAPTIVE_V1.md` |
| `LEGACY_ADAPTIVE_V1.md` | Pre-existing archive entry: where the v1 configs/scripts went |
| `ADAPTIVE_CARTPOLE_PIPELINE.md` | 513 lines stepping through the adaptive epoch loop for cartpole. Every CLI example misfires (`conformal.*` knobs and `sampling_mode=` no longer exist). Its one unique asset was the loop map — but `adaptive_v2/engine.py` is ~300 lines and is true. Retired rather than left flagged: a doc with a warning label is still a doc people read |
| `ADAPTIVE_ROA_METHOD.md` | 296 lines of system-agnostic method math. The math is durable but the prose is partly stale, and `METHODS.md` (the Feb-27 deep dive) covers the same ground and is current. Retired as redundant, not as wrong |

## A warning that outlived its doc

`adaptive_roa/adaptive/` **is not legacy and must not be removed.** The v1 *loop* was retired (see
`LEGACY_ADAPTIVE_V1.md`), but its **data layer survives and `adaptive_v2` depends on it** — its own
`__init__.py` says "Remaining components (used by adaptive_v2)". All three acquisition strategies
import `UncertainSampler` from it, and it holds the repo's **only** DATA_DIR write guard. The name
misleads; the code is load-bearing.

## Where things live now

`docs/INDEX.md` routes everything current. `README.md` is the honest front page.
`CLAUDE.md` is the operating manual.
