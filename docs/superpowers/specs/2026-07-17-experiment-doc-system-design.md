# Experiment Documentation System — Design

**Date:** 2026-07-17
**Status:** Approved (design phase)
**Branch:** context-engineering

## Goal

Port the **experiment-documentation *system*** used in
`/common/home/dm1487/robotics_research/ktamp/namo` into this repo — the structure, template,
workflow, enforcement, and registry conventions — **not** namo's actual experiment content. The
result is a pre-registered experiment loop (Hypothesis → Plan → Run → Result+Verdict →
Discussion) whose verdicts are decided on numbers only, whose cards are lint-validated, and whose
results feed a curated paper-ready layer.

This repo already has ~70% of the namo system (`docs/INDEX.md` intent-router with status tags;
`docs/superpowers/specs/` dated design docs; `docs/research_journal/` frozen evidence +
`job_registry.tsv`; a codified frozen-never-edit philosophy). The gap this design closes is the
**experiment-card lifecycle** and its enforcement — the part namo has that we do not.

### What we are porting (the system)

- The per-experiment **card** with fixed sections and queryable frontmatter.
- The `idea → live → done` **lifecycle** with `log/` → `archive/` movement.
- The **numbers-only verdict** rule and mandatory result-splitting.
- A **two-layer results** model (verbose card vs curated `RESULTS.md`).
- **Lint enforcement** (`scripts/docs_lint.py` + a pytest wrapper).
- A **run registry** as the single source of truth for run paths (never glob).

### What we are NOT porting

- namo's experiment content, threads, metrics, or test-set specifics (difficulty×horizon,
  onepush/pure2push, horizon-Q registry, etc.).
- The Obsidian Bases dependency (`experiments.base`, `DASHBOARD.md` as an Obsidian view). We stay
  portable: `DASHBOARD.md` is plain generated markdown.
- namo's hardcoded global split axis. Ours is **per-card declared** (see D1).

## Key design decisions

### D1 — Per-card declared split axis (not a global rule)
namo enforces one global split — *always by difficulty × horizon*. Our mandatory-split axis is
not yet fixed; candidate axes are **system**, **method/predictor**, **controller**, and **noise
level**, and which ones matter will depend on the experiment. So instead of hardcoding a global
rule, **each card declares its required split axes in frontmatter** (`splits: [system, method]`),
and the linter enforces that the card's own declaration is non-empty (for `done` cards) and is the
contract the Result section must honor. This is strictly more general than namo's fixed rule and
matches "we'll figure out the axes as things move along."

*Rejected:* hardcoding `system × method` globally — premature; controller-chaining and
stochastic-ROA lines will want different axes.

### D2 — New `docs/experiments/` tree, integrated with existing docs (no duplication)
A parallel `docs/experiments/` tree holds the card loop, mirroring namo's layout. It does **not**
duplicate infrastructure we already have:
- `docs/superpowers/specs/` stays the design-doc home; a card **links to** its spec, it does not
  replace it.
- `docs/research_journal/` + `job_registry.tsv` remain the **frozen-journal + run-registry**
  layer (namo's journals/registry role). Cards reference journal entries; a finished experiment
  *line* is frozen there.
- `docs/INDEX.md` gains exactly one new route row pointing at `experiments/WORKFLOW.md`.

*Rejected:* folding cards into `docs/superpowers/` (blends two systems, muddies the specs dir);
building a second parallel registry (splits the source of truth — see D4).

### D3 — Enforcement is both a standalone script and a pytest test
`scripts/docs_lint.py` is runnable by hand (fast feedback, `--dashboard` regeneration). A thin
`tests/test_docs_experiments.py` wrapper invokes the same checker so violations fail loudly under
`pytest`, matching this repo's existing pytest-based CI and namo's lint discipline.

*Rejected:* pytest-only (loses manual/dashboard use); script-only (loses CI enforcement).

### D4 — One run registry: extend the existing `job_registry.tsv`
Runs are logged in the existing `docs/research_journal/job_registry.tsv`, not a new
experiments-local registry. One registry, one source of truth for run paths; the "never glob
`EXP_DIR`" rule points everyone at the same file.

*Rejected:* a dedicated `experiments/run_registry.tsv` — splits the registry, reintroduces the
glob temptation.

### D5 — Portable dashboard, no Obsidian
`DASHBOARD.md` is plain generated markdown, emitted by `docs_lint.py --dashboard` (the linter
already parses every card's frontmatter, so the table is free). No Obsidian Bases dependency, so
the system works for anyone with the repo.

*Rejected:* porting `experiments.base` — adds a hard Obsidian dependency for a table we can
generate.

## Target layout

```
docs/experiments/
├── log/                     active cards, one experiment per file
│   └── EXP-YYYY-MM-DD-<slug>.md
├── archive/                 done cards (git mv here on completion)
├── _templates/
│   └── experiment.md        the card template
├── WORKFLOW.md              the operating loop + enforced rules (HUB)
├── RESULTS.md               curated, paper-ready findings (starts empty)
└── DASHBOARD.md             generated status table (docs_lint.py --dashboard)
```

Existing, unchanged in role:
- `docs/superpowers/specs/` — design docs (cards link to these).
- `docs/research_journal/` — frozen evidence + `job_registry.tsv` (the journal + registry layer).
- `docs/INDEX.md` — gains one route row to `experiments/WORKFLOW.md`.

## Component specifications

### Card template — `docs/experiments/_templates/experiment.md`

Frontmatter (queryable):

```yaml
---
type: experiment
status: idea            # idea → live → done
created: {{date}}
commit:                 # SHA, stamped before status flips to live
splits: []              # REQUIRED axes for this experiment, e.g. [system, method]
metric:                 # the exact quantity that decides accept/reject
systems: []             # e.g. [pendulum, cartpole]
predictor:              # classifier | flow-matching | gp | n/a
thread:                 # controller-chaining | stochastic-roa | ...
tags: [experiment]
---
```

Fixed sections (author in parentheses):

| Section | Author | Content |
|---|---|---|
| **Hypothesis** | user | Falsifiable claim + expected direction. |
| **Plan** | Claude | Code / data / config + exact run command. |
| **Run** | Claude (auto) | job id · commit · config · date; cross-refs `job_registry.tsv`. |
| **Result + Verdict** | Claude (auto) | Numbers **split by every axis in `splits:`**; accept/reject on numbers only. |
| **Next** | either | Implication + follow-up experiment. |
| **Discussion** | user ↔ Claude | Dated inline `**[who YYYY-MM-DD]**`, newest at bottom. |

Repo-specific adaptations vs namo:
- `splits:` is per-card (D1); the linter treats it as the Result-splitting contract.
- `predictor:` and `systems:` frontmatter capture this repo's core comparison dimensions.
- **Cross-predictor verdicts must state abstention/coverage.** F1 is computed on a retained
  subset and is not comparable across methods with different abstention rates (see `CLAUDE.md`,
  `docs/TARGET.md`); a verdict comparing predictors without stating coverage is invalid.

Role separation (as namo): the user writes **Hypothesis** and **Discussion**; Claude writes
**Plan**, **Run**, **Result+Verdict**, and the registry row.

### Operating doc — `docs/experiments/WORKFLOW.md`

The loop:
1. User creates `log/EXP-<date>-<slug>.md` (status `idea`) with Hypothesis, `splits:`, `metric:`.
2. Claude writes Plan; user approves.
3. Commit code, stamp `commit:` SHA, flip status → `live`, launch; log the job in
   `docs/research_journal/job_registry.tsv`.
4. On completion, Claude fills Run + Result+Verdict, split by declared axes, numbers-only
   accept/reject.
5. Flip status → `done`; `git mv` card `log/ → archive/`; lift a curated finding into
   `RESULTS.md`; freeze a finished line into `docs/research_journal/`.

Enforced rules (repo-specific):
- Status enum is `idea → live → done` only.
- Result is **split by every axis in `splits:`** — no aggregate-only reporting.
- `commit:` SHA is stamped **before** status becomes `live`.
- Cross-predictor verdicts **must state abstention/coverage** (F1-not-comparable trap).
- The registry (`job_registry.tsv`) is the source of truth for run paths — **never glob**
  `EXP_DIR`.
- Writes go only under `EXP_DIR` / `outputs/`; **never** `DATA_DIR` (shared, read-only).

### Curated results — `docs/experiments/RESULTS.md`

Starts empty with a header explaining the two-layer contract. One tight block per **accepted**
finding: a MAIN table (split by the card's axes) + a one-paragraph finding + a link back to the
archived card. Verbose detail (all tables, caveats, rejected variants) stays in the card. This is
the layer that lifts into a paper.

### Dashboard — `docs/experiments/DASHBOARD.md`

Generated markdown table, columns: `card · status · thread · systems · predictor · metric ·
verdict`. Regenerated by `docs_lint.py --dashboard`. Never hand-edited (a header notes this).

### Linter — `scripts/docs_lint.py`

Parses every card under `log/` and `archive/` and checks:
- Frontmatter parses; `type: experiment`; `status` ∈ {idea, live, done}.
- `commit:` is set when status ≠ `idea`.
- `metric:` is set when status ∈ {live, done}.
- `splits:` is non-empty when status == `done`.
- All six fixed section headings are present, in order.
- `docs/INDEX.md` contains a route to `experiments/WORKFLOW.md`.

Modes:
- default: validate; exit non-zero on any violation, printing `file: problem` lines.
- `--dashboard`: regenerate `DASHBOARD.md` from parsed frontmatter (implies validate).

The parser must not depend on a YAML library beyond what the repo already vendors; if PyYAML is
available in the `arcmg` env it may be used, otherwise a minimal frontmatter parser is included.

### Pytest wrapper — `tests/test_docs_experiments.py`

Thin test that imports the checker from `scripts/docs_lint.py` and asserts zero violations across
the experiments tree, so CI (`pytest`) fails loudly on a malformed card. Does not re-implement
checks — it calls the same function the script does.

### INDEX.md route

Add one row under an appropriate section of `docs/INDEX.md`:

| I need to… | Read | Status |
|---|---|---|
| **run or track an experiment** — the pre-registered loop, verdict rules | **`experiments/WORKFLOW.md`** | current |

### Seed content

- `_templates/experiment.md`, `WORKFLOW.md`, empty `RESULTS.md`, empty-ish `DASHBOARD.md`,
  `scripts/docs_lint.py`, `tests/test_docs_experiments.py`.
- One **example card** `log/EXP-2026-07-17-example.md` (status `idea`), a shape demonstration —
  not a real experiment, deletable — so the linter runs green against a real file and the
  dashboard renders a row.

## Testing plan

- **Lint self-test:** `python scripts/docs_lint.py` exits 0 against the seeded tree; `pytest
  tests/test_docs_experiments.py` passes.
- **Negative cases** (in the pytest module, against fixtures under a temp dir): a card with an
  invalid status, a `live` card missing `commit:`, a `done` card with empty `splits:`, and a card
  missing a fixed heading each produce exactly one violation.
- **Dashboard round-trip:** `docs_lint.py --dashboard` regenerates `DASHBOARD.md`; a second run is
  a no-op (idempotent), and the generated table lists the example card.
- **INDEX route:** the linter fails if the `experiments/WORKFLOW.md` route row is removed.

## Risks / notes

- **Two doc systems side by side.** `docs/superpowers/specs/` (designs) and `docs/experiments/`
  (experiments) are distinct on purpose: a spec answers "what are we building"; a card answers
  "what did this run show." The INDEX route and WORKFLOW.md must state this so future readers do
  not file experiments as specs or vice versa.
- **Frontmatter drift vs code.** Per `CLAUDE.md`, docs route and code is evidence. Cards are
  point-in-time records; once `done` and archived they are frozen (like `research_journal/`) and
  must not be "fixed" to match later code.
- **YAML dependency.** The linter must degrade gracefully if PyYAML is absent in `arcmg`; a
  minimal frontmatter parser removes that dependency.
- **Registry is shared.** Extending `job_registry.tsv` (D4) means experiment runs and any other
  logged runs coexist; the schema addition must be backward-compatible with existing rows.
