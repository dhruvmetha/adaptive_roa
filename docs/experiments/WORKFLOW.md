# Experiment Workflow

This tree holds **experiment cards**: point-in-time records of what a run showed. It is distinct
from `docs/superpowers/specs/` (which answers "what are we building") — a card *links to* a spec,
it does not replace one. Finished experiment *lines* are frozen into `docs/research_journal/`.

## The loop

1. Create `log/EXP-<date>-<slug>.md` from `_templates/experiment.md` (status `idea`). Write the
   **Hypothesis**, declare `splits:` and `metric:`.
2. Claude writes the **Plan**; you approve it.
3. Commit code, stamp `commit:` with the SHA, flip status to `live`, launch. Log the job in
   `docs/research_journal/job_registry.tsv`.
4. On completion, Claude fills **Run** and **Result + Verdict**, split by the declared axes,
   accept/reject on numbers only, and sets `verdict:`.
5. Flip status to `done`; `git mv` the card `log/ → archive/`; lift a curated finding into
   `RESULTS.md`; freeze a finished line into `docs/research_journal/`.

## Enforced rules (checked by `scripts/docs_lint.py`)

- Status is one of `idea → live → done`.
- `commit:` is stamped **before** status becomes `live`.
- `metric:` is set once status is `live`/`done`.
- `splits:` is non-empty for a `done` card, and the **Result** is split by every axis it names —
  never aggregate-only.
- Cross-predictor verdicts **state abstention/coverage** (the F1-not-comparable trap; see
  `docs/TARGET.md`).
- The registry (`docs/research_journal/job_registry.tsv`) is the source of truth for run paths —
  **never glob** `EXP_DIR`.
- Writes go only under `EXP_DIR` / `outputs/`; never `DATA_DIR` (shared, read-only).

## Roles

You write **Hypothesis** and **Discussion**. Claude writes **Plan**, **Run**, **Result+Verdict**,
and the registry row.

## Commands

```bash
python scripts/docs_lint.py                 # validate every card + INDEX route
python scripts/docs_lint.py --dashboard     # regenerate DASHBOARD.md, then validate
pytest tests/test_docs_experiments.py       # CI enforcement
```

`DASHBOARD.md` is generated — never hand-edit it. `RESULTS.md` is the curated, paper-ready layer;
the verbose detail stays in the card.
