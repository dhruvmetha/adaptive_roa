# Experiment Documentation System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pre-registered experiment-card system under `docs/experiments/` with a lint-validated card lifecycle, a portable generated dashboard, and pytest enforcement.

**Architecture:** A minimal-dependency Python linter (`scripts/docs_lint.py`) parses experiment-card frontmatter and validates the card lifecycle; it also regenerates a plain-markdown dashboard. A thin pytest module wraps the same checker so CI fails on malformed cards. The docs tree (`docs/experiments/`) holds the template, workflow, curated results, dashboard, and cards, integrating with existing `docs/superpowers/specs/`, `docs/research_journal/`, and `docs/INDEX.md` rather than duplicating them.

**Tech Stack:** Python 3 stdlib only (no PyYAML dependency — a minimal frontmatter parser is included); pytest (already in repo); markdown docs.

## Global Constraints

- Package is `adaptive_roa/`; there is no `src/`. New script lives at `scripts/docs_lint.py`.
- Python is the `arcmg` conda env (`/common/users/dm1487/envs/arcmg`); run tests with `pytest`.
- Linter uses **Python 3 stdlib only** — must run even if PyYAML is absent.
- Writes go only under the repo / `EXP_DIR` / `outputs/`; never `DATA_DIR` (shared, read-only).
- Never delete docs — archive via `git mv`. Card completion moves `log/ → archive/`.
- Valid status enum, verbatim: `idea`, `live`, `done`.
- Required card sections, verbatim and in this order: `Hypothesis`, `Plan`, `Run`,
  `Result + Verdict`, `Next`, `Discussion`.
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_018hHsL4f2BqbeVdAztD5L2V
  ```

---

### Task 1: Frontmatter parser + card/index validators

**Files:**
- Create: `scripts/docs_lint.py`
- Test: `tests/test_docs_lint.py`

**Interfaces:**
- Produces:
  - `parse_frontmatter(text: str) -> dict | None` — returns frontmatter dict (scalars as `str`,
    inline `[a, b]` as `list[str]`, `[]` as `[]`), or `None` if no leading `---` block.
  - `split_frontmatter(text: str) -> tuple[dict | None, str]` — returns `(frontmatter, body)`
    where `body` is the text after the closing `---`.
  - `card_violations(path: str, text: str) -> list[str]` — lifecycle + section checks for one card.
  - `index_violations(index_text: str) -> list[str]` — checks the INDEX route exists.
  - Module constants `VALID_STATUS: list[str]`, `REQUIRED_SECTIONS: list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docs_lint.py
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "docs_lint",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "docs_lint.py"),
)
docs_lint = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(docs_lint)


VALID_CARD = """\
---
type: experiment
status: done
created: 2026-07-17
commit: abc1234
splits: [system, method]
metric: F1 on retained subset
systems: [pendulum]
predictor: classifier
thread: controller-chaining
verdict: accept
tags: [experiment]
---
# Example

## Hypothesis
Claim.

## Plan
Plan.

## Run
Run.

## Result + Verdict
Numbers.

## Next
Next.

## Discussion
Talk.
"""


def test_parse_frontmatter_scalars_and_lists():
    fm = docs_lint.parse_frontmatter(VALID_CARD)
    assert fm["type"] == "experiment"
    assert fm["status"] == "done"
    assert fm["splits"] == ["system", "method"]
    assert fm["tags"] == ["experiment"]


def test_parse_frontmatter_empty_list_and_missing_block():
    assert docs_lint.parse_frontmatter("no frontmatter here") is None
    fm = docs_lint.parse_frontmatter("---\nsplits: []\ncommit:\n---\nbody\n")
    assert fm["splits"] == []
    assert fm["commit"] == ""


def test_valid_card_has_no_violations():
    assert docs_lint.card_violations("log/EXP-2026-07-17-example.md", VALID_CARD) == []


def test_index_route_present_and_absent():
    assert docs_lint.index_violations("see experiments/WORKFLOW.md for runs") == []
    problems = docs_lint.index_violations("no route here")
    assert len(problems) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_lint.py -v`
Expected: FAIL — `ModuleNotFoundError`/`FileNotFoundError` (no `scripts/docs_lint.py` yet).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/docs_lint.py
"""Lint experiment cards under docs/experiments/ and regenerate the dashboard.

Stdlib only: must run without PyYAML. Frontmatter is a restricted subset —
`key: value` scalars and inline `[a, b]` / `[]` lists, one per line.
"""
from __future__ import annotations

VALID_STATUS = ["idea", "live", "done"]
REQUIRED_SECTIONS = ["Hypothesis", "Plan", "Run", "Result + Verdict", "Next", "Discussion"]
INDEX_ROUTE = "experiments/WORKFLOW.md"


def split_frontmatter(text):
    """Return (frontmatter_dict_or_None, body_text)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm = _parse_block(lines[1:i])
            body = "\n".join(lines[i + 1:])
            return fm, body
    return None, text


def _parse_block(block_lines):
    fm = {}
    for line in block_lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        key = key.strip()
        val = raw.strip()
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            fm[key] = [p.strip() for p in inner.split(",") if p.strip()] if inner else []
        else:
            fm[key] = val
    return fm


def parse_frontmatter(text):
    fm, _ = split_frontmatter(text)
    return fm


def card_violations(path, text):
    problems = []
    fm, body = split_frontmatter(text)
    if fm is None:
        return [f"{path}: missing YAML frontmatter"]
    if fm.get("type") != "experiment":
        problems.append(f"{path}: frontmatter type must be 'experiment'")
    status = fm.get("status")
    if status not in VALID_STATUS:
        problems.append(f"{path}: status '{status}' not in {VALID_STATUS}")
    if status in ("live", "done") and not fm.get("commit"):
        problems.append(f"{path}: commit SHA required once status is live/done")
    if status in ("live", "done") and not fm.get("metric"):
        problems.append(f"{path}: metric required once status is live/done")
    if status == "done" and not fm.get("splits"):
        problems.append(f"{path}: splits must be non-empty for a done card")
    # Section presence + order.
    last = -1
    for section in REQUIRED_SECTIONS:
        idx = body.find(f"## {section}")
        if idx == -1:
            problems.append(f"{path}: missing section '## {section}'")
        elif idx < last:
            problems.append(f"{path}: section '## {section}' out of order")
        else:
            last = idx
    return problems


def index_violations(index_text):
    if INDEX_ROUTE in index_text:
        return []
    return [f"docs/INDEX.md: missing route to {INDEX_ROUTE}"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_docs_lint.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/docs_lint.py tests/test_docs_lint.py
git commit -m "$(cat <<'EOF'
feat(docs-lint): frontmatter parser + card/index validators

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018hHsL4f2BqbeVdAztD5L2V
EOF
)"
```

---

### Task 2: Tree walker, dashboard renderer, and CLI

**Files:**
- Modify: `scripts/docs_lint.py` (append functions + `main`)
- Test: `tests/test_docs_lint.py` (append tests)

**Interfaces:**
- Consumes: `card_violations`, `index_violations`, `split_frontmatter`, `REQUIRED_SECTIONS` (Task 1).
- Produces:
  - `collect_cards(experiments_dir: str) -> list[tuple[str, dict]]` — `(relpath, frontmatter)` for
    every `*.md` under `log/` and `archive/`, sorted by relpath.
  - `check_tree(experiments_dir: str, index_path: str) -> list[str]` — aggregated violations across
    all cards plus the INDEX route check.
  - `render_dashboard(cards: list[tuple[str, dict]]) -> str` — markdown table string with a
    "generated — do not edit" header. Columns: card, status, thread, systems, predictor, metric,
    verdict.
  - `main(argv: list[str] | None = None) -> int` — CLI: default validates (exit 1 on violations);
    `--dashboard` writes `DASHBOARD.md` then validates.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_docs_lint.py
import textwrap


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_check_tree_and_dashboard(tmp_path):
    exp = tmp_path / "experiments"
    _write(str(exp / "log" / "EXP-2026-07-17-example.md"), VALID_CARD)
    index = tmp_path / "INDEX.md"
    _write(str(index), "route: experiments/WORKFLOW.md\n")

    assert docs_lint.check_tree(str(exp), str(index)) == []

    cards = docs_lint.collect_cards(str(exp))
    table = docs_lint.render_dashboard(cards)
    assert "do not edit" in table.lower()
    assert "EXP-2026-07-17-example.md" in table
    assert "| done |" in table
    # Idempotent rendering.
    assert docs_lint.render_dashboard(cards) == table


def test_check_tree_reports_bad_status(tmp_path):
    bad = VALID_CARD.replace("status: done", "status: cooking")
    exp = tmp_path / "experiments"
    _write(str(exp / "log" / "EXP-bad.md"), bad)
    index = tmp_path / "INDEX.md"
    _write(str(index), "route: experiments/WORKFLOW.md\n")
    problems = docs_lint.check_tree(str(exp), str(index))
    assert any("not in" in p for p in problems)


def test_main_dashboard_writes_file(tmp_path):
    exp = tmp_path / "experiments"
    _write(str(exp / "log" / "EXP-2026-07-17-example.md"), VALID_CARD)
    _write(str(exp / "DASHBOARD.md"), "old\n")
    index = tmp_path / "INDEX.md"
    _write(str(index), "route: experiments/WORKFLOW.md\n")
    rc = docs_lint.main(["--experiments", str(exp), "--index", str(index), "--dashboard"])
    assert rc == 0
    with open(str(exp / "DASHBOARD.md")) as f:
        assert "EXP-2026-07-17-example.md" in f.read()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_lint.py -k "tree or dashboard or main" -v`
Expected: FAIL — `AttributeError: module 'docs_lint' has no attribute 'check_tree'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/docs_lint.py
import argparse
import os
import sys

DEFAULT_EXPERIMENTS = os.path.join("docs", "experiments")
DEFAULT_INDEX = os.path.join("docs", "INDEX.md")
_DASH_HEADER = "<!-- generated by scripts/docs_lint.py --dashboard — do not edit by hand -->\n"


def _iter_card_paths(experiments_dir):
    for sub in ("log", "archive"):
        base = os.path.join(experiments_dir, sub)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if name.endswith(".md"):
                yield os.path.join(base, name)


def collect_cards(experiments_dir):
    cards = []
    for path in _iter_card_paths(experiments_dir):
        with open(path) as f:
            fm, _ = split_frontmatter(f.read())
        rel = os.path.relpath(path, experiments_dir)
        cards.append((rel, fm or {}))
    return sorted(cards, key=lambda c: c[0])


def check_tree(experiments_dir, index_path):
    problems = []
    for path in _iter_card_paths(experiments_dir):
        with open(path) as f:
            problems.extend(card_violations(path, f.read()))
    if os.path.exists(index_path):
        with open(index_path) as f:
            problems.extend(index_violations(f.read()))
    else:
        problems.append(f"{index_path}: file not found")
    return problems


def _cell(fm, key):
    val = fm.get(key, "")
    if isinstance(val, list):
        return ", ".join(val)
    return val or ""


def render_dashboard(cards):
    cols = ["card", "status", "thread", "systems", "predictor", "metric", "verdict"]
    lines = [_DASH_HEADER, "# Experiment Dashboard\n",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for rel, fm in cards:
        row = [os.path.basename(rel),
               _cell(fm, "status"), _cell(fm, "thread"), _cell(fm, "systems"),
               _cell(fm, "predictor"), _cell(fm, "metric"), _cell(fm, "verdict")]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lint experiment cards; regenerate dashboard.")
    parser.add_argument("--experiments", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--dashboard", action="store_true", help="regenerate DASHBOARD.md")
    args = parser.parse_args(argv)

    if args.dashboard:
        cards = collect_cards(args.experiments)
        with open(os.path.join(args.experiments, "DASHBOARD.md"), "w") as f:
            f.write(render_dashboard(cards))

    problems = check_tree(args.experiments, args.index)
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_docs_lint.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/docs_lint.py tests/test_docs_lint.py
git commit -m "$(cat <<'EOF'
feat(docs-lint): tree walker, dashboard renderer, CLI

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018hHsL4f2BqbeVdAztD5L2V
EOF
)"
```

---

### Task 3: Seed the experiments tree + INDEX route

**Files:**
- Create: `docs/experiments/_templates/experiment.md`
- Create: `docs/experiments/WORKFLOW.md`
- Create: `docs/experiments/RESULTS.md`
- Create: `docs/experiments/log/EXP-2026-07-17-example.md`
- Create: `docs/experiments/DASHBOARD.md` (generated in Step 3)
- Modify: `docs/INDEX.md` (add one route row)

**Interfaces:**
- Consumes: `main(["--dashboard", ...])` (Task 2) to generate `DASHBOARD.md`.
- Produces: a real experiments tree the Task 4 pytest asserts is lint-clean.

- [ ] **Step 1: Create the card template**

Create `docs/experiments/_templates/experiment.md`:

```markdown
---
type: experiment
status: idea
created: {{date}}
commit:
splits: []
metric:
systems: []
predictor:
thread:
verdict:
tags: [experiment]
---
# {{title}}

## Hypothesis
_(you)_ The falsifiable claim — what we're testing and expected direction.

## Plan
_(Claude)_ Code / data / config + the exact run command.

## Run
_(Claude, auto)_ job id · commit · config · date. Cross-ref research_journal/job_registry.tsv.

## Result + Verdict
_(Claude, auto)_ Numbers, split by every axis in `splits:`. Accept/reject on numbers only.
Cross-predictor verdicts MUST state abstention/coverage (F1 is not comparable across
methods with different abstention rates).

## Next
What this implies; the follow-up experiment.

## Discussion
_(you ↔ Claude — ask here; answered inline, dated `**[who YYYY-MM-DD]**`, newest at bottom.)_
```

- [ ] **Step 2: Create WORKFLOW.md**

Create `docs/experiments/WORKFLOW.md`:

```markdown
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
```

- [ ] **Step 3: Create RESULTS.md**

Create `docs/experiments/RESULTS.md`:

```markdown
# Results

Curated, paper-ready findings. One tight block per **accepted** experiment: a MAIN table (split by
the card's declared axes) + a one-paragraph finding + a link back to the archived card. Verbose
detail — every table, caveats, rejected variants — stays in the card, not here.

_No accepted findings yet._
```

- [ ] **Step 4: Create the example card**

Create `docs/experiments/log/EXP-2026-07-17-example.md`:

```markdown
---
type: experiment
status: idea
created: 2026-07-17
commit:
splits: []
metric:
systems: []
predictor:
thread:
verdict:
tags: [experiment, example]
---
# Example card — shape demonstration (delete me)

## Hypothesis
_(you)_ This card exists only to demonstrate the shape and keep the linter green; delete it once a
real experiment exists.

## Plan
_(Claude)_ None — not a real experiment.

## Run
_(Claude, auto)_ —

## Result + Verdict
_(Claude, auto)_ —

## Next
Replace with a real experiment card from `_templates/experiment.md`.

## Discussion
_(you ↔ Claude)_
```

- [ ] **Step 5: Add the INDEX route**

In `docs/INDEX.md`, add this row under the "Orient (start here)" table (after the `TARGET.md` row):

```markdown
| **run or track an experiment** — the pre-registered loop, verdict rules | **`experiments/WORKFLOW.md`** | current |
```

- [ ] **Step 6: Generate the dashboard and validate**

Run:
```bash
python scripts/docs_lint.py --dashboard
```
Expected: exit 0, no output; `docs/experiments/DASHBOARD.md` now lists `EXP-2026-07-17-example.md`.

- [ ] **Step 7: Commit**

```bash
git add docs/experiments docs/INDEX.md
git commit -m "$(cat <<'EOF'
feat(docs): seed experiments tree (template, workflow, results, dashboard, INDEX route)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018hHsL4f2BqbeVdAztD5L2V
EOF
)"
```

---

### Task 4: Pytest CI enforcement over the real tree

**Files:**
- Create: `tests/test_docs_experiments.py`

**Interfaces:**
- Consumes: `check_tree`, `card_violations`, `render_dashboard`, `collect_cards` (Tasks 1–2); the
  seeded real tree (Task 3).
- Produces: a CI test that fails on any malformed card or a stale dashboard.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docs_experiments.py
import importlib.util
import os

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_SPEC = importlib.util.spec_from_file_location(
    "docs_lint", os.path.join(_ROOT, "scripts", "docs_lint.py"))
docs_lint = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(docs_lint)

_EXPERIMENTS = os.path.join(_ROOT, "docs", "experiments")
_INDEX = os.path.join(_ROOT, "docs", "INDEX.md")


def test_real_experiments_tree_is_clean():
    problems = docs_lint.check_tree(_EXPERIMENTS, _INDEX)
    assert problems == [], "\n".join(problems)


def test_dashboard_is_up_to_date():
    cards = docs_lint.collect_cards(_EXPERIMENTS)
    expected = docs_lint.render_dashboard(cards)
    with open(os.path.join(_EXPERIMENTS, "DASHBOARD.md")) as f:
        assert f.read() == expected, "run: python scripts/docs_lint.py --dashboard"


def test_done_card_requires_nonempty_splits():
    card = """\
---
type: experiment
status: done
created: 2026-07-17
commit: abc1234
splits: []
metric: F1
systems: [pendulum]
predictor: classifier
verdict: accept
tags: [experiment]
---
# X
## Hypothesis
h
## Plan
p
## Run
r
## Result + Verdict
v
## Next
n
## Discussion
d
"""
    problems = docs_lint.card_violations("log/x.md", card)
    assert any("splits must be non-empty" in p for p in problems)


def test_live_card_requires_commit():
    card = """\
---
type: experiment
status: live
created: 2026-07-17
commit:
splits: [system]
metric: F1
systems: [pendulum]
predictor: classifier
verdict:
tags: [experiment]
---
# X
## Hypothesis
h
## Plan
p
## Run
r
## Result + Verdict
v
## Next
n
## Discussion
d
"""
    problems = docs_lint.card_violations("log/x.md", card)
    assert any("commit SHA required" in p for p in problems)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_docs_experiments.py -v`
Expected: PASS (4 tests) — the real seeded tree is clean, the dashboard matches, and the two
negative fixtures each surface their violation.

Note: this test asserts on already-correct seeded content (Task 3), so it passes on first run
rather than starting red; the negative-case tests (`test_done_card_requires_nonempty_splits`,
`test_live_card_requires_commit`) are the ones that would have failed against a checker that did
not implement those rules, and they exercise Task 1's logic against fresh fixtures.

- [ ] **Step 3: Run the full suite to confirm no regressions**

Run: `pytest tests/test_docs_lint.py tests/test_docs_experiments.py -v`
Expected: PASS (all tests from Tasks 1, 2, 4).

- [ ] **Step 4: Commit**

```bash
git add tests/test_docs_experiments.py
git commit -m "$(cat <<'EOF'
test(docs): CI enforcement of experiment-card lint + dashboard freshness

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018hHsL4f2BqbeVdAztD5L2V
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- D1 per-card `splits:` → Task 1 `card_violations` (done-card non-empty check), Task 4 negative test. ✓
- D2 new tree + no duplication + INDEX route → Task 3 (files + route row), WORKFLOW.md states the specs/journal split. ✓
- D3 script + pytest wrapper → Tasks 1–2 (script), Task 4 (pytest). ✓
- D4 one registry (extend `job_registry.tsv`) → WORKFLOW.md references it; no new registry file created. ✓
- D5 portable generated dashboard → Task 2 `render_dashboard` + `--dashboard`, Task 3 generation, Task 4 freshness test; no Obsidian. ✓
- Template fixed sections + frontmatter → Task 3 template; validated by Task 1. ✓
- WORKFLOW enforced rules → Task 3 WORKFLOW.md; the machine-checkable subset in Task 1. ✓
- RESULTS.md two-layer → Task 3. ✓
- Cross-predictor abstention rule → template + WORKFLOW copy (documentation-enforced; not
  machine-checkable, noted in spec risks). ✓
- Seed example card → Task 3; keeps linter green (Task 4). ✓
- Testing plan (lint self-test, negative cases, dashboard idempotency, INDEX route) → Tasks 1, 2, 4. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The `{{date}}`/`{{title}}`
tokens are intentional template placeholders in a doc file, not plan gaps.

**Type consistency:** `split_frontmatter`, `parse_frontmatter`, `card_violations`,
`index_violations`, `collect_cards`, `check_tree`, `render_dashboard`, `main` — names and
signatures match across Tasks 1, 2, and 4. Dashboard columns (`card, status, thread, systems,
predictor, metric, verdict`) match the `verdict:` frontmatter field added to the template.

**Note — refinement past the approved spec:** a `verdict:` frontmatter field was added to the
template so the dashboard's verdict column is queryable (the numeric result still lives in the card
body; `verdict:` holds only the accept/reject flag). Small, in the spirit of D5; flag if unwanted.
