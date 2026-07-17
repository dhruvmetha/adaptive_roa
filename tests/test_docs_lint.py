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
