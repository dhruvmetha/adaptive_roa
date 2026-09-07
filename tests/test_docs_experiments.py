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
