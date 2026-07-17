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
