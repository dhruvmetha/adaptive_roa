"""Living docs must not point at things that don't exist.

The docs rotted once already: the src/ -> adaptive_roa/ rename left ~56 dead path references
across the guides, and nothing failed. This test makes that class of rot loud.

Doc classes (see docs/INDEX.md):
  LIVING   -- must track the tree; checked here.
  FROZEN   -- dated evidence (run notes, journal). Point-in-time records; a file they cite may
              legitimately no longer exist. NEVER checked, never "fixed".
  ARCHIVE  -- retired and untrue by definition. Never checked.

Only backticked references with a real file extension are checked, so prose like
"there is no `src/`" is naturally ignored.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# LIVING docs: repo root entry points + docs/ top level, minus the frozen run notes.
FROZEN = {
    "mc_sample_errors_report.md",   # generated evidence tables
    "partx_pendulum_run.md",        # dated run note
    "partx_cartpole_run.md",        # dated run note
}

LIVING = [REPO / "CLAUDE.md", REPO / "README.md"] + [
    p for p in sorted((REPO / "docs").glob("*.md")) if p.name not in FROZEN
]

# `adaptive_roa/...py`, `configs/...yaml`, `docs/...md`, etc. Requiring an extension skips
# intentional non-paths like `src/` and `src/...`.
PATH_RE = re.compile(
    r"`((?:adaptive_roa|scripts|configs|docs|tests)/[A-Za-z0-9_/.\-]+"
    r"\.(?:py|md|yaml|yml|sh|tsv|json|txt|pdf))`"
)


def referenced_paths(doc: Path):
    return sorted(set(PATH_RE.findall(doc.read_text())))


@pytest.mark.parametrize("doc", LIVING, ids=lambda p: str(p.relative_to(REPO)))
def test_all_referenced_paths_exist(doc):
    missing = [ref for ref in referenced_paths(doc) if not (REPO / ref).exists()]
    assert not missing, (
        f"{doc.relative_to(REPO)} references paths that do not exist: {missing}. "
        "Fix the reference, or archive the doc (git mv to docs/archive/) -- never leave a "
        "living doc pointing at nothing."
    )


def test_living_set_is_current():
    """If a new .md lands in docs/ it gets checked automatically; this just guards the
    FROZEN allowlist against typos (a frozen name that no longer exists)."""
    for name in FROZEN:
        assert (REPO / "docs" / name).exists(), f"FROZEN allowlist entry gone: {name}"


def test_no_dead_cli_keys_in_living_docs():
    """`sampling_mode=` selected the acquisition strategy until 2026-06-29, and its ghost
    keeps reappearing in prose as if it still did.

    It came back on 2026-07-26 (275bb30) as a `@package _global_` label set by each
    `configs/adaptive_v2/acquisition/*.yaml`, but it is interpolated into `output_dir` and
    nothing else -- the loop branches on `self.acquisition.mode`. So the key exists and
    overriding it is *silently* inert, which is worse than a hard error.

    Mentions are therefore fine only when the line says the key is removed or cosmetic.
    Selection is `acquisition=...`, then and now.
    """
    pat = re.compile(r"sampling_mode=")
    allowed = ("removed", "cosmetic")
    for doc in LIVING:
        for i, line in enumerate(doc.read_text().splitlines(), 1):
            if pat.search(line) and not any(w in line.lower() for w in allowed):
                pytest.fail(
                    f"{doc.relative_to(REPO)}:{i} presents `sampling_mode=` as a live CLI "
                    "selector. It only renames output_dir. Mark it removed or cosmetic; "
                    "the selector is `acquisition=...`."
                )
