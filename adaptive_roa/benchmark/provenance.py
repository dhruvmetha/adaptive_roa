"""Which code produced a run.

Pooling results from before and after a correctness fix is the single most
likely way this benchmark produces a confident wrong table, and it is invisible
in the numbers. That detector needs each run to record its own commit.
"""
from __future__ import annotations

import subprocess

import pandas as pd


def git_sha(repo_root=None) -> str | None:
    """Short SHA of the working tree, or None if this is not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        # OSError covers FileNotFoundError (no `git` binary), NotADirectoryError
        # (repo_root exists but isn't a directory), PermissionError, etc. -- any
        # reason `cwd` or the subprocess launch itself fails means "can't tell",
        # which this function reports as None, never by raising.
        return None
    return out.stdout.strip() or None


def assert_post_fix(df: pd.DataFrame, fix_sha: str, repo_root=None) -> None:
    """Every row must come from code at or after `fix_sha`."""
    commits = df["commit"].unique()
    for commit in commits:
        if commit is None or (isinstance(commit, float) and pd.isna(commit)):
            raise ValueError(
                "a run reports unknown provenance (no recorded commit). "
                "Refusing to assume it postdates the fix -- that assumption is "
                "what this guard exists to prevent. Re-run it, or exclude it "
                "explicitly."
            )
        try:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", fix_sha, str(commit)],
                cwd=str(repo_root) if repo_root else None,
                capture_output=True, check=True,
            )
        except subprocess.CalledProcessError:
            raise ValueError(
                f"run at commit {commit} predates fix {fix_sha} (or is not "
                f"found in this repository). Its results are not comparable to "
                f"runs after that fix."
            ) from None
