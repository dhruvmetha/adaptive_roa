import numpy as np
import pandas as pd
import pytest
from adaptive_roa.benchmark.provenance import git_sha, assert_post_fix


def test_git_sha_returns_a_sha_in_a_real_checkout():
    sha = git_sha()
    assert sha is not None and len(sha) >= 7


def test_git_sha_returns_none_outside_a_checkout(tmp_path):
    assert git_sha(repo_root=tmp_path) is None


def test_git_sha_returns_none_when_repo_root_is_not_a_directory(tmp_path):
    # repo_root that exists but is a file (not a dir) raises NotADirectoryError
    # from the subprocess `cwd=` kwarg -- must be reported as None, not raised.
    not_a_dir = tmp_path / "not_a_directory"
    not_a_dir.write_text("")
    assert git_sha(repo_root=not_a_dir) is None


def test_unknown_provenance_is_refused_not_assumed_valid():
    df = pd.DataFrame({"arm": ["bnn_mfvi"], "commit": [None]})
    with pytest.raises(ValueError, match="unknown"):
        assert_post_fix(df, fix_sha="HEAD")


def test_unknown_provenance_via_real_nan_is_refused_not_assumed_valid():
    # A genuine pandas/numpy float NaN (not Python None) must hit the same
    # "unknown provenance" branch -- not fall through to the git-subprocess
    # ancestor check, which would misreport it as "predates fix".
    df = pd.DataFrame({"arm": ["bnn_mfvi"], "commit": [np.nan]})
    assert df["commit"].dtype == np.float64
    with pytest.raises(ValueError, match="unknown"):
        assert_post_fix(df, fix_sha="HEAD")


def test_a_commit_that_is_not_an_ancestor_of_the_fix_is_refused():
    # A run from before the fix must not be pooled with runs after it.
    df = pd.DataFrame({"arm": ["bnn_mfvi"], "commit": ["0000000"]})
    with pytest.raises(ValueError, match="predates|not found|unknown"):
        assert_post_fix(df, fix_sha="HEAD")


def test_a_run_at_head_passes_against_an_ancestor_fix():
    head = git_sha()
    df = pd.DataFrame({"arm": ["bnn_mfvi"], "commit": [head]})
    assert_post_fix(df, fix_sha="HEAD~1")
