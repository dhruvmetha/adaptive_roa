"""git_sha records the commit whatever directory a job starts in."""
import subprocess

from adaptive_roa.benchmark.provenance import COMMIT_ENV, PACKAGE_ROOT, git_dirty, git_sha


def test_sha_from_the_package_checkout_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv(COMMIT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)                    # not a git checkout
    expected = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PACKAGE_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    assert git_sha() == expected


def test_launcher_can_pin_the_commit(monkeypatch):
    monkeypatch.setenv(COMMIT_ENV, "abc1234")
    assert git_sha() == "abc1234"


def test_explicit_repo_root_ignores_the_pin(tmp_path, monkeypatch):
    monkeypatch.setenv(COMMIT_ENV, "abc1234")
    assert git_sha(repo_root=tmp_path) is None


def test_dirty_flag_is_a_bool_in_a_checkout():
    assert git_dirty() in (True, False)
