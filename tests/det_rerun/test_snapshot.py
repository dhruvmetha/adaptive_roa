"""scripts/det_rerun/snapshot.sh freezes a commit into a read-only dir outside any repo."""
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/det_rerun/snapshot.sh"
WT = Path("/common/home/st1122/Projects/adaptive_roa/.claude/worktrees/det-paper")

pytestmark = pytest.mark.skipif(not (WT / ".env").exists(), reason="needs the det-paper worktree")


def _snap(root, ref="HEAD"):
    out = subprocess.run(["bash", str(SCRIPT), ref], env={**os.environ, "DET_CODE_ROOT": str(root)},
                         capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def test_snapshot_holds_the_commit_and_is_read_only(tmp_path):
    head = subprocess.run(["git", "-C", str(WT), "rev-parse", "--short", "HEAD~1"],
                          capture_output=True, text=True, check=True).stdout.strip()
    code = _snap(tmp_path, "HEAD~1")
    try:
        assert code == tmp_path / head
        assert (code / "COMMIT").read_text().strip() == head
        assert (code / ".env").exists() and (code / "scripts/run_adaptive.py").exists()
        assert not (code / ".git").exists()
        tracked = subprocess.run(["git", "-C", str(WT), "show", f"{head}:scripts/run_adaptive.py"],
                                 capture_output=True, text=True, check=True).stdout
        assert (code / "scripts/run_adaptive.py").read_text() == tracked
        with pytest.raises(PermissionError):
            (code / "new_file").write_text("x")
        assert _snap(tmp_path, "HEAD~1") == code          # a second call reuses it
        assert not list(tmp_path.glob(".tmp_*"))
    finally:
        subprocess.run(["chmod", "-R", "u+w", str(tmp_path)], check=True)
