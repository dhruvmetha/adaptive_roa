"""retry.sh reruns a failing command, so a dropped ssh connection does not fail the sync.

The Amarel log rsync in sync_amarel.sh broke three times on 2026-11-03 (broken pipe,
rsync 255/19/20) and each manual rerun went through.
"""
import subprocess
from pathlib import Path

RETRY = Path(__file__).resolve().parents[2] / "scripts/det_rerun/retry.sh"


def _bash(body: str, tmp_path) -> subprocess.CompletedProcess:
    script = f"set -euo pipefail\nsource {RETRY}\nRETRY_SLEEP=0\n{body}\n"
    return subprocess.run(["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True)


FLAKY = 'flaky() { echo x >> tries; [ "$(wc -l < tries)" -ge "$1" ]; }'


def test_a_command_that_fails_twice_succeeds_on_the_third_try(tmp_path):
    r = _bash(f"{FLAKY}\nretry 3 flaky 3\necho done", tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == "done"
    assert (tmp_path / "tries").read_text().count("x") == 3


def test_it_gives_up_after_n_tries_with_the_last_status(tmp_path):
    r = _bash("fail() { echo x >> tries; return 23; }\nretry 2 fail", tmp_path)
    assert r.returncode == 23
    assert (tmp_path / "tries").read_text().count("x") == 2


def test_a_first_try_success_runs_once(tmp_path):
    r = _bash(f"{FLAKY}\nretry 3 flaky 1", tmp_path)
    assert r.returncode == 0
    assert (tmp_path / "tries").read_text().count("x") == 1
