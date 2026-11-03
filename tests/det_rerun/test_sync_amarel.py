"""sync_amarel.sh shares one ssh connection and appends to the Amarel logs.

On 2026-11-03 a sync took 373-497 s of a 10-minute cycle: ~200 s went to the log rsync,
whose delta pass re-read every growing FM log (they pass 1 GB) to send a few MB of new
lines, and ~2.4 s of ssh handshake went to each of ~50 calls.
"""
import os
import stat
import subprocess
from pathlib import Path

SYNC = Path(__file__).resolve().parents[2] / "scripts/det_rerun/sync_amarel.sh"
NAME = "dr_q3d30_fm_ranked_fprcap1pct_s99512"


def _fake(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text(f"#!/bin/bash\nprintf '%s\\0' {name} \"$@\" >> \"$CALLS\"; printf '\\n' >> \"$CALLS\"\n{body}\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def _sync(tmp_path):
    here, exp, bin_dir = tmp_path / "here", tmp_path / "exp", tmp_path / "bin"
    for d in (here, exp, bin_dir):
        d.mkdir()
    out = exp / "q3d30" / NAME
    (here / "manifest.tsv").write_text(
        "name\tsystem\tarm\tmode\tseed\tmem\tpriority\toverrides\tscript\n"
        f"{NAME}\tq3d30\tfm\tranked\t99512\t40\t1\toutput_dir={out} n_epochs=11\ttrain.py\n")
    (here / "launched.tsv").write_text(
        f"{NAME}\tamarel\t--partition=gpu\t63486718\t20261103_1300\t{here}/x.out\tf205193\n")
    # ssh answers the two remote listings the script asks for; rsync only records its call.
    _fake(bin_dir, "ssh", f'case "$*" in *"ls -d"*) echo q3d30/{NAME};; *squeue*) echo "{NAME} RUNNING";; esac')
    _fake(bin_dir, "rsync", "exit 0")
    calls = tmp_path / "calls"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "CALLS": str(calls),
           "DET_RERUN_HERE": str(here), "DET_RERUN_EXP": str(exp), "RETRY_SLEEP": "0"}
    r = subprocess.run(["bash", str(SYNC)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [line.strip("\0").split("\0") for line in calls.read_text().split("\n") if line.strip("\0")]
    return r.stdout, rows, here


def test_every_ssh_and_rsync_rides_one_shared_connection(tmp_path):
    out, rows, _ = _sync(tmp_path)
    assert f"synced {NAME}" in out
    ssh = [r for r in rows if r[0] == "ssh"]
    rsync = [r for r in rows if r[0] == "rsync"]
    assert len(ssh) == 2 and len(rsync) == 2
    paths = set()
    for r in ssh:
        assert "ControlMaster=auto" in r
        paths |= {a for a in r if a.startswith("ControlPath=")}
    for r in rsync:
        e = r[r.index("-e") + 1]
        assert "ControlMaster=auto" in e
        paths |= {w for w in e.split() if w.startswith("ControlPath=")}
    assert len(paths) == 1


def test_the_log_rsync_appends_and_the_cell_rsync_does_not(tmp_path):
    _, rows, here = _sync(tmp_path)
    logs = [r for r in rows if r[0] == "rsync" and r[-1] == f"{here}/amarel_logs/"]
    cells = [r for r in rows if r[0] == "rsync" and r not in logs]
    assert len(logs) == 1 and "--append" in logs[0]
    # A cell dir holds checkpoints rewritten in place; appending would corrupt them.
    assert len(cells) == 1 and "--append" not in cells[0] and "--delete" in cells[0]


def test_the_queue_snapshot_lands_in_the_overridden_dir(tmp_path):
    _, _, here = _sync(tmp_path)
    assert (here / "amarel_queue.txt").read_text().split() == [NAME, "RUNNING"]
