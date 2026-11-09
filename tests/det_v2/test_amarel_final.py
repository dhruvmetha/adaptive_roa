"""Resuming and relaunching Amarel runs while Amarel is down (maintenance-202611, 2026-11-10 08:00).

final_pull_amarel.sh pulls each run whose job squeue no longer lists and writes .amarel_final
("<job> <time> pulled|absent") into the shared run dir. is_live and leave_amarel accept the marker
for that same job instead of asking Amarel, so resume_*.sh and DET_V2_RELAUNCH launches work with
Amarel unreachable, for either account's runs.
"""
from __future__ import annotations

import os
import stat
import subprocess

from det_v2_world import GPU_RUN, SMALL_RUN, STUBS, _exe


def _arun(world, run_dir):
    return world.amarel / "exp" / run_dir.relative_to(world.exp)


def _amarel_down(world):
    """ssh to the Amarel host fails as an unreachable host does (exit 255); other hosts still answer."""
    _exe(world.bin / "ssh", STUBS["ssh"].replace(
        'host=$1; shift\n', 'host=$1; shift\n[ "$host" = amarelhost ] && { echo "amarelhost: down" >&2; exit 255; }\n'))


def _amarel_calls(world):
    p = world.calls / "ssh"
    return [l for l in p.read_text().splitlines() if l.startswith("amarelhost\t")] if p.exists() else []


def test_final_pull_marks_an_ended_job_and_resume_then_skips_amarel(world):
    run_dir = world.run_dir(GPU_RUN)
    world.bank(_arun(world, run_dir), 3)
    world.ledger_row(GPU_RUN, "amarel", "55", "gpu", run_dir)
    r = world.run("final_pull_amarel.sh", GPU_RUN)
    assert f"final {GPU_RUN}: job 55, pulled, 3 banked" in r.stdout
    marker = (run_dir / ".amarel_final").read_text().split()
    assert marker[0] == "55" and marker[2] == "pulled"
    assert stat.S_IMODE(os.stat(run_dir / ".amarel_final").st_mode) & 0o060 == 0o060   # the other account reads it
    _amarel_down(world)
    n = len(_amarel_calls(world))
    world.run("resume_ilab.sh", GPU_RUN, "1:00:00", "a4000")
    assert len(_amarel_calls(world)) == n                       # nothing asked Amarel
    assert (world.calls / "sbatch").read_text().splitlines()[-1].startswith(GPU_RUN)
    assert world.ledger()[-1][2] == "ilab" and world.ledger()[-1][9] == "resume"
    assert not (run_dir / ".amarel_mirror").exists()


def test_a_job_cancelled_while_pending_relaunches_without_amarel(world):
    run_dir = world.run_dir(SMALL_RUN)
    world.ledger_row(SMALL_RUN, "amarel", "56", "gpu", run_dir)
    world.ledger_row(SMALL_RUN, "amarel", "56", "gpu", run_dir, action="cancel")
    r = world.run("final_pull_amarel.sh", SMALL_RUN)
    assert f"final {SMALL_RUN}: job 56, absent, 0 banked" in r.stdout
    assert (run_dir / ".amarel_final").read_text().split()[2] == "absent"
    _amarel_down(world)
    world.run("launch.sh", SMALL_RUN, "ilab", "a4000", extra={"DET_V2_RELAUNCH": "1"})
    assert world.ledger()[-1][2] == "ilab" and world.ledger()[-1][9] == "launch"


def test_another_accounts_marked_run_resumes_here(world):
    """Dhruv's side marks its own runs (only that account can read its scratch); this account can then
    resume one with Amarel down, where a pull would otherwise refuse another account's scratch."""
    run_dir = world.run_dir(GPU_RUN)
    world.bank(run_dir, 3)
    world.ledger_row(GPU_RUN, "amarel", "57", "gpu", run_dir, user="other")
    (run_dir / ".amarel_final").write_text("57 2026-11-10T07:15:00 pulled\n")
    _amarel_down(world)
    world.run("resume_ilab.sh", GPU_RUN, "1:00:00", "a4000")
    assert world.ledger()[-1][1] == "tester" and world.ledger()[-1][9] == "resume"


def test_a_marker_for_an_older_job_does_not_count(world):
    run_dir = world.run_dir(GPU_RUN)
    world.bank(run_dir, 3)
    world.ledger_row(GPU_RUN, "amarel", "55", "gpu", run_dir)
    (run_dir / ".amarel_final").write_text("55 2026-11-10T07:15:00 pulled\n")
    world.ledger_row(GPU_RUN, "amarel", "58", "gpu", run_dir, action="resume")   # resumed on Amarel since
    _amarel_down(world)
    r = world.run("resume_ilab.sh", GPU_RUN, "1:00:00", "a4000", check=False)
    assert r.returncode != 0 and "could not reach" in r.stderr


def test_final_pull_leaves_a_job_squeue_still_lists(world):
    run_dir = world.run_dir(GPU_RUN)
    world.bank(_arun(world, run_dir), 2)
    world.ledger_row(GPU_RUN, "amarel", "55", "gpu", run_dir)
    (world.calls / "squeue_out").write_text("COMPLETING\n")
    r = world.run("final_pull_amarel.sh", GPU_RUN, check=False)
    assert r.returncode != 0 and f"skip {GPU_RUN}: squeue still lists job 55 (COMPLETING)" in r.stdout
    assert not (run_dir / ".amarel_final").exists()


def test_final_pull_writes_no_marker_when_the_pull_fails(world):
    run_dir = world.run_dir(GPU_RUN)
    world.bank(_arun(world, run_dir), 2)
    world.ledger_row(GPU_RUN, "amarel", "55", "gpu", run_dir)
    _exe(world.bin / "rsync", "exit 23\n")
    r = world.run("final_pull_amarel.sh", GPU_RUN, check=False)
    assert r.returncode != 0 and f"FAILED {GPU_RUN} (rsync 23)" in r.stdout
    assert not (run_dir / ".amarel_final").exists()


def test_final_pull_writes_no_marker_when_amarel_cannot_be_asked(world):
    run_dir = world.run_dir(GPU_RUN)
    world.ledger_row(GPU_RUN, "amarel", "55", "gpu", run_dir)
    _amarel_down(world)
    r = world.run("final_pull_amarel.sh", GPU_RUN, check=False)
    assert r.returncode != 0 and f"FAILED {GPU_RUN}: could not ask Amarel about job 55" in r.stdout
    assert not (run_dir / ".amarel_final").exists()


def test_final_pull_refuses_a_dir_it_does_not_own(world):
    run_dir = world.run_dir(GPU_RUN)
    world.bank(_arun(world, run_dir), 1)
    world.bank(run_dir, 3)                                             # a local run, no mirror marker
    world.ledger_row(GPU_RUN, "amarel", "55", "gpu", run_dir)
    r = world.run("final_pull_amarel.sh", GPU_RUN, check=False)
    assert r.returncode != 0 and f"REFUSED {GPU_RUN}" in r.stdout
    assert len(list(run_dir.glob("epoch_*"))) == 3 and not (run_dir / ".amarel_final").exists()


def test_default_names_are_this_accounts_unfinished_amarel_runs(world):
    gpu, small = world.run_dir(GPU_RUN), world.run_dir(SMALL_RUN)
    world.bank(_arun(world, gpu), 1)
    world.ledger_row(GPU_RUN, "amarel", "55", "gpu", gpu)
    world.ledger_row(GPU_RUN, "amarel", "55", "gpu", gpu, action="cancel")   # a later cancel row still counts
    world.ledger_row(SMALL_RUN, "amarel", "56", "gpu", small, user="other")
    out = world.run("final_pull_amarel.sh").stdout
    assert f"final {GPU_RUN}" in out and SMALL_RUN not in out


def test_a_later_sync_keeps_the_marker(world):
    run_dir = world.run_dir(GPU_RUN)
    world.bank(_arun(world, run_dir), 2)
    world.ledger_row(GPU_RUN, "amarel", "55", "gpu", run_dir)
    world.run("final_pull_amarel.sh", GPU_RUN)
    world.run("sync_amarel.sh")
    assert (run_dir / ".amarel_final").read_text().startswith("55 ")


def test_det_v3_seeds_has_the_wrappers():
    from det_v2_world import ROOT
    for name in ("final_pull_amarel.sh", "amarel_outage_rows.sh"):
        twin = ROOT / "scripts/det_v3_seeds" / name
        assert twin.read_text() == (ROOT / "scripts/det_v3_seeds/sync_amarel.sh").read_text(), name
        assert os.access(twin, os.X_OK), name


# ---------------------------------------- a later cancel row for the same job (Dhruv's side, driver 07:10)
# rebalance.move and cancel_for_outage.py append a cancel row only after sacct shows the job CANCELLED, so
# is_live answers "not live" for that job without asking its cluster, which may be down.

def test_a_later_cancel_row_for_the_job_skips_the_probe(world):
    run_dir = world.run_dir(SMALL_RUN)
    world.ledger_row(SMALL_RUN, "amarel", "61", "gpu", run_dir, user="other")
    world.ledger_row(SMALL_RUN, "amarel", "61", "gpu", run_dir, action="cancel", user="other")
    _amarel_down(world)
    world.run("launch.sh", SMALL_RUN, "ilab", "a4000", extra={"DET_V2_RELAUNCH": "1"})
    assert _amarel_calls(world) == []
    assert world.ledger()[-1][2] == "ilab" and world.ledger()[-1][9] == "launch"


def test_without_a_cancel_row_is_live_still_probes(world):
    run_dir = world.run_dir(SMALL_RUN)
    world.ledger_row(SMALL_RUN, "amarel", "62", "gpu", run_dir)
    _amarel_down(world)
    r = world.run("launch.sh", SMALL_RUN, "ilab", "a4000", check=False, extra={"DET_V2_RELAUNCH": "1"})
    assert r.returncode != 0 and "could not reach amarel to check job 62" in r.stderr
    _exe(world.bin / "ssh", STUBS["ssh"])                        # Amarel back: the probe answers
    (world.calls / "squeue_out").write_text("PENDING\n")
    r = world.run("launch.sh", SMALL_RUN, "ilab", "a4000", check=False, extra={"DET_V2_RELAUNCH": "1"})
    assert r.returncode != 0 and "still runs" in r.stderr        # the probe ran and saw it PENDING


def test_a_cancel_row_for_an_older_job_does_not_count(world):
    run_dir = world.run_dir(SMALL_RUN)
    world.ledger_row(SMALL_RUN, "amarel", "63", "gpu", run_dir)
    world.ledger_row(SMALL_RUN, "amarel", "63", "gpu", run_dir, action="cancel")
    world.ledger_row(SMALL_RUN, "amarel", "64", "gpu", run_dir)                   # relaunched there since
    _amarel_down(world)
    r = world.run("launch.sh", SMALL_RUN, "ilab", "a4000", check=False, extra={"DET_V2_RELAUNCH": "1"})
    assert r.returncode != 0 and "could not reach amarel to check job 64" in r.stderr


# ------------------------- 'outage' rows inside the maintenance window (Dhruv's side, driver 07:2x)
# amarel_outage_rows.sh writes one per job the last squeue snapshot showed RUNNING, at or after
# AMAREL_OUTAGE_START (2026-11-10T08:00:00) and before AMAREL_OUTAGE_END (2026-11-11T23:59:00): no Amarel job
# runs inside the reservation, so is_live takes the row as "not live" without asking Amarel.

def _row_at(world, name, where, job, run_dir, action, ts, user="tester"):
    world.ledger_row(name, where, job, "gpu", run_dir, action=action, user=user)
    p = world.exp / "det_v2" / "launched.tsv"
    lines = p.read_text().splitlines()
    f = lines[-1].split("\t")
    f[5] = ts
    p.write_text("\n".join(lines[:-1] + ["\t".join(f)]) + "\n")


def test_an_outage_row_in_the_window_skips_the_probe(world):
    run_dir = world.run_dir(SMALL_RUN)
    world.ledger_row(SMALL_RUN, "amarel", "71", "gpu", run_dir, user="other")
    _row_at(world, SMALL_RUN, "amarel", "71", run_dir, "outage", "2026-11-10T08:05:00", user="other")
    _amarel_down(world)
    world.run("launch.sh", SMALL_RUN, "ilab", "a4000", extra={"DET_V2_RELAUNCH": "1"})
    assert _amarel_calls(world) == []


def test_an_outage_row_for_another_job_does_not_count(world):
    run_dir = world.run_dir(SMALL_RUN)
    _row_at(world, SMALL_RUN, "amarel", "72", run_dir, "launch", "2026-11-09T20:00:00")
    _row_at(world, SMALL_RUN, "amarel", "70", run_dir, "outage", "2026-11-10T08:05:00")
    _amarel_down(world)
    r = world.run("launch.sh", SMALL_RUN, "ilab", "a4000", check=False, extra={"DET_V2_RELAUNCH": "1"})
    assert r.returncode != 0 and "could not reach amarel to check job 72" in r.stderr


def test_an_outage_row_outside_the_window_is_ignored(world):
    run_dir = world.run_dir(SMALL_RUN)
    world.ledger_row(SMALL_RUN, "amarel", "73", "gpu", run_dir)
    _amarel_down(world)
    for ts in ("2026-11-10T07:59:59", "2026-11-11T23:59:00"):
        _row_at(world, SMALL_RUN, "amarel", "73", run_dir, "outage", ts)
        r = world.run("launch.sh", SMALL_RUN, "ilab", "a4000", check=False, extra={"DET_V2_RELAUNCH": "1"})
        assert r.returncode != 0 and "could not reach amarel to check job 73" in r.stderr, ts


def _snapshot(world, text):
    p = world.exp / "det_v2" / "logs" / "amarel_queue_tester.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_outage_rows_are_written_for_running_jobs_inside_the_window_only(world):
    gpu, small = world.run_dir(GPU_RUN), world.run_dir(SMALL_RUN)
    world.ledger_row(GPU_RUN, "amarel", "81", "gpu", gpu)
    world.ledger_row(SMALL_RUN, "amarel", "82", "gpu", small)
    _snapshot(world, f"{GPU_RUN} RUNNING\n{SMALL_RUN} PENDING\nsomething_else RUNNING\n")
    r = world.run("amarel_outage_rows.sh", check=False, extra={"DET_V2_NOW": "2026-11-10T07:30:00"})
    assert r.returncode != 0 and "outside" in r.stderr
    assert not [l for l in world.ledger() if l[9] == "outage"]
    world.run("amarel_outage_rows.sh", extra={"DET_V2_NOW": "2026-11-10T08:00:00"})
    world.run("amarel_outage_rows.sh", extra={"DET_V2_NOW": "2026-11-10T09:00:00"})   # once per job id
    rows = [l for l in world.ledger() if l[9] == "outage"]
    assert len(rows) == 1 and rows[0][0] == GPU_RUN and rows[0][3] == "81" and rows[0][5] == "2026-11-10T08:00:00"
    assert rows[0][6] == "squeue:RUNNING"
    r = world.run("amarel_outage_rows.sh", check=False, extra={"DET_V2_NOW": "2026-11-11T23:59:00"})
    assert r.returncode != 0 and "outside" in r.stderr


def test_outage_rows_only_for_this_accounts_amarel_placements(world):
    gpu = world.run_dir(GPU_RUN)
    world.ledger_row(GPU_RUN, "amarel", "83", "gpu", gpu, user="other")
    _snapshot(world, f"{GPU_RUN} RUNNING\n")
    world.run("amarel_outage_rows.sh", extra={"DET_V2_NOW": "2026-11-10T08:30:00"})
    assert not [l for l in world.ledger() if l[9] == "outage"]


def _is_live(world, name):
    """is_live on the name's latest placement: 0 live, 1 not live, 2 the host could not be asked."""
    common = world.repo / "scripts" / "det_v2" / "common.sh"
    r = subprocess.run(["bash", "-c", f'source {common}; is_live "$(ledger_last_placed {name})"; echo "rc=$?"'],
                       env=world.env, capture_output=True, text=True, timeout=120)
    return int(r.stdout.strip().rsplit("rc=", 1)[-1])


def test_a_held_timeout_job_gets_an_outage_row_carrying_its_sacct_state(world):
    """Ended TIMEOUT between the 06:05 cancel and 08:00: no cancel row, not RUNNING in the last squeue snapshot.
    The last sacct snapshot (sync_amarel.sh, before Amarel went down) saw it end."""
    gpu, small = world.run_dir(GPU_RUN), world.run_dir(SMALL_RUN)
    world.bank(gpu, 4)
    world.ledger_row(GPU_RUN, "amarel", "91", "gpu", gpu)
    world.ledger_row(SMALL_RUN, "amarel", "92", "gpu", small)                     # still PENDING: no row
    (world.calls / "sacct_out").write_text(f"90|{GPU_RUN}|CANCELLED by 1\n91|{GPU_RUN}|TIMEOUT\n"
                                           f"92|{SMALL_RUN}|PENDING\n")
    world.run("sync_amarel.sh")
    world.run("amarel_outage_rows.sh", extra={"DET_V2_NOW": "2026-11-10T08:01:00"})
    rows = [l for l in world.ledger() if l[9] == "outage"]
    assert [(l[0], l[3], l[6]) for l in rows] == [(GPU_RUN, "91", "sacct:TIMEOUT")]
    _amarel_down(world)
    n = len(_amarel_calls(world))                                                  # the sync's, before 08:00
    assert _is_live(world, GPU_RUN) == 1 and len(_amarel_calls(world)) == n
    assert _is_live(world, SMALL_RUN) == 2                                         # no row: still probes


def test_an_ended_state_for_an_older_job_writes_no_row(world):
    gpu = world.run_dir(GPU_RUN)
    world.ledger_row(GPU_RUN, "amarel", "93", "gpu", gpu)                        # resumed since as job 93
    (world.calls / "sacct_out").write_text(f"91|{GPU_RUN}|TIMEOUT\n93|{GPU_RUN}|PENDING\n")
    world.run("sync_amarel.sh")
    world.run("amarel_outage_rows.sh", extra={"DET_V2_NOW": "2026-11-10T08:01:00"})
    assert not [l for l in world.ledger() if l[9] == "outage"]


def test_an_outage_row_resumes_from_the_ilab_copy_with_a_note(world):
    """Driver 07:3x: a job RUNNING at the window or ended TIMEOUT/PREEMPTED/NODE_FAIL before it, with no final pull,
    resumes from the iLab copy, repeating whatever only Amarel had; waiting for Amarel costs ~40 h of the row."""
    run_dir = world.run_dir(GPU_RUN)
    world.bank(run_dir, 3)
    world.ledger_row(GPU_RUN, "amarel", "95", "gpu", run_dir, user="other")
    _row_at(world, GPU_RUN, "amarel", "95", run_dir, "outage", "2026-11-10T08:10:00", user="other")
    _amarel_down(world)
    r = world.run("resume_ilab.sh", GPU_RUN, "1:00:00", "a4000")
    assert "resumed from iLab copy; Amarel may hold later epochs" in r.stderr
    assert world.ledger()[-1][2] == "ilab" and world.ledger()[-1][9] == "resume"
