"""status.py counts from disk; snapshot.sh freezes a commit; sync_amarel.sh mirrors safely."""
import os
import stat
import subprocess
import sys
import time

from det_v2_world import GPU_RUN, PARTX_RUN, SMALL_RUN


def _status(world, *args):
    r = subprocess.run([sys.executable, str(world.repo / "scripts/det_v2/status.py"), *args], env=world.env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _tsv(world):
    lines = _status(world, "--tsv").splitlines()
    head = lines[0].split("\t")
    return {r[1]: dict(zip(head, r)) for r in (l.split("\t") for l in lines[1:])}


def test_finished_only_when_all_eleven_epochs_are_on_disk(world):
    gpu, small = world.run_dir(GPU_RUN), world.run_dir(SMALL_RUN)
    world.ledger_row(GPU_RUN, "ilab", "11", "a4000", gpu)
    world.ledger_row(SMALL_RUN, "direct", "999999999", "boxa:1", small)
    world.bank(gpu, 11)
    world.bank(small, 11)
    (small / "epoch_004" / "artifacts_v2.json").unlink()          # a hole: 10 of 11
    (small / "epoch_011").mkdir()
    (small / "epoch_011" / "artifacts_v2.json").write_text("{}")  # beyond epoch_010 does not count
    s = _tsv(world)
    assert s[GPU_RUN]["state"] == "finished" and s[GPU_RUN]["banked"] == "11"
    assert s[SMALL_RUN]["state"] == "running" and s[SMALL_RUN]["banked"] == "10"
    assert s[PARTX_RUN]["state"] == "not launched" and s[PARTX_RUN]["where"] == "-"
    assert s[SMALL_RUN]["where"] == "direct" and s[SMALL_RUN]["hostgpu"] == "boxa:1" and s[SMALL_RUN]["user"] == "tester"


def test_states_from_the_ledger_and_the_amarel_queue(world):
    gpu = world.run_dir(GPU_RUN)
    world.ledger_row(GPU_RUN, "amarel", "321", "gpu", gpu)
    world.ledger_row(SMALL_RUN, "ilab", "-", "a4000", world.run_dir(SMALL_RUN), action="claim")
    world.ledger_row(PARTX_RUN, "ilabcpu", "-", "cpu", world.run_dir(PARTX_RUN), action="abandon")
    s = _tsv(world)
    assert s[GPU_RUN]["state"] == "launched" and s[SMALL_RUN]["state"] == "claimed"
    assert s[PARTX_RUN]["state"] == "not launched"
    logs = world.exp / "det_v2" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "amarel_queue_other.txt").write_text(f"{GPU_RUN} PENDING\n")
    assert _tsv(world)[GPU_RUN]["state"] == "queued"
    old = time.time() - 3600
    os.utime(logs / "amarel_queue_other.txt", (old, old))           # stale snapshot says nothing
    assert _tsv(world)[GPU_RUN]["state"] == "launched"


def test_a_silent_run_reads_stalled(world):
    small = world.run_dir(SMALL_RUN)
    world.ledger_row(SMALL_RUN, "direct", "999999999", "boxa:1", small)
    world.bank(small, 3)
    old = time.time() - 4 * 3600
    for f in small.rglob("*"):
        os.utime(f, (old, old))
    (small / "epoch_003").mkdir()
    (small / "epoch_003" / "metrics.csv").write_text("x")
    os.utime(small / "epoch_003" / "metrics.csv", (old, old))
    assert _tsv(world)[SMALL_RUN]["state"] == "stalled?"


def test_an_unledgered_run_dir_is_found_by_name(world):
    world.bank(world.run_dir(SMALL_RUN), 11)
    s = _tsv(world)
    assert s[SMALL_RUN]["state"] == "finished" and s[SMALL_RUN]["run_dir"] == str(world.run_dir(SMALL_RUN))
    out = _status(world)
    assert "finished=1" in out and "not launched=2" in out


def test_snapshot_is_read_only_and_reused(world):
    snap = world.repo / "scripts/det_v2/snapshot.sh"
    run = lambda: subprocess.run(["bash", str(snap)], env=world.env, capture_output=True, text=True, check=True).stdout.strip()
    code = run()
    assert code == str(world.tmp / "snap" / world.head())
    assert open(f"{code}/COMMIT").read().strip() == world.head()
    env = open(f"{code}/.env").read()
    assert "NET_ID=tester" in env and f"EXP_DIR={world.exp}" in env and f"DATA_DIR={world.data}" in env
    assert not os.path.exists(f"{code}/.git") and not os.path.exists(f"{code}/scripts/det_v2/user.env")
    assert not os.stat(code).st_mode & stat.S_IWUSR and os.stat(code).st_mode & stat.S_IRGRP
    assert run() == code and not list((world.tmp / "snap").glob(".tmp_*"))


def _sync(world):
    r = world.run("sync_amarel.sh")
    return r.stdout


def test_sync_mirrors_this_accounts_amarel_runs_with_hydra_and_group_bits(world):
    gpu = world.run_dir(GPU_RUN)
    agpu = world.amarel / "exp" / gpu.relative_to(world.exp)
    world.bank(agpu, 2)
    (agpu / ".hydra").mkdir()
    (agpu / ".hydra" / "config.yaml").write_text("a: 1\n")
    os.chmod(agpu / ".hydra" / "config.yaml", 0o600)
    world.ledger_row(GPU_RUN, "amarel", "5", "gpu", gpu)
    world.ledger_row(SMALL_RUN, "amarel", "6", "gpu", world.run_dir(SMALL_RUN), user="other")
    out = _sync(world)
    assert f"synced {GPU_RUN}" in out and SMALL_RUN not in out
    assert (gpu / ".amarel_mirror").exists() and (gpu / ".hydra" / "config.yaml").read_text() == "a: 1\n"
    assert stat.S_IMODE(os.stat(gpu / ".hydra" / "config.yaml").st_mode) & 0o060 == 0o060
    rs = (world.calls / "rsync").read_text().splitlines()
    assert all("-rlptD" in l and "--chmod=Dg+rwxs,Fg+rw" in l for l in rs)
    cell = [l for l in rs if "amarel_logs" not in l]
    assert len(cell) == 1 and "--delete" in cell[0] and "--append" not in cell[0]
    assert (world.exp / "det_v2" / "logs" / "amarel_queue_tester.txt").exists()
    # a resumed cell dropped its crashed partial epoch: the mirror follows (sole writer)
    (gpu / "epoch_002").mkdir()
    (gpu / "epoch_002" / "partial").write_text("x")
    _sync(world)
    assert not (gpu / "epoch_002").exists()


def test_sync_refuses_a_dir_it_does_not_own(world):
    gpu = world.run_dir(GPU_RUN)
    world.bank(world.amarel / "exp" / gpu.relative_to(world.exp), 1)
    world.bank(gpu, 3)                                             # a local run, no marker
    world.ledger_row(GPU_RUN, "amarel", "5", "gpu", gpu)
    assert f"REFUSED {GPU_RUN}" in _sync(world)
    assert len(list(gpu.glob("epoch_*"))) == 3
