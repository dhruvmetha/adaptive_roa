"""Fixes from the driver's harness review of d0875c6..dc13c45 (2026-11-06)."""
import shutil
import subprocess
from pathlib import Path

import numpy as np

from det_v2_world import GPU_RUN, PARTX_RUN, ROOT


def test_resume_ilab_uses_the_short_wall_and_the_placement_gres(world):
    run_dir = world.run_dir(GPU_RUN)
    world.bank(run_dir, 2)
    world.ledger_row(GPU_RUN, "ilab", "77", "a4000", run_dir)
    # A later move attempt that gave up leaves an abandon row carrying host:gpu.
    world.ledger_row(GPU_RUN, "direct", "-", "boxa:3", run_dir, action="abandon")
    (world.calls / "squeue_out").write_text("CANCELLED\n")
    world.run("resume_ilab.sh", GPU_RUN)
    submit = (world.calls / "ssh").read_text().splitlines()[-1]
    assert "--gres=gpu:a4000:1" in submit and "boxa" not in submit
    assert "--time=06:00:00" in submit


def test_rsync_pulls_omit_directory_times():
    common = (ROOT / "scripts/det_v2/common.sh").read_text()
    sync = (ROOT / "scripts/det_v2/sync_amarel.sh").read_text()
    pull = [l for l in common.splitlines() if l.strip().startswith("rsync")]
    assert pull and all("-rlptDO" in l for l in pull)
    assert "RSYNC_PERMS=(-rlptDO" in sync


def test_partx_rerun_never_claims_over_an_in_flight_claim(world):
    world.run("launch.sh", PARTX_RUN, "ilabcpu")
    run_dir = world.run_dir(PARTX_RUN)
    world.ledger_row(PARTX_RUN, "ilabcpu", "-", "cpu", run_dir, action="claim", user="other")
    r = world.run("launch.sh", PARTX_RUN, "ilabcpu", check=False)
    assert r.returncode != 0 and "already in the ledger" in r.stderr
    # Once that claim ends, the rerun goes through.
    world.ledger_row(PARTX_RUN, "ilabcpu", "-", "cpu", run_dir, action="abandon", user="other")
    world.run("launch.sh", PARTX_RUN, "ilabcpu")


def test_user_env_cannot_unexclude_gpu029_or_pick_cgpu(world):
    def source(**over):
        env = {**world.env, **over}
        return subprocess.run(["bash", "-c", f"source {world.repo}/scripts/det_v2/common.sh && echo OK"],
                              env=env, capture_output=True, text=True)
    assert source().stdout.strip().endswith("OK")
    r = source(AMAREL_EXCLUDE="gpu046")
    assert r.returncode != 0 and "gpu029" in r.stderr
    r = source(AMAREL_PARTITION="cgpu")
    assert r.returncode != 0 and "cgpu" in r.stderr


def test_amarel_launch_refuses_a_missing_data_directory(world):
    import sys
    out = subprocess.run([sys.executable, str(world.repo / "scripts/det_v2/preflight.py"), GPU_RUN, "--list",
                          "--code", str(world.repo)], env=world.env, capture_output=True, text=True, check=True).stdout
    dirs = [l.split("\t", 1)[1] for l in out.splitlines() if l.startswith("d\t")]
    assert dirs, out
    staged = Path(dirs[0].replace(str(world.data), str(world.amarel / "data"), 1))
    assert staged.is_dir()
    shutil.rmtree(staged)
    r = world.run("launch.sh", GPU_RUN, "amarel", check=False)
    assert r.returncode != 0 and "not staged on Amarel" in r.stderr and str(staged) in r.stderr


def test_split_states_write_leaves_no_temp_file(tmp_path):
    from adaptive_roa.adaptive_v2.eval import prediction_store as ps
    X, y = np.arange(6, dtype=np.float32).reshape(3, 2), np.array([1, -1, 1])
    ps.write_split_states_once(tmp_path, "val", X, y)
    ps.write_split_states_once(tmp_path, "val", X * 0, y)       # first write wins
    assert sorted(p.name for p in tmp_path.iterdir()) == ["prediction_store_states_val.npz"]
    z = np.load(tmp_path / "prediction_store_states_val.npz")
    assert np.array_equal(z["states"], X) and np.array_equal(z["true_labels"], y)
