"""DIRECT_TMP_ROOT="host=/path ..." moves one direct box's per-job TMPDIR off its /tmp (arrakis's
/tmp tmpfs fills to 100% and kills checkpoints with ENOSPC). Only start_direct reads it: other
direct hosts and every iLab and Amarel job keep TMP_ROOT, since Amarel can't create a /common
path. A malformed entry refuses the direct launch before tmux starts."""
from pathlib import Path

import pytest

from det_v2_world import GPU_RUN, PARTX_RUN, SMALL_RUN


def _map(world, value):
    with open(world.user_env, "a") as f:
        f.write(f'DIRECT_TMP_ROOT="{value}"\n')


def _tmp_parent(world) -> Path:
    dumps = world.dumps_list()
    assert len(dumps) == 1, dumps
    return Path(dumps[0]["TMPDIR"]).parent


def _cmd_files(world):
    return sorted((world.exp / "det_v2" / "logs" / "cmd").glob("*.sh"))


@pytest.mark.parametrize("script, args, bank", [
    ("launch.sh", (SMALL_RUN, "direct", "boxa", "2"), 0),
    ("resume_direct.sh", (GPU_RUN, "boxa", "3"), 2),
])
def test_a_mapped_direct_host_gets_its_tmp_root(world, script, args, bank):
    nfs = world.tmp / "nfs_tmp"
    _map(world, f"boxq=/elsewhere boxa={nfs}")
    if bank:
        world.bank(world.run_dir(args[0]), bank)
        world.ledger_row(args[0], "direct", "999999999", "boxz:0", world.run_dir(args[0]))
    world.run(script, *args)
    assert _tmp_parent(world) == nfs
    [cmd] = _cmd_files(world)
    assert f"DET_V2_TMP_ROOT={nfs} " in cmd.read_text()


@pytest.mark.parametrize("script, args, bank", [
    ("launch.sh", (SMALL_RUN, "direct", "boxb", "2"), 0),
    ("launch.sh", (GPU_RUN, "ilab", "a4000"), 0),
    ("launch.sh", (GPU_RUN, "amarel"), 0),
    ("launch.sh", (PARTX_RUN, "ilabcpu"), 0),
    ("resume_direct.sh", (GPU_RUN, "boxb", "3"), 2),
    ("resume_ilab.sh", (GPU_RUN, "3-00:00:00", "a4000"), 2),
    ("resume_amarel.sh", (GPU_RUN,), 2),
])
def test_unmapped_hosts_and_every_slurm_job_keep_tmp_root(world, script, args, bank):
    _map(world, f"boxa={world.tmp / 'nfs_tmp'}")
    if bank:
        world.bank(world.run_dir(args[0]), bank)
        world.ledger_row(args[0], "direct", "999999999", "boxz:0", world.run_dir(args[0]))
    world.run(script, *args)
    assert _tmp_parent(world) == world.tmp / "tmproot"
    sb = world.calls / "sbatch"
    if sb.exists():
        assert f"DET_V2_TMP_ROOT={world.tmp / 'tmproot'}" in sb.read_text()
        assert "nfs_tmp" not in sb.read_text()


@pytest.mark.parametrize("entry", ["boxa", "=/x", "boxa=", "boxa=relative/dir", "boxa=b=/c"])
@pytest.mark.parametrize("host", ["boxa", "boxb"])
def test_a_malformed_entry_refuses_the_direct_launch(world, entry, host):
    _map(world, f"boxq=/fine {entry}")
    r = world.run("launch.sh", SMALL_RUN, "direct", host, "2", check=False)
    assert r.returncode != 0 and "DIRECT_TMP_ROOT" in r.stderr, r.stderr
    assert world.dumps_list() == [] and _cmd_files(world) == []
    tmux = world.calls / "tmux"
    assert not tmux.exists() or "new-session" not in tmux.read_text()
    assert [row[-1] for row in world.ledger()[1:]] == ["claim", "abandon"]


def test_a_malformed_map_does_not_touch_ilab_jobs(world):
    _map(world, "boxa")
    world.run("launch.sh", GPU_RUN, "ilab", "a4000")
    assert _tmp_parent(world) == world.tmp / "tmproot"
