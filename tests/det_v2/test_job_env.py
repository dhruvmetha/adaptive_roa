"""Every launch and resume path runs its job with umask 002 and the det_v2 environment.

The stub world runs each path end to end (ssh, sbatch, tmux stubbed) and the job's
python dumps what it was started with. A path that bypassed job.sh, or a job.sh that
dropped a variable, fails here.
"""
import re
from pathlib import Path

import pytest

from det_v2_world import GPU_RUN, PARTX_RUN, SMALL_RUN, SRC

REQUIRED = {"Q3D_CACHE_TRANSFORMED": "1", "Q2D_CACHE_TRANSFORMED": "1", "PYTHONNOUSERSITE": "1"}


def _check(dump: dict, world, threads: str) -> None:
    assert dump["UMASK"] == "0002"
    for k, v in REQUIRED.items():
        assert dump.get(k) == v, k
    assert dump["OMP_NUM_THREADS"] == threads and dump["MKL_NUM_THREADS"] == threads
    assert dump["ADAPTIVE_ROA_COMMIT"] == world.head()
    assert dump["PYTHONPATH"] == dump["PWD"] and dump["PWD"].endswith("/" + world.head())
    tmpdir = Path(dump["TMPDIR"])
    assert tmpdir.parent == world.tmp / "tmproot" and tmpdir.name.startswith("tester_") and tmpdir.is_dir()


def _only_dump(world) -> dict:
    dumps = world.dumps_list()
    assert len(dumps) == 1, dumps
    return dumps[0]


@pytest.mark.parametrize("args, threads", [
    ((GPU_RUN, "ilab", "a4000"), "1"),
    ((SMALL_RUN, "direct", "boxa", "2"), "1"),
    ((GPU_RUN, "amarel"), "4"),
    ((PARTX_RUN, "ilabcpu"), "8"),
])
def test_each_launch_path(world, args, threads):
    world.run("launch.sh", *args)
    dump = _only_dump(world)
    _check(dump, world, threads)
    name = args[0]
    run_dir = world.run_dir(name)
    if args[1] == "amarel":       # runs on the scratch twin, from the Amarel snapshot
        run_dir = Path(str(run_dir).replace(str(world.exp), str(world.amarel / "exp")))
        assert dump["PWD"].startswith(str(world.amarel / "snap"))
    assert dump["ARGS"].split()[1:] == [f"+experiment=det_v2/runs/{name}", f"output_dir={run_dir}"]
    if args[1] == "direct":
        assert dump["CUDA_VISIBLE_DEVICES"] == "2"


@pytest.mark.parametrize("script, args, threads", [
    ("resume_ilab.sh", ("3-00:00:00", "a4000"), "1"),
    ("resume_direct.sh", ("boxa", "3"), "1"),
    ("resume_amarel.sh", (), "4"),
])
def test_each_resume_path(world, script, args, threads):
    run_dir = world.run_dir(GPU_RUN)
    world.bank(run_dir, 2)
    world.ledger_row(GPU_RUN, "direct", "999999999", "boxz:0", run_dir)
    world.run(script, GPU_RUN, *args)
    dump = _only_dump(world)
    _check(dump, world, threads)
    assert dump["ARGS"].split()[0] == "scripts/resume_adaptive.py"
    target = Path(dump["ARGS"].split()[2])
    assert target.name == run_dir.name and (target / "epoch_001" / "artifacts_v2.json").exists()


def test_job_sh_sets_everything_and_every_path_goes_through_it():
    job = (SRC / "job.sh").read_text()
    for line in ("umask 002", "export PYTHONNOUSERSITE=1", "export Q3D_CACHE_TRANSFORMED=1",
                 "export Q2D_CACHE_TRANSFORMED=1", "export MKL_NUM_THREADS=$OMP_NUM_THREADS",
                 'export PYTHONPATH="$DET_V2_CODE"', "export ADAPTIVE_ROA_COMMIT", "export TMPDIR="):
        assert line in job, line
    common = (SRC / "common.sh").read_text()
    for fn in ("ilab_submit", "amarel_submit", "start_direct"):
        body = common.split(f"{fn}() {{", 1)[1].split("\n}\n", 1)[0]
        assert "scripts/det_v2/job.sh" in body, fn
        assert "umask 002" in body, fn
    for script in ("launch.sh", "resume_ilab.sh", "resume_direct.sh", "resume_amarel.sh"):
        text = (SRC / script).read_text()
        assert "source \"$(dirname \"$0\")/common.sh\"" in text
        assert re.search(r"\b(ilab_submit|amarel_submit|start_direct)\b", text), script
        assert "sbatch" not in text.replace("sbatch failed", ""), f"{script} submits around job.sh"
    assert "\numask 002\n" in common


def test_amarel_snapshot_env_points_at_amarel(world):
    world.run("launch.sh", GPU_RUN, "amarel")
    env = (world.amarel / "snap" / world.head() / ".env").read_text()
    assert f"EXP_DIR={world.amarel / 'exp'}" in env and f"DATA_DIR={world.amarel / 'data'}" in env
    assert f"SHARED_DATA_BASE={world.amarel / 'data'}" in env
