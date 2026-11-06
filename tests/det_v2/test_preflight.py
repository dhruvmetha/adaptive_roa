"""preflight.py finds data the shared group cannot read, and --fix repairs only what it owns."""
import grp
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from det_v2_world import GPU_RUN, ROOT

GROUP = grp.getgrgid(os.getgid()).gr_name      # a group the test can chgrp to


def _preflight(world, *args, env=None):
    return subprocess.run([sys.executable, str(world.repo / "scripts/det_v2/preflight.py"), *args, "--group", GROUP],
                          env=env or world.env, capture_output=True, text=True)


def _problems_in(out: str, root: Path) -> list[str]:
    """Problem lines whose own path (the last word before the reason) is under root."""
    hits = []
    for l in out.splitlines():
        if l.strip().split(" ")[0] in ("WOULD", "FIXED", "OUTSIDE", "NOT", "MISSING"):
            path = l.split("  (")[0].split()[-1]
            if path == str(root) or path.startswith(str(root) + "/"):
                hits.append(l)
    return hits


def test_dry_run_reports_and_fix_repairs(world):
    splits = world.data / "toy" / "splits"
    os.chmod(splits / "test_set.txt", 0o600)        # group cannot read
    os.chmod(splits, 0o700)                         # group cannot list or search
    os.chmod(world.data / "toy", 0o700)             # a parent the group cannot search
    before = {p: stat.S_IMODE(os.stat(p).st_mode) for p in (splits / "test_set.txt", splits, world.data / "toy")}
    r = _preflight(world, GPU_RUN)
    assert r.returncode == 1
    would = [l for l in r.stdout.splitlines() if "WOULD FIX" in l]
    assert any(f" {splits / 'test_set.txt'}  (group cannot read;" in l for l in would)
    assert any(f" {splits}  (group cannot read or search" in l for l in would)
    assert any(f" {world.data / 'toy'}  (group cannot search (parent)" in l for l in would)
    assert f"chmod g+r {splits / 'test_set.txt'}" in r.stdout and f"chmod g+rx {splits}" in r.stdout
    assert {p: stat.S_IMODE(os.stat(p).st_mode) for p in before} == before, "a dry run changed modes"

    r = _preflight(world, GPU_RUN, "--fix")
    assert r.stdout.count("FIXED") == 3
    assert os.stat(splits / "test_set.txt").st_mode & stat.S_IRGRP
    assert os.stat(splits).st_mode & stat.S_IRGRP and os.stat(splits).st_mode & stat.S_IXGRP
    assert os.stat(world.data / "toy").st_mode & stat.S_IXGRP
    again = _preflight(world, GPU_RUN)
    assert not _problems_in(again.stdout, world.data), again.stdout


def test_missing_files_and_list(world):
    (world.data / "toy" / "splits" / "train.npz").unlink()
    r = _preflight(world, GPU_RUN)
    assert r.returncode == 1 and f"MISSING {world.data / 'toy/splits/train.npz'}" in r.stdout
    r = _preflight(world, GPU_RUN, "--list")
    rows = [l.split("\t") for l in r.stdout.splitlines()]
    assert ["-", str(world.data / "toy/splits/train.npz")] in rows
    assert ["f", str(world.data / "toy/splits/test_set.txt")] in rows
    assert ["d", str(world.data / "toy/splits")] in rows


def test_rows_without_a_config_are_skipped(world):
    (world.repo / "configs/adaptive_v2/experiment/det_v2/runs" / f"{GPU_RUN}.yaml").unlink()
    r = _preflight(world)
    assert f"{GPU_RUN}: no config yet" in r.stdout


SHARED = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories")


@pytest.mark.skipif(not (SHARED / "deterministic").is_dir(), reason="shared data tree not mounted")
def test_composes_a_det_rerun_cell_like_run_adaptive(world, tmp_path):
    """The real configs: pendulum's splits live under lqr_failset, not under a base_dir."""
    env_file = tmp_path / "real.env"
    env_file.write_text(world.user_env.read_text().replace(f"DATA_DIR={world.data}", f"DATA_DIR={SHARED}"))
    env = {**world.env, "DET_V2_USER_ENV": str(env_file)}
    r = subprocess.run([sys.executable, str(world.repo / "scripts/det_v2/preflight.py"), "--list", "--code", str(ROOT),
                        "--overrides", "system=pendulum predictor=det_mlp_clf acquisition=ranked +experiment=det_rerun seed=99512"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    paths = dict((p, k) for k, p in (l.split("\t") for l in r.stdout.splitlines()))
    split = SHARED / "deterministic/pendulum/lqr_failset/splits_det_rerun"
    for f in ("pool_order_99512.txt", "pool_labels_99512.txt", "cal_set.txt", "val_set.txt", "test_set.txt", "train.npz"):
        assert paths.get(str(split / f)) == "f", f
