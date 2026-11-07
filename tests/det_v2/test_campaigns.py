"""One harness, several campaigns: DET_CAMPAIGN (det_v2 by default; scripts/det_v3/ sets det_v3)
picks the shared root, ledger, logs, manifest, Amarel log dir and tmux session tag."""
import csv
import shutil
import subprocess
import sys

import pytest

from det_v2_world import GPU_RUN, MANIFEST, PARTX_RUN, ROOT, RUN_YAML, SMALL_RUN, git


def _env(world, campaign=None):
    extra = {} if campaign is None else {"DET_CAMPAIGN": campaign}
    r = subprocess.run(["bash", "-c", f"source {world.repo}/scripts/det_v2/common.sh && "
                        'echo "$DET_V2_ROOT|$LEDGER|$LOG_DIR|$MANIFEST|$AMAREL_LOG_DIR|$SESSION_TAG"'],
                       env={**world.env, **extra}, capture_output=True, text=True)
    return r


def _add_det_v3(world):
    """The world's repo gains the real det_v3 wrappers, a det_v3 manifest and det_v3 run configs."""
    r = world.repo
    d = r / "scripts" / "det_v3"
    d.mkdir()
    for f in (ROOT / "scripts" / "det_v3").iterdir():
        if f.suffix == ".sh" or f.name in ("status.py", "rebalance.py", "preflight.py", "run_dir.py"):
            shutil.copy2(f, d / f.name)
    (d / "manifest.tsv").write_text(MANIFEST.replace("+experiment=det_v2/", "+experiment=det_v3/"))
    runs = r / "configs" / "adaptive_v2" / "experiment" / "det_v3" / "runs"
    runs.mkdir(parents=True)
    for name, setting in ((GPU_RUN, "q3d-lqr"), (SMALL_RUN, "cp-lqr"), (PARTX_RUN, "cp-lqr")):
        (runs / f"{name}.yaml").write_text(RUN_YAML.replace("det_v2/", "det_v3/").format(setting=setting, name=name))
    (world.exp / "det_v3").mkdir()
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "det_v3")
    git(r, "push", "-q", "origin", "HEAD:main")


def _run_v3(world, script, *args, check=True):
    r = subprocess.run(["bash", str(world.repo / "scripts" / "det_v3" / script), *args],
                       env=world.env, capture_output=True, text=True, timeout=300)
    if check and r.returncode != 0:
        raise AssertionError(f"det_v3/{script} {args} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
    return r


def _rows(path):
    return list(csv.DictReader(open(path), delimiter="\t")) if path.exists() else []


def test_default_campaign_keeps_the_det_v2_paths(world):
    r = _env(world)
    assert r.returncode == 0, r.stderr
    root, ledger, logs, manifest, alogs, tag = r.stdout.strip().split("|")
    assert root == f"{world.exp}/det_v2" and ledger == f"{root}/launched.tsv" and logs == f"{root}/logs"
    assert manifest == f"{world.repo}/scripts/det_v2/manifest.tsv"
    assert alogs == str(world.amarel / "logs") and tag == ""


def test_det_v3_campaign_gets_its_own_root_manifest_logs_and_tag(world):
    r = _env(world, "det_v3")
    assert r.returncode == 0, r.stderr
    root, ledger, logs, manifest, alogs, tag = r.stdout.strip().split("|")
    assert root == f"{world.exp}/det_v3" and ledger == f"{root}/launched.tsv" and logs == f"{root}/logs"
    assert manifest == f"{world.repo}/scripts/det_v3/manifest.tsv"
    assert alogs == f"{world.amarel / 'logs'}/det_v3" and tag == "v3_"
    r = _env(world, "v3")
    assert r.returncode != 0 and "DET_CAMPAIGN" in r.stderr


def test_det_v3_launch_writes_only_the_det_v3_ledger_even_when_det_v2_has_the_name(world):
    _add_det_v3(world)
    world.ledger_row(GPU_RUN, "ilab", "77", "a4000", world.run_dir(GPU_RUN))      # a det_v2 run of the same name
    v2 = (world.exp / "det_v2" / "launched.tsv").read_text()
    out = _run_v3(world, "launch.sh", GPU_RUN, "ilab", "a4000")
    assert "iLab job" in out.stdout
    rows = _rows(world.exp / "det_v3" / "launched.tsv")
    assert [r["action"] for r in rows] == ["claim", "launch"]
    assert rows[-1]["run_dir"] == str(world.exp / "det_v3" / "q3d-lqr" / f"{GPU_RUN}_s7")
    assert (world.exp / "det_v2" / "launched.tsv").read_text() == v2       # det_v2 ledger untouched
    log = rows[-1]["log"]
    assert log.startswith(str(world.exp / "det_v3" / "logs"))


def test_det_v3_direct_launch_tags_its_tmux_session(world):
    _add_det_v3(world)
    out = _run_v3(world, "launch.sh", SMALL_RUN, "direct", "boxa", "2")
    assert f"session gpu_v3_{SMALL_RUN}" in out.stdout
    assert f"-s gpu_v3_{SMALL_RUN}" in (world.calls / "tmux").read_text()


def test_det_v3_python_wrappers_read_the_det_v3_campaign(world):
    _add_det_v3(world)
    _run_v3(world, "launch.sh", GPU_RUN, "ilab", "a4000")
    st = subprocess.run([sys.executable, str(world.repo / "scripts" / "det_v3" / "status.py")],
                        env=world.env, capture_output=True, text=True, timeout=300)
    assert st.returncode == 0, st.stderr
    line = next(l for l in st.stdout.splitlines() if GPU_RUN in l)
    assert "not launched" not in line                                       # it reads the det_v3 ledger
    v2 = subprocess.run([sys.executable, str(world.repo / "scripts" / "det_v2" / "status.py")],
                        env=world.env, capture_output=True, text=True, timeout=300)
    assert next(l for l in v2.stdout.splitlines() if GPU_RUN in l).split()[2:4] == ["not", "launched"]
    rd = subprocess.run([sys.executable, str(world.repo / "scripts" / "det_v3" / "run_dir.py"), GPU_RUN],
                        env=world.env, capture_output=True, text=True, timeout=300)
    assert rd.stdout.strip().endswith(f"det_v3/q3d-lqr/{GPU_RUN}_s7"), rd.stderr


@pytest.mark.parametrize("campaign,n", [("det_v2", 309), ("det_v3", 420)])
def test_manifests_regenerate_byte_for_byte_per_campaign(tmp_path, campaign, n):
    out = tmp_path / "m.tsv"
    subprocess.run([sys.executable, str(ROOT / "scripts/det_v2/make_manifest.py"), "--campaign", campaign,
                    "--out", str(out)], check=True, capture_output=True)
    assert out.read_text() == (ROOT / "scripts" / campaign / "manifest.tsv").read_text()
    rows = _rows(out)
    assert len(rows) == n and sorted(int(r["rank"]) for r in rows) == list(range(1, n + 1))
    assert all(r["config"] == f"+experiment={campaign}/runs/{r['name']}" for r in rows)


def test_det_v3_manifest_follows_its_plan_and_owner_split():
    import json
    plan = json.loads((ROOT / "scripts/det_v3/run_plan.json").read_text())
    runs = {r["name"]: r for r in plan["main"] + plan["abl"]}
    rows = _rows(ROOT / "scripts/det_v3/manifest.tsv")
    assert {r["name"] for r in rows} == set(runs)
    for r in rows:
        p = runs[r["name"]]
        assert int(r["rank"]) == p["rank"] and r["owner"] == p["owner"] and float(r["est_hours"]) == p["hours"]
        assert (ROOT / "configs/adaptive_v2/experiment/det_v3/runs" / f"{r['name']}.yaml").exists()
    owners = {o: sum(r["owner"] == o for r in rows) for o in ("Sumanth", "Dhruv")}
    assert owners == {"Sumanth": 140, "Dhruv": 280}
    assert {runs[n]["setting"] for n in runs if runs[n]["owner"] == "Sumanth"} == {"q3d-ppo", "q2d-ppo"}
