"""The shared ledger, the committed manifest, user.env sourcing, and no per-user paths."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from det_v2_world import SRC

sys.path.insert(0, str(SRC))
import det_v2_lib as lib  # noqa: E402
import make_manifest  # noqa: E402

COLUMNS = ["id", "name", "rank", "owner", "config", "script", "mem", "est_hours"]


def _bash(world, body: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f"set -euo pipefail\nsource {world.repo}/scripts/det_v2/common.sh\n{body}"],
                          env=world.env, capture_output=True, text=True)


# --- ledger ------------------------------------------------------------------------------

def test_ledger_append_writes_one_header_and_whole_rows_under_contention(world):
    n = 40
    body = " & ".join(f"ledger_append r{i} tester ilab {i} a4000 ts log c rd launch" for i in range(n)) + " & wait"
    r = _bash(world, body)
    assert r.returncode == 0, r.stderr
    lines = (world.exp / "det_v2" / "launched.tsv").read_text().splitlines()
    assert lines[0].split("\t") == lib.LEDGER_COLUMNS and len(lines) == n + 1
    assert sorted(l.split("\t")[0] for l in lines[1:]) == sorted(f"r{i}" for i in range(n))
    assert all(len(l.split("\t")) == 10 for l in lines)


def test_python_and_shell_appends_share_the_format(world):
    led = world.exp / "det_v2" / "launched.tsv"
    lib.append_ledger(led, {"name": "a", "user": "u", "where": "direct", "job": "1", "hostgpu": "h:0",
                            "timestamp": "t", "log": "l", "commit": "c", "run_dir": "/r", "action": "launch"})
    assert _bash(world, "ledger_append b tester ilab 2 a4000 ts log c /r2 resume").returncode == 0
    rows = lib.read_ledger(led)
    assert [r["name"] for r in rows] == ["a", "b"] and rows[1]["action"] == "resume"
    assert lib.latest_rows(led)["b"]["run_dir"] == "/r2"
    assert led.stat().st_mode & 0o020, "ledger must stay group-writable"


def test_ledger_rows_name_who_ran_what_where(world):
    world.run("launch.sh", "m02_cplqr_mlpclf_qa", "direct", "boxa", "2")
    row = dict(zip(lib.LEDGER_COLUMNS, world.ledger()[-1]))
    assert row["user"] == "tester" and row["where"] == "direct" and row["hostgpu"] == "boxa:2"
    assert row["job"] == "999999999" and row["commit"] == world.head() and row["action"] == "launch"
    assert row["log"].startswith(str(world.exp / "det_v2" / "logs")) and row["run_dir"] == str(world.run_dir("m02_cplqr_mlpclf_qa"))
    assert row["timestamp"][:4].isdigit()


# --- manifest ----------------------------------------------------------------------------

def _manifest():
    with open(SRC / "manifest.tsv") as f:
        return f.readline().rstrip("\n").split("\t"), list(csv.DictReader(open(SRC / "manifest.tsv"), delimiter="\t"))


def test_manifest_columns_rows_names_and_ranks():
    header, rows = _manifest()
    assert header == COLUMNS
    assert len(rows) == 306
    assert len({r["name"] for r in rows}) == 306
    assert sorted(int(r["rank"]) for r in rows) == list(range(1, 307))
    for r in rows:
        assert r["config"] == f"+experiment=det_v2/runs/{r['name']}"
        assert r["mem"][:-1].isdigit() and r["mem"].endswith("G")
        assert r["owner"] in ("Sumanth", "Dhruv")
        float(r["est_hours"])


def test_manifest_scripts_and_mem_follow_the_plan():
    plan = json.loads((SRC / "run_plan.json").read_text())
    runs = {r["name"]: r for r in plan["main"] + plan["abl"]}
    mem = make_manifest.read_mem_table(SRC / "det_rerun_mem.tsv")
    _, rows = _manifest()
    for r in rows:
        run = runs[r["name"]]
        assert run["rank"] == int(r["rank"]) and not run["reused"]
        partx = run.get("arm") == "Part-X"
        assert r["script"] == ("scripts/run_partx_faithful.py" if partx else "scripts/run_adaptive.py")
        key = (make_manifest.SETTING_TO_OLD[run["setting"]], make_manifest.MODEL_TO_ARM[run["model"]])
        assert r["mem"] == mem[key]
    assert sum(r["script"].endswith("partx_faithful.py") for r in rows) == 3
    todo = {n for n, run in runs.items() if run["rank"] is not None}
    assert todo == {r["name"] for r in rows}            # every ranked row, no reused one


def test_manifest_regenerates_byte_for_byte(tmp_path):
    out = tmp_path / "m.tsv"
    subprocess.run([sys.executable, str(SRC / "make_manifest.py"), "--out", str(out)], check=True, capture_output=True)
    assert out.read_text() == (SRC / "manifest.tsv").read_text()


DET_RERUN_MANIFEST = Path("/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun/manifest.tsv")


@pytest.mark.skipif(not DET_RERUN_MANIFEST.exists(), reason="needs the det_rerun manifest")
def test_mem_table_matches_the_det_rerun_manifest():
    assert make_manifest.extract_mem(DET_RERUN_MANIFEST) == make_manifest.read_mem_table(SRC / "det_rerun_mem.tsv")


# --- user.env ------------------------------------------------------------------------------

def test_user_env_is_sourced_with_defaults(world):
    r = _bash(world, 'echo "$NET_ID|$LEDGER|$LOG_DIR|$DET_V2_ROOT|$SHARED_DATA_BASE|$(umask)|$AMAREL_USER|$DET_V2_REPO"')
    assert r.returncode == 0, r.stderr
    net, ledger, logs, root, sdb, um, auser, repo = r.stdout.strip().split("|")
    assert net == "tester" and root == f"{world.exp}/det_v2"
    assert ledger == f"{root}/launched.tsv" and logs == f"{root}/logs"
    assert sdb == str(world.data) and um == "0002" and auser == "tester" and repo == str(world.repo)


def test_python_tools_read_the_same_user_env(world, monkeypatch):
    for k, v in world.env.items():
        monkeypatch.setenv(k, v)
    env = lib.load_user_env()
    assert env["NET_ID"] == "tester" and env["DET_V2_ROOT"] == f"{world.exp}/det_v2"
    assert env["AMAREL_EXP_DIR"] == str(world.amarel / "exp")


def test_missing_user_env_or_key_is_an_error(world, tmp_path):
    r = subprocess.run(["bash", str(world.repo / "scripts/det_v2/launch.sh"), "x", "ilab", "a4000"],
                       env={**world.env, "DET_V2_USER_ENV": str(tmp_path / "nope.env")}, capture_output=True, text=True)
    assert r.returncode != 0 and "user.env.example" in r.stderr
    bad = tmp_path / "bad.env"
    bad.write_text(world.user_env.read_text().replace("NET_ID=tester\n", ""))
    r = subprocess.run(["bash", "-c", f"source {world.repo}/scripts/det_v2/common.sh"],
                       env={**world.env, "DET_V2_USER_ENV": str(bad)}, capture_output=True, text=True)
    assert r.returncode != 0 and "NET_ID" in r.stderr


def test_example_documents_every_key_the_scripts_read():
    example = (SRC / "user.env.example").read_text()
    keys = {l.lstrip("#").split("=", 1)[0] for l in example.splitlines() if "=" in l and not l.startswith("# ")}
    common = (SRC / "common.sh").read_text()
    used = {"NET_ID", "PYTHON", "SNAPSHOT_DIR", "DATA_DIR", "SHARED_DATA_BASE", "EXP_DIR", "LEDGER", "LOG_DIR",
            "HOLD_HOSTS", "SHARED_GROUP", "TMP_ROOT", "ILAB_SUBMIT_HOST", "ILAB_PARTITION", "ILAB_EXCLUDE",
            "DIRECT_HOST_SUFFIX", "DIRECT_TASKSET", "DIRECT_JOBS_TSV", "SHARED_CARDS", "GPU_FREE", "SSH_CLEAR_AGENT",
            "AMAREL_HOST", "AMAREL_USER", "AMAREL_ACCOUNT", "AMAREL_CLONE", "AMAREL_PYTHON", "AMAREL_SNAPSHOT_DIR",
            "AMAREL_EXP_DIR", "AMAREL_DATA_DIR", "AMAREL_LOG_DIR", "AMAREL_PARTITION", "AMAREL_EXCLUDE", "AMAREL_CPUS"}
    assert used <= keys, used - keys
    for k in used:
        assert k in common, k


def test_user_env_is_gitignored():
    root = SRC.parents[1]
    r = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "scripts/det_v2/user.env"])
    assert r.returncode == 0
    r = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "scripts/det_v2/user.env.example"])
    assert r.returncode == 1


def test_no_det_v2_file_hardcodes_a_users_home():
    for f in sorted(SRC.iterdir()):
        if not f.is_file() or f.name == "user.env" or f.suffix == ".pyc":
            continue
        text = f.read_text()
        for bad in ("st1122", "/common/home/", "dm1487"):
            assert bad not in text, f"{f.name} mentions {bad}"
