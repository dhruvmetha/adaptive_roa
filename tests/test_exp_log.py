import json
import subprocess
import sys
from pathlib import Path

SCRIPT = "/common/home/st1122/Projects/adaptive_roa/scripts/exp_log.py"
PY = "/common/home/st1122/Projects/adaptive_roa/env/bin/python"


def _run(*args, cwd):
    return subprocess.run([PY, SCRIPT, *args], cwd=cwd, capture_output=True, text=True)


def test_append_then_report_roundtrip(tmp_path):
    log = tmp_path / "runs.jsonl"
    r = _run("append", "--log", str(log), "--run-id", "fm_high_epi_var",
             "--system", "pendulum_stoch", "--level", "high", "--predictor", "fm",
             "--arm", "epi_var", "--cluster", "amarel", "--job-id", "123",
             "--output-dir", "/scratch/x", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    rec = json.loads(log.read_text().strip())
    assert rec["run_id"] == "fm_high_epi_var"
    assert rec["status"] == "launched"
    assert len(rec["code_hash"]) == 12          # short content hash
    assert rec["launched_at"].startswith("20")


def test_update_status_rewrites_only_that_run(tmp_path):
    log = tmp_path / "runs.jsonl"
    for rid in ("a", "b"):
        _run("append", "--log", str(log), "--run-id", rid, "--system", "s",
             "--level", "high", "--predictor", "clf", "--arm", "total",
             "--cluster", "ilab", "--job-id", "1", "--output-dir", "/x", cwd=tmp_path)
    _run("update-status", "--log", str(log), "--run-id", "a", "--status", "preempted",
         cwd=tmp_path)
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert {r["run_id"]: r["status"] for r in recs} == {"a": "preempted", "b": "launched"}


def test_report_groups_by_status(tmp_path):
    log = tmp_path / "runs.jsonl"
    _run("append", "--log", str(log), "--run-id", "a", "--system", "s", "--level",
         "high", "--predictor", "clf", "--arm", "total", "--cluster", "ilab",
         "--job-id", "1", "--output-dir", "/x", cwd=tmp_path)
    out = _run("report", "--log", str(log), cwd=tmp_path).stdout
    assert "launched" in out and "a" in out


def test_missing_run_id_on_update_is_an_error(tmp_path):
    log = tmp_path / "runs.jsonl"
    log.write_text("")
    r = _run("update-status", "--log", str(log), "--run-id", "ghost",
             "--status", "done", cwd=tmp_path)
    assert r.returncode != 0
    assert "ghost" in (r.stderr + r.stdout)
