"""A queued Amarel resume reads "queued", not "stalled?": its silence is expected.

Resumes wait hours for an Amarel slot, and status.py judged stalls from log silence
alone, so every cycle flagged them (2026-11-02).
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _mod():
    spec = importlib.util.spec_from_file_location("det_status", ROOT / "scripts/det_rerun/status.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell(here: Path, name: str = "dr_q3d200_bnnclf_ranked_s99512") -> str:
    """A 20-epoch Amarel cell with two banked epochs, silent for three hours."""
    out = here / "out"
    old = time.time() - 3 * 3600
    for e in range(2):
        (out / f"epoch_{e:03d}").mkdir(parents=True)
        art = out / f"epoch_{e:03d}" / "artifacts_v2.json"
        art.write_text(json.dumps({"train_trajectories": 10000 * (e + 1)}))
        os.utime(art, (old + 60 * e, old + 60 * e))
    partial = out / "epoch_002" / "metrics.csv"
    partial.parent.mkdir()
    partial.write_text("x\n")
    os.utime(partial, (old, old))
    (here / "manifest.tsv").write_text(f"name\tarm\toverrides\n{name}\tdet_bnn_clf\toutput_dir={out} n_epochs=20\n")
    (here / "launched.tsv").write_text(
        f"{name}\tamarel\t--partition=gpu\t63454758\t20261102_1954\t{here}/not_started.out\t907fefc\n")
    return name


def _run(mod, here, monkeypatch, capsys):
    monkeypatch.setattr(mod, "HERE", here)
    monkeypatch.setattr(sys, "argv", ["status.py"])
    mod.main()
    return capsys.readouterr().out


def test_a_pending_resume_reads_queued(tmp_path, monkeypatch, capsys):
    mod, name = _mod(), _cell(tmp_path)
    (tmp_path / "amarel_queue.txt").write_text(f"{name} PENDING\nother_cell RUNNING\n")
    out = _run(mod, tmp_path, monkeypatch, capsys)
    assert "queued" in out and "stalled?" not in out and "queued=1" in out


def test_without_a_fresh_snapshot_the_silence_still_reads_stalled(tmp_path, monkeypatch, capsys):
    mod, name = _mod(), _cell(tmp_path)
    assert "stalled?=1" in _run(mod, tmp_path, monkeypatch, capsys)       # no snapshot
    snap = tmp_path / "amarel_queue.txt"
    snap.write_text(f"{name} PENDING\n")
    old = time.time() - 2 * mod.QUEUE_MAX_AGE
    os.utime(snap, (old, old))
    assert "stalled?=1" in _run(mod, tmp_path, monkeypatch, capsys)       # stale snapshot
    snap.write_text(f"{name} RUNNING\n")
    assert "stalled?=1" in _run(mod, tmp_path, monkeypatch, capsys)       # running and silent


def test_amarel_pending_parses_only_pending_rows(tmp_path):
    mod = _mod()
    snap = tmp_path / "q.txt"
    snap.write_text("a PENDING\nb RUNNING\n\nc PENDING\nmalformed\n")
    assert mod.amarel_pending(snap) == {"a", "c"}
    assert mod.amarel_pending(tmp_path / "missing.txt") == set()
