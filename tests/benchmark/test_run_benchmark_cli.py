"""The shipped command, end to end over a synthetic on-disk campaign.

Every other test module in this package exercises a library function.
Nothing exercised ``scripts/run_benchmark.py report`` itself -- which is
how the branch ended up with a reporting path that called none of the
provenance guards: ``build_report`` deliberately does not call
``validate_frame`` (see report.py), and the only shipped caller did not
either, so the check existed and ran nowhere.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from adaptive_roa.benchmark.provenance import git_sha

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_benchmark.py"
_PRE_FIX_COMMIT = "2b78b44"      # older than every INVALIDATING_COMMITS entry


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_benchmark_cli", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


def _write_run(root, run_id, *, arm, system="pendulum", acquisition="ranked",
               seed=42, n_epochs=2, commit=None, accuracy=0.8, complete=True):
    commit = commit or git_sha()
    for epoch in range(n_epochs):
        ed = root / run_id / f"epoch_{epoch:03d}"
        (ed / ".hydra").mkdir(parents=True)
        (ed / ".hydra" / "config.yaml").write_text(
            f"predictor:\n  name: {arm}\n"
            f"adaptive_v2:\n  system_name: {system}\n"
            f"seed: {seed}\nn_epochs: {n_epochs}\n")
        (ed / "artifacts_v2.json").write_text(json.dumps({
            "epoch": epoch, "sampling_mode": acquisition,
            "eval_metrics": {"accuracy": accuracy + 0.01 * epoch,
                             "f1": accuracy - 0.02 + 0.01 * epoch},
            "extra": {"commit": commit},
        }))
    if complete:
        (root / run_id / "final_results.json").write_text("{}")


def _campaign(root, *, commit=None, arms=("bnn_mfvi", "gp")):
    # Two arms x two acquisition levels, matched seeds and coverage.
    # bnn_mfvi is gated on INVALIDATING_COMMITS["bnn_mfvi"], which is what
    # makes the provenance test below able to fail at all.
    for i, arm in enumerate(arms):
        for j, acq in enumerate(("ranked", "direct")):
            _write_run(root, f"{arm}_{acq}", arm=arm, acquisition=acq,
                       commit=commit, accuracy=0.80 + 0.05 * i - 0.03 * j)
    return root


def _args(exp_root, **kw):
    base = dict(exp_root=exp_root, metric="accuracy", tier=None, system=None,
                epochs="final", control=None, require_complete=False,
                validate=True, out=None)
    base.update(kw)
    return type("Args", (), base)


def test_report_renders_a_synthetic_campaign(tmp_path, capsys):
    cli._run_report(_args(_campaign(tmp_path)))
    out = capsys.readouterr().out
    assert "Benchmark report" in out
    assert "| bnn_mfvi |" in out and "| gp |" in out
    assert "delta (ranked − direct)" in out


def test_report_checks_provenance_by_default(tmp_path):
    # THE follow-up this module exists for. Both arms' rows carry a commit
    # that predates the seeding fix; validate_frame must refuse before a
    # single table line is printed.
    _campaign(tmp_path, commit=_PRE_FIX_COMMIT)
    with pytest.raises(ValueError, match="predates|not found|unknown"):
        cli._run_report(_args(tmp_path))


def test_no_validate_skips_the_provenance_check_and_says_so(tmp_path, capsys):
    _campaign(tmp_path, commit=_PRE_FIX_COMMIT)
    cli._run_report(_args(tmp_path, validate=False))
    out = capsys.readouterr().out
    assert "provenance" in out and "NOT checked" in out
    assert "Benchmark report" in out


def test_report_refuses_an_arm_that_skipped_a_system(tmp_path):
    root = _campaign(tmp_path)
    for acq in ("ranked", "direct"):
        _write_run(root, f"bnn_mfvi_quad_{acq}", arm="bnn_mfvi", system="quadrotor3d",
                   acquisition=acq, accuracy=0.5)
    with pytest.raises(ValueError, match="do not cover the same"):
        cli._run_report(_args(root))


def test_the_system_filter_narrows_to_a_slice_every_arm_covers(tmp_path, capsys):
    root = _campaign(tmp_path)
    for acq in ("ranked", "direct"):
        _write_run(root, f"bnn_mfvi_quad_{acq}", arm="bnn_mfvi", system="quadrotor3d",
                   acquisition=acq, accuracy=0.5)
    cli._run_report(_args(root, system="pendulum"))
    out = capsys.readouterr().out
    assert "system: pendulum" in out
    assert "quadrotor3d" not in out


def test_an_unknown_system_is_refused_not_reported_as_empty(tmp_path):
    root = _campaign(tmp_path)
    with pytest.raises(SystemExit, match="systems present"):
        cli._run_report(_args(root, system="pendulm"))


def test_require_complete_refuses_an_unfinished_run(tmp_path):
    root = _campaign(tmp_path)
    (root / "bnn_mfvi_ranked" / "final_results.json").unlink()
    with pytest.raises(ValueError, match="final_results|run_complete"):
        cli._run_report(_args(root, require_complete=True))


def test_the_epoch_selection_reaches_build_report(tmp_path, capsys):
    root = _campaign(tmp_path)
    cli._run_report(_args(root, epochs="all"))
    assert "ALL epochs" in capsys.readouterr().out
    cli._run_report(_args(root, epochs="final"))
    assert "FINAL epoch" in capsys.readouterr().out


def test_epochs_argument_accepts_final_all_and_an_integer():
    assert cli._parse_epochs("final") == "final"
    assert cli._parse_epochs("all") == "all"
    assert cli._parse_epochs("9") == 9
    with pytest.raises(Exception, match="not an epoch selection"):
        cli._parse_epochs("last")


def test_an_empty_exp_root_short_circuits_before_the_guards(tmp_path, capsys):
    cli._run_report(_args(tmp_path))
    assert "no runs found" in capsys.readouterr().out
