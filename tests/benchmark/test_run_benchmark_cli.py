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
import yaml

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
                validate=True, out=None, separatrix=False, separatrix_k=3,
                fidelity_vs=None, rhat_threshold=1.1)
    base.update(kw)
    return type("Args", (), base)


def _add_per_point(root, run_id, *, epoch, probs_sign=1.0, diagnostics=None):
    """The per-point artifacts --separatrix / --fidelity-vs read off disk."""
    import numpy as np

    grid = np.linspace(-1.0, 1.0, 40)
    directory = root / run_id / f"epoch_{epoch:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    probs = np.where(grid * probs_sign > 0, 0.9, 0.1)
    np.savez(directory / "full_roa_per_point.npz",
             start_states=np.stack([grid, np.zeros_like(grid)], axis=1),
             p_success=probs, p_failure=1.0 - probs,
             p_invalid=np.zeros_like(probs),
             true_labels=np.where(grid > 0, 1, -1))
    if diagnostics is not None:
        (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
        (directory / "checkpoints" / "hmc_diagnostics.json").write_text(
            json.dumps(diagnostics))


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


def test_the_written_artifact_itself_records_that_validation_was_skipped(tmp_path):
    # NEW-2: the stdout warning is not attached to the file that gets saved,
    # read weeks later and pasted into a thread. grep the artifact.
    root = tmp_path / "campaign"
    root.mkdir()
    _campaign(root, commit=_PRE_FIX_COMMIT)
    out_path = tmp_path / "report.md"
    cli._run_report(_args(root, validate=False, out=out_path))
    text = out_path.read_text()
    assert "PROVENANCE NOT CHECKED" in text
    assert "--no-validate" in text


def test_a_validated_written_artifact_carries_no_such_banner(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    _campaign(root)
    out_path = tmp_path / "report.md"
    cli._run_report(_args(root, out=out_path))
    assert "PROVENANCE NOT CHECKED" not in out_path.read_text()


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


# ---------------------------------------------------------------------------
# I5: Tasks 5 and 6 reachable from a real command. Before this, separatrix.py
# was imported by nothing outside its own test and fidelity_vs_reference had
# zero call sites anywhere -- two scientific deliverables that no shipped
# command could produce.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NEW-4: a typo'd GROUP NAME in a manifest `overrides:` entry. expand_manifest
# validates a group's VALUE (system=cartpole) but cannot judge a KEY: a
# non-dotted key that is not a group directory is legitimately a config value
# (seed=, n_epochs=, controller=, exp_id=), and only Hydra's struct-mode check
# can say whether it exists. That check used to happen inside SLURM, per job.
# ---------------------------------------------------------------------------

def _launch_args(manifest, exp_root, **kw):
    base = dict(manifest=manifest, exp_root=exp_root, launch=False)
    base.update(kw)
    return type("Args", (), base)


def _manifest(tmp_path, **extra):
    body = dict(tier="production", arms=["gp"], systems=["pendulum"],
                acquisition=["ranked"], seeds=[42], n_epochs=10)
    body.update(extra)
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


def test_a_valid_manifest_composes_and_dry_runs(tmp_path, capsys):
    manifest = _manifest(tmp_path)
    cli._run_launch(_launch_args(manifest, tmp_path / "exp"))
    out = capsys.readouterr().out
    assert "dry run" in out
    assert "sbatch" in out          # printed, never executed


def test_a_typod_override_key_is_refused_before_anything_is_submitted(tmp_path):
    # `acquisiton=` names no config group, so it is treated as a config
    # VALUE -- and there is no such key. Hydra refuses it, and now that
    # refusal happens here rather than ~450 times inside SLURM.
    manifest = _manifest(tmp_path, overrides=["acquisiton=ranked"])
    with pytest.raises(SystemExit, match="does not compose"):
        cli._run_launch(_launch_args(manifest, tmp_path / "exp"))


def test_a_typod_override_value_is_still_caught_earlier_by_the_manifest(tmp_path):
    # The group EXISTS, so expand_manifest's file check fires first and gives
    # the better message (it lists the groups that do exist).
    manifest = _manifest(tmp_path, overrides=["acquisition=rankd"])
    with pytest.raises(ValueError, match="no acquisition/rankd.yaml"):
        cli._run_launch(_launch_args(manifest, tmp_path / "exp"))


def test_the_shipped_pilot_manifest_passes_the_launch_time_compose_check(tmp_path):
    pilot = Path(cli.__file__).resolve().parents[1] / "configs/benchmark/pilot.yaml"
    cli._run_launch(_launch_args(pilot, tmp_path / "exp"))


def test_separatrix_flag_adds_the_near_boundary_section(tmp_path, capsys):
    root = _campaign(tmp_path)
    for run_id in ("bnn_mfvi_ranked", "bnn_mfvi_direct", "gp_ranked", "gp_direct"):
        _add_per_point(root, run_id, epoch=1)
    cli._run_report(_args(root, separatrix=True))
    out = capsys.readouterr().out
    assert "Near-boundary conditioned accuracy" in out
    assert "near boundary" in out
    assert "not a decomposition of it" in out


def test_without_the_flag_no_near_boundary_section_is_rendered(tmp_path, capsys):
    root = _campaign(tmp_path)
    for run_id in ("bnn_mfvi_ranked", "bnn_mfvi_direct", "gp_ranked", "gp_direct"):
        _add_per_point(root, run_id, epoch=1)
    cli._run_report(_args(root))
    assert "Near-boundary" not in capsys.readouterr().out


def test_a_run_missing_its_per_point_artifact_is_named_not_dropped(tmp_path, capsys):
    root = _campaign(tmp_path)
    for run_id in ("bnn_mfvi_ranked", "bnn_mfvi_direct", "gp_ranked"):
        _add_per_point(root, run_id, epoch=1)   # gp_direct writes nothing
    cli._run_report(_args(root, separatrix=True))
    out = capsys.readouterr().out
    assert "Runs refused while computing the band" in out
    assert "full_roa_per_point.npz" in out


def test_fidelity_flag_adds_the_posterior_fidelity_table(tmp_path, capsys):
    root = tmp_path
    _write_run(root, "hmc_r", arm="hmc", acquisition="ranked")
    _write_run(root, "bnn_r", arm="bnn_mfvi", acquisition="ranked")
    _add_per_point(root, "hmc_r", epoch=1,
                   diagnostics={"rhat_max": 1.02, "converged": True})
    _add_per_point(root, "bnn_r", epoch=1)
    cli._run_report(_args(root, fidelity_vs="hmc"))
    out = capsys.readouterr().out
    assert "Posterior fidelity vs the HMC reference" in out
    assert "| bnn_mfvi |" in out


def test_fidelity_withholds_against_a_non_converged_reference(tmp_path, capsys):
    root = tmp_path
    _write_run(root, "hmc_r", arm="hmc", acquisition="ranked")
    _write_run(root, "bnn_r", arm="bnn_mfvi", acquisition="ranked")
    _add_per_point(root, "hmc_r", epoch=1,
                   diagnostics={"rhat_max": 91.4, "converged": False})
    _add_per_point(root, "bnn_r", epoch=1, probs_sign=-1.0)
    cli._run_report(_args(root, fidelity_vs="hmc"))
    out = capsys.readouterr().out
    assert "withheld" in out and "91.4" in out
