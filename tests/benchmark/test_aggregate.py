import json
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from adaptive_roa.benchmark.aggregate import collect_runs
from adaptive_roa.benchmark.manifest import RunSpec

ADAPTIVE_V2_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs/adaptive_v2")


def _write_run(root, run_id, arm, n_epochs=2, commit="abc1234", seed=42):
    for e in range(n_epochs):
        ed = root / run_id / f"epoch_{e:03d}"
        (ed / ".hydra").mkdir(parents=True)
        (ed / ".hydra" / "config.yaml").write_text(
            f"predictor:\n  name: {arm}\nsystem:\n  name: pendulum\nseed: {seed}\n")
        (ed / "artifacts_v2.json").write_text(json.dumps({
            "epoch": e, "sampling_mode": "ranked",
            "eval_metrics": {"accuracy": 0.8 + 0.01 * e, "f1": 0.7},
            "extra": {"commit": commit},
        }))


# --- Step 1 tests, verbatim from the brief -----------------------------------

def test_one_row_per_run_epoch(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi", n_epochs=3)
    df = collect_runs(tmp_path)
    assert len(df) == 3
    assert set(df["epoch"]) == {0, 1, 2}


def test_arm_comes_from_the_config_not_the_directory_name(tmp_path):
    # A renamed directory must not relabel the arm.
    _write_run(tmp_path, "misleading_name_gp_reg", "bnn_laplace")
    df = collect_runs(tmp_path)
    assert set(df["arm"]) == {"bnn_laplace"}


def test_metrics_are_flattened_into_columns(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi")
    df = collect_runs(tmp_path)
    assert "accuracy" in df.columns and "f1" in df.columns
    assert df.loc[df.epoch == 1, "accuracy"].iloc[0] == pytest.approx(0.81)


def test_commit_provenance_is_carried_per_row(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi", commit="aaaaaaa")
    _write_run(tmp_path, "r2", "bnn_mfvi", commit="bbbbbbb")
    df = collect_runs(tmp_path)
    assert set(df["commit"]) == {"aaaaaaa", "bbbbbbb"}


def test_a_partial_run_contributes_only_its_finished_epochs(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi", n_epochs=2)
    (tmp_path / "r1" / "epoch_002").mkdir()      # preempted: dir but no artifact
    df = collect_runs(tmp_path)
    assert len(df) == 2


def test_empty_root_returns_an_empty_frame_not_a_crash(tmp_path):
    df = collect_runs(tmp_path)
    assert len(df) == 0


# --- Malformed-artifact decision: skipped, like a missing one ----------------
# See adaptive_roa/benchmark/aggregate.py's module docstring for the argument.
# Mirrors launcher.completed_epochs(), which already treats a truncated
# artifacts_v2.json as "not finished" rather than an error.

def test_a_malformed_artifact_is_skipped_not_raised(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi", n_epochs=2)
    bad_dir = tmp_path / "r1" / "epoch_002"
    (bad_dir / ".hydra").mkdir(parents=True)
    (bad_dir / ".hydra" / "config.yaml").write_text("predictor:\n  name: bnn_mfvi\n")
    (bad_dir / "artifacts_v2.json").write_text('{"epoch": 2, "eval_metrics": {"f1"')  # truncated
    df = collect_runs(tmp_path)   # must not raise
    assert len(df) == 2
    assert set(df["epoch"]) == {0, 1}


def test_an_empty_artifact_file_is_skipped_not_raised(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi", n_epochs=1)
    empty_dir = tmp_path / "r1" / "epoch_001"
    empty_dir.mkdir(parents=True)
    (empty_dir / "artifacts_v2.json").write_text("")
    df = collect_runs(tmp_path)
    assert len(df) == 1


# --- tier: the brief's column list requires it, but its own Step-3 sample ----
# code and tests never populate or check it. Nothing in config.yaml itself
# says "reference" -- it is recovered from the raw CLI overrides Hydra
# preserves in .hydra/overrides.yaml.

def test_tier_defaults_to_production_without_an_experiment_override(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi")
    df = collect_runs(tmp_path)
    assert set(df["tier"]) == {"production"}


def test_tier_is_recovered_from_the_reference_tier_override(tmp_path):
    ed = tmp_path / "r1" / "epoch_000"
    (ed / ".hydra").mkdir(parents=True)
    (ed / ".hydra" / "config.yaml").write_text("predictor:\n  name: hmc\nseed: 42\n")
    (ed / ".hydra" / "overrides.yaml").write_text(yaml.dump([
        "system=pendulum", "predictor=hmc", "seed=42", "n_epochs=10",
        "+experiment=reference_tier",
    ]))
    (ed / "artifacts_v2.json").write_text(json.dumps({
        "epoch": 0, "sampling_mode": "ranked", "eval_metrics": {}, "extra": {},
    }))
    df = collect_runs(tmp_path)
    assert set(df["tier"]) == {"reference"}


# --- arm: bare-string legacy predictor shape ---------------------------------
# adaptive_roa/probabilistic_classifier/export.py's resolve_predictor_name
# falls back to the family `type` tag when a dict-shaped predictor has no
# `name`; mirrored here rather than imported (that module pulls in torch and
# registers classifier subclasses as an import side effect).

def test_arm_falls_back_to_a_bare_string_predictor(tmp_path):
    ed = tmp_path / "r1" / "epoch_000"
    (ed / ".hydra").mkdir(parents=True)
    (ed / ".hydra" / "config.yaml").write_text("predictor: classifier\nseed: 42\n")
    (ed / "artifacts_v2.json").write_text(json.dumps({
        "epoch": 0, "sampling_mode": "ranked", "eval_metrics": {}, "extra": {},
    }))
    df = collect_runs(tmp_path)
    assert set(df["arm"]) == {"classifier"}


def test_arm_falls_back_to_predictor_type_when_name_is_absent(tmp_path):
    ed = tmp_path / "r1" / "epoch_000"
    (ed / ".hydra").mkdir(parents=True)
    (ed / ".hydra" / "config.yaml").write_text(
        "predictor:\n  type: classifier\nseed: 42\n")
    (ed / "artifacts_v2.json").write_text(json.dumps({
        "epoch": 0, "sampling_mode": "ranked", "eval_metrics": {}, "extra": {},
    }))
    df = collect_runs(tmp_path)
    assert set(df["arm"]) == {"classifier"}


# --- system: real run configs carry identity at adaptive_v2.system_name -----
# not at system.name (the real `system:` block only has _target_/dataset_dir).
# A system.name fallback is kept for schemas (including this file's simpler
# fixtures) that do record it there.

def test_system_is_recovered_from_adaptive_v2_system_name(tmp_path):
    ed = tmp_path / "r1" / "epoch_000"
    (ed / ".hydra").mkdir(parents=True)
    (ed / ".hydra" / "config.yaml").write_text(
        "predictor:\n  name: bnn_mfvi\n"
        "system:\n  _target_: adaptive_roa.systems.cartpole.CartPoleSystem\n"
        "adaptive_v2:\n  system_name: cartpole_pybullet\n"
        "seed: 42\n")
    (ed / "artifacts_v2.json").write_text(json.dumps({
        "epoch": 0, "sampling_mode": "ranked", "eval_metrics": {}, "extra": {},
    }))
    df = collect_runs(tmp_path)
    assert set(df["system"]) == {"cartpole_pybullet"}


# --- metrics: real eval_metrics nest the useful numbers ----------------------

def test_metrics_flatten_recursively_with_dotted_names(tmp_path):
    ed = tmp_path / "r1" / "epoch_000"
    (ed / ".hydra").mkdir(parents=True)
    (ed / ".hydra" / "config.yaml").write_text("predictor:\n  name: bnn_mfvi\nseed: 42\n")
    (ed / "artifacts_v2.json").write_text(json.dumps({
        "epoch": 0, "sampling_mode": "ranked",
        "eval_metrics": {
            "n_total": 1000,
            "lambda_delta": {"accuracy": 0.9, "f1": 0.85, "_doc": "skip me"},
            "fixed_threshold": {"accuracy": 0.7},
        },
        "extra": {"commit": "abc1234"},
    }))
    df = collect_runs(tmp_path)
    assert df.loc[0, "n_total"] == 1000
    assert df.loc[0, "lambda_delta.accuracy"] == pytest.approx(0.9)
    assert df.loc[0, "fixed_threshold.accuracy"] == pytest.approx(0.7)
    # Nested string / non-numeric leaves are dropped, not coerced.
    assert "lambda_delta._doc" not in df.columns


# --- n_epochs: not in the brief's column list, but Task 4's budget guard ----
# needs it and it is derivable directly from the run's own Hydra config (see
# RunSpec.hydra_overrides(), which always sets a top-level n_epochs=).

def test_n_epochs_is_read_from_the_run_config(tmp_path):
    ed = tmp_path / "r1" / "epoch_000"
    (ed / ".hydra").mkdir(parents=True)
    (ed / ".hydra" / "config.yaml").write_text(
        "predictor:\n  name: bnn_mfvi\nseed: 42\nn_epochs: 10\n")
    (ed / "artifacts_v2.json").write_text(json.dumps({
        "epoch": 0, "sampling_mode": "ranked", "eval_metrics": {}, "extra": {},
    }))
    df = collect_runs(tmp_path)
    assert df.loc[0, "n_epochs"] == 10


# --- Realistic end-to-end regression against the actual config tree ---------
# The hand-written fixtures above use simplified shapes; this drives a REAL
# RunSpec through REAL hydra.compose() against the actual configs/adaptive_v2
# tree (same pattern as test_manifest.py's test_hydra_overrides_actually_compose)
# so a schema drift in the real predictor/system configs is caught here rather
# than only by a fixture that encodes the same wrong assumption as the code.

@pytest.mark.parametrize("tier,expected_tier", [("production", "production"),
                                                  ("reference", "reference")])
def test_collect_runs_against_a_real_composed_hydra_config(tmp_path, tier, expected_tier):
    from hydra import compose, initialize_config_dir

    spec = RunSpec(arm="bnn_mfvi", system="pendulum", tier=tier,
                    acquisition="ranked", seed=42, n_epochs=10)
    overrides = spec.hydra_overrides()
    with initialize_config_dir(config_dir=ADAPTIVE_V2_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default", overrides=overrides)

    epoch_dir = tmp_path / spec.run_id / "epoch_000"
    (epoch_dir / ".hydra").mkdir(parents=True)
    (epoch_dir / ".hydra" / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    (epoch_dir / ".hydra" / "overrides.yaml").write_text(yaml.dump(overrides))
    (epoch_dir / "artifacts_v2.json").write_text(json.dumps({
        "epoch": 0, "sampling_mode": "ranked",
        "eval_metrics": {"lambda_delta": {"accuracy": 0.9}},
        "extra": {"commit": "deadbee"},
    }))

    df = collect_runs(tmp_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["arm"] == "bnn_mfvi"
    assert row["system"] == "pendulum"
    assert row["tier"] == expected_tier
    assert row["seed"] == 42
    assert row["n_epochs"] == 10
    assert row["lambda_delta.accuracy"] == pytest.approx(0.9)


# --- Empty campaign still yields guard-friendly columns ----------------------

def test_empty_root_still_carries_provenance_columns(tmp_path):
    df = collect_runs(tmp_path)
    for col in ("run_id", "arm", "system", "tier", "acquisition", "seed",
                "epoch", "n_epochs", "commit"):
        assert col in df.columns
