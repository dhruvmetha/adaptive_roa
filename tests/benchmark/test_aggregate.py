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
                "epoch", "n_epochs", "n_epochs_collected", "run_complete",
                "commit"):
        assert col in df.columns


# --- n_epochs_collected: the corrupt-INTERIOR-epoch gap ----------------------
# run_complete answers "did this run reach the end". It cannot answer "did it
# lose an epoch on the way": a corrupt interior artifact, or an epoch on which
# eval did not run, is skipped above and the run still writes
# final_results.json, so run_complete stays True while the population any
# reported mean is taken over silently shrinks.

def test_n_epochs_collected_counts_the_rows_a_run_actually_contributed(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi", n_epochs=3)
    df = collect_runs(tmp_path)
    assert set(df["n_epochs_collected"]) == {3}


def test_n_epochs_collected_records_a_lost_interior_epoch(tmp_path):
    # epoch_001 is corrupt; epochs 0 and 2 are fine, and the run finished.
    # run_complete is True and `epoch` alone would not say anything is wrong.
    _write_run(tmp_path, "r1", "bnn_mfvi", n_epochs=3)
    (tmp_path / "r1" / "epoch_001" / "artifacts_v2.json").write_text('{"epoch": 1')
    (tmp_path / "r1" / "final_results.json").write_text(json.dumps({}))
    df = collect_runs(tmp_path)
    assert set(df["run_complete"]) == {True}
    assert set(df["n_epochs_collected"]) == {2}
    assert set(df["n_epochs"]) == {None} or df["n_epochs"].isna().all()


def test_n_epochs_collected_is_per_run_not_per_campaign(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi", n_epochs=3)
    _write_run(tmp_path, "r2", "bnn_mfvi", n_epochs=1)
    df = collect_runs(tmp_path)
    assert set(df.loc[df.run_id == "r1", "n_epochs_collected"]) == {3}
    assert set(df.loc[df.run_id == "r2", "n_epochs_collected"]) == {1}


# --- run_complete: the ONLY true completion signal ---------------------------
# collect_runs never reads final_results.json for anything else, so a run
# whose last epoch is corrupt is otherwise indistinguishable from a run that
# is simply still in progress -- which, per launcher.py's own docs, is the
# campaign's normal steady state, not an edge case. run_complete carries that
# distinction explicitly per row.

def test_run_complete_is_true_when_final_results_json_exists(tmp_path):
    _write_run(tmp_path, "r1", "bnn_mfvi", n_epochs=2)
    (tmp_path / "r1" / "final_results.json").write_text(json.dumps({"final_stats": {}}))
    df = collect_runs(tmp_path)
    assert set(df["run_complete"]) == {True}


def test_run_complete_is_false_when_final_results_json_is_absent(tmp_path):
    # The launcher's normal steady state: a run that simply has not gotten
    # to its last epoch yet. Must NOT be confused with a corrupted tail.
    _write_run(tmp_path, "r1", "bnn_mfvi", n_epochs=2)
    df = collect_runs(tmp_path)
    assert set(df["run_complete"]) == {False}


def test_run_complete_is_false_when_final_results_json_is_malformed(tmp_path):
    # Same "skip, don't raise" treatment as a malformed epoch artifact.
    _write_run(tmp_path, "r1", "bnn_mfvi", n_epochs=2)
    (tmp_path / "r1" / "final_results.json").write_text('{"final_stats": {')  # truncated
    df = collect_runs(tmp_path)   # must not raise
    assert set(df["run_complete"]) == {False}


def test_run_complete_is_computed_per_run_not_shared_across_runs(tmp_path):
    _write_run(tmp_path, "done", "bnn_mfvi", n_epochs=2)
    (tmp_path / "done" / "final_results.json").write_text(json.dumps({}))
    _write_run(tmp_path, "still_running", "bnn_mfvi", n_epochs=2)
    df = collect_runs(tmp_path)
    assert set(df.loc[df.run_id == "done", "run_complete"]) == {True}
    assert set(df.loc[df.run_id == "still_running", "run_complete"]) == {False}


# --- tier: must not be fooled by an UNRELATED +experiment= override ----------
# _resolve_tier matches the override's VALUE ("reference_tier"), not merely
# the presence of an `experiment=` key. mlp_det always composes under its own
# +experiment=mlp_det_baseline (manifest.py) and is never valid in the
# reference tier (expand_manifest rejects that combination outright) -- a
# broadened match that fires on ANY `experiment=` override would mislabel
# every real mlp_det run as "reference".

@pytest.mark.parametrize("overrides,expected_tier", [
    (["system=pendulum", "predictor=hmc", "seed=42", "n_epochs=10",
      "+experiment=reference_tier", "acquisition=ranked"], "reference"),
    (["system=pendulum", "predictor=mlp_det", "seed=42", "n_epochs=10",
      "+experiment=mlp_det_baseline"], "production"),
    (["system=pendulum", "predictor=bnn_mfvi", "seed=42", "n_epochs=10",
      "acquisition=ranked"], "production"),
], ids=["reference_tier", "mlp_det_baseline-is-not-reference", "no-experiment-override"])
def test_tier_distinguishes_reference_tier_from_other_experiment_overrides(
        tmp_path, overrides, expected_tier):
    ed = tmp_path / "r1" / "epoch_000"
    (ed / ".hydra").mkdir(parents=True)
    (ed / ".hydra" / "config.yaml").write_text("predictor:\n  name: mlp_det\nseed: 42\n")
    (ed / ".hydra" / "overrides.yaml").write_text(yaml.dump(overrides))
    (ed / "artifacts_v2.json").write_text(json.dumps({
        "epoch": 0, "sampling_mode": "ranked", "eval_metrics": {}, "extra": {},
    }))
    df = collect_runs(tmp_path)
    assert set(df["tier"]) == {expected_tier}


# --- metrics: dotted-key collisions must be loud, not silently order-dependent -

def test_flattening_raises_on_a_dotted_key_collision(tmp_path):
    ed = tmp_path / "r1" / "epoch_000"
    (ed / ".hydra").mkdir(parents=True)
    (ed / ".hydra" / "config.yaml").write_text("predictor:\n  name: bnn_mfvi\nseed: 42\n")
    (ed / "artifacts_v2.json").write_text(json.dumps({
        "epoch": 0, "sampling_mode": "ranked",
        # {"a": {"b": 1}} flattens to "a.b"; the literal key "a.b" collides.
        "eval_metrics": {"a": {"b": 1}, "a.b": 2},
        "extra": {},
    }))
    with pytest.raises(ValueError, match="collision"):
        collect_runs(tmp_path)


# --- metrics: *_per_dim lists are a documented, tested exclusion -------------
# adaptive_v2/eval/full_roa.py's _compute_geodesic_error_stats emits
# mean_per_dim / median_per_dim / variance_per_dim as lists (one entry per
# state dimension). Pinned here so the drop reads as intentional, not an
# accident a future reader has to rediscover.

def test_per_dim_list_metrics_are_dropped_not_coerced(tmp_path):
    ed = tmp_path / "r1" / "epoch_000"
    (ed / ".hydra").mkdir(parents=True)
    (ed / ".hydra" / "config.yaml").write_text("predictor:\n  name: fm\nseed: 42\n")
    (ed / "artifacts_v2.json").write_text(json.dumps({
        "epoch": 0, "sampling_mode": "ranked",
        "eval_metrics": {
            "endpoint_errors": {
                "mean": 0.05,
                "mean_per_dim": [0.01, 0.02, 0.03, 0.04],
                "median_per_dim": [0.01, 0.02, 0.03, 0.04],
                "variance_per_dim": [0.001, 0.002, 0.003, 0.004],
            },
        },
        "extra": {},
    }))
    df = collect_runs(tmp_path)   # must not raise
    assert df.loc[0, "endpoint_errors.mean"] == pytest.approx(0.05)
    assert "endpoint_errors.mean_per_dim" not in df.columns
    assert "endpoint_errors.median_per_dim" not in df.columns
    assert "endpoint_errors.variance_per_dim" not in df.columns


# --- metrics: flattening recurses past a single level -------------------------

def test_metrics_flatten_at_depth_two(tmp_path):
    ed = tmp_path / "r1" / "epoch_000"
    (ed / ".hydra").mkdir(parents=True)
    (ed / ".hydra" / "config.yaml").write_text("predictor:\n  name: bnn_mfvi\nseed: 42\n")
    (ed / "artifacts_v2.json").write_text(json.dumps({
        "epoch": 0, "sampling_mode": "ranked",
        "eval_metrics": {"endpoint_errors": {"mc_sample_errors": {"mean": 0.42}}},
        "extra": {},
    }))
    df = collect_runs(tmp_path)
    assert df.loc[0, "endpoint_errors.mc_sample_errors.mean"] == pytest.approx(0.42)
