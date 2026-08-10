import json
import pytest
from adaptive_roa.benchmark.manifest import RunSpec
from adaptive_roa.benchmark.launcher import run_state, plan_launch, sbatch_command


def _spec(seed=42, arm="bnn_mfvi"):
    return RunSpec(arm=arm, system="pendulum", tier="production",
                   acquisition="ranked", seed=seed, n_epochs=3)


def _make_run(root, spec, n_epochs_done):
    d = root / spec.run_id
    for e in range(n_epochs_done):
        ed = d / f"epoch_{e:03d}"
        ed.mkdir(parents=True)
        (ed / "artifacts_v2.json").write_text(json.dumps({"epoch": e}))
    return d


def test_absent_when_nothing_on_disk(tmp_path):
    assert run_state(_spec(), tmp_path) == "absent"


def test_partial_when_some_epochs_present(tmp_path):
    s = _spec(); _make_run(tmp_path, s, 2)
    assert run_state(s, tmp_path) == "partial"


def test_complete_when_all_epochs_present(tmp_path):
    s = _spec(); _make_run(tmp_path, s, 3)
    assert run_state(s, tmp_path) == "complete"


def test_an_empty_epoch_dir_is_not_a_finished_epoch(tmp_path):
    # A preempted job leaves the directory but no artifact. Counting DIRS
    # instead of artifacts would call this run complete and skip it forever.
    s = _spec(); _make_run(tmp_path, s, 2)
    (tmp_path / s.run_id / "epoch_002").mkdir()
    assert run_state(s, tmp_path) == "partial"


def test_a_quiet_log_does_not_imply_completion(tmp_path):
    s = _spec(); _make_run(tmp_path, s, 1)
    (tmp_path / s.run_id / "job.out").write_text("Epoch 3/3 done\nfinished\n")
    assert run_state(s, tmp_path) == "partial"


def test_plan_launch_skips_complete_and_relaunches_partial(tmp_path):
    done, partial, fresh = _spec(seed=1), _spec(seed=2), _spec(seed=3)
    _make_run(tmp_path, done, 3)
    _make_run(tmp_path, partial, 1)
    to_launch, states = plan_launch([done, partial, fresh], tmp_path)
    assert [s.run_id for s in to_launch] == [partial.run_id, fresh.run_id]
    assert states[done.run_id] == "complete"


def test_relaunching_twice_launches_nothing_new(tmp_path):
    specs = [_spec(seed=i) for i in range(3)]
    for s in specs:
        _make_run(tmp_path, s, 3)
    to_launch, _ = plan_launch(specs, tmp_path)
    assert to_launch == []


def test_sbatch_targets_the_correct_partition_and_carries_overrides(tmp_path):
    cmd = sbatch_command(_spec(), tmp_path)
    joined = " ".join(cmd)
    assert "gpu-redhat" in joined
    assert "cgpu-redhat" not in joined          # Camden: never submit
    assert "predictor=bnn_mfvi" in joined
    assert "seed=42" in joined
    assert f"output_dir={tmp_path / _spec().run_id}" in joined
