import json
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


def test_an_empty_artifact_is_not_counted_as_complete(tmp_path):
    # A job preempted mid-write (before the atomic-write fix in engine.py, or
    # by any other non-atomic writer) can leave a 0-byte artifacts_v2.json.
    # Counting file *presence* alone would call this run complete and skip it
    # forever -- exactly the failure this launcher exists to prevent.
    s = _spec(); _make_run(tmp_path, s, 2)
    epoch_dir = tmp_path / s.run_id / "epoch_002"
    epoch_dir.mkdir()
    (epoch_dir / "artifacts_v2.json").write_text("")
    assert run_state(s, tmp_path) == "partial"


def test_a_truncated_artifact_is_not_counted_as_complete(tmp_path):
    # Mid-write truncation: a valid-looking JSON prefix that never closes.
    s = _spec(); _make_run(tmp_path, s, 2)
    epoch_dir = tmp_path / s.run_id / "epoch_002"
    epoch_dir.mkdir()
    (epoch_dir / "artifacts_v2.json").write_text('{"epoch": 2, "eval_metrics": {"f1"')
    assert run_state(s, tmp_path) == "partial"


def test_a_valid_nested_artifact_is_counted_as_complete(tmp_path):
    s = _spec()
    d = tmp_path / s.run_id
    for e in range(3):
        ed = d / f"epoch_{e:03d}"
        ed.mkdir(parents=True)
        (ed / "artifacts_v2.json").write_text(json.dumps({
            "epoch": e,
            "eval_metrics": {"f1": 0.9, "coverage": [0.1, 0.2]},
            "extra": {"commit": "abc123"},
        }))
    assert run_state(s, tmp_path) == "complete"


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


def test_plan_launch_dedupes_specs_that_share_a_run_id(tmp_path):
    # n_epochs is deliberately excluded from run_id (extending a run resumes
    # it rather than starting a new one), so two specs that agree on
    # everything else but n_epochs collide on the same run_id and the same
    # output_dir. Without dedup both get queued and two jobs write into that
    # directory concurrently.
    first = _spec(seed=7)
    second = RunSpec(arm=first.arm, system=first.system, tier=first.tier,
                      acquisition=first.acquisition, seed=first.seed,
                      n_epochs=first.n_epochs + 5)
    assert first.run_id == second.run_id  # sanity: same identity despite n_epochs differing

    to_launch, states = plan_launch([first, second], tmp_path)
    assert [s.run_id for s in to_launch] == [first.run_id]
    assert to_launch == [first]           # first-seen spec wins, never both
    assert len(states) == 1


def test_plan_launch_dedup_preserves_first_seen_order(tmp_path):
    a, b = _spec(seed=1), _spec(seed=2)
    a_dup = RunSpec(arm=a.arm, system=a.system, tier=a.tier,
                     acquisition=a.acquisition, seed=a.seed, n_epochs=a.n_epochs + 1)
    to_launch, _ = plan_launch([a, b, a_dup], tmp_path)
    assert [s.run_id for s in to_launch] == [a.run_id, b.run_id]


def test_sbatch_targets_the_correct_partition_and_carries_overrides(tmp_path):
    cmd = sbatch_command(_spec(), tmp_path)
    joined = " ".join(cmd)
    assert "gpu-redhat" in joined
    assert "cgpu-redhat" not in joined          # Camden: never submit
    assert "predictor=bnn_mfvi" in joined
    assert "seed=42" in joined
    assert f"output_dir={tmp_path / _spec().run_id}" in joined
