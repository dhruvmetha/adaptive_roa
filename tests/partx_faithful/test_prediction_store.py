"""Part-X writes the classifier prediction store the adaptive runs write (val and cal)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval import fpr_cap, store_conformal, store_score
from adaptive_roa.partx.tree import build_root
from adaptive_roa.partx_faithful import runner
from adaptive_roa.partx_faithful.algorithm import PartX, PartXConfig
from adaptive_roa.systems.pendulum import PendulumSystem

N_VAL, N_CAL, MAX_ROWS = 40, 25, 30      # val is longer than the row cap, cal shorter


def _p_true(X):
    return 1.0 / (1.0 + np.exp(-4.0 * (1.0 - np.abs(X[:, 0]))))


def _det_split(path, rng, low, high, n):
    """Deterministic layout (start, end, label 0/1), a few states outside the pool box."""
    X = rng.uniform(low - 0.5, high + 0.5, size=(n, 2))
    y = (rng.random(n) < _p_true(X)).astype(int)
    np.savetxt(path, np.column_stack([X, X, y]), delimiter=",", fmt="%.6f")


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    droot = tmp_path_factory.mktemp("data")
    system = PendulumSystem()
    root = build_root(system)
    rng = np.random.default_rng(0)
    starts = rng.uniform(root.low, root.high, size=(3000, 2))
    labels = np.where(rng.random(3000) < _p_true(starts), 1.0, -1.0)
    g0, g1 = np.meshgrid(np.linspace(-3.0, 3.0, 15), np.linspace(-6.0, 6.0, 15))
    grid = np.round(np.column_stack([g0.ravel(), g1.ravel()]), 6)
    np.savetxt(droot / "test_set.txt", np.column_stack([grid, _p_true(grid)]), delimiter=",", fmt="%.6f")
    lo, hi = starts.min(axis=0), starts.max(axis=0)
    _det_split(droot / "val_set.txt", rng, lo, hi, N_VAL)
    _det_split(droot / "cal_set.txt", rng, lo, hi, N_CAL)
    return system, starts, labels, droot


def _go(toy, out, n_epochs, store=True):
    system, starts, labels, droot = toy
    files = (dict(val_set_file=str(droot / "val_set.txt"), cal_set_file=str(droot / "cal_set.txt"),
                  max_eval_rows=MAX_ROWS) if store else {})
    return runner.run_partx_faithful(
        system=system, starts=starts, labels=labels, eval_states_file=str(droot / "test_set.txt"),
        output_dir=str(out), initial=60, samples_per_epoch=30, n_epochs=n_epochs,
        pcfg=PartXConfig(n0=5, n_bo=2, n_c=10, R=4, M=50, seed=0),
        gp_kwargs=dict(n_restarts=1, max_points=500), falsify="failure",
        attractor_radius=0.1, extra={"test": True}, **files)


def _split(droot, name):
    data = np.loadtxt(droot / f"{name}_set.txt", delimiter=",", ndmin=2)[:MAX_ROWS]
    return data[:, :2].astype(np.float32), np.where(data[:, -1] > 0.5, 1, -1).astype(np.int8)


def _through(handle, X):
    """p_success as the store computes it: sigmoid of the handle's float32 logits."""
    return torch.sigmoid(handle(torch.as_tensor(X, dtype=torch.float32))).to(torch.float64).numpy()


@pytest.fixture(scope="module")
def stored(toy, tmp_path_factory):
    """A 2-epoch run with the store on, and the handle that scored each epoch's test split."""
    out = tmp_path_factory.mktemp("stored") / "run"
    handles = {}
    real = runner.evaluate_full_roa_classifier

    def spy(**kw):
        handles[Path(kw["output_dir"]).name] = kw["classifier"]
        return real(**kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner, "evaluate_full_roa_classifier", spy)
        _go(toy, out, n_epochs=2)
    return out, handles


def test_each_epoch_stores_val_and_cal_through_the_handle_that_scored_test(toy, stored):
    out, handles = stored
    droot = toy[3]
    assert sorted(handles) == ["epoch_000", "epoch_001"]
    X_test = np.loadtxt(droot / "test_set.txt", delimiter=",", ndmin=2)[:, :2].astype(np.float32)
    for name, handle in handles.items():
        d = out / name
        with np.load(d / "full_roa_per_point.npz") as z:     # the spy caught the test handle
            np.testing.assert_array_equal(z["p_success"], _through(handle, X_test))
        for split, n in (("val", MAX_ROWS), ("cal", N_CAL)):
            X, y = _split(droot, split)
            assert len(X) == n
            with np.load(d / f"predictions_{split}.npz") as z:
                assert sorted(z.files) == ["p_success", "true_labels"]      # the classifier layout
                assert z["p_success"].dtype == np.float32 and z["p_success"].shape == (n,)
                assert z["true_labels"].dtype == np.int8
                np.testing.assert_array_equal(z["true_labels"], y)
                np.testing.assert_array_equal(z["p_success"], _through(handle, X).astype(np.float32))
                # and that is the readout's own probability, up to the float32 logit
                np.testing.assert_allclose(z["p_success"], handle.pw.p_success(X), rtol=1e-5, atol=1e-6)
    for split, n in (("val", MAX_ROWS), ("cal", N_CAL)):
        X, y = _split(droot, split)
        with np.load(out / f"prediction_store_states_{split}.npz") as z:
            np.testing.assert_array_equal(z["states"], X)
            np.testing.assert_array_equal(z["true_labels"], y)


def test_the_store_changes_nothing_else_and_is_off_without_split_files(toy, stored, tmp_path):
    out, _ = stored
    plain = tmp_path / "plain"
    _go(toy, plain, n_epochs=2, store=False)
    assert not list(plain.rglob("predictions_*.npz")) and not list(plain.glob("prediction_store_*"))
    for e in range(2):
        a, b = out / f"epoch_{e:03d}", plain / f"epoch_{e:03d}"
        assert json.loads((a / "artifacts_v2.json").read_text()) == json.loads((b / "artifacts_v2.json").read_text())
        for f in ("full_roa_per_point.npz", "partx_leaves.npz"):
            with np.load(a / f) as za, np.load(b / f) as zb:
                assert za.files == zb.files
                for k in za.files:
                    np.testing.assert_array_equal(za[k], zb[k])


def test_post_hoc_tools_read_the_store(stored):
    out, _ = stored
    (out / "resolved_config.yaml").write_text("decision_rule: one_sided\n")
    rows = fpr_cap.score_run(out, alphas=[0.05])
    assert [r["epoch"] for r in rows] == [0, 1] and all(r["val_k"] == "classifier" for r in rows)
    rows = store_score.score_run(out, "run")
    assert [r["epoch"] for r in rows] == [0, 1]
    assert {r["epoch"] for r in store_conformal.sweep_run(out)} == {0, 1}


def _store(d):
    out = {}
    for split in ("val", "cal"):
        with np.load(d / f"predictions_{split}.npz") as z:
            out[split] = {k: z[k].copy() for k in z.files}
    return out


def _written(d):
    """An epoch's files as first written. d2_indices is left out: the next epoch fills it in."""
    art = json.loads((d / "artifacts_v2.json").read_text())
    art["acquisition"].pop("d2_indices")
    return art, (d / "full_roa_per_point.npz").read_bytes(), (d / "partx_leaves.npz").read_bytes()


@pytest.fixture(scope="module")
def reference(toy, tmp_path_factory):
    out = tmp_path_factory.mktemp("reference") / "run"
    _go(toy, out, n_epochs=4)
    return out


@pytest.mark.parametrize("store_before", [True, False], ids=["store_on_throughout", "store_turned_on_at_resume"])
def test_a_resumed_run_keeps_written_epochs_and_stores_the_later_ones(toy, reference, tmp_path, monkeypatch,
                                                                      store_before):
    out = tmp_path / "run"
    real_take = PartX._take
    marker = out / "epoch_001" / "artifacts_v2.json"

    def flaky(self, region, handles):
        ids = real_take(self, region, handles)
        if marker.exists():
            raise RuntimeError("simulated preemption")
        return ids

    monkeypatch.setattr(PartX, "_take", flaky)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        _go(toy, out, n_epochs=4, store=store_before)
    monkeypatch.setattr(PartX, "_take", real_take)
    before = {e: _written(out / f"epoch_{e:03d}") for e in (0, 1)}
    _go(toy, out, n_epochs=4, store=True)
    for e in (0, 1):                                  # written before the crash: not rewritten
        assert _written(out / f"epoch_{e:03d}") == before[e]
    for e in range(4):
        a, b = reference / f"epoch_{e:03d}", out / f"epoch_{e:03d}"
        art_a, art_b = (json.loads((x / "artifacts_v2.json").read_text()) for x in (a, b))
        assert art_a == art_b, f"epoch {e}"
        has = (b / "predictions_val.npz").exists()
        assert has == (store_before or e >= 2) and has == (b / "predictions_cal.npz").exists()
        if has:
            sa, sb = _store(a), _store(b)
            for split in sa:
                for k in sa[split]:
                    np.testing.assert_array_equal(sa[split][k], sb[split][k])


# --- the cfg gate ---------------------------------------------------------------

_CFGDIR = str(Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2")


def _compose(overrides):
    from hydra import compose, initialize_config_dir

    from adaptive_roa.utils.env_config import get_data_dir, get_exp_dir, get_net_id
    for name, fn in (("data_dir", get_data_dir), ("exp_dir", get_exp_dir), ("net_id", get_net_id)):
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, lambda fn=fn: fn())
    with initialize_config_dir(config_dir=_CFGDIR, version_base=None):
        return compose(config_name="partx_faithful", overrides=overrides)


@pytest.mark.parametrize("system", ["pendulum", "cartpole_ppo", "quadrotor3d"])
def test_the_gate_reads_the_split_files_the_adaptive_runs_use(system, tmp_path):
    cfg = _compose([f"system={system}", "+experiment=det_rerun", f"output_dir={tmp_path}"])
    got = runner.prediction_store_files(cfg)
    assert got == {"val_set_file": cfg.data_source.val_set_file, "cal_set_file": cfg.data_source.cal_set_file,
                   "max_eval_rows": cfg.eval.max_eval_rows}
    assert got["val_set_file"].endswith("val_set.txt") and got["cal_set_file"].endswith("cal_set.txt")


@pytest.mark.parametrize("extra", [[], ["prediction_store.enabled=false"], ["data_source.val_set_file=null"]],
                         ids=["no_store_key", "disabled", "no_val_file"])
def test_the_gate_is_off_unless_enabled_with_both_files(extra, tmp_path):
    base = ["system=pendulum", f"output_dir={tmp_path}"]
    cfg = _compose(base + (["+experiment=det_rerun"] + extra if extra else []))
    if not extra:
        assert cfg.get("prediction_store", None) is None      # the plain Part-X config has no store
    assert runner.prediction_store_files(cfg) == {}


class _FakeSource:
    def __init__(self, cfg):
        rng = np.random.default_rng(0)
        self.start_states = rng.uniform(-1.0, 1.0, size=(50, 2))
        self.labels = np.where(rng.random(50) < 0.5, 1.0, -1.0)


@pytest.mark.parametrize("enabled", [True, False])
def test_run_from_cfg_passes_the_store_through(enabled, tmp_path, monkeypatch):
    import hydra

    cfg = _compose(["system=pendulum", "+experiment=det_rerun", f"output_dir={tmp_path}",
                    f"prediction_store.enabled={str(enabled).lower()}"])
    seen = {}
    monkeypatch.setattr("adaptive_roa.adaptive.npz_data_source.NpzTrajectoryDataSource", _FakeSource)
    monkeypatch.setattr(hydra.utils, "instantiate", lambda c: PendulumSystem())
    monkeypatch.setattr(runner, "run_partx_faithful", lambda **kw: seen.update(kw) or {})
    runner.run_from_cfg(cfg)
    if enabled:
        assert seen["val_set_file"] == cfg.data_source.val_set_file
        assert seen["cal_set_file"] == cfg.data_source.cal_set_file
        assert seen["max_eval_rows"] is None
        assert OmegaConf.load(tmp_path / "resolved_config.yaml").decision_rule == "two_sided"
    else:
        assert not {"val_set_file", "cal_set_file", "max_eval_rows"} & set(seen)
        assert not (tmp_path / "resolved_config.yaml").exists()
