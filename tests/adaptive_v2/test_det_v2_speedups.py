"""det_v2 speedups: each fast path checked against the path it replaces.

d  batched indexing      same batches, same order, same RNG use; same trained weights
e  reuse_cal_for_qhat    q_hat_eval from the stored cal prediction; cal predicted once
f  endpoint_error=false  skipped with a reason (engine test)
g  reuse_body_features   same endpoints and RNG use for deterministic and Laplace posteriors
h  load once             FinalStateTrainer parses each split once
i  plot_epochs           plots only at the listed epochs; npz files unchanged
   progress bar / mae_val_frequency overrides reach the trainers

All CPU. The tests marked slow at the end run one real adaptive epoch per
predictor with the flags off and on and compare every result file.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from adaptive_roa.data.adaptive_classification_data import AdaptiveClassificationDataModule
from adaptive_roa.data.batched_indexing import identity_collate, materialize_rows
from adaptive_roa.data.cartpole_endpoint_data import CartPoleEndpointDataModule, CartPoleEndpointDataset
from adaptive_roa.data.pendulum_endpoint_data import PendulumEndpointDataModule
from adaptive_roa.data.quadrotor2d_endpoint_data import Quadrotor2DEndpointDataModule
from adaptive_roa.data.quadrotor3d_endpoint_data import Quadrotor3DEndpointDataModule

# ---------------------------------------------------------------- data files


def _pendulum_rows(rng, n):
    # Angles well outside [-pi, pi] so the wrap does real work.
    return np.c_[rng.uniform(-12, 12, n), rng.normal(0, 3, n), rng.uniform(-12, 12, n), rng.normal(0, 3, n)]


def _cartpole_rows(rng, n):
    s = rng.normal(0, 2, (n, 4))
    e = rng.normal(0, 2, (n, 4))
    s[:, 1] = rng.uniform(-12, 12, n)
    e[:, 1] = rng.uniform(-12, 12, n)
    return np.c_[s, e]


def _q2d_rows(rng, n):
    s = rng.normal(0, 2, (n, 6))
    e = rng.normal(0, 2, (n, 6))
    s[:, 2] = rng.uniform(-12, 12, n)
    e[:, 2] = rng.uniform(-12, 12, n)
    return np.c_[s, e]


def _q3d_rows(rng, n):
    # Non-unit quaternions, half with qw < 0, and one row whose quaternion is
    # below the 1e-8 norm floor (normalize_quaternion's identity fallback).
    s = rng.normal(0, 1, (n, 13))
    e = rng.normal(0, 1, (n, 13))
    s[:, 3:7] *= rng.uniform(0.2, 3.0, (n, 1))
    s[3, 3:7] = 1e-10
    return np.c_[s, e]


_ROWS = {"pendulum": _pendulum_rows, "cartpole": _cartpole_rows, "quadrotor2d": _q2d_rows,
         "quadrotor3d": _q3d_rows}


def _write(path: Path, rows: np.ndarray, delimiter=" ") -> str:
    np.savetxt(path, rows, delimiter=delimiter, fmt="%.9g")
    return str(path)


def _endpoint_files(tmp_path, system, n_train=301, n_val=77):
    rng = np.random.default_rng(0)
    delim = "," if system.startswith("quadrotor") else " "
    return (_write(tmp_path / "train.txt", _ROWS[system](rng, n_train), delim),
            _write(tmp_path / "val.txt", _ROWS[system](rng, n_val), delim))


def _endpoint_dm(system, train, val, tmp_path, batched, num_workers=0):
    kw = dict(data_file=train, validation_file=val, test_file=val, batch_size=64,
              val_batch_size=50, num_workers=num_workers, batched_indexing=batched)
    if system == "pendulum":
        return PendulumEndpointDataModule(**kw)
    if system == "cartpole":
        return CartPoleEndpointDataModule(dataset_dir=str(tmp_path), **kw)
    if system == "quadrotor2d":
        return Quadrotor2DEndpointDataModule(dataset_dir=str(tmp_path), **kw)
    return Quadrotor3DEndpointDataModule(dataset_dir=str(tmp_path), **kw)


def _epochs(dm, n_epochs=2, seed=0):
    """Every batch of n_epochs train passes and one val pass, plus the RNG state after."""
    torch.manual_seed(seed)
    out = []
    for _ in range(n_epochs):
        out.append([{k: v.clone() for k, v in b.items()} for b in dm.train_dataloader()])
    out.append([{k: v.clone() for k, v in b.items()} for b in dm.val_dataloader()])
    return out, torch.get_rng_state()


def _assert_same_batches(a, b):
    assert len(a) == len(b)
    for ea, eb in zip(a, b):
        assert len(ea) == len(eb)
        for ba, bb in zip(ea, eb):
            assert ba.keys() == bb.keys()
            for k in ba:
                assert ba[k].dtype == bb[k].dtype and ba[k].shape == bb[k].shape
                assert torch.equal(ba[k], bb[k]), k


# ---------------------------------------------------------------- d: batched indexing


@pytest.mark.parametrize("system", ["pendulum", "cartpole", "quadrotor2d", "quadrotor3d"])
def test_batched_indexing_gives_the_same_batches_in_the_same_order(system, tmp_path):
    train, val = _endpoint_files(tmp_path, system)
    old = _endpoint_dm(system, train, val, tmp_path, batched=False)
    new = _endpoint_dm(system, train, val, tmp_path, batched=True)
    old.setup("fit")
    new.setup("fit")
    assert new.train_dataset.batched_indexing and not old.train_dataset.batched_indexing
    assert new.train_dataloader().collate_fn is identity_collate
    a, rng_a = _epochs(old)
    b, rng_b = _epochs(new)
    _assert_same_batches(a, b)
    assert torch.equal(rng_a, rng_b)           # the sampler drew the same seeds
    # Two epochs differ (the permutation is redrawn), so the check is not vacuous.
    assert not torch.equal(a[0][0]["start_state"], a[1][0]["start_state"])


@pytest.mark.parametrize("system,env", [("quadrotor2d", "Q2D_CACHE_TRANSFORMED"),
                                        ("quadrotor3d", "Q3D_CACHE_TRANSFORMED")])
def test_batched_indexing_matches_with_the_env_transform_cache(system, env, tmp_path, monkeypatch):
    train, val = _endpoint_files(tmp_path, system)
    old = _endpoint_dm(system, train, val, tmp_path, batched=False)
    old.setup("fit")
    a, rng_a = _epochs(old)
    monkeypatch.setenv(env, "1")
    new = _endpoint_dm(system, train, val, tmp_path, batched=True)
    new.setup("fit")
    b, rng_b = _epochs(new)
    _assert_same_batches(a, b)
    assert torch.equal(rng_a, rng_b)


def test_batched_indexing_matches_with_dataloader_workers(tmp_path):
    train, val = _endpoint_files(tmp_path, "pendulum", n_train=200, n_val=40)
    runs = []
    for batched in (False, True):
        dm = _endpoint_dm("pendulum", train, val, tmp_path, batched=batched, num_workers=2)
        dm.setup("fit")
        runs.append(_epochs(dm, n_epochs=1))
    _assert_same_batches(runs[0][0], runs[1][0])
    assert torch.equal(runs[0][1], runs[1][1])


def test_materialized_rows_equal_default_collate_of_every_row(tmp_path):
    train, _ = _endpoint_files(tmp_path, "cartpole", n_train=130)
    ds = CartPoleEndpointDataset(train, dataset_dir=str(tmp_path))
    from torch.utils.data import default_collate
    want = default_collate([ds[i] for i in range(len(ds))])
    got = materialize_rows(ds, chunk=17)          # chunk boundaries do not matter
    for k in want:
        assert torch.equal(want[k], got[k])
    # Disabled, __getitems__ is the fetcher's own per-row fallback.
    rows = ds.__getitems__([5, 0, 5])
    assert isinstance(rows, list)
    for r, i in zip(rows, [5, 0, 5]):
        assert all(torch.equal(r[k], ds[i][k]) for k in r)


def _classification_files(tmp_path, n_train=257, n_val=63, d=4):
    rng = np.random.default_rng(3)
    def rows(n):
        x = rng.normal(0, 1, (n, d))
        return np.c_[x, (x[:, 0] > 0).astype(float)]
    return _write(tmp_path / "ctrain.txt", rows(n_train)), _write(tmp_path / "cval.txt", rows(n_val))


def test_classification_batched_indexing_gives_the_same_batches(tmp_path):
    train, val = _classification_files(tmp_path)
    runs = []
    for batched in (False, True):
        dm = AdaptiveClassificationDataModule(train, val, batch_size=64, num_workers=0,
                                              batched_indexing=batched)
        dm.setup()
        runs.append(_epochs(dm))
    _assert_same_batches(runs[0][0], runs[1][0])
    assert torch.equal(runs[0][1], runs[1][1])
    assert runs[1][0][0][0]["label"].shape == (64,)


# ------------------------------------------- d + h + g through the real trainers


def _cartpole_files(tmp_path):
    from adaptive_roa.systems.cartpole import CartPoleSystem
    rng = np.random.default_rng(0)
    system = CartPoleSystem()

    def rows(n):
        z = rng.uniform(-1, 1, (n, 4))
        z[:, 1] *= np.pi
        s = system.denormalize_state(torch.as_tensor(z, dtype=torch.float32)).numpy()
        e = 0.5 * s + rng.normal(0, 0.05, s.shape)
        return np.c_[s, e]
    return {"train": _write(tmp_path / "train.txt", rows(300)), "val": _write(tmp_path / "val.txt", rows(90))}


def _fs_cfg(posterior, head, batched, reuse=False, **extra):
    fs = {"posterior": posterior, "hidden_dims": [16, 16], "lr": 1e-2, "weight_decay": 1e-5,
          "max_epochs": 3, "patience": 5, "prior_sigma": 1.0, "beta_nll": 0.0,
          "reuse_body_features": reuse}
    if head == "mixture":
        fs["head"] = {"kind": "mixture", "n_components": 3}
    if posterior == "laplace":
        fs["prior_precision"] = "auto" if head == "mixture" else 1.0
    fs.update(extra)
    return OmegaConf.create({"device": "cpu", "seed": 0, "predictor": {
        "type": "generative", "name": "fs_test", "batch_size": 64, "val_batch_size": 64,
        "batched_indexing": batched, "final_state": fs}})


def _fit_final_state(cfg, files, out):
    from adaptive_roa.adaptive_v2.trainers.final_state_trainer import FinalStateTrainer
    from adaptive_roa.systems.cartpole import CartPoleSystem
    torch.manual_seed(0)
    return FinalStateTrainer(cfg, CartPoleSystem(), "cartpole_pybullet").fit(files, str(out))


def _state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _assert_same_state(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def _draws(handle, x, k=6, seed=11):
    torch.manual_seed(seed)
    out = [handle.predict_endpoint(x) for _ in range(k)]
    return out, torch.get_rng_state()


ARMS = [("deterministic", "gaussian"), ("deterministic", "mixture"),
        ("laplace", "gaussian"), ("laplace", "mixture")]


@pytest.mark.parametrize("posterior,head", ARMS)
def test_final_state_trainer_batched_indexing_trains_the_same_weights(posterior, head, tmp_path):
    files = _cartpole_files(tmp_path)
    h_old = _fit_final_state(_fs_cfg(posterior, head, batched=False), files, tmp_path / "old")
    h_new = _fit_final_state(_fs_cfg(posterior, head, batched=True), files, tmp_path / "new")
    _assert_same_state(_state(h_old.posterior), _state(h_new.posterior))
    if posterior == "laplace":
        if head == "mixture":       # block mode, fitted on _full_training_tensors
            for La, Lb in zip(h_old.posterior._block_chols, h_new.posterior._block_chols):
                assert torch.equal(La, Lb)
        else:
            assert torch.equal(h_old.posterior.precision_cholesky, h_new.posterior.precision_cholesky)
    x = torch.randn(32, 4)
    a, ra = _draws(h_old, x)
    b, rb = _draws(h_new, x)
    assert all(torch.equal(p, q) for p, q in zip(a, b)) and torch.equal(ra, rb)


@pytest.mark.parametrize("posterior", ["deterministic", "laplace"])
def test_bnn_trainer_batched_indexing_trains_the_same_weights(posterior, tmp_path):
    from adaptive_roa.adaptive_v2.trainers.bayesian_mlp_trainer import BayesianMLPTrainer
    from adaptive_roa.systems.cartpole import CartPoleSystem
    train, val = _classification_files(tmp_path)
    files = {"train": train, "val": val}
    handles = []
    for batched in (False, True):
        bnn = {"posterior": posterior, "hidden_dims": [16, 16], "lr": 1e-2, "weight_decay": 1e-5,
               "max_epochs": 3, "patience": 5, "pos_weight": 1.0, "n_marginal_samples": 8}
        if posterior == "laplace":
            bnn["prior_precision"] = "auto"
        cfg = OmegaConf.create({"device": "cpu", "seed": 0, "predictor": {
            "type": "classifier", "name": "bnn_test", "batch_size": 64, "batched_indexing": batched,
            "bnn": bnn}})
        torch.manual_seed(0)
        handles.append(BayesianMLPTrainer(cfg, CartPoleSystem(), "cartpole_pybullet")
                       .fit(files, str(tmp_path / f"b{int(batched)}")))
    _assert_same_state(_state(handles[0].posterior), _state(handles[1].posterior))
    x = torch.randn(40, 4)
    assert torch.equal(handles[0](x), handles[1](x))


# ---------------------------------------------------------------- h: load once


def test_final_state_trainer_parses_each_split_once(tmp_path, monkeypatch):
    files = _cartpole_files(tmp_path)
    loaded = []
    orig = CartPoleEndpointDataset.__init__

    def spy(self, data_file, *a, **kw):
        loaded.append(Path(data_file).name)
        orig(self, data_file, *a, **kw)
    monkeypatch.setattr(CartPoleEndpointDataset, "__init__", spy)
    _fit_final_state(_fs_cfg("deterministic", "gaussian", batched=False), files, tmp_path / "o")
    # Before: setup() read train, val and test(=val), then Lightning's setup("fit")
    # read train and val again.
    assert sorted(loaded) == ["train.txt", "val.txt"]


def test_endpoint_datamodule_setup_is_idempotent(tmp_path):
    train, val = _endpoint_files(tmp_path, "quadrotor3d")
    dm = _endpoint_dm("quadrotor3d", train, val, tmp_path, batched=False)
    dm.setup()
    ids = (id(dm.train_dataset), id(dm.val_dataset), id(dm.test_dataset))
    dm.setup("fit")
    dm.setup("test")
    assert (id(dm.train_dataset), id(dm.val_dataset), id(dm.test_dataset)) == ids


# ---------------------------------------------------------------- g: reuse body features


def _handle_pair(posterior, head, tmp_path):
    """(old handle, new handle) over the SAME trained posterior."""
    from adaptive_roa.predictors.final_state_handle import FinalStateModelHandle
    files = _cartpole_files(tmp_path)
    old = _fit_final_state(_fs_cfg(posterior, head, batched=False), files, tmp_path / "fit")
    new = FinalStateModelHandle(old.posterior, old.head, old.system, reuse_body_features=True).eval()
    return old, new


@pytest.mark.parametrize("posterior,head", ARMS)
def test_reuse_body_features_gives_the_same_endpoints_and_rng_use(posterior, head, tmp_path):
    old, new = _handle_pair(posterior, head, tmp_path)
    if posterior == "laplace":
        assert old.posterior.is_fitted
    x = torch.randn(48, 4)
    body = old.posterior.net if posterior == "deterministic" else old.posterior.body
    calls = []
    hook = body.register_forward_hook(lambda *a: calls.append(1))
    a, ra = _draws(old, x, k=8)
    n_old = len(calls)
    b, rb = _draws(new, x, k=8)
    hook.remove()
    assert all(torch.equal(p, q) for p, q in zip(a, b)) and torch.equal(ra, rb)
    assert n_old == 8 and len(calls) - n_old == 1          # body once per batch, not per draw
    assert not torch.equal(a[0], a[1])                      # draws still differ


@pytest.mark.parametrize("posterior,head", [("deterministic", "mixture"), ("laplace", "mixture")])
def test_reuse_body_features_through_the_mc_loops(posterior, head, tmp_path):
    """The loops that call predict_endpoint K times per batch: the store's
    mc_predict and evaluate_full_roa_fast. Labels and p_success unchanged."""
    from adaptive_roa.adaptive_v2.eval import prediction_store as ps
    from adaptive_roa.adaptive_v2.eval.full_roa import evaluate_full_roa_fast
    old, new = _handle_pair(posterior, head, tmp_path)
    system = old.system
    X = torch.randn(150, 4).numpy()
    outs = []
    for h in (old, new):
        torch.manual_seed(5)
        lab, ep = ps.mc_predict(h, system, X, 12, 0.3, batch_size=64, keep_rows=np.arange(10))
        outs.append((lab, ep))
    assert np.array_equal(outs[0][0], outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])

    f = tmp_path / "eval.txt"
    np.savetxt(f, np.c_[X, X, (X[:, 0] > 0).astype(int)], delimiter=",")
    res = []
    for h, name in ((old, "o"), (new, "n")):
        torch.manual_seed(9)
        evaluate_full_roa_fast(h, system, str(f), num_mc_samples=10, batch_size=64, lambda_star=0.5,
                               delta=0.1, attractor_radius=0.3, device="cpu", decision_rule="two_sided",
                               output_dir=str(tmp_path / name), verbose=False, binary_outcomes=False)
        res.append(np.load(tmp_path / name / "full_roa_per_point.npz"))
    for k in ("p_success", "p_failure", "p_invalid"):
        assert np.array_equal(res[0][k], res[1][k])


def test_reuse_body_features_recomputes_after_an_input_or_weight_change(tmp_path):
    old, new = _handle_pair("deterministic", "gaussian", tmp_path)
    x = torch.randn(20, 4)
    new.predict_endpoint(x)                       # cache x's body output
    x.mul_(2.0)                                   # in place: same object, new version
    torch.manual_seed(1)
    a = old.predict_endpoint(x)
    torch.manual_seed(1)
    b = new.predict_endpoint(x)
    assert torch.equal(a, b)
    with torch.no_grad():                         # a weight update invalidates as well
        next(old.posterior.parameters()).add_(0.5)
    torch.manual_seed(2)
    a = old.predict_endpoint(x)
    torch.manual_seed(2)
    b = new.predict_endpoint(x)
    assert torch.equal(a, b)
    v = x[:10]                                    # a view, then a write through the base
    new.predict_endpoint(v)
    x[:5] += 1.0
    torch.manual_seed(3)
    a = old.predict_endpoint(v)
    torch.manual_seed(3)
    assert torch.equal(a, new.predict_endpoint(v))


def test_reuse_body_features_leaves_other_posteriors_alone(tmp_path):
    from adaptive_roa.predictors.final_state_handle import FinalStateModelHandle
    files = _cartpole_files(tmp_path)
    old = _fit_final_state(_fs_cfg("mfvi", "gaussian", batched=False, kl_weight=1.0), files, tmp_path / "m")
    new = FinalStateModelHandle(old.posterior, old.head, old.system, reuse_body_features=True).eval()
    assert new._split_posterior() is None
    x = torch.randn(20, 4)
    a, ra = _draws(old, x)
    b, rb = _draws(new, x)
    assert all(torch.equal(p, q) for p, q in zip(a, b)) and torch.equal(ra, rb)


def test_laplace_forward_sample_is_body_then_sample_from_features():
    from adaptive_roa.predictors.posteriors import LastLayerLaplacePosterior
    torch.manual_seed(0)
    body = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU())
    post = LastLayerLaplacePosterior(body, torch.nn.Linear(8, 2))
    x = torch.randn(30, 3)
    post.fit(body(x), torch.randn(30, 2), task="final_state", sigma=0.5)
    torch.manual_seed(4)
    a = post.forward_sample(x)
    torch.manual_seed(4)
    assert torch.equal(a, post.sample_from_features(body(x)))


# ---------------------------------------------------------------- e: reuse cal for q_hat


class _Clf:
    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, x):
        return 4.0 * x[:, 0]


class _Endpoint:
    """Deterministic endpoint model, so the old and new cal passes see the same labels."""

    def eval(self):
        return self

    def predict_endpoint(self, x):
        return x * 1.5


class _NoisyEndpoint(_Endpoint):
    def predict_endpoint(self, x):
        return x + 0.6 * torch.randn(x.shape)


class _SignSystem:
    state_dim = 2

    def classify_attractor(self, pred, radius=None):
        lab = torch.zeros(pred.shape[0], dtype=torch.long)
        lab[pred[:, 0] > 0.3] = 1
        lab[pred[:, 0] < -0.3] = -1
        return lab


def _cal_backend(rule, k, model, predictor_type):
    from adaptive_roa.adaptive_v2.calibration.conformal_calibration import ConformalCalibrationBackend
    cfg = OmegaConf.create({"delta": 0.05, "w": 0.9, "alpha": 0.1, "alpha_eval": 0.1, "decision_rule": rule,
                            "attractor_radius": 0.1, "num_mc_samples": k, "verbose": False})
    cb = ConformalCalibrationBackend(cfg, system=_SignSystem(), device="cpu")
    cb.bind_model(model, predictor_type=predictor_type)
    return cb


def _fake_engine(cb, predictor_type, k_eval):
    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine
    eng = SimpleNamespace(
        calibration_backend=cb, predictor_type=predictor_type, system=_SignSystem(), device="cpu",
        evaluator=SimpleNamespace(num_mc_samples_eval=k_eval, attractor_radius=0.1),
        cfg=OmegaConf.create({"val_batch_size": 64}))
    eng.calibrate = lambda *a, **kw: AdaptiveEngine._calibrate_eval_on_store_prediction(eng, *a, **kw)
    return eng


def _cal_set(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
    y = np.where(X[:, 0] + rng.normal(0, 0.3, n) > 0, 1, -1)
    return X, y


@pytest.mark.parametrize("rule", ["two_sided", "one_sided"])
@pytest.mark.parametrize("predictor_type,model", [("classifier", _Clf()), ("generative", _Endpoint())])
def test_reuse_cal_q_hat_equals_calibrate_eval_and_store_file_is_unchanged(rule, predictor_type, model, tmp_path):
    from adaptive_roa.adaptive_v2.eval import prediction_store as ps
    from adaptive_roa.adaptive_v2.types import ThresholdState
    X, y = _cal_set()
    state = ThresholdState(lambda_star=0.55, delta_star=0.1)
    q_old = _cal_backend(rule, 20, model, predictor_type).calibrate_eval(X, y, state)
    ps.write_split(tmp_path / "old", "cal", X, y, predictor_type=predictor_type, model=model,
                   system=_SignSystem(), num_mc_samples=20, attractor_radius=0.1, batch_size=64)
    eng = _fake_engine(_cal_backend(rule, 20, model, predictor_type), predictor_type, 20)
    q_new, pred = eng.calibrate(model, X, y, state, keep_endpoints=False)
    assert q_new == q_old
    ps.write_split_prediction(tmp_path / "new", "cal", pred, sub_k=pred.get("qhat_sub_k"))
    zo, zn = np.load(tmp_path / "old" / "predictions_cal.npz"), np.load(tmp_path / "new" / "predictions_cal.npz")
    assert zo.files == zn.files
    for k in zo.files:
        assert np.array_equal(zo[k], zn[k], equal_nan=True), k


def test_reuse_cal_q_hat_is_replayable_from_the_stored_counts(tmp_path):
    """Noisy model: old and new draw differently, but the new q_hat is exactly the
    calibrator on the stored counts, including at a sub-K."""
    from adaptive_roa.adaptive_v2.eval import prediction_store as ps
    from adaptive_roa.adaptive_v2.types import ThresholdState
    from adaptive_roa.conformal.calibrator import Calibrator
    from adaptive_roa.conformal import ConformalConfig
    X, y = _cal_set()
    state = ThresholdState(lambda_star=0.5, delta_star=0.08)
    for k_cal, k_eval in ((30, 30), (10, 30)):
        torch.manual_seed(0)
        cb = _cal_backend("two_sided", k_cal, _NoisyEndpoint(), "generative")
        q, pred = _fake_engine(cb, "generative", k_eval).calibrate(_NoisyEndpoint(), X, y, state, False)
        out = tmp_path / f"k{k_cal}"
        ps.write_split_prediction(out, "cal", pred, sub_k=pred.get("qhat_sub_k"))
        z = np.load(out / "predictions_cal.npz")
        assert int(z["num_mc_samples"]) == k_eval
        sfx = "" if k_cal == k_eval else f"_k{k_cal}"
        assert ("n_success_k10" in z.files) == (k_cal < k_eval)
        cal = Calibrator(ConformalConfig(delta=0.05, alpha=0.1, decision_rule="two_sided"))
        q2 = cal.calibrate(z["n_success" + sfx] / k_cal, z["true_labels"], 0.5, 0.08,
                           z["n_failure" + sfx] / k_cal, verbose=False)
        assert q == q2


def test_reuse_cal_refuses_a_cal_k_above_the_eval_k():
    from adaptive_roa.adaptive_v2.types import ThresholdState
    X, y = _cal_set(20)
    eng = _fake_engine(_cal_backend("two_sided", 50, _Endpoint(), "generative"), "generative", 30)
    with pytest.raises(ValueError, match="exceeds"):
        eng.calibrate(_Endpoint(), X, y, ThresholdState(lambda_star=0.5, delta_star=0.1), False)


# ---------------------------------------------------------------- i: plot_epochs


def _evaluator(plot_epochs, predictor_type="classifier"):
    from adaptive_roa.adaptive_v2.eval.full_roa import FullROAEvaluator
    cfg = OmegaConf.create({"predictor_type": predictor_type, "alpha_eval": 0.1, "attractor_radius": 0.1,
                            "num_mc_samples_eval": 8, "decision_rule": "two_sided", "refine_invalids": False,
                            "refine_t_min": 0.7, "refine_t_max": 0.9, "refine_num_steps": 10,
                            "refine_max_attempts": 1, "max_eval_rows": None, "verbose": False,
                            "binary_outcomes": False, "plot_epochs": plot_epochs})
    return FullROAEvaluator(cfg, _SignSystem(), "cpu")


@pytest.mark.parametrize("predictor_type,model", [("classifier", _Clf()), ("generative", _NoisyEndpoint())])
def test_plot_epochs_plots_only_the_listed_epochs(predictor_type, model, tmp_path):
    from adaptive_roa.adaptive_v2.types import ThresholdState
    X, y = _cal_set(200)
    f = tmp_path / "eval.txt"
    np.savetxt(f, np.c_[X, X, (y == 1).astype(int)], delimiter=",", fmt=["%.6f"] * 4 + ["%d"])
    state = ThresholdState(lambda_star=0.5, delta_star=0.1)
    pngs = {"full_roa_projections.png", "full_roa_heatmap.png"}
    per_point = {}
    for plot_epochs, epoch, want in ((None, 3, True), ([0, 5], 5, True), ([0, 5], 3, False),
                                     ([], 0, False), ([0, 5], None, True)):
        out = tmp_path / f"{plot_epochs}_{epoch}"
        ctx = {"eval_states_file": str(f), "output_dir": str(out), "verbose": False, "batch_size": 64}
        if epoch is not None:
            ctx["epoch"] = epoch
        torch.manual_seed(0)
        _evaluator(plot_epochs, predictor_type).evaluate_epoch(model, state, ctx)
        assert pngs.issubset({p.name for p in out.iterdir()}) == want, (plot_epochs, epoch)
        z = np.load(out / "full_roa_per_point.npz")
        per_point[(str(plot_epochs), epoch)] = z["p_success"]
    ref = per_point[("None", 3)]
    assert all(np.array_equal(ref, v) for v in per_point.values())


def test_plot_epochs_default_passes_no_new_kwarg(monkeypatch):
    """The default config makes the very call it made before (no make_plots kwarg)."""
    from adaptive_roa.adaptive_v2.eval import full_roa
    from adaptive_roa.adaptive_v2.types import ThresholdState
    seen = {}
    monkeypatch.setattr(full_roa, "evaluate_full_roa_fast", lambda **kw: seen.update(kw) or {})
    _evaluator(None, "generative").evaluate_epoch(None, ThresholdState(0.5, 0.1),
                                                  {"eval_states_file": "x", "epoch": 4})
    assert "make_plots" not in seen
    _evaluator([1], "generative").evaluate_epoch(None, ThresholdState(0.5, 0.1),
                                                 {"eval_states_file": "x", "epoch": 4})
    assert seen["make_plots"] is False


# ------------------------------------------------- Hydra overrides reach the trainers

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2"
DET_PREDICTORS = ["fm_det", "det_mlp_reg", "det_bnn_reg", "det_mlp_clf", "det_bnn_clf"]


def _register_resolvers() -> None:
    """Mirrors scripts/run_adaptive.py (as test_fpr_cap_engine.py does); compose() registers none."""
    import os
    from adaptive_roa.utils.env_config import (
        get_data_dir, get_env_config, get_exp_dir, get_net_id, get_shared_data_base,
    )
    if not OmegaConf.has_resolver("net_id"):
        OmegaConf.register_new_resolver("net_id", lambda: get_net_id())
    if not OmegaConf.has_resolver("exp_dir"):
        OmegaConf.register_new_resolver("exp_dir", lambda: get_exp_dir())
    if not OmegaConf.has_resolver("data_dir"):
        OmegaConf.register_new_resolver("data_dir", lambda: get_data_dir())
    if not OmegaConf.has_resolver("shared_data_base"):
        OmegaConf.register_new_resolver(
            "shared_data_base", lambda default="": get_shared_data_base() or default)
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver(
            "env", lambda key, default="": os.environ.get(key, get_env_config().get(key, default)))


def _compose(*overrides):
    from hydra import compose, initialize_config_dir
    _register_resolvers()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="default", overrides=list(overrides))


def _det_v2_overrides(pred):
    """The override keys, as they must be spelled on the CLI, per predictor."""
    o = ["predictor.lightning_trainer.enable_progress_bar=false" if pred == "fm_det"
         else "+predictor.lightning_trainer.enable_progress_bar=false",
         "+predictor.batched_indexing=true", "+prediction_store.reuse_cal_for_qhat=true",
         "adaptive_v2.endpoint_error=false", "eval.plot_epochs=[0,5,10]"]
    if pred == "fm_det":
        o.append("predictor.flow_matching.mae_val_frequency=1000000")
    if pred in ("det_mlp_reg", "det_bnn_reg"):
        o.append("+predictor.final_state.reuse_body_features=true")
    return o


@pytest.mark.parametrize("pred", DET_PREDICTORS)
def test_det_v2_override_keys_compose(pred):
    cfg = _compose("system=pendulum", f"predictor={pred}", "acquisition=direct", "+experiment=det_rerun",
                   "output_dir=/tmp/unused", *_det_v2_overrides(pred))
    assert cfg.predictor.lightning_trainer.enable_progress_bar is False
    assert cfg.predictor.batched_indexing is True
    assert cfg.prediction_store.reuse_cal_for_qhat is True
    assert cfg.adaptive_v2.endpoint_error is False and list(cfg.eval.plot_epochs) == [0, 5, 10]
    if pred == "fm_det":
        assert cfg.predictor.flow_matching.mae_val_frequency == 1000000
    # The defaults, with no override: every new flag off.
    base = _compose("system=pendulum", f"predictor={pred}", "acquisition=direct", "+experiment=det_rerun",
                    "output_dir=/tmp/unused")
    assert base.adaptive_v2.endpoint_error is True and base.eval.plot_epochs is None
    assert "batched_indexing" not in base.predictor and "reuse_cal_for_qhat" not in base.prediction_store


class _StopFit(Exception):
    pass


def _capture_trainer(monkeypatch, module):
    seen = {}

    class FakeTrainer:
        def __init__(self, **kw):
            seen["kw"] = kw

        def fit(self, model, datamodule=None, **kw):
            seen["model"], seen["dm"] = model, datamodule
            raise _StopFit
    monkeypatch.setattr(module.pl, "Trainer", FakeTrainer)
    return seen


@pytest.mark.parametrize("pred", DET_PREDICTORS)
@pytest.mark.parametrize("override", [False, True])
def test_progress_bar_and_batching_overrides_reach_the_trainer(pred, override, tmp_path, monkeypatch):
    import importlib
    import hydra
    over = (_det_v2_overrides(pred) if override else []) + ["device=cpu", "num_workers=0"]
    cfg = _compose("system=pendulum", f"predictor={pred}", "acquisition=direct", "+experiment=det_rerun",
                   f"output_dir={tmp_path}", *over)
    system = hydra.utils.instantiate(cfg.system)
    if pred.endswith("_clf"):
        train, val = _classification_files(tmp_path, d=2)
    else:
        train, val = _endpoint_files(tmp_path, "pendulum", n_train=64, n_val=16)
    trainer_mod = importlib.import_module(cfg.predictor.trainer_target.rsplit(".", 1)[0])
    seen = _capture_trainer(monkeypatch, trainer_mod)
    trainer = hydra.utils.get_class(cfg.predictor.trainer_target)(cfg, system, "pendulum")
    with pytest.raises(_StopFit):
        trainer.fit({"train": train, "val": val}, str(tmp_path / "out"))
    # FM's default is a progress bar; the MLP/BNN trainers never had one.
    assert seen["kw"]["enable_progress_bar"] is ((pred == "fm_det") and not override)
    dm = seen["dm"]
    assert dm.batched_indexing is override
    if pred == "fm_det":
        assert seen["model"].mae_val_frequency == (1000000 if override else 10)


# ------------------------------------------- engine: one real adaptive epoch (slow)
# Same layout as test_fpr_cap_engine.py's engine tests: the deterministic pendulum
# det_rerun splits, skipped without them. Run with CUDA_VISIBLE_DEVICES= for CPU.

PEN_SPLITS = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/pendulum/"
                  "lqr_failset/splits_det_rerun")


def _engine_run(pred, out, *extra):
    if not (PEN_SPLITS / "val_set.txt").exists():
        pytest.skip("pendulum det_rerun splits not mounted")
    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine
    o = ["system=pendulum", f"predictor={pred}", "acquisition=direct", "+experiment=det_rerun",
         "device=cpu", "n_epochs=1", "initial_train_size=100", "samples_per_epoch=50",
         "eval.max_eval_rows=200", "eval.num_mc_samples_eval=8", "acquisition_mc_samples=4",
         "prediction_store.endpoint_epochs=[0]", "num_workers=0", f"output_dir={out}",
         "acquisition.max_samples_per_epoch=200"]
    if pred == "fm_det":
        o += ["predictor.lightning_trainer.max_epochs=2"]
    elif pred.endswith("_clf"):
        o += ["predictor.bnn.max_epochs=3"]
    else:
        o += ["predictor.final_state.max_epochs=3"]
    cfg = _compose(*o, *extra)
    AdaptiveEngine(cfg).run()
    return Path(out), cfg


def _ckpt_state(path):
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def _assert_same_run(a: Path, b: Path):
    """Every result file of run b equals run a's. Skips the files that record the
    run's own path or wall clock (provenance.json, resolved_config.yaml, logs)."""
    import filecmp
    import json
    names = lambda d: sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()
                             and "fs_logs" not in p.parts and "bnn_logs" not in p.parts
                             and "version_" not in p.as_posix() and "events.out" not in p.name
                             and p.name not in ("provenance.json", "resolved_config.yaml"))
    fa, fb = names(a), names(b)
    assert fa == fb
    assert any(f.endswith("full_roa_per_point.npz") for f in fa)
    for rel in fa:
        pa, pb = a / rel, b / rel
        if rel.endswith(".npz"):
            za, zb = np.load(pa), np.load(pb)
            assert za.files == zb.files, rel
            for k in za.files:
                assert np.array_equal(za[k], zb[k], equal_nan=True), (rel, k)
        elif rel.endswith(".ckpt"):
            sa, sb = _ckpt_state(pa), _ckpt_state(pb)
            assert sa.keys() == sb.keys()
            assert all(torch.equal(sa[k], sb[k]) for k in sa), rel
        elif rel.endswith(".pt"):
            ta, tb = torch.load(pa, weights_only=False), torch.load(pb, weights_only=False)
            assert repr(ta) == repr(tb), rel
        elif rel.endswith(".json"):
            assert json.loads(pa.read_text()) == json.loads(pb.read_text()), rel
        elif rel.endswith(".png"):
            continue                          # matplotlib output; the npz behind it is compared
        else:
            assert filecmp.cmp(pa, pb, shallow=False), rel


IDENTITY_CELLS = [
    ("det_mlp_reg", ["+predictor.final_state.reuse_body_features=true"]),
    ("det_bnn_reg", ["+predictor.final_state.reuse_body_features=true"]),
    ("det_mlp_clf", []),
    ("det_bnn_clf", []),
    ("fm_det", []),
]


@pytest.mark.slow
@pytest.mark.parametrize("pred,extra", IDENTITY_CELLS, ids=[c[0] for c in IDENTITY_CELLS])
def test_bit_identical_flags_leave_a_whole_epoch_unchanged(pred, extra, tmp_path):
    """d (batched indexing), g (reuse_body_features) and h (load once, always on):
    training, threshold fit, acquisition, eval and the prediction store all match."""
    a, _ = _engine_run(pred, tmp_path / "a")
    b, _ = _engine_run(pred, tmp_path / "b", "+predictor.batched_indexing=true", *extra)
    _assert_same_run(a, b)


@pytest.mark.slow
def test_reuse_cal_skip_endpoint_error_and_plot_epochs_in_one_epoch(tmp_path, monkeypatch):
    import json
    from adaptive_roa.adaptive_v2.calibration.conformal_calibration import ConformalCalibrationBackend
    from adaptive_roa.adaptive_v2.eval import prediction_store as ps
    from adaptive_roa.conformal import ConformalConfig
    from adaptive_roa.conformal.calibrator import Calibrator

    old, _ = _engine_run("det_mlp_reg", tmp_path / "old")

    def second_pass(*a, **kw):
        raise AssertionError("calibrate_eval ran its own MC pass on the cal split")
    written = []
    orig_write = ps.write_split
    monkeypatch.setattr(ConformalCalibrationBackend, "calibrate_eval", second_pass)
    monkeypatch.setattr(ps, "write_split", lambda ep, split, *a, **kw: (written.append(split),
                                                                       orig_write(ep, split, *a, **kw)))
    new, cfg = _engine_run("det_mlp_reg", tmp_path / "new", "+prediction_store.reuse_cal_for_qhat=true",
                           "adaptive_v2.endpoint_error=false", "eval.plot_epochs=[]")
    assert written == ["val"]                         # cal was written from the q_hat prediction

    ao = json.loads((old / "epoch_000" / "artifacts_v2.json").read_text())
    an = json.loads((new / "epoch_000" / "artifacts_v2.json").read_text())
    # f: skipped with a reason; computed by default.
    assert an["endpoint_error"] == {"skipped": True, "reason": "adaptive_v2.endpoint_error=false"}
    assert ao["endpoint_error"]["n_endpoint_pairs"] > 0
    # i: no plots at an epoch plot_epochs leaves out; plots by default.
    pngs = {"full_roa_projections.png", "full_roa_heatmap.png"}
    assert not pngs & {p.name for p in (new / "epoch_000").iterdir()}
    assert pngs <= {p.name for p in (old / "epoch_000").iterdir()}
    # e: same store files as before, and q_hat_eval replays from the stored counts.
    for name in ("predictions_cal.npz", "endpoints_cal.npz"):
        assert np.load(old / "epoch_000" / name).files == np.load(new / "epoch_000" / name).files
    z = np.load(new / "epoch_000" / "predictions_cal.npz")
    k = int(z["num_mc_samples"])
    assert k == cfg.calibration.num_mc_samples == cfg.eval.num_mc_samples_eval
    rule = str(cfg.calibration.decision_rule)
    thr = an["threshold_state"]
    cal = Calibrator(ConformalConfig(delta=float(cfg.calibration.delta), alpha=float(cfg.calibration.alpha_eval),
                                     decision_rule=rule))
    q = cal.calibrate(z["n_success"] / k, z["true_labels"], thr["lambda_star"], thr["delta_star"],
                      z["n_failure"] / k if rule == "two_sided" else None, verbose=False)
    assert q == thr["q_hat_eval"] == an["eval_metrics"]["q_hat_eval"]
    assert an["extra"]["n_cal_eval"] == len(z["true_labels"])


@pytest.mark.slow
def test_reuse_cal_changes_nothing_for_a_classifier(tmp_path):
    """A classifier's cal prediction draws nothing from the global RNG, and with
    the cal set inside one forward batch (200 rows here) both paths see the same
    logits, so the whole epoch is unchanged."""
    a, _ = _engine_run("det_bnn_clf", tmp_path / "a")
    b, _ = _engine_run("det_bnn_clf", tmp_path / "b", "+prediction_store.reuse_cal_for_qhat=true")
    _assert_same_run(a, b)
