"""threshold=fpr_cap in the v2 engine (det rerun, Idea 6), and the lambda/delta path left as it was.

The slow tests run one real adaptive epoch on the deterministic pendulum for FM and
the MLP classifier, under ranked and direct, modelled on test_bald_arms_smoke.py.
They need the shared det_rerun splits and skip without them. Run them with
CUDA_VISIBLE_DEVICES= to keep them on the CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval import fpr_cap
from adaptive_roa.adaptive_v2.types import ThresholdState

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "adaptive_v2"
PEN_SPLITS = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/pendulum/"
                  "lqr_failset/splits_det_rerun")


def _register_resolvers() -> None:
    """Mirrors scripts/run_adaptive.py (and test_bald_arms_smoke.py); compose() registers none."""
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
    _register_resolvers()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="default", overrides=list(overrides))


# ---------------------------------------------------------------- configs


@pytest.mark.parametrize("acq", ["ranked", "direct"])
def test_existing_det_rerun_cell_resolves_to_lambda_delta(acq):
    from adaptive_roa.adaptive_v2.strategy.direct import DirectAcquisitionStrategy
    from adaptive_roa.adaptive_v2.strategy.ranked import RankedAcquisitionStrategy
    from adaptive_roa.adaptive_v2.threshold.conformal_threshold import ConformalThresholdBackend

    cfg = _compose("system=pendulum", "predictor=fm_det", f"acquisition={acq}", "+experiment=det_rerun",
                   "output_dir=/tmp/unused")
    assert cfg.threshold._target_.endswith("conformal_threshold.ConformalThresholdBackend")
    assert "rule" not in cfg.acquisition and "alpha_s" not in cfg.threshold
    assert not hasattr(ConformalThresholdBackend, "fit_split")
    s = (RankedAcquisitionStrategy if acq == "ranked" else DirectAcquisitionStrategy)(cfg.acquisition)
    assert s.rule == "lambda_delta"


@pytest.mark.parametrize("pred", ["fm_det", "det_mlp_clf", "det_bnn_clf", "det_mlp_reg", "det_bnn_reg"])
@pytest.mark.parametrize("acq", ["ranked", "direct"])
def test_fprcap_experiment_composes(pred, acq):
    cfg = _compose("system=pendulum", f"predictor={pred}", f"acquisition={acq}", "+experiment=det_rerun_fprcap")
    assert cfg.threshold._target_.endswith("fpr_cap_threshold.FprCapThresholdBackend")
    assert cfg.threshold.alpha_s == 0.005 and cfg.threshold.fit_mc_samples == 20
    assert cfg.acquisition.rule == "tau"
    assert cfg.acquisition.num_mc_samples == 20 and cfg.eval.num_mc_samples_eval == 100
    assert cfg.acquisition.fill_shortfall is True and cfg.prediction_store.enabled is True
    raw = OmegaConf.to_container(cfg, resolve=False)["output_dir"]
    assert "${acquisition.rule}" in raw and "${threshold.alpha_s}" in raw
    for knob in ("${predictor.name}", "${sampling_mode}", "${acquisition.d2_ratio}", "${seed}"):
        assert knob in raw


def test_fprcap_output_dir_never_matches_a_det_rerun_dir():
    over = ["system=pendulum", "predictor=fm_det", "acquisition=ranked"]
    a = OmegaConf.to_container(_compose(*over, "+experiment=det_rerun"), resolve=False)["output_dir"]
    b = OmegaConf.to_container(_compose(*over, "+experiment=det_rerun_fprcap"), resolve=False)["output_dir"]
    assert a != b and "_rule_" not in a and "_rule_${acquisition.rule}_alpha_s_${threshold.alpha_s}" in b


def test_strategies_refuse_a_mismatched_rule():
    from adaptive_roa.adaptive_v2.strategy.direct import taus_for_rule
    ld = ThresholdState(lambda_star=0.5, delta_star=0.1)
    tau = ThresholdState(lambda_star=0.3, delta_star=-0.1, tau_s=0.5, tau_f=0.3, tau_inv=1.01, alpha_s=0.005)
    assert taus_for_rule("lambda_delta", ld) is None
    assert taus_for_rule("tau", tau) == (0.5, 0.3, 1.01)
    with pytest.raises(ValueError):
        taus_for_rule("tau", ld)
    with pytest.raises(ValueError):
        taus_for_rule("lambda_delta", tau)


# ---------------------------------------------------------------- backend, eval, store, q_hat


def test_backend_fits_on_given_probabilities_and_converts():
    from adaptive_roa.adaptive_v2.threshold.fpr_cap_threshold import FprCapThresholdBackend
    cfg = OmegaConf.create({"predictor_type": "generative", "decision_rule": "two_sided", "alpha_s": 0.01,
                            "fit_mc_samples": 20, "verbose": False})
    b = FprCapThresholdBackend(cfg, system=None, device="cpu")
    assert b.fit_split == "val" and b.predictor is None and b.optimize_mode == "fpr_cap"
    rng = np.random.default_rng(0)
    q = rng.beta(0.5, 0.5, 2000)
    n = rng.binomial(20, q)
    y = np.where(rng.random(2000) < q, 1, -1)
    st = b.optimize_val(np.zeros((2000, 2)), y, n / 20, 1 - n / 20, np.zeros(2000), 20)
    want = fpr_cap.fit(n / 20, 1 - n / 20, np.zeros(2000), y, [0.01])[0.01]
    assert (st.tau_s, st.tau_f, st.tau_inv, st.alpha_s) == (want["tau_s"], want["tau_f"], want["tau_inv"], 0.01)
    assert np.isclose(st.lambda_star, (1 + st.tau_s - st.tau_f) / 2)
    assert np.isclose(st.delta_star, (st.tau_s + st.tau_f - 1) / 2)
    assert b.last_fit["num_mc_samples"] == 20 and len(b.last_fit["labels"]) == 2000
    with pytest.raises(RuntimeError):
        b.optimize(np.zeros((3, 2)), np.ones(3))


def test_unset_taus_leave_the_artifact_payload_as_it_was():
    from dataclasses import asdict
    from adaptive_roa.adaptive_v2.engine import _drop_unset_taus
    d = asdict(ThresholdState(lambda_star=0.5, delta_star=0.1))
    _drop_unset_taus(d)
    assert list(d) == ["lambda_star", "delta_star", "q_hat", "q_hat_eval", "q_hat_success_eval",
                       "q_hat_failure_eval"]
    d = asdict(ThresholdState(lambda_star=0.3, delta_star=-0.1, tau_s=0.5, tau_f=0.3, tau_inv=0.9, alpha_s=0.005))
    _drop_unset_taus(d)
    assert d["tau_s"] == 0.5 and d["alpha_s"] == 0.005


class _Clf:
    """A fixed classifier: logit = 4 * x0."""
    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, x):
        return 4.0 * x[:, 0]


def _eval_file(tmp_path, n=400, seed=0):
    """start (2), end (2), label 0/1 -- the deterministic eval layout."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
    y01 = (X[:, 0] + rng.normal(0, 0.3, n) > 0).astype(int)
    f = tmp_path / "eval.txt"
    np.savetxt(f, np.c_[X, X, y01], delimiter=",", fmt=["%.6f"] * 4 + ["%d"])
    X = np.loadtxt(f, delimiter=",")[:, :2].astype(np.float32)
    return f, X, np.where(y01 == 1, 1, -1)


def test_classifier_eval_labels_with_the_tau_rule(tmp_path):
    import torch
    from adaptive_roa.adaptive_v2.eval.full_roa import FullROAEvaluator
    f, X, y = _eval_file(tmp_path)
    cfg = OmegaConf.create({"predictor_type": "classifier", "alpha_eval": 0.1, "attractor_radius": None,
                            "num_mc_samples_eval": 1, "decision_rule": "one_sided", "refine_invalids": False,
                            "refine_t_min": 0.7, "refine_t_max": 0.9, "refine_num_steps": 10,
                            "refine_max_attempts": 1, "max_eval_rows": None, "verbose": False,
                            "binary_outcomes": False})

    class Sys:
        state_dim = 2
    ev = FullROAEvaluator(cfg, Sys(), "cpu")
    ts, tf, ti = 0.7, 0.2, 1.01               # tau_s + tau_f < 1: both can fire, delta < 0
    lam, dl = fpr_cap.to_lambda_delta(ts, tf)
    state = ThresholdState(lambda_star=lam, delta_star=dl, tau_s=ts, tau_f=tf, tau_inv=ti, alpha_s=0.005)
    out = tmp_path / "ep"
    m = ev.evaluate_epoch(_Clf(), state, {"eval_states_file": str(f), "output_dir": str(out), "verbose": False})
    p = torch.sigmoid(4.0 * torch.from_numpy(X[:, 0])).to(torch.float64).numpy()
    want = fpr_cap.predict(p, 1 - p, np.zeros(len(p)), ts, tf, ti)
    ld = m["lambda_delta"]
    assert ld["rule"] == "fpr_cap" and (ld["tau_s"], ld["tau_f"], ld["tau_inv"]) == (ts, tf, ti)
    assert ld["n_uncertain"] == int(np.sum(want == -1)) and ld["n_both_fire"] > 0
    assert ld["tp"] == int(np.sum((want == 1) & (y == 1))) and ld["fp"] == int(np.sum((want == 1) & (y != 1)))
    z = np.load(out / "full_roa_per_point.npz")
    assert float(z["tau_s"]) == ts and float(z["delta"]) == pytest.approx(dl)
    # lambda/delta call unchanged: no tau keys in the per-point file.
    m2 = ev.evaluate_epoch(_Clf(), ThresholdState(lambda_star=0.5, delta_star=0.1),
                           {"eval_states_file": str(f), "output_dir": str(tmp_path / "ep2"), "verbose": False})
    assert "rule" not in m2["lambda_delta"]
    assert "tau_s" not in np.load(tmp_path / "ep2" / "full_roa_per_point.npz").files


def test_eval_q_hat_runs_on_a_negative_converted_delta():
    """The eval-time q_hat (calibration backend) sets lambda*/delta* on its predictor
    directly; its ConformalConfig keeps calibration.delta, so a negative converted delta
    never reaches ConformalConfig's 0 < delta < 0.5 assertion."""
    from adaptive_roa.adaptive_v2.calibration.conformal_calibration import ConformalCalibrationBackend
    from adaptive_roa.adaptive_v2.eval.full_roa import _predict_qhat_prediction_sets
    from adaptive_roa.conformal import ConformalConfig, ConformalPredictor

    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 500)
    y = np.where(rng.random(500) < p, 1, -1)

    class Est:
        def estimate(self, X, **kw):
            idx = np.asarray(X, int).ravel()
            return p[idx], 1 - p[idx], np.zeros(len(idx))

    lam, dl = fpr_cap.to_lambda_delta(0.5, 0.16)          # pen FM golden cell: delta = -0.17
    assert dl < 0
    for rule in ("two_sided", "one_sided"):
        cfg = OmegaConf.create({"delta": 0.05, "w": 0.9, "alpha": 0.1, "alpha_eval": 0.1, "decision_rule": rule,
                                "attractor_radius": 0.1, "num_mc_samples": 10, "verbose": False})
        cb = ConformalCalibrationBackend(cfg, system=None, device="cpu")
        conf = ConformalConfig(delta=cb.delta, w=cb.w, alpha=cb.alpha_eval, attractor_radius=0.1,
                               num_mc_samples=10, decision_rule=rule)
        cb._predictor_eval = ConformalPredictor(flow_matcher=None, system=None, config=conf, device="cpu",
                                                probability_estimator=Est())
        q = cb.calibrate_eval(np.arange(500)[:, None], y,
                              ThresholdState(lambda_star=lam, delta_star=dl, tau_s=0.5, tau_f=0.16,
                                             tau_inv=0.9, alpha_s=0.005))
        assert np.isfinite(q)
        pred, _ = _predict_qhat_prediction_sets(p, 1 - p, np.zeros(500), y, lam, dl, q, rule, None)
        assert pred.shape == (500,)
    with pytest.raises(AssertionError):                    # the hazard this avoids
        ConformalConfig(delta=dl)


def test_store_check_replays_the_tau_rule(tmp_path):
    from adaptive_roa.adaptive_v2.eval.full_roa import _classification_metrics_from_predictions
    from adaptive_roa.adaptive_v2.eval.store_check import check_run
    rng = np.random.default_rng(1)
    run = tmp_path / "run"
    ep = run / "epoch_000"
    ep.mkdir(parents=True)
    (run / "resolved_config.yaml").write_text("decision_rule: two_sided\n")
    k = 20
    c = np.array([rng.multinomial(k, [0.45, 0.45, 0.1]) for _ in range(500)])
    ps_, pf_, pi_ = c[:, 0] / k, c[:, 1] / k, c[:, 2] / k
    y = np.where(rng.random(500) < ps_, 1, -1)
    np.savez(ep / "full_roa_per_point.npz", p_success=ps_, p_failure=pf_, p_invalid=pi_, true_labels=y)
    taus = (0.5, 0.3, 0.2)                                   # both-fire states and a free tau_inv
    pred = fpr_cap.predict(ps_, pf_, pi_, *taus)
    ld = _classification_metrics_from_predictions(pred, y)
    lam, dl = fpr_cap.to_lambda_delta(taus[0], taus[1])
    ld.update(lambda_star=lam, delta=dl)
    art = {"epoch": 0, "eval_metrics": {"lambda_delta": ld},
           "threshold_state": {"lambda_star": lam, "delta_star": dl, "tau_s": taus[0], "tau_f": taus[1],
                               "tau_inv": taus[2], "alpha_s": 0.005}}
    (ep / "artifacts_v2.json").write_text(json.dumps(art))
    assert check_run(run) == []
    del art["threshold_state"]["tau_s"]                      # replayed as lambda/delta, the counts differ
    (ep / "artifacts_v2.json").write_text(json.dumps(art))
    assert any("replayed counts differ" in p for p in check_run(run))


# ---------------------------------------------------------------- one real epoch


ENGINE_CELLS = [("fm_det", "ranked"), ("fm_det", "direct"), ("det_mlp_clf", "ranked"), ("det_mlp_clf", "direct")]


def _engine_overrides(pred, acq, out):
    o = [f"system=pendulum", f"predictor={pred}", f"acquisition={acq}", "+experiment=det_rerun_fprcap",
         "device=cpu", "n_epochs=1", "initial_train_size=100", "samples_per_epoch=50",
         "eval.max_eval_rows=300", "eval.num_mc_samples_eval=30", "'prediction_store.endpoint_epochs=[0]'",
         "num_workers=0", f"output_dir={out}"]
    o += ["acquisition.n_ranked_candidates=300"] if acq == "ranked" else ["acquisition.max_samples_per_epoch=300"]
    if pred == "fm_det":
        o += ["predictor.lightning_trainer.max_epochs=2"]
    else:
        o += ["predictor.bnn.max_epochs=3"]
    return [x.strip("'") for x in o]


@pytest.mark.slow
@pytest.mark.parametrize("pred,acq", ENGINE_CELLS, ids=[f"{p}-{a}" for p, a in ENGINE_CELLS])
def test_one_fprcap_epoch_runs_writes_taus_and_acquires_with_them(pred, acq, tmp_path, monkeypatch):
    if not (PEN_SPLITS / "val_set.txt").exists():
        pytest.skip("pendulum det_rerun splits not mounted")
    from adaptive_roa.adaptive.balanced_sampler import UncertainSampler
    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine
    from adaptive_roa.adaptive_v2.eval.store_check import check_run

    seen = []
    name = "sample_ranked" if acq == "ranked" else "sample_direct"
    orig = getattr(UncertainSampler, name)

    def spy(self, *a, **kw):
        res = orig(self, *a, **kw)
        seen.append((kw.get("rule"), kw.get("taus"), res))
        return res
    monkeypatch.setattr(UncertainSampler, name, spy)

    out = tmp_path / f"{pred}_{acq}"
    cfg = _compose(*_engine_overrides(pred, acq, out))
    AdaptiveEngine(cfg).run()

    ep = out / "epoch_000"
    art = json.loads((ep / "artifacts_v2.json").read_text())
    thr = art["threshold_state"]
    taus = (thr["tau_s"], thr["tau_f"], thr["tau_inv"])
    assert thr["alpha_s"] == 0.005 and thr["tau_s"] >= 0.5
    assert thr["tau_s"] in fpr_cap.GRID and thr["tau_f"] in fpr_cap.GRID
    assert np.isclose(thr["lambda_star"], (1 + taus[0] - taus[1]) / 2)
    assert np.isclose(thr["delta_star"], (taus[0] + taus[1] - 1) / 2)

    # Acquisition read the fitted taus.
    assert len(seen) == 1 and seen[0][0] == "tau" and tuple(seen[0][1]) == taus
    diag = art["acquisition"]["diagnostics"]
    assert diag["rule"] == "tau" and tuple(diag["taus"]) == taus
    res = seen[0][2]
    picked = res.selected_indices if acq == "ranked" else res.uncertain_indices
    assert list(picked) == art["acquisition"]["d2_indices"]

    # The in-run evaluation labelled test with the tau rule; the store replays it.
    ld = art["eval_metrics"]["lambda_delta"]
    assert ld["rule"] == "fpr_cap" and (ld["tau_s"], ld["tau_f"], ld["tau_inv"]) == taus
    assert check_run(out) == []

    # The val store: fit-K counts for endpoint models, the probability for classifiers;
    # a refit from what was stored gives back the recorded thresholds.
    z = np.load(ep / "predictions_val.npz")
    if pred == "fm_det":
        assert int(z["num_mc_samples"]) == 30 and "n_success_k20" in z.files
        import torch
        from adaptive_roa.systems.criteria import optional_radius
        e = np.load(ep / "endpoints_val.npz")
        import hydra
        system = hydra.utils.instantiate(cfg.system)
        radius = optional_radius(cfg.eval.attractor_radius)
        lab = np.stack([system.classify_attractor(torch.from_numpy(e["endpoints"][:, s]), radius).numpy()
                        for s in range(20)], 1)
        assert np.array_equal(z["n_success_k20"], (lab == 1).sum(1))
        assert np.array_equal(z["n_failure_k20"], (lab == -1).sum(1))
        (row,) = fpr_cap.score_run(out, [0.005], fit_k=20)
    else:
        assert z.files == ["p_success", "true_labels"]
        (row,) = fpr_cap.score_run(out, [0.005])
    assert (row["tau_s"], row["tau_f"], row["tau_inv"]) == taus and row["matches_run"] is True
    assert (out / "epoch_000" / "predictions_thrfit.npz").exists()
