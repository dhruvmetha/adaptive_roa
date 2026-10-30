"""Store scoring (tracker H01): denominators, threshold-free scores, fixed-precision point."""
import numpy as np
import pytest

from adaptive_roa.adaptive_v2.eval import store_score as ss

sklearn = pytest.importorskip("sklearn.metrics")


def test_auroc_and_auprc_match_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    p = np.round(np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1), 1)   # many ties
    assert ss._auroc(p, y) == pytest.approx(sklearn.roc_auc_score(y, p))
    assert ss._auprc(p, y) == pytest.approx(sklearn.average_precision_score(y, p))


def test_recovery_and_fpr_use_all_states_as_denominators():
    # 4 successes, 4 failures; one success abstains (uncertain), one failure admitted
    p = np.array([.9, .9, .9, .5, .1, .1, .1, .9])
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    r = ss.score_epoch(p, 1 - p, np.zeros(8), y, lam=0.5, delta=0.2, rule="one_sided", inv_thr=None)
    assert r["success_precision"] == pytest.approx(3 / 4)
    assert r["success_recovery"] == pytest.approx(3 / 4)      # abstained success still counts
    assert r["fpr_all_failures"] == pytest.approx(1 / 4)
    assert r["abstain_uncertain"] == pytest.approx(1 / 8)


def test_precision_threshold_is_chosen_on_val_only():
    p_val = np.array([.95, .9, .8, .7, .6])
    y_val = np.array([1, 1, 1, 0, 1])
    t = ss.threshold_for_precision(p_val, y_val, 0.99)
    assert t == pytest.approx(0.8)
    assert ss.threshold_for_precision(np.array([.9, .8]), np.array([0, 0]), 0.99) is None


def test_val_thresholds_refit_on_the_stored_val_split(tmp_path):
    import json

    from omegaconf import OmegaConf

    from adaptive_roa.adaptive_v2.eval.store_threshold import refit
    tcfg = {"predictor_type": "fm", "decision_rule": "two_sided", "delta": 0.05, "w": 0.9, "alpha": 0.1,
            "attractor_radius": 0.2, "num_mc_samples": 10, "optimize_mode": "joint", "optimize_objective": "loss",
            "target_f1": 0.9, "lambda_grid_size": 20, "delta_grid_size": 10, "delta_min": 0.05, "delta_max": 0.45,
            "fixed_lambda_star": 0.5, "fixed_delta_star": 0.1, "use_p_invalid_veto": True, "verbose": False}
    OmegaConf.save(OmegaConf.create({"decision_rule": "two_sided", "threshold": tcfg}), tmp_path / "resolved_config.yaml")
    rng = np.random.default_rng(1)
    ep = tmp_path / "epoch_000"
    ep.mkdir()
    y = np.where(rng.random(400) < 0.4, 1, -1)
    n_s = np.clip(np.where(y == 1, 80, 20) + rng.integers(-15, 15, 400), 0, 95)
    n_i = rng.integers(0, 5, 400)
    np.savez(ep / "predictions_val.npz", true_labels=y, n_success=n_s, n_failure=100 - n_s - n_i, n_invalid=n_i,
             num_mc_samples=np.int32(100))
    p = rng.random(50)
    np.savez(ep / "full_roa_per_point.npz", p_success=p, p_failure=1 - p, p_invalid=np.zeros(50),
             true_labels=np.where(p > 0.5, 1, -1))
    (ep / "artifacts_v2.json").write_text(json.dumps({"epoch": 0, "train_trajectories": 10, "eval_metrics": {
        "lambda_delta": {"lambda_star": 0.7, "delta": 0.2, "invalid_threshold": 0.5}}}))
    (run_row,) = ss.score_run(tmp_path, "run")
    (val_row,) = ss.score_run(tmp_path, "val")
    assert (run_row["lambda_star"], run_row["delta_star"], run_row["thresholds"]) == (0.7, 0.2, "run")
    lam, delta = refit(tcfg, 0.9, n_s / 100, (100 - n_s - n_i) / 100, n_i / 100, y)
    assert (val_row["lambda_star"], val_row["delta_star"], val_row["thresholds"]) == (lam, delta, "val")
    with pytest.raises(ValueError):
        ss.score_run(tmp_path, "cal")
