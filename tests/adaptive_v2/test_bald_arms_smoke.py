"""One tiny adaptive epoch on the real engine, per BALD arm added 2026-09-13.

Guards the failure mode that costs a whole campaign: an acquisition strategy
that raises on a predictor family, or silently scores zero on it, is otherwise
discovered only after every arm has burned its GPU time.

The non-zero `epistemic_mean` assertion is the load-bearing one. A BALD arm
whose members agree everywhere selects by an arbitrary tie-break while
reporting a fully adaptive configuration, and nothing in the artifacts says so.

Runs on pendulum_stoch because it is the cheapest pool with a ground-truth grid;
the backend, strategy and trainer code under test is system-independent, and
`test_bald_arm_configs.py` separately checks the per-system width resolves on
all three campaign systems.
"""
import json
import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = "/common/home/st1122/Projects/adaptive_roa/configs/adaptive_v2"

ARMS = ["bnn_mfvi_reg_bald", "bnn_ens_reg_bald", "fm_outcome_bald"]


def _register_resolvers() -> None:
    """Mirrors scripts/run_adaptive.py; a bare hydra.compose() registers none."""
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


def _overrides(arm, tmp_path):
    common = [
        "system=pendulum_stoch", "noise_level=high",
        f"predictor={arm}", "acquisition=decomp_epi_bald",
        "acquisition.d2_ratio=1.0", "acquisition.selection_rule=greedy",
        "acquisition.n_candidates=500",
        "n_epochs=1", "initial_train_size=200", "samples_per_epoch=200",
        "eval.max_eval_rows=200", "eval.num_mc_samples_eval=4",
        "num_workers=0", "adaptive_v2.filter_confident_pairs=false",
        # The campaign's timeout policy. Exercised here so a run that cannot
        # build its dataset under `drop` fails in seconds rather than in queue.
        "++data_source.timeout_intermediates=drop",
        f"output_dir={tmp_path}/{arm}",
    ]
    if arm.startswith("bnn_"):
        # num_mc_samples=100, well above the shipped 20, ON PURPOSE. K sets
        # BALD's noise floor at ~(1/2K)(1-1/M), and the assertion below has to
        # be able to tell signal from that floor. Simulated null (members
        # identical, K-sample noise only) at M=4: K=10 gives mean 0.043, K=100
        # gives 0.004. At 500 candidates the extra draws are cheap, and they are
        # what make this test able to fail on a degenerate arm.
        common += ["predictor.final_state.max_epochs=2",
                   "probability.n_posterior_samples=4",
                   "probability.num_mc_samples=100"]
        if arm == "bnn_ens_reg_bald":
            common += ["predictor.final_state.n_members=2"]
    else:
        # NOT 2 epochs. Measured budget sweep (2026-09-13), pendulum/high:
        #   epochs  M  ode   epistemic_mean   epistemic_max
        #        2  2   10        5.2e-07         2.5e-06     <- degenerate
        #       20  3   50        9.3e-04         1.1e-02
        #       60  5   50        2.9e-03         1.3e-02
        # At 2 epochs the flow predicts p ~ 1.0 everywhere, members agree, and
        # BOTH uncertainty terms vanish; the arm looks broken when it is only
        # untrained. 20 epochs is the cheapest row with signal to assert on.
        common += ["predictor.outcome_fm.max_epochs=20",
                   "predictor.ensemble.n_members=3",
                   "predictor.outcome_fm.num_ode_steps=50"]
    return common


@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_one_adaptive_epoch_runs(arm, tmp_path):
    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine
    _register_resolvers()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        AdaptiveEngine(compose(config_name="default",
                               overrides=_overrides(arm, tmp_path))).run()
    assert (tmp_path / arm / "epoch_000" / "full_roa_per_point.npz").exists()


def _bald_noise_floor(diag) -> float:
    """What BALD reads when members carry ONLY K-sample MC noise.

    ~(1/2K)(1 - 1/M), from uncertainty_scores.py's finite-sample warning. At the
    shipped K=10, M=5 that is 0.04 nats, so `epistemic_mean > 0` is not evidence
    of anything on an MC-member arm: pure sampling noise clears it comfortably.
    Backends whose members carry no sampling noise report member_sample_size
    None and have no floor.
    """
    k = diag.get("member_sample_size")
    m = int(diag.get("n_members", 0))
    if not k or m < 2:
        return 0.0
    return (1.0 / (2.0 * float(k))) * (1.0 - 1.0 / m)


@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS)
def test_acquisition_found_real_epistemic_disagreement(arm, tmp_path):
    from adaptive_roa.adaptive_v2.engine import AdaptiveEngine
    _register_resolvers()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        AdaptiveEngine(compose(config_name="default",
                               overrides=_overrides(arm, tmp_path))).run()

    art = json.loads((tmp_path / arm / "epoch_000" / "artifacts_v2.json").read_text())
    diag = art.get("acquisition", {}).get("diagnostics", {})
    assert diag.get("score_mode") == "epistemic_bald", diag
    assert diag.get("n_members", 0) >= 2, diag

    # The MEAN over candidates, against the floor. Deliberately not score_max:
    # a max over 500 candidates is an extreme-value statistic and clears a
    # mean-valued floor by ~7x even on a fully degenerate arm (measured), so an
    # assertion on it has almost no power. The mean is what the floor describes.
    floor = _bald_noise_floor(diag)
    assert diag["epistemic_mean"] > floor, (
        f"{arm}: mean BALD {diag['epistemic_mean']:.5g} does not clear its own "
        f"MC-noise floor {floor:.5g}, so the score is consistent with members that "
        f"agree everywhere. Selection would be an arbitrary tie-break while the "
        f"config reports an adaptive arm. {diag}")

    # NOT asserted: epistemic_mean_selected > epistemic_mean. With
    # score_mode=epistemic_bald, `score` and `epi` in decomposition._diagnostics
    # are the SAME array and greedy takes the top 200 of 500, so
    # mean-of-top-40% > overall mean holds unless every score ties exactly. That
    # tests select_greedy, not acquisition, and it passes on a degenerate arm.
