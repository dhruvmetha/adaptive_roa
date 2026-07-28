import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ARMS = ["mlp_det", "bnn_mfvi_reg", "bnn_ensemble_reg", "bnn_laplace_reg"]

# Deliberately NOT the wrapper's hardcoded fallbacks (0.2 and 10), so a test
# that passes proves the config was read rather than the default guessed.
_RADIUS = 0.37
_K = 7

_POSTERIOR_OF = {
    "mlp_det": "deterministic",
    "bnn_mfvi_reg": "mfvi",
    "bnn_ensemble_reg": "ensemble",
    "bnn_laplace_reg": "laplace",
}


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_is_registered_for_export(arm):
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class

    cls = get_probabilistic_classifier_class(arm)
    assert cls.predictor_name == arm
    assert cls.predictor_type == "generative"
    assert cls.native_probs == ("p_success", "p_failure", "p_invalid")


def test_arms_do_not_steal_the_legacy_generative_alias():
    """Runs written before arm names existed resolve via the family alias."""
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.probabilistic_classifier.flow_matching import FMProbabilisticClassifier

    assert get_probabilistic_classifier_class("generative") is FMProbabilisticClassifier


# --- the export wrapper itself -------------------------------------------
#
# Nothing exercised load_from_run or predict before these. Everything they
# cover -- a probability vector that does not sum to 1, a silently-defaulted
# attractor radius, an inverted success/failure label convention -- produces
# plausible-looking numbers rather than an error.


def _endpoint_file(path, n=256, seed=0):
    """8-column cartpole endpoint pairs spanning the system's own bounds."""
    from adaptive_roa.systems.cartpole import CartPoleSystem

    rng = np.random.default_rng(seed)
    normalized = rng.uniform(-1.0, 1.0, size=(n, 4))
    normalized[:, 1] *= np.pi
    start = CartPoleSystem().denormalize_state(
        torch.as_tensor(normalized, dtype=torch.float32)
    ).numpy()
    np.savetxt(path, np.column_stack([start, start * 0.5]))
    return str(path)


def _cfg(arm):
    return OmegaConf.create({
        "device": "cpu",
        "probability": {"attractor_radius": _RADIUS, "num_mc_samples": _K},
        "predictor": {
            "type": "generative", "name": arm, "batch_size": 64, "val_batch_size": 64,
            "final_state": {
                "posterior": _POSTERIOR_OF[arm], "hidden_dims": [16, 16], "lr": 1e-2,
                "weight_decay": 1e-5, "max_epochs": 2, "patience": 5,
                "prior_sigma": 1.0, "n_members": 2, "kl_weight": 1.0, "beta_nll": 0.5,
            },
        },
    })


def _train_and_load(arm, tmp_path):
    from adaptive_roa.adaptive_v2.trainers.final_state_trainer import FinalStateTrainer
    from adaptive_roa.probabilistic_classifier import get_probabilistic_classifier_class
    from adaptive_roa.systems.cartpole import CartPoleSystem

    cfg = _cfg(arm)
    files = {"train": _endpoint_file(tmp_path / "train.txt"),
             "val": _endpoint_file(tmp_path / "val.txt", n=128, seed=1)}
    run_dir = tmp_path / "run"
    FinalStateTrainer(cfg, CartPoleSystem(), "cartpole_pybullet").fit(
        files, str(run_dir / "epoch_000")
    )
    cls = get_probabilistic_classifier_class(arm)
    loaded = cls.load_from_run(str(run_dir), 0, cfg, CartPoleSystem(), device="cpu")
    return loaded, cfg


@pytest.mark.parametrize("arm", ARMS)
def test_export_wrapper_round_trips_and_returns_a_normalized_distribution(arm, tmp_path):
    """load_from_run -> predict must work for every arm and return a genuine
    distribution. The three counts come from three separate accumulators over
    the same K draws, so a label that matches none of them (or two of them)
    silently yields probabilities that do not sum to 1."""
    loaded, _ = _train_and_load(arm, tmp_path)

    states = np.loadtxt(tmp_path / "val.txt")[:64, :4]
    probs = loaded.predict(states)

    total = probs.p_success + probs.p_failure + probs.p_invalid
    np.testing.assert_allclose(total, np.ones(len(states)), rtol=0, atol=1e-12)
    for p in (probs.p_success, probs.p_failure, probs.p_invalid):
        assert p.shape == (len(states),)
        assert ((p >= 0.0) & (p <= 1.0)).all()
        # K draws counted -> every probability is a multiple of 1/K.
        np.testing.assert_allclose(p * _K, np.round(p * _K), atol=1e-9)


@pytest.mark.parametrize("arm", ARMS)
def test_export_wrapper_resolves_radius_and_k_from_the_config(arm, tmp_path):
    """Both wrapper fallbacks (radius 0.2, K 10) are wrong for pendulum and both
    quadrotors, and a wrong radius relabels every MC endpoint with no
    downstream signal."""
    loaded, cfg = _train_and_load(arm, tmp_path)

    assert loaded.attractor_radius == pytest.approx(_RADIUS)
    assert loaded.num_mc_samples == _K
    # Guard the guard: these must not coincide with the fallbacks, or the
    # assertions above would pass without the config ever being read.
    assert _RADIUS != 0.2 and _K != 10
    assert loaded.attractor_radius == pytest.approx(float(cfg.probability.attractor_radius))


def test_export_wrapper_refuses_to_guess_an_attractor_radius():
    from adaptive_roa.probabilistic_classifier.bayesian_final_state import (
        _resolve_probability_cfg,
    )

    with pytest.raises(ValueError, match="attractor_radius"):
        _resolve_probability_cfg(OmegaConf.create({"probability": {"num_mc_samples": 4}}))


def test_export_wrapper_maps_labels_to_the_right_outcome():
    """classify_attractor returns 1=success, -1=failure, 0=invalid. Inverting
    success and failure changes no shape, keeps the sum at 1, and would be
    invisible in every other assertion here -- so pin the mapping directly with
    a system whose labels are known by construction."""
    from adaptive_roa.probabilistic_classifier.bayesian_final_state import (
        DeterministicFinalStateProbabilisticClassifier as Cls,
    )

    # 4 query points, one per label plus a repeat, fixed for every draw.
    labels = torch.tensor([1, -1, 0, 1])

    class StubSystem:
        def classify_attractor(self, endpoints, radius):
            assert radius == _RADIUS
            return labels

    class StubHandle:
        def predict_endpoint(self, batch):
            return torch.zeros(len(batch), 4)

    clf = Cls(StubHandle(), StubSystem(), "cpu", _RADIUS, _K)
    probs = clf.predict(np.zeros((4, 4)))

    np.testing.assert_allclose(probs.p_success, [1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(probs.p_failure, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(probs.p_invalid, [0.0, 0.0, 1.0, 0.0])


def test_export_wrapper_handles_an_empty_query():
    from adaptive_roa.probabilistic_classifier.bayesian_final_state import (
        DeterministicFinalStateProbabilisticClassifier as Cls,
    )

    probs = Cls(None, None, "cpu", _RADIUS, _K).predict(np.zeros((0, 4)))
    assert probs.p_success.shape == (0,)
