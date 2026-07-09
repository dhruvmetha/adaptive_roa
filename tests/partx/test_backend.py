import numpy as np
from omegaconf import OmegaConf
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.gp_classifier import GPClassifier
from adaptive_roa.partx.model_handle import GPModelHandle
from adaptive_roa.partx.backend import GPProbabilityBackend


def test_backend_estimate_and_latent():
    system = PendulumSystem()
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-3, 3, 200), rng.uniform(-8, 8, 200)])
    y = ((X[:, 0] ** 2 + (X[:, 1] / 3) ** 2) < 1.0).astype(int)
    gp = GPClassifier(system, n_inducing=64, n_iters=150).fit(X, y)
    backend = GPProbabilityBackend(OmegaConf.create({}), system, "cpu")
    backend.bind_model(GPModelHandle(gp, system))

    probs = backend.estimate(X[:10])
    assert probs.p_success.shape == (10,)
    assert np.all((probs.p_success >= 0) & (probs.p_success <= 1))
    assert np.allclose(probs.p_failure, 1.0 - probs.p_success)
    assert np.allclose(probs.p_invalid, 0.0)

    m, s2 = backend.latent_posterior(X[:10])
    assert m.shape == (10,) and np.all(s2 > 0)
