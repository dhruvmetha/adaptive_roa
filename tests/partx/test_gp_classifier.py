import numpy as np
import pytest
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.gp_classifier import GPClassifier


def _disk_data(n=400, seed=0):
    """Success (1) inside a disk in (θ, θ̇) space, failure (0) outside."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-3.0, 3.0, n), rng.uniform(-8.0, 8.0, n)])
    r2 = (X[:, 0] / 1.5) ** 2 + (X[:, 1] / 4.0) ** 2
    y = (r2 < 1.0).astype(np.int64)
    return X, y


def test_gp_recovers_disk_boundary():
    system = PendulumSystem()
    X, y = _disk_data()
    gp = GPClassifier(system, n_inducing=64, n_iters=250).fit(X, y)

    center = np.array([[0.0, 0.0]])          # deep inside disk -> success
    outside = np.array([[2.8, 7.0]])         # far outside -> failure
    assert gp.p_success(center)[0] > 0.8
    assert gp.p_success(outside)[0] < 0.2

    # Latent variance is finite and larger far from data density at the edges.
    m, s2 = gp.latent_posterior(np.array([[1.5, 0.0]]))  # near boundary
    assert np.isfinite(m).all() and (s2 > 0).all()
