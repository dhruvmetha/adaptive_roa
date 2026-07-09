import numpy as np
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.tree import PartitionTree
from adaptive_roa.partx.bounds import roa_volume_bound


def test_volume_matches_known_halfspace():
    system = PendulumSystem()
    # RoA = {θ < 0}: exactly half the (normalized) support -> volume ~ 0.5.
    def latent_fn(X):
        return -X[:, 0], np.full(len(X), 0.01)   # f>0 iff θ<0
    tree = PartitionTree(system, delta=0.1, m_class=128)
    for _ in range(4):
        tree.refine(latent_fn)
    out = roa_volume_bound(tree, latent_fn, R=100, M=64)
    assert 0.4 < out["volume"] < 0.6
    assert out["ci_low"] <= out["volume"] <= out["ci_high"]
