import os
import numpy as np
from adaptive_roa.systems.pendulum import PendulumSystem
from adaptive_roa.partx.tree import PartitionTree
from adaptive_roa.partx.viz import plot_region_tree


def test_plot_region_tree_writes_png(tmp_path):
    system = PendulumSystem()
    def latent_fn(X):
        return -X[:, 0], np.full(len(X), 0.02)
    tree = PartitionTree(system, delta=0.2, m_class=64)
    for _ in range(3):
        tree.refine(latent_fn)
    out = plot_region_tree(tree, str(tmp_path / "tree.png"))
    assert os.path.exists(out) and os.path.getsize(out) > 0
