import os

import numpy as np
from adaptive_roa.partx.eval import PartXEvaluator, merge_region_bounds
from adaptive_roa.partx.tree import PartitionTree
from adaptive_roa.systems.pendulum import PendulumSystem


def test_merge_region_bounds():
    base = {"coverage": 0.9, "f1": 0.8}
    diag = {"roa_volume": 0.42, "roa_volume_ci": [0.4, 0.44], "n_leaves": 12}
    out = merge_region_bounds(base, diag)
    assert out["coverage"] == 0.9
    assert out["partx_roa_volume"] == 0.42
    assert out["partx_roa_volume_ci"] == [0.4, 0.44]
    assert out["partx_n_leaves"] == 12


def test_merge_region_bounds_none_diag():
    base = {"coverage": 0.9}
    assert merge_region_bounds(base, None) == {"coverage": 0.9}


class _FakeBaseEvaluator:
    """Tiny stub standing in for the wrapped RoA evaluator."""

    def __init__(self, cfg, system, device):
        self.max_eval_rows = 1000

    def evaluate_epoch(self, model_handle, threshold_state, epoch_context):
        return {"coverage": 0.9, "f1": 0.8}


class _FakeModelHandle:
    def __init__(self, tree, diag):
        self.partx_tree = tree
        self.partx_diag = diag


def test_evaluate_epoch_writes_region_tree_png(tmp_path):
    system = PendulumSystem()

    def latent_fn(X):
        return -X[:, 0], np.full(len(X), 0.02)

    tree = PartitionTree(system, delta=0.2, m_class=64)
    for _ in range(3):
        tree.refine(latent_fn)

    diag = {"roa_volume": 0.5, "roa_volume_ci": [0.4, 0.6], "n_leaves": len(tree.leaves())}
    model_handle = _FakeModelHandle(tree, diag)

    cfg = type("Cfg", (), {"base": object()})()
    evaluator = PartXEvaluator.__new__(PartXEvaluator)
    evaluator._base = _FakeBaseEvaluator(cfg, system, "cpu")
    evaluator.max_eval_rows = evaluator._base.max_eval_rows
    evaluator.system = system

    output_dir = str(tmp_path)
    epoch_context = {"output_dir": output_dir}

    metrics = evaluator.evaluate_epoch(model_handle, None, epoch_context)

    assert metrics["coverage"] == 0.9
    assert metrics["partx_roa_volume"] == 0.5

    png_path = os.path.join(output_dir, "partx_region_tree.png")
    assert os.path.exists(png_path)
    assert os.path.getsize(png_path) > 0


def test_merge_region_bounds_carries_acquisition_health():
    # The acquisition-health keys are the only on-disk evidence that an epoch
    # acquired nothing. Dropping them is what made q3d_nd048_partx's 18 empty
    # epochs indistinguishable from healthy ones in artifacts_v2.json.
    diag = {
        "roa_volume": 0.42,
        "n_eligible": 0,
        "n_candidates": 250000,
        "n_outside_tree": 250000,
        "n_selected": 5000,
        "fallback_used": True,
        "fallback_reason": "no candidate in an unresolved leaf",
        "n_topup_outside_unresolved": 0,
    }
    out = merge_region_bounds({}, diag)
    assert out["partx_fallback_used"] is True
    assert out["partx_n_eligible"] == 0
    assert out["partx_n_outside_tree"] == 250000
    assert out["partx_n_selected"] == 5000
    assert "no candidate" in out["partx_fallback_reason"]
