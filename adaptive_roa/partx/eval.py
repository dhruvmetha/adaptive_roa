from __future__ import annotations

import os
from typing import Any

from hydra.utils import get_class

from adaptive_roa.partx.viz import plot_region_tree


def merge_region_bounds(base_metrics: dict, diag: dict | None) -> dict:
    out = dict(base_metrics)
    if not diag:
        return out
    for key in ("roa_volume", "roa_volume_ci", "n_leaves", "n_remaining_leaves"):
        if key in diag:
            out[f"partx_{key}"] = diag[key]
    return out


class PartXEvaluator:
    """Wraps the base RoA evaluator and appends partx region-bound metrics."""

    def __init__(self, cfg: Any, system: Any, device: str):
        base_target = cfg.base._target_
        base_cls = get_class(base_target)
        self._base = base_cls(cfg.base, system, device)
        self.max_eval_rows = self._base.max_eval_rows
        self.system = system

    def evaluate_epoch(self, model_handle, threshold_state, epoch_context) -> dict:
        base_metrics = self._base.evaluate_epoch(model_handle, threshold_state, epoch_context)
        diag = getattr(model_handle, "partx_diag", None)
        metrics = merge_region_bounds(base_metrics, diag)

        tree = getattr(model_handle, "partx_tree", None)
        output_dir = epoch_context.get("output_dir")
        if tree is not None and self.system.state_dim == 2 and output_dir:
            try:
                png_path = os.path.join(output_dir, "partx_region_tree.png")
                plot_region_tree(tree, png_path)
                metrics["partx_region_tree_png"] = png_path
            except Exception:
                pass

        return metrics
