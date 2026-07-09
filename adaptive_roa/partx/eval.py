from __future__ import annotations

from typing import Any

from hydra.utils import get_class


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

    def evaluate_epoch(self, model_handle, threshold_state, epoch_context) -> dict:
        base_metrics = self._base.evaluate_epoch(model_handle, threshold_state, epoch_context)
        diag = getattr(model_handle, "partx_diag", None)
        return merge_region_bounds(base_metrics, diag)
