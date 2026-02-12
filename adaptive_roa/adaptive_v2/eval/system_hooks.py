"""System-specific eval hooks for adaptive v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SystemEvalHook:
    system_name: str
    decision_rule: str
    attractor_radius_default: float
    projection_dims: tuple[int, int] = (0, 1)
    projection_labels: tuple[str, str] = ("x0", "x1")

    def maybe_plot(
        self,
        start_states: np.ndarray,
        p_success: np.ndarray,
        true_labels: np.ndarray,
        lambda_star: float,
        delta: float,
        output_file: str,
    ) -> None:
        if start_states.ndim != 2 or start_states.shape[1] < 2:
            return

        i, j = self.projection_dims
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        pred_labels = np.full(len(p_success), 0)
        pred_labels[p_success > (lambda_star + delta)] = 1
        pred_labels[p_success < (lambda_star - delta)] = -1

        colors = np.array(["gold"] * len(pred_labels), dtype=object)
        colors[pred_labels == 1] = "blue"
        colors[pred_labels == -1] = "red"

        axes[0].scatter(start_states[:, i], start_states[:, j], c=colors, s=2, alpha=0.6)
        axes[0].set_title("Predicted ROA")
        axes[0].set_xlabel(self.projection_labels[0])
        axes[0].set_ylabel(self.projection_labels[1])
        axes[0].grid(True, alpha=0.3)

        gt_colors = np.array(["gold"] * len(true_labels), dtype=object)
        gt_colors[true_labels == 1] = "blue"
        gt_colors[true_labels == -1] = "red"
        axes[1].scatter(start_states[:, i], start_states[:, j], c=gt_colors, s=2, alpha=0.6)
        axes[1].set_title("Ground Truth ROA")
        axes[1].set_xlabel(self.projection_labels[0])
        axes[1].set_ylabel(self.projection_labels[1])
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"{self.system_name} (lambda={lambda_star:.3f}, delta={delta:.3f})")
        fig.tight_layout()
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)


_HOOKS = {
    "pendulumsystem": SystemEvalHook(
        system_name="pendulum",
        decision_rule="one_sided",
        attractor_radius_default=0.1,
        projection_dims=(0, 1),
        projection_labels=("theta", "theta_dot"),
    ),
    "cartpolesystem": SystemEvalHook(
        system_name="cartpole_pybullet",
        decision_rule="two_sided",
        attractor_radius_default=0.2,
        projection_dims=(0, 1),
        projection_labels=("x", "theta"),
    ),
    "quadrotor2dsystem": SystemEvalHook(
        system_name="quadrotor2d",
        decision_rule="two_sided",
        attractor_radius_default=0.3,
        projection_dims=(0, 1),
        projection_labels=("x", "z"),
    ),
    "quadrotor3dsystem": SystemEvalHook(
        system_name="quadrotor3d",
        decision_rule="two_sided",
        attractor_radius_default=0.2,
        projection_dims=(0, 2),
        projection_labels=("x", "z"),
    ),
}


def resolve_system_hook(system: Any) -> SystemEvalHook:
    key = type(system).__name__.lower()
    return _HOOKS.get(
        key,
        SystemEvalHook(
            system_name=type(system).__name__,
            decision_rule="two_sided",
            attractor_radius_default=0.2,
        ),
    )
