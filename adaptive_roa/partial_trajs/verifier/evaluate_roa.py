"""ROA evaluation for the partial-trajectory verifier.

Runs the verifier over a dataset's ``(init, final, label)`` eval rows and reports
ROA success-detection scores (metric set), plus the two dynamics-accuracy
diagnostics: metric #2 (rollout final-state error) here, and metric #1
(per-horizon T-step error) during training (see ``eval/metrics.py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from adaptive_roa.partial_trajs.eval.metrics import manifold_state_distance, roa_scores
from adaptive_roa.partial_trajs.verifier.rollout import (
    resolve_outcome,
    resolve_probabilistic,
    rollout_final_state,
)


def load_eval_states(
    path: Union[str, Path], state_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a comma-delimited ``[init(D), final(D), label]`` eval file."""
    rows = np.loadtxt(Path(path), delimiter=",")
    if rows.ndim == 1:
        rows = rows[None, :]
    init = torch.as_tensor(rows[:, :state_dim], dtype=torch.float32)
    terminal = torch.as_tensor(rows[:, state_dim : 2 * state_dim], dtype=torch.float32)
    label = torch.as_tensor(rows[:, -1], dtype=torch.long)
    return init, terminal, label


def evaluate_roa(
    model,
    system,
    init: torch.Tensor,
    terminal: torch.Tensor,
    label: torch.Tensor,
    K: int,
    circular_indices: Sequence[int] = (),
    radius: Optional[float] = None,
    num_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Verify each query and score against ground-truth labels.

    Deterministic (``num_samples=None``): one rollout per query. Probabilistic
    (``num_samples`` set): N rollouts per query; a query is predicted success if
    ``p_success >= 0.5``.
    """
    result: Dict[str, float] = {}

    if num_samples is None:
        pred_labels = resolve_outcome(model, system, init, K, radius)
    else:
        probs = resolve_probabilistic(model, system, init, K, num_samples, radius)
        pred_labels = torch.where(
            probs["p_success"] >= 0.5,
            torch.ones_like(label),
            torch.zeros_like(label),
        )
        result["p_success_mean"] = float(probs["p_success"].mean())

    result.update(roa_scores(pred_labels, label))

    final = rollout_final_state(model, system, init, K, radius)
    err = manifold_state_distance(final, terminal, circular_indices)
    result["rollout_final_state_error"] = float(err.mean())

    return result
