"""
Endpoint dataset utilities for adaptive sampling pipeline.

Provides functions for:
1. Sampling endpoint data for lambda/delta optimization
2. Computing endpoint prediction error on the training dataset
"""
import numpy as np
import torch
from typing import Tuple, Dict


def sample_endpoint_data_for_optimization(
    dataset_builder,
    sample_fraction: float = 1.0,
    max_samples: int = 0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get endpoint data for lambda/delta optimization.

    Uses the full expanded endpoint dataset (get_training_data) which has
    multiple pairs per trajectory.

    Args:
        dataset_builder: AdaptiveDatasetBuilder with training data
        sample_fraction: Unused, kept for API compatibility.
        max_samples: Unused, kept for API compatibility.
        seed: Unused, kept for API compatibility.

    Returns:
        Tuple of (X [N, state_dim], y [N])
        where X are start states and y are labels
    """
    start_states, end_states, labels = dataset_builder.get_training_data()

    n_total = len(labels)
    n_success = int(np.sum(labels == 1))
    n_failure = int(np.sum(labels == -1))
    print(f"    Endpoint dataset: {n_total} total pairs (using all)")
    print(f"    Labels: {n_success} success, {n_failure} failure")

    return start_states, labels


@torch.no_grad()
def compute_endpoint_prediction_error(
    flow_matcher,
    dataset_builder,
    batch_size: int = 512,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict:
    """
    Compute endpoint prediction error on the full training endpoint dataset.

    Uses manifold geodesic distances (per-component) to measure how well
    the flow matcher predicts end states from start states.

    Args:
        flow_matcher: Trained flow matcher model (in eval mode)
        dataset_builder: AdaptiveDatasetBuilder with training data
        batch_size: Batch size for predict_endpoint calls
        device: Device for inference
        verbose: Print detailed error report

    Returns:
        Dict with per-component and overall MAE statistics
    """
    start_states, end_states, labels = dataset_builder.get_training_data()

    n_total = len(labels)
    component_names = flow_matcher.get_manifold_component_names()
    n_components = len(component_names)

    all_distances = np.zeros((n_total, n_components), dtype=np.float32)

    flow_matcher.eval()
    start_tensor = torch.from_numpy(start_states).float().to(device)
    end_tensor = torch.from_numpy(end_states).float().to(device)

    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch_starts = start_tensor[batch_start:batch_end]
        batch_ends = end_tensor[batch_start:batch_end]

        predicted = flow_matcher.predict_endpoint(batch_starts)

        distances = flow_matcher.compute_manifold_distance_per_component(
            predicted, batch_ends
        )
        all_distances[batch_start:batch_end] = distances.cpu().numpy()

    # Per-component statistics
    per_component_mae = np.mean(all_distances, axis=0)
    per_component_median = np.median(all_distances, axis=0)

    # Overall statistics (L2 norm across components, then aggregate)
    l2_norms = np.linalg.norm(all_distances, axis=1)
    overall_mae = float(np.mean(l2_norms))
    overall_median = float(np.median(l2_norms))

    # Split by label
    success_mask = labels == 1
    failure_mask = labels == -1

    success_mae = float(np.mean(l2_norms[success_mask])) if np.any(success_mask) else 0.0
    failure_mae = float(np.mean(l2_norms[failure_mask])) if np.any(failure_mask) else 0.0

    result = {
        "n_endpoint_pairs": int(n_total),
        "component_names": list(component_names),
        "per_component_mae": [float(x) for x in per_component_mae],
        "per_component_median": [float(x) for x in per_component_median],
        "overall_mae": overall_mae,
        "overall_median": overall_median,
        "success_mae": success_mae,
        "failure_mae": failure_mae,
        "n_success": int(np.sum(success_mask)),
        "n_failure": int(np.sum(failure_mask)),
    }

    if verbose:
        print(f"\n    --- Training Endpoint Prediction Error ---")
        print(f"    {n_total} endpoint pairs ({int(np.sum(success_mask))} success, {int(np.sum(failure_mask))} failure)")
        print(f"    Overall geodesic MAE: {overall_mae:.6f} (median: {overall_median:.6f})")
        print(f"    By label: success={success_mae:.6f}, failure={failure_mae:.6f}")
        print(f"    Per-component MAE:")
        for name, mae, med in zip(component_names, per_component_mae, per_component_median):
            print(f"      {name}: MAE={mae:.6f}, median={med:.6f}")

    return result
