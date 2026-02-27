"""
MC prediction cache for fast re-evaluation.

Caches raw endpoint predictions and attractor labels from Monte Carlo sampling,
allowing re-evaluation with different thresholds (λ*, δ, q_hat) or attractor
radii without re-running the flow matcher.

Cache format (.npz):
    mc_endpoints: [N, K, state_dim] float32 - raw predicted endpoints
    mc_labels:    [N, K] int8              - classify_attractor results {-1, 0, 1}
    start_states: [N, state_dim] float32   - input states (for validation)
    metadata keys: num_mc_samples, attractor_radius, state_dim, n_states
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from adaptive_roa.conformal.refinement import RefinementStats, refine_invalid_endpoints


@dataclass
class MCCache:
    """Cached MC endpoint predictions for a set of states."""

    mc_endpoints: np.ndarray   # [N, K, state_dim] float32
    mc_labels: np.ndarray      # [N, K] int8 — {-1, 0, 1}
    start_states: np.ndarray   # [N, state_dim] float32
    attractor_radius: float
    num_mc_samples: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return self.mc_endpoints.shape[0]

    @property
    def state_dim(self) -> int:
        return self.mc_endpoints.shape[2]

    def probabilities(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (p_success, p_failure, p_invalid) from cached labels."""
        K = self.num_mc_samples
        p_success = (self.mc_labels == 1).sum(axis=1).astype(np.float64) / K
        p_failure = (self.mc_labels == -1).sum(axis=1).astype(np.float64) / K
        p_invalid = (self.mc_labels == 0).sum(axis=1).astype(np.float64) / K
        return p_success, p_failure, p_invalid

    def reclassify(self, system, new_radius: float) -> MCCache:
        """Re-run classify_attractor on cached endpoints with a new radius.

        Returns a new MCCache with updated labels and radius.
        """
        N, K, D = self.mc_endpoints.shape
        new_labels = np.zeros((N, K), dtype=np.int8)

        # Process in chunks to avoid GPU OOM
        chunk = 4096
        for i in range(0, N, chunk):
            end = min(i + chunk, N)
            # Flatten [chunk, K, D] -> [chunk*K, D] for batch classify
            flat = self.mc_endpoints[i:end].reshape(-1, D)
            flat_t = torch.from_numpy(flat).float()
            labels_t = system.classify_attractor(flat_t, radius=new_radius)
            new_labels[i:end] = labels_t.cpu().numpy().reshape(end - i, K)

        return MCCache(
            mc_endpoints=self.mc_endpoints,
            mc_labels=new_labels,
            start_states=self.start_states,
            attractor_radius=new_radius,
            num_mc_samples=self.num_mc_samples,
            metadata={**self.metadata, "reclassified_from_radius": self.attractor_radius},
        )


def compute_mc_predictions(
    flow_matcher,
    system,
    states: np.ndarray,
    num_mc_samples: int,
    attractor_radius: float,
    batch_size: int = 2048,
    device: str = "cuda",
    verbose: bool = True,
    refine_invalids: bool = False,
    refine_t_range: tuple[float, float] = (0.7, 0.9),
    refine_num_steps: int = 100,
    refine_max_attempts: int = 5,
) -> MCCache:
    """Run MC sampling and return a cacheable MCCache object.

    This performs the expensive GPU work: for each state, runs K forward passes
    through the flow matcher, classifies endpoints, and stores everything.

    Args:
        flow_matcher: Trained flow matching model.
        system: Dynamical system with classify_attractor().
        states: [N, state_dim] numpy array of initial states.
        num_mc_samples: Number of MC samples per state.
        attractor_radius: Radius for attractor classification.
        batch_size: GPU batch size.
        device: Compute device.
        verbose: Show progress bar.
        refine_invalids: Whether to refine invalid endpoints.
        refine_t_range: Time range for refinement.
        refine_num_steps: ODE steps for refinement.
        refine_max_attempts: Max refinement attempts.

    Returns:
        MCCache with mc_endpoints [N, K, state_dim] and mc_labels [N, K].
    """
    N = len(states)
    K = num_mc_samples
    state_dim = states.shape[1]

    mc_endpoints = np.zeros((N, K, state_dim), dtype=np.float32)
    mc_labels = np.zeros((N, K), dtype=np.int8)

    X_tensor = torch.from_numpy(states).float().to(device)

    cumulative_rstats = RefinementStats(
        per_attempt_resolved=[0] * refine_max_attempts
    ) if refine_invalids else None

    n_batches = (N + batch_size - 1) // batch_size
    total_steps = n_batches * K

    flow_matcher.eval()
    with torch.no_grad():
        with tqdm(total=total_steps, desc="MC cache", disable=not verbose) as pbar:
            for batch_start in range(0, N, batch_size):
                batch_end = min(batch_start + batch_size, N)
                batch_inputs = X_tensor[batch_start:batch_end]

                for sample_idx in range(K):
                    pred = flow_matcher.predict_endpoint(batch_inputs)
                    labels_tensor = system.classify_attractor(pred, attractor_radius)

                    if refine_invalids:
                        rstats = refine_invalid_endpoints(
                            flow_matcher, system,
                            pred, labels_tensor, batch_inputs,
                            attractor_radius=attractor_radius,
                            t_range=refine_t_range,
                            num_steps=refine_num_steps,
                            max_attempts=refine_max_attempts,
                        )
                        cumulative_rstats.accumulate(rstats)

                    mc_endpoints[batch_start:batch_end, sample_idx] = pred.cpu().numpy()
                    mc_labels[batch_start:batch_end, sample_idx] = labels_tensor.cpu().numpy()

                    pbar.update(1)

    if refine_invalids and verbose and cumulative_rstats.n_initially_invalid > 0:
        total_original = cumulative_rstats.n_initially_invalid
        total_resolved = cumulative_rstats.n_resolved
        resolve_rate = total_resolved / total_original * 100
        print(f"  [Refine] {total_resolved}/{total_original} invalid MC samples "
              f"resolved ({resolve_rate:.1f}%)")

    return MCCache(
        mc_endpoints=mc_endpoints,
        mc_labels=mc_labels,
        start_states=states.astype(np.float32),
        attractor_radius=attractor_radius,
        num_mc_samples=K,
    )


def save_mc_cache(cache: MCCache, path: str | Path) -> Path:
    """Save MCCache to an .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        mc_endpoints=cache.mc_endpoints,
        mc_labels=cache.mc_labels,
        start_states=cache.start_states,
        attractor_radius=np.float64(cache.attractor_radius),
        num_mc_samples=np.int64(cache.num_mc_samples),
    )
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  [Cache] Saved {path.name}: "
          f"{cache.n_states} states × {cache.num_mc_samples} MC × {cache.state_dim}D "
          f"({size_mb:.1f} MB)")
    return path


def load_mc_cache(path: str | Path) -> MCCache:
    """Load MCCache from an .npz file."""
    path = Path(path)
    data = np.load(str(path))
    cache = MCCache(
        mc_endpoints=data["mc_endpoints"],
        mc_labels=data["mc_labels"],
        start_states=data["start_states"],
        attractor_radius=float(data["attractor_radius"]),
        num_mc_samples=int(data["num_mc_samples"]),
    )
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  [Cache] Loaded {path.name}: "
          f"{cache.n_states} states × {cache.num_mc_samples} MC × {cache.state_dim}D "
          f"({size_mb:.1f} MB)")
    return cache
