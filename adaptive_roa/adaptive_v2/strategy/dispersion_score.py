"""Pure scoring functions for dispersion-based acquisition.

Nothing here imports a model, a pipeline component, or the success criteria:
every function operates on plain arrays so it can be unit-tested without a
trained model and reasoned about in isolation.
"""

from __future__ import annotations

import numpy as np
import torch


def _wrap_circular_(diff: torch.Tensor, circular_mask: torch.Tensor) -> torch.Tensor:
    """Wrap circular columns of `diff` into [-pi, pi], in place.

    Uses atan2(sin, cos) to match the convention in
    adaptive_roa/systems/pendulum.py:130. Only the circular columns are
    materialized, which keeps the temporaries small for wide state vectors.
    """
    if not bool(circular_mask.any()):
        return diff
    circ = diff[..., circular_mask]
    diff[..., circular_mask] = torch.atan2(torch.sin(circ), torch.cos(circ))
    return diff


def mean_pairwise_dispersion(
    endpoints: np.ndarray,
    scales: np.ndarray,
    circular_mask: np.ndarray,
    chunk_size: int = 2048,
    device: str = "cpu",
) -> np.ndarray:
    """Score each candidate by the mean pairwise distance among its endpoints.

    score(x) = 2 / (K(K-1)) * sum_{i<j} || wrap(e_i - e_j) / scales ||_2

    The full pairwise matrix holds each pair twice plus a zero diagonal, so
    sum(all d_ij) == 2 * sum_{i<j} d_ij and the score reduces to
    sum(all d_ij) / (K(K-1)).

    Args:
        endpoints: Predicted endpoint clouds [M, K, D]
        scales: Per-dimension distance scales [D], all > 0
        circular_mask: Boolean [D], True where the dimension wraps
        chunk_size: Candidates processed per block (bounds the [m,K,K,D] temp)
        device: Torch device for the computation

    Returns:
        np.ndarray: Scores [M] as float64. NaN for any candidate with a
            non-finite endpoint; such candidates are excluded downstream.
    """
    endpoints = np.asarray(endpoints)
    if endpoints.ndim != 3:
        raise ValueError(f"endpoints must be [M, K, D], got shape {endpoints.shape}")

    M, K, D = endpoints.shape
    if K < 2:
        raise ValueError(f"dispersion needs at least 2 endpoint samples, got K={K}")
    if scales.shape != (D,):
        raise ValueError(f"scales must be [{D}], got {scales.shape}")
    if circular_mask.shape != (D,):
        raise ValueError(f"circular_mask must be [{D}], got {circular_mask.shape}")

    scales_t = torch.as_tensor(np.asarray(scales), dtype=torch.float32, device=device)
    circ_t = torch.as_tensor(np.asarray(circular_mask), dtype=torch.bool, device=device)
    out = np.empty(M, dtype=np.float64)

    for start in range(0, M, chunk_size):
        stop = min(start + chunk_size, M)
        chunk = torch.as_tensor(
            endpoints[start:stop], dtype=torch.float32, device=device
        )

        finite = torch.isfinite(chunk).all(dim=2).all(dim=1)          # [m]
        diff = chunk.unsqueeze(2) - chunk.unsqueeze(1)                # [m, K, K, D]
        diff = _wrap_circular_(diff, circ_t)
        dists = torch.linalg.vector_norm(diff / scales_t, dim=-1)     # [m, K, K]
        scores = dists.sum(dim=(1, 2)) / (K * (K - 1))                # [m]

        scores = torch.where(finite, scores, torch.full_like(scores, float("nan")))
        out[start:stop] = scores.double().cpu().numpy()

    return out


def normalized_distances(
    states: np.ndarray,
    reference: np.ndarray,
    scales: np.ndarray,
    circular_mask: np.ndarray,
) -> np.ndarray:
    """Distance from every state to one reference state, same metric as above.

    Args:
        states: States [P, D]
        reference: Single state [D]
        scales: Per-dimension distance scales [D]
        circular_mask: Boolean [D], True where the dimension wraps

    Returns:
        np.ndarray: Distances [P] as float64
    """
    diff = np.asarray(states, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    circular_mask = np.asarray(circular_mask)
    if circular_mask.any():
        circ = diff[:, circular_mask]
        diff[:, circular_mask] = np.arctan2(np.sin(circ), np.cos(circ))
    return np.linalg.norm(diff / np.asarray(scales, dtype=np.float64), axis=1)


def _finite_order(scores: np.ndarray) -> np.ndarray:
    """Positions of finite scores, ordered by score descending (stable)."""
    valid = np.flatnonzero(np.isfinite(scores))
    return valid[np.argsort(-scores[valid], kind="stable")]


def select_greedy(scores: np.ndarray, n_select: int) -> np.ndarray:
    """Take the n_select highest-scoring candidates.

    Args:
        scores: Dispersion scores [M]; NaN entries are skipped
        n_select: Number of candidates to select

    Returns:
        np.ndarray: Positions into `scores`, highest score first
    """
    return _finite_order(scores)[:n_select]


def select_greedy_diverse(
    scores: np.ndarray,
    states: np.ndarray,
    scales: np.ndarray,
    circular_mask: np.ndarray,
    n_select: int,
    pool_multiplier: int = 5,
) -> np.ndarray:
    """Farthest-point-sample n_select from the top-scoring shortlist.

    Takes the top `pool_multiplier * n_select` candidates by score, then
    greedily picks points that are far apart in *initial-state* space, seeded
    at the highest-scoring candidate. This stops the batch collapsing onto a
    single high-uncertainty pocket.

    Args:
        scores: Dispersion scores [M]; NaN entries are skipped
        states: Candidate initial states [M, D]
        scales: Per-dimension distance scales [D]
        circular_mask: Boolean [D], True where the dimension wraps
        n_select: Number of candidates to select
        pool_multiplier: Shortlist size as a multiple of n_select

    Returns:
        np.ndarray: Positions into `scores`, seed candidate first
    """
    order = _finite_order(scores)
    shortlist = order[: max(n_select * pool_multiplier, n_select)]
    if len(shortlist) <= n_select:
        return shortlist

    X = np.asarray(states)[shortlist]
    selected = [0]  # shortlist is score-sorted, so 0 is the highest scorer
    dist = normalized_distances(X, X[0], scales, circular_mask)
    dist[0] = -np.inf

    while len(selected) < n_select:
        nxt = int(np.argmax(dist))
        selected.append(nxt)
        dist = np.minimum(dist, normalized_distances(X, X[nxt], scales, circular_mask))
        dist[selected] = -np.inf

    return shortlist[np.asarray(selected, dtype=int)]


def select_proportional(
    scores: np.ndarray,
    n_select: int,
    temperature: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """Sample n_select candidates with probability rising in the score.

    Scores are min-max normalized within the batch before the softmax, so
    `temperature` carries the same meaning across systems and epochs. Sampling
    without replacement uses the Gumbel-top-k trick, which is exact and needs
    no renormalization loop.

    Args:
        scores: Dispersion scores [M]; NaN entries are skipped
        n_select: Number of candidates to select
        temperature: Softmax temperature; lower concentrates on top scores
        seed: Seed for reproducibility; None draws fresh entropy

    Returns:
        np.ndarray: Positions into `scores`, unordered
    """
    valid = np.flatnonzero(np.isfinite(scores))
    if len(valid) <= n_select:
        return valid

    s = np.asarray(scores, dtype=np.float64)[valid]
    lo, hi = s.min(), s.max()
    s_norm = np.zeros_like(s) if hi <= lo else (s - lo) / (hi - lo)

    logits = s_norm / max(float(temperature), 1e-12)
    rng = np.random.default_rng(seed)
    keys = logits + rng.gumbel(size=len(s))

    top = np.argpartition(-keys, n_select - 1)[:n_select]
    return valid[top]
