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
        chunk_size: Candidates processed per block (bounds the [m,K,K,D] temp).
            The default of 2048 was sized for D=12 (~39 MB per chunk); the
            temporary scales linearly in D, so wider state spaces should use a
            smaller chunk_size to hold the same memory budget.
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
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

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
        # diff is a fresh, unaliased temporary (the broadcasted subtraction
        # above always allocates), so dividing in place is safe and avoids a
        # second full-size [m,K,K,D] allocation on top of it.
        diff.div_(scales_t)
        dists = torch.linalg.vector_norm(diff, dim=-1)                # [m, K, K]
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


def _pairwise(chunk: torch.Tensor, scales_t: torch.Tensor, circ_t: torch.Tensor) -> torch.Tensor:
    """Normalized, circular-aware pairwise distances for a [m,K,D] block -> [m,K,K]."""
    diff = chunk.unsqueeze(2) - chunk.unsqueeze(1)
    diff = _wrap_circular_(diff, circ_t)
    return torch.linalg.vector_norm(diff.div_(scales_t), dim=-1)


def mode_separation(
    endpoints: np.ndarray,
    scales: np.ndarray,
    circular_mask: np.ndarray,
    chunk_size: int = 2048,
    device: str = "cpu",
) -> np.ndarray:
    """Score how strongly a candidate's endpoint cloud splits into two modes.

        s(x) = 4*p*(1-p) * G / (G + W)

    where the cloud is partitioned in two (seeded at the farthest pair, then
    refined), p is the smaller side's fraction, G the GAP between the sides
    (smallest distance across the split) and W the mean within-side pairwise
    distance.

    This is the quantity mean_pairwise_dispersion cannot express. Dispersion is
    the sum of within-mode spread and between-mode separation, and only the
    latter carries information about a decision boundary: a basin boundary is a
    discontinuity of the endpoint map, so a model of that map must place mass on
    two separated modes there. A cloud that is merely wide -- the model unsure
    *where* within one basin, or a genuinely diffuse divergence region -- has
    W ~ B and scores near zero.

    Being a ratio, the score is invariant to the cloud's overall scale, so the
    fuzz of an undertrained model cancels rather than dominating.

    Args:
        endpoints: Predicted endpoint clouds [M, K, D]
        scales: Per-dimension distance scales [D]
        circular_mask: Boolean [D], True where the dimension wraps
        chunk_size: Candidates per block
        device: Torch device

    Returns:
        np.ndarray: Scores [M] in [0, 1); NaN where a cloud has non-finite points
    """
    endpoints = np.asarray(endpoints)
    if endpoints.ndim != 3:
        raise ValueError(f"endpoints must be [M, K, D], got shape {endpoints.shape}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    M, K, D = endpoints.shape
    if K < 4:
        raise ValueError(f"mode separation needs at least 4 endpoint samples, got K={K}")

    scales_t = torch.as_tensor(np.asarray(scales), dtype=torch.float32, device=device)
    circ_t = torch.as_tensor(np.asarray(circular_mask), dtype=torch.bool, device=device)
    out = np.empty(M, dtype=np.float64)

    for start in range(0, M, chunk_size):
        stop = min(start + chunk_size, M)
        chunk = torch.as_tensor(endpoints[start:stop], dtype=torch.float32, device=device)
        finite = torch.isfinite(chunk).all(dim=2).all(dim=1)
        safe = torch.where(finite.view(-1, 1, 1), chunk, torch.zeros_like(chunk))

        d = _pairwise(safe, scales_t, circ_t)                       # [m,K,K]
        m = d.shape[0]
        flat = d.view(m, -1).argmax(dim=1)                          # farthest pair seeds
        i0, i1 = flat // K, flat % K
        rows = torch.arange(m, device=d.device)
        for _ in range(2):                                          # Lloyd refinement
            side = d[rows, i1] < d[rows, i0]                        # True -> side B
            # recentre each side on its medoid (min summed within-side distance)
            for sel, idx in ((~side, "a"), (side, "b")):
                # summed distance from each point to the points on its own side;
                # only points on that side are eligible to be its medoid
                w = torch.where(sel.unsqueeze(1), d, torch.zeros_like(d))
                cost = w.sum(dim=2)
                cost = torch.where(sel, cost, torch.full_like(cost, float("inf")))
                med = cost.argmin(dim=1)
                if idx == "a": i0 = med
                else: i1 = med

        nb = side.sum(dim=1).float()
        p = torch.minimum(nb, K - nb) / K
        balance = 4.0 * p * (1.0 - p)

        # G: the GAP -- smallest distance across the split. Deliberately not the
        # centroid/medoid separation: splitting any cloud at its farthest pair puts
        # the medoids far apart relative to within-side spread, so a contiguous blob
        # would score as high as two genuinely separated modes. Only the nearest
        # cross-pair distinguishes "there is empty space between two groups" from
        # "this is one group that happens to be wide".
        cross = side.unsqueeze(2) != side.unsqueeze(1)
        G = torch.where(cross, d, torch.full_like(d, float("inf"))).amin(dim=(1, 2))
        same = side.unsqueeze(2) == side.unsqueeze(1)
        eye = torch.eye(K, dtype=torch.bool, device=d.device).unsqueeze(0)
        wmask = same & ~eye
        wcount = wmask.sum(dim=(1, 2)).clamp(min=1).float()
        W = (d * wmask).sum(dim=(1, 2)) / wcount

        s = balance * G / (G + W).clamp(min=1e-12)
        s = torch.where(finite, s, torch.full_like(s, float("nan")))
        out[start:stop] = s.double().cpu().numpy()

    return out


def idempotence_defect(
    endpoints: np.ndarray,
    remapped: np.ndarray,
    scales: np.ndarray,
    circular_mask: np.ndarray,
) -> np.ndarray:
    """Mean distance each predicted endpoint moves when fed back through the model.

    An attractor is a fixed point of the endpoint map, so E(e) == e for a true
    attractor. A large defect means the cloud sits somewhere the learned map does
    not consider settled -- off-attractor or divergent. This distinguishes "the
    cloud is diffuse because the outcome is genuinely unresolved" from "the cloud
    is diffuse because it straddles two attractors", which raw spread conflates
    and which is what starved the success class on quadrotor2d.

    It identifies attractors as the fixed-point set of the learned map, without
    being told what they are.

    Args:
        endpoints: Predicted endpoint clouds [M, K, D]
        remapped: The model applied again to each endpoint, same shape
        scales: Per-dimension distance scales [D]
        circular_mask: Boolean [D], True where the dimension wraps

    Returns:
        np.ndarray: Mean normalized displacement [M]; NaN where either input is
            non-finite for that candidate
    """
    endpoints, remapped = np.asarray(endpoints, dtype=np.float64), np.asarray(remapped, dtype=np.float64)
    if endpoints.shape != remapped.shape:
        raise ValueError(f"shape mismatch: {endpoints.shape} vs {remapped.shape}")

    diff = remapped - endpoints
    circular_mask = np.asarray(circular_mask)
    if circular_mask.any():
        c = diff[:, :, circular_mask]
        diff[:, :, circular_mask] = np.arctan2(np.sin(c), np.cos(c))
    d = np.linalg.norm(diff / np.asarray(scales, dtype=np.float64), axis=2)   # [M,K]

    ok = np.isfinite(endpoints).all(axis=(1, 2)) & np.isfinite(remapped).all(axis=(1, 2))
    out = d.mean(axis=1)
    return np.where(ok, out, np.nan)
