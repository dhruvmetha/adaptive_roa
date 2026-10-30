"""Per-epoch prediction store (det paper, EXPERIMENT_TRACKER sections 9 and 10).

Every post-training analysis reads these files instead of loading a model, so a
run writes, per epoch and next to ``full_roa_per_point.npz`` (the test split):

- ``predictions_<split>.npz`` for the held-out val and cal splits. Endpoint
  models store per-state outcome COUNTS out of K, not probabilities: the K
  labels at one state are exchangeable, so a K' < K subsample follows a
  hypergeometric law given the counts, and smaller-K analyses need no
  inference. Classifiers store p_success.
- ``predictions_thrfit.npz``: the probabilities the lambda*/delta* optimizer saw,
  so thresholds can be refit offline (ablation A-W1).
- ``endpoints_<split>.npz`` at the reporting epochs only: all K generated
  endpoints per state as float32, for reclassifying at another buffer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def endpoint_test_rows(n_test: int, cap: int | None) -> np.ndarray | None:
    """Fixed test rows whose endpoints are stored: all, or a seed-0 subsample.

    Seeded by nothing run-specific, so every run on a system stores the same rows.
    """
    if cap is None or n_test <= cap:
        return np.arange(n_test)
    return np.sort(np.random.default_rng(0).permutation(n_test)[:cap])


def mc_predict(
    model: Any,
    system: Any,
    X: np.ndarray,
    num_mc_samples: int,
    attractor_radius: float,
    batch_size: int = 2048,
    device: str = "cpu",
    keep_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """K endpoint draws per state: labels [N, K] int8 and, for keep_rows, endpoints [n, K, d]."""
    X = np.asarray(X, dtype=np.float32)
    n, k = len(X), int(num_mc_samples)
    labels = np.zeros((n, k), dtype=np.int8)
    slot = np.full(n, -1, dtype=np.int64)
    endpoints = None
    if keep_rows is not None:
        keep_rows = np.asarray(keep_rows, dtype=np.int64)
        slot[keep_rows] = np.arange(len(keep_rows))
        endpoints = np.zeros((len(keep_rows), k, X.shape[1]), dtype=np.float32)
    n_members = getattr(model, "n_members", None)
    model.eval()
    with torch.no_grad():
        for b0 in range(0, n, batch_size):
            b1 = min(b0 + batch_size, n)
            xb = torch.from_numpy(X[b0:b1]).to(device)
            local = np.flatnonzero(slot[b0:b1] >= 0)
            for s in range(k):
                # Ensembles pin the member by draw index, as evaluate_full_roa_fast does.
                pred = (model.predict_endpoint_member(s % n_members, xb) if n_members
                        else model.predict_endpoint(xb))
                labels[b0:b1, s] = system.classify_attractor(pred, attractor_radius).cpu().numpy()
                if endpoints is not None and local.size:
                    endpoints[slot[b0:b1][local], s] = pred[local].cpu().numpy()
    return labels, endpoints


def counts_from_labels(labels: np.ndarray) -> dict[str, np.ndarray]:
    k = labels.shape[1]
    if k > np.iinfo(np.int16).max:
        raise ValueError(f"K={k} does not fit the int16 counts")
    return {
        "n_success": (labels == 1).sum(axis=1).astype(np.int16),
        "n_failure": (labels == -1).sum(axis=1).astype(np.int16),
        "n_invalid": (labels == 0).sum(axis=1).astype(np.int16),
        "num_mc_samples": np.int32(k),
    }


def write_split(
    epoch_dir: str | Path,
    split: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    predictor_type: str,
    model: Any,
    system: Any,
    num_mc_samples: int,
    attractor_radius: float,
    batch_size: int = 2048,
    device: str = "cpu",
    keep_endpoints: bool = False,
) -> None:
    """Write predictions_<split>.npz (and endpoints_<split>.npz when asked)."""
    out = Path(epoch_dir)
    out.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y).astype(np.int8)
    if predictor_type == "classifier":
        from adaptive_roa.conformal.classifier_probability_estimator import (
            ClassifierProbabilityEstimator,
        )
        est = ClassifierProbabilityEstimator(model, system, None, device)
        p_success, _pf, _pi = est.estimate(np.asarray(X, dtype=np.float32))
        np.savez(out / f"predictions_{split}.npz",
                 p_success=np.asarray(p_success, dtype=np.float32), true_labels=y)
        return
    keep = np.arange(len(X)) if keep_endpoints else None
    labels, endpoints = mc_predict(model, system, X, num_mc_samples, attractor_radius,
                                   batch_size=batch_size, device=device, keep_rows=keep)
    np.savez(out / f"predictions_{split}.npz", true_labels=y,
             attractor_radius=np.float64(np.nan if attractor_radius is None else attractor_radius), **counts_from_labels(labels))
    if endpoints is not None:
        np.savez(out / f"endpoints_{split}.npz", endpoints=endpoints, rows=keep)


def write_split_states_once(run_dir: str | Path, split: str, X: np.ndarray, y: np.ndarray) -> None:
    """States and labels of a fixed split, written on first use only."""
    path = Path(run_dir) / f"prediction_store_states_{split}.npz"
    if not path.exists():
        np.savez(path, states=np.asarray(X, dtype=np.float32), true_labels=np.asarray(y).astype(np.int8))


def write_threshold_fit(epoch_dir: str | Path, fit: dict[str, Any] | None) -> None:
    """The states, labels and probabilities the lambda*/delta* optimizer used."""
    if not fit:
        return
    np.savez(
        Path(epoch_dir) / "predictions_thrfit.npz",
        states=np.asarray(fit["states"], dtype=np.float32),
        true_labels=np.asarray(fit["labels"]).astype(np.int8),
        p_success=np.asarray(fit["p_success"], dtype=np.float32),
        p_failure=np.asarray(fit["p_failure"], dtype=np.float32),
        p_invalid=np.asarray(fit["p_invalid"], dtype=np.float32),
        num_mc_samples=np.int32(fit["num_mc_samples"]),
    )
