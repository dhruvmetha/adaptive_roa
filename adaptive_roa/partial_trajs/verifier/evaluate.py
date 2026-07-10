"""Standalone ROA evaluation for the partial-trajectory verifier.

Loads a trained checkpoint, rolls the T-step dynamics model out over a dataset's
``(init, final, label)`` eval rows, and writes:

- ``metrics.json``: ROA success-detection scores (extended ``roa_scores``) plus the
  stratified rollout final-state error (metric #2), and run metadata.
- ``predictions.npz``: per-query arrays for downstream analysis.

The autoregressive horizon ``K`` and ``state_dim`` are read from the dataset's
``dataset_description.json``; the success/failure ``radius`` is the adaptive
per-system ``attractor_radius`` (config default).

Usage:
    python adaptive_roa/partial_trajs/verifier/evaluate.py \
        system=pendulum run_dir=outputs/partial_trajs/pendulum/2026-07-09_12-00-00
    python adaptive_roa/partial_trajs/verifier/evaluate.py \
        system=humanoid_standup_reach run_dir=... eval_split=test num_samples=20
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from adaptive_roa.partial_trajs.data.description import load_dataset_description
from adaptive_roa.partial_trajs.eval.metrics import (
    manifold_state_distance,
    roa_scores,
    stratified_error_report,
)
from adaptive_roa.partial_trajs.systems import make_verifier_system
from adaptive_roa.partial_trajs.train import build_model
from adaptive_roa.partial_trajs.verifier.evaluate_roa import load_eval_states
from adaptive_roa.partial_trajs.verifier.rollout import (
    _rollout,
    resolve_probabilistic,
)


def _classify(system, state: torch.Tensor, radius: Optional[float]) -> torch.Tensor:
    if radius is None:
        return system.classify_attractor(state)
    return system.classify_attractor(state, radius=radius)


@torch.no_grad()
def run_verifier_eval(
    model,
    system,
    init: torch.Tensor,
    terminal: torch.Tensor,
    label: torch.Tensor,
    K: int,
    *,
    radius: Optional[float] = None,
    circular_indices: Sequence[int] = (),
    num_samples: Optional[int] = None,
    device: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    """Roll out the verifier over eval rows -> (metrics, per-query arrays).

    Deterministic (``num_samples=None``): one rollout per query. Probabilistic
    (``num_samples`` set): N rollouts; a query is predicted success if
    ``p_success >= 0.5``.

    The rollout (the expensive part: K autoregressive steps, each an ODE
    integration for the generative backend) runs on ``device`` (e.g. ``"cuda"``)
    in row-chunks of ``chunk_size`` to bound VRAM. Per-query outputs are gathered
    back on CPU, where the cheap O(N) metrics are then computed. ``device=None``
    keeps everything on CPU (default).
    """
    if device is not None:
        model = model.to(device)

    n = int(init.shape[0])
    cs = int(chunk_size) if chunk_size else n
    pred_parts, final_parts = [], []
    prob_parts = {"p_success": [], "p_failure": [], "p_unresolved": []} if num_samples else None

    for s in range(0, n, cs):
        xi = init[s : s + cs]
        if device is not None:
            xi = xi.to(device)
        # One rollout yields both the absorbing labels and the terminal state.
        pl, fin = _rollout(model.predict, system, xi, K, radius)
        if num_samples:
            probs = resolve_probabilistic(model, system, xi, K, int(num_samples), radius)
            pl = torch.where(
                probs["p_success"] >= 0.5,
                torch.ones_like(probs["p_success"], dtype=torch.long),
                torch.zeros_like(probs["p_success"], dtype=torch.long),
            )
            for key in prob_parts:
                prob_parts[key].append(probs[key].cpu())
        pred_parts.append(pl.cpu())
        final_parts.append(fin.cpu())

    pred_labels = torch.cat(pred_parts)
    final = torch.cat(final_parts)

    # Metrics on CPU (init/terminal/label are CPU eval tensors).
    err = manifold_state_distance(final, terminal, circular_indices)
    terminal_class = _classify(system, terminal, radius)
    motion = manifold_state_distance(terminal, init, circular_indices)

    metrics: Dict[str, object] = {
        "roa": roa_scores(pred_labels, label),
        "rollout_final_state_error": stratified_error_report(
            err, label, pred_labels, terminal_class, motion
        ),
    }
    per_query: Dict[str, torch.Tensor] = {
        "init": init,
        "pred_label": pred_labels,
        "true_label": label,
        "final_state": final,
        "terminal_class": terminal_class,
    }
    if prob_parts is not None:
        probs_cat = {key: torch.cat(vals) for key, vals in prob_parts.items()}
        metrics["p_success_mean"] = float(probs_cat["p_success"].mean())
        per_query["p_success"] = probs_cat["p_success"]
        per_query["p_failure"] = probs_cat["p_failure"]
        per_query["p_unresolved"] = probs_cat["p_unresolved"]

    return metrics, per_query


def write_eval_outputs(
    out_dir: Union[str, Path],
    metrics: Dict[str, object],
    per_query: Dict[str, torch.Tensor],
    metadata: Dict[str, object],
    export_predictions: bool = True,
) -> Path:
    """Write ``metrics.json`` (with ``metadata``) and optionally ``predictions.npz``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {"metadata": metadata, **metrics}
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))

    if export_predictions:
        arrays = {
            key: (val.detach().cpu().numpy() if torch.is_tensor(val) else np.asarray(val))
            for key, val in per_query.items()
        }
        np.savez(out_dir / "predictions.npz", **arrays)

    return metrics_path


def eval_filename_for_split(eval_split_file: str, split: str) -> str:
    """Map a split name to its eval filename, honoring the ``_fps`` (humanoid) naming.

    ``eval_split_file`` is the system's test file (``test_set.txt`` or
    ``test_set_fps.txt``); ``split`` is ``test`` | ``cal`` | ``eval``.
    """
    fps = eval_split_file.endswith("_fps.txt")
    if split == "test":
        return eval_split_file
    if split == "cal":
        return "cal_set_fps.txt" if fps else "cal_set.txt"
    if split == "eval":
        return "eval_fps.txt" if fps else "eval_states.txt"
    raise ValueError(f"unknown split {split!r} (expected test|cal|eval)")


def _find_checkpoint(run_dir: Union[str, Path]) -> str:
    """Best-val checkpoint in ``<run_dir>/checkpoints``, else ``last.ckpt``."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    best = sorted(glob.glob(str(ckpt_dir / "best*.ckpt")))
    if best:
        return best[0]
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return str(last)
    raise FileNotFoundError(f"no checkpoint found under {ckpt_dir}")


def _load_model(cfg: DictConfig, system) -> Tuple[object, str]:
    """Rebuild the model from cfg and load checkpoint weights (state_dict)."""
    ckpt_path = cfg.get("checkpoint") or _find_checkpoint(cfg.run_dir)
    model = build_model(cfg, system)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()
    return model, ckpt_path


@hydra.main(
    version_base=None,
    config_path="../../../configs/partial_trajs",
    config_name="evaluate_partial_trajs",
)
def main(cfg: DictConfig) -> None:
    system = make_verifier_system(str(cfg.system), cfg.get("system_dataset_dir"))
    model, ckpt_path = _load_model(cfg, system)

    desc = load_dataset_description(cfg.dataset_dir)
    split = str(cfg.get("eval_split", "test"))
    eval_file = Path(cfg.dataset_dir) / eval_filename_for_split(cfg.eval_split_file, split)
    init, terminal, label = load_eval_states(eval_file, desc.state_dim)

    max_rows = cfg.get("max_eval_rows")
    if max_rows is not None:
        init, terminal, label = init[: int(max_rows)], terminal[: int(max_rows)], label[: int(max_rows)]

    radius = cfg.get("radius")
    num_samples = cfg.get("num_samples")
    # GPU by default (falls back to CPU if unavailable); rollout runs in row-chunks.
    device = str(cfg.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    chunk_size = cfg.get("eval_chunk_size")
    metrics, per_query = run_verifier_eval(
        model, system, init, terminal, label, desc.autoregressive_K,
        radius=None if radius is None else float(radius),
        circular_indices=system.get_circular_indices(),
        num_samples=None if num_samples is None else int(num_samples),
        device=device,
        chunk_size=None if chunk_size is None else int(chunk_size),
    )

    metadata = {
        "system": str(cfg.system),
        "backend": str(cfg.get("backend", "generative")),
        "dataset_dir": str(cfg.dataset_dir),
        "eval_file": str(eval_file),
        "eval_split": split,
        "horizon_T": desc.horizon_T,
        "K": desc.autoregressive_K,
        "radius": None if radius is None else float(radius),
        "num_samples": None if num_samples is None else int(num_samples),
        "checkpoint": str(ckpt_path),
        "n_eval_rows": int(label.numel()),
        "in_sample": desc.in_sample,
        "seed": int(cfg.get("seed", 0)),
        "device": device,
    }

    out_dir = cfg.get("output_dir") or (Path(cfg.run_dir) / "eval" / split)
    metrics_path = write_eval_outputs(
        out_dir, metrics, per_query, metadata, bool(cfg.get("export_predictions", True))
    )
    print("Wrote", metrics_path)
    print(json.dumps({"roa": metrics["roa"], **({"p_success_mean": metrics["p_success_mean"]} if "p_success_mean" in metrics else {})}, indent=2))


if __name__ == "__main__":
    main()
