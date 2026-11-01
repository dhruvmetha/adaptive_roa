#!/usr/bin/env python3
"""Re-score an epoch's test metrics three ways (success / failure / neither) from its checkpoint.

Cells launched before 49c7ab2 on the stochastic-tree zero-noise datasets, run where the
shared tree is mounted (iLab, direct boxes), scored the TEST set two ways: eval.binary_outcomes
was null, eval_success_prob.npz sat beside the dataset, and every 'neither' endpoint was folded
into failure. Training, acquisition and the threshold fit never read the flag, so the model,
the training set and the epoch's lambda*/delta*/q_hat are all valid; only the test scoring is
redone here, with the engine's own evaluator and the epoch's stored threshold state.

Writes <run>/rescored_ternary/epoch_NNN/{artifacts_v2.json, full_roa_per_point.npz}; the
originals are never touched. plot_progress.py reads the re-scored artifact in place of a
folded epoch.

    rescore_ternary.py --run-dir <run> [--epochs 0,1,2 | --folded] [--device cuda:0] [--max-rows N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

ENDPOINT_PREDICTORS = ("fm", "fm_det", "det_mlp_reg", "det_bnn_reg")   # fm_det runs record name "fm"


def load_handle(run_dir: Path, epoch: int, cfg, system, device: str):
    """The epoch's model handle, as the engine's evaluator takes it."""
    name = str(cfg.predictor.name)
    if name in ("fm", "fm_det"):
        from adaptive_roa.probabilistic_classifier.flow_matching import FMProbabilisticClassifier
        return FMProbabilisticClassifier.load_from_run(str(run_dir), epoch, cfg, system, device).flow_matcher
    from adaptive_roa.probabilistic_classifier import bayesian_final_state as bfs
    cls = {"det_mlp_reg": bfs.DetMLPRegProbabilisticClassifier,
           "det_bnn_reg": bfs.DetBNNRegProbabilisticClassifier}.get(name)
    if cls is None:
        raise SystemExit(f"{run_dir}: predictor {name!r} has no endpoints to re-score")
    return cls.load_from_run(str(run_dir), epoch, cfg, system, device).handle


def is_folded(epoch_dir: Path) -> bool:
    f = epoch_dir / "full_roa_per_point.npz"
    if not f.exists():
        return False
    d = np.load(f)
    return bool(d["p_invalid"].max() == 0 and np.allclose(d["p_success"] + d["p_failure"], 1.0))


def rescore(run_dir: Path, epoch: int, device: str, max_rows: int | None = None, seed: int = 0) -> dict:
    import hydra
    from adaptive_roa.adaptive_v2.eval.full_roa import FullROAEvaluator
    from adaptive_roa.adaptive_v2.types import ThresholdState

    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    with open_dict(cfg):
        cfg.eval.binary_outcomes = False
        if max_rows is not None:
            cfg.eval.max_eval_rows = int(max_rows)
    system = hydra.utils.instantiate(cfg.system)
    evaluator = FullROAEvaluator(cfg.eval, system, device)
    epoch_dir = run_dir / f"epoch_{epoch:03d}"
    art = json.loads((epoch_dir / "artifacts_v2.json").read_text())
    ts = ThresholdState(**art["threshold_state"])
    handle = load_handle(run_dir, epoch, cfg, system, device)
    out = run_dir / "rescored_ternary" / f"epoch_{epoch:03d}"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed + epoch)
    np.random.seed(seed + epoch)
    t0 = time.time()
    metrics = evaluator.evaluate_epoch(handle, ts, {
        "eval_states_file": cfg.data_source.test_set_file,
        "batch_size": int(cfg.get("val_batch_size", 2048)),
        "output_dir": str(out),
        "store_endpoints": False,
        "binary_outcomes": False,
        "verbose": False,
    })
    old = art.get("eval_metrics") or {}
    for k in ("q_hat_training", "q_hat_eval", "n_cal_eval"):      # set by the engine around the call
        if k in old:
            metrics[k] = old[k]
    from adaptive_roa.adaptive_v2.engine import _convert_numpy
    new = dict(art)
    new["eval_metrics"] = _convert_numpy(metrics)
    new.setdefault("extra", {})
    new["extra"] = dict(new["extra"] or {}, rescored_ternary={
        "from": str(epoch_dir / "artifacts_v2.json"), "reason": "eval.binary_outcomes folded invalid into failure",
        "seed": seed + epoch, "max_rows": max_rows, "seconds": round(time.time() - t0, 1),
        "commit": os.environ.get("ADAPTIVE_ROA_COMMIT", "worktree")})
    tmp = out / "artifacts_v2.json.tmp"
    tmp.write_text(json.dumps(new, indent=2))
    os.replace(tmp, out / "artifacts_v2.json")
    ld = new["eval_metrics"]["lambda_delta"]
    return {"epoch": epoch, "invalid_pct": ld["invalid_pct"], "uncertain_pct": ld["uncertain_pct"], "f1": ld["f1"],
            "seconds": new["extra"]["rescored_ternary"]["seconds"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--epochs", default=None, help="comma list; default with --folded: every folded epoch")
    ap.add_argument("--folded", action="store_true", help="re-score every folded epoch not yet re-scored")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-rows", type=int, default=None, help="tests only: cap the test rows")
    a = ap.parse_args()
    cfg = OmegaConf.load(a.run_dir / "resolved_config.yaml")
    if str(cfg.predictor.name) not in ENDPOINT_PREDICTORS:
        raise SystemExit(f"{a.run_dir}: {cfg.predictor.name} has no endpoints; nothing to re-score")
    if a.epochs:
        epochs = [int(e) for e in a.epochs.split(",")]
    else:
        epochs = [int(p.name[-3:]) for p in sorted(a.run_dir.glob("epoch_*"))
                  if is_folded(p) and not (a.run_dir / "rescored_ternary" / p.name / "artifacts_v2.json").exists()]
    for e in epochs:
        r = rescore(a.run_dir, e, a.device, a.max_rows)
        print(f"{a.run_dir.name} epoch {e:03d}: invalid {100 * r['invalid_pct']:.1f}%  "
              f"uncertain {100 * r['uncertain_pct']:.1f}%  F1 {r['f1']:.3f}  ({r['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
