"""Shared loaders and scorers for the det-rerun post-hoc analyses.

Reads the stored per-epoch predictions of each cell (no model is loaded):
  test   full_roa_per_point.npz        eval-grid p_success / p_failure / p_invalid
  thrfit predictions_thrfit.npz        the set the run's lambda/delta were fit on (K = 10)
  val    predictions_val.npz           held-out starts at the eval K (counts), or a
                                       classifier's p_success (p_failure = 1 - p, no invalid)
Labels are {-1, 1}. Scorers mirror adaptive_roa/adaptive_v2/eval/full_roa.py.
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval.full_roa import (
    _classification_metrics_from_predictions,
    _conservative_metrics,
    _predict_lambda_delta,
)

MANIFEST = Path("/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun/manifest.tsv")
SETTINGS = ["pen", "cp", "q2d", "q3d30", "q3d60", "q3d110"]
LEARNED = ["fm_det", "det_mlp_clf", "det_bnn_clf", "det_mlp_reg", "det_bnn_reg"]
SHORT = {"fm_det": "FM", "det_mlp_clf": "MLP clf", "det_bnn_clf": "BNN clf", "det_mlp_reg": "MLP reg",
         "det_bnn_reg": "BNN reg", "partx_faithful": "Part-X"}
NEEDED = ("full_roa_per_point.npz", "predictions_thrfit.npz", "predictions_val.npz", "artifacts_v2.json")


def run_dirs() -> dict:
    out = {}
    for r in csv.DictReader(open(MANIFEST), delimiter="\t"):
        if r["name"].startswith("dr_") and "startpool" not in r["name"]:
            out[(r["system"], r["arm"], r["mode"])] = Path(re.search(r"output_dir=(\S+)", r["overrides"]).group(1))
    return out


def complete_epochs(d: Path, needed=NEEDED) -> list[int]:
    eps = []
    for p in glob.glob(f"{d}/epoch_*"):
        if all((Path(p) / f).exists() for f in needed):
            eps.append(int(p.rsplit("_", 1)[1]))
    return sorted(eps)


def matched_epoch(dirs: dict, setting: str, mode: str = "ranked", arms=LEARNED) -> int | None:
    """Latest epoch every learned arm of this setting has complete (the comparison point)."""
    eps = [complete_epochs(dirs[(setting, a, mode)]) for a in arms if (setting, a, mode) in dirs]
    if not eps or any(not e for e in eps):
        return None
    return min(max(e) for e in eps)


def _probs_from_counts(z):
    k = float(z["num_mc_samples"])
    return z["n_success"] / k, z["n_failure"] / k, z["n_invalid"] / k


def load_cell(d: Path, epoch: int) -> dict:
    ep = d / f"epoch_{epoch:03d}"
    pp = np.load(ep / "full_roa_per_point.npz")
    tf = np.load(ep / "predictions_thrfit.npz")
    zv = np.load(ep / "predictions_val.npz")
    art = json.loads((ep / "artifacts_v2.json").read_text())
    counts = "n_success" in zv.files
    if counts:
        vs, vf, vi = _probs_from_counts(zv)
    else:
        vs = np.asarray(zv["p_success"], float)
        vf, vi = 1.0 - vs, np.zeros_like(vs)
    cfg = OmegaConf.load(d / "resolved_config.yaml")
    return {
        "counts": counts,
        "k_test": int(art["eval_metrics"].get("num_mc_samples") or 0) if counts else None,
        "test": tuple(np.asarray(pp[k], float) for k in ("p_success", "p_failure", "p_invalid")) + (pp["true_labels"],),
        "thrfit": tuple(np.asarray(tf[k], float) for k in ("p_success", "p_failure", "p_invalid")) + (tf["true_labels"],),
        "val": (vs, vf, vi, zv["true_labels"]),
        "lam": float(pp["lambda_star"]), "delta": float(pp["delta"]),
        "rule": str(cfg.decision_rule),
        "threshold": OmegaConf.to_container(cfg.threshold, resolve=True),
        "eval_metrics": art["eval_metrics"],
    }


def score(pred: np.ndarray, y: np.ndarray) -> dict:
    """pred: 1 success, 0 failure, -1 outcome-uncertain, -2 invalid."""
    m = _classification_metrics_from_predictions(pred, y)
    c = _conservative_metrics(pred, y)
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
    return {"F1c": m["f1"], "F1all": c["f1"], "unc": 100 * m["separatrix_pct"], "inv": 100 * m["invalid_pct"],
            "ounc": 100 * m["uncertain_pct"], "FPR": fp / (fp + tn) if fp + tn else float("nan"),
            "FNR": fn / (fn + tp) if fn + tp else float("nan")}


def as_run(cell: dict) -> np.ndarray:
    ps, pf, pi, _ = cell["test"]
    pred, _ = _predict_lambda_delta(ps, pf, pi, cell["lam"], cell["delta"], cell["rule"], None)
    return pred
