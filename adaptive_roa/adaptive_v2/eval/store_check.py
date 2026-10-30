"""Refuse a run whose stored predictions do not reproduce its own artifacts (C08).

For every evaluated epoch, the lambda*/delta* rule is replayed on
``full_roa_per_point.npz`` and its confusion counts must equal the
``eval_metrics.lambda_delta`` block of ``artifacts_v2.json`` exactly. The val,
cal and threshold-fit files must exist and be internally consistent, and the
reporting epochs must carry their endpoints. A cell counts as done only when
this returns no problems (EXPERIMENT_TRACKER rules).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval.full_roa import (
    _classification_metrics_from_predictions,
    _predict_lambda_delta,
)

COUNT_KEYS = ("tp", "tn", "fp", "fn", "n_invalid", "n_uncertain")


def _check_counts(npz: Path, n_rows: int, problems: list[str], tag: str) -> None:
    z = np.load(npz)
    if "p_success" in z.files and "n_success" not in z.files:
        if len(z["p_success"]) != n_rows:
            problems.append(f"{tag}: {len(z['p_success'])} probabilities for {n_rows} states")
        return
    k = int(z["num_mc_samples"])
    total = z["n_success"].astype(int) + z["n_failure"] + z["n_invalid"]
    if len(total) != n_rows:
        problems.append(f"{tag}: {len(total)} count rows for {n_rows} states")
    elif not np.all(total == k):
        problems.append(f"{tag}: counts do not sum to K={k}")


def check_run(run_dir: str | Path) -> list[str]:
    run = Path(run_dir)
    problems: list[str] = []
    hydra_cfg = run / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
        return [f"{run}: no .hydra/config.yaml, so the decision rule and store settings are unknown"]
    cfg = OmegaConf.load(hydra_cfg)
    rule = str(cfg.get("decision_rule"))
    store = cfg.get("prediction_store", None)
    endpoint_epochs = {int(e) for e in (store.get("endpoint_epochs", []) if store else [])}

    epochs = sorted(run.glob("epoch_*/artifacts_v2.json"))
    if not epochs:
        return [f"{run}: no epoch artifacts"]
    split_rows = {}
    for split in ("val", "cal"):
        f = run / f"prediction_store_states_{split}.npz"
        if store and store.get("enabled", False):
            if not f.exists():
                problems.append(f"{run}: missing {f.name}")
            else:
                split_rows[split] = len(np.load(f)["true_labels"])

    for art_path in epochs:
        ep_dir = art_path.parent
        ep = int(ep_dir.name.split("_")[1])
        art = json.loads(art_path.read_text())
        ev = art.get("eval_metrics") or {}
        if ev.get("skipped") or "lambda_delta" not in ev:
            continue
        ld = ev["lambda_delta"]
        pp_path = ep_dir / "full_roa_per_point.npz"
        if not pp_path.exists():
            problems.append(f"{ep_dir.name}: missing full_roa_per_point.npz")
            continue
        pp = np.load(pp_path)
        pred, _ = _predict_lambda_delta(
            pp["p_success"], pp["p_failure"], pp["p_invalid"],
            float(ld["lambda_star"]), float(ld["delta"]), rule,
            ld.get("invalid_threshold"),
        )
        got = _classification_metrics_from_predictions(pred, pp["true_labels"])
        diff = {k: (got[k], ld[k]) for k in COUNT_KEYS if got[k] != ld[k]}
        if diff:
            problems.append(f"{ep_dir.name}: replayed counts differ from artifacts_v2.json: {diff}")

        if not (store and store.get("enabled", False)):
            continue
        for split, n_rows in split_rows.items():
            f = ep_dir / f"predictions_{split}.npz"
            if not f.exists():
                problems.append(f"{ep_dir.name}: missing {f.name}")
            else:
                _check_counts(f, n_rows, problems, f"{ep_dir.name}/{f.name}")
        lam = art.get("threshold_state", {}).get("lambda_star")
        if lam is not None and not (isinstance(lam, float) and math.isnan(lam)):
            if not (ep_dir / "predictions_thrfit.npz").exists():
                problems.append(f"{ep_dir.name}: thresholds were fit but predictions_thrfit.npz is missing")
        if ep in endpoint_epochs and "n_success" in np.load(ep_dir / "predictions_val.npz").files:
            for split in ("test", "val", "cal"):
                f = ep_dir / f"endpoints_{split}.npz"
                if not f.exists():
                    problems.append(f"{ep_dir.name}: reporting epoch without {f.name}")
                    continue
                e = np.load(f)
                if e["endpoints"].shape[0] != len(e["rows"]):
                    problems.append(f"{ep_dir.name}/{f.name}: endpoints and rows disagree")
    return problems
