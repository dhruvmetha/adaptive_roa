"""Refit lambda*/delta* at other w from the stored threshold-fit predictions (A-W1, H03).

The engine stores, per epoch, the states, labels and probabilities the optimizer
fit on (predictions_thrfit.npz). This refits the same grid search, through the
same ConformalPredictor._optimize_loss, with only w changed, then scores the
refit rule on the test split. Refitting at the run's own w must give back the
run's own lambda*/delta*; `matches_run` records that check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval.store_score import score_epoch
from adaptive_roa.conformal import ConformalConfig, ConformalPredictor

WS = (0.5, 0.75, 0.9, 0.95, 0.99)


def _config(threshold_cfg, w: float) -> ConformalConfig:
    t = threshold_cfg
    return ConformalConfig(
        delta=float(t["delta"]), w=float(w), alpha=float(t.get("alpha", 0.1)),
        attractor_radius=float(t["attractor_radius"]), num_mc_samples=int(t["num_mc_samples"]),
        optimize_mode=str(t["optimize_mode"]), optimize_objective=str(t["optimize_objective"]),
        target_f1=float(t["target_f1"]), decision_rule=str(t["decision_rule"]),
        lambda_grid_size=int(t["lambda_grid_size"]), delta_grid_size=int(t["delta_grid_size"]),
        delta_min=float(t["delta_min"]), delta_max=float(t["delta_max"]),
        fixed_lambda_star=float(t["fixed_lambda_star"]), fixed_delta_star=float(t["fixed_delta_star"]),
        use_p_invalid_veto=bool(t["use_p_invalid_veto"]),
    )


def refit(threshold_cfg, w: float, p_s, p_f, p_i, y) -> tuple[float, float]:
    pred = ConformalPredictor(flow_matcher=None, system=None, config=_config(threshold_cfg, w),
                              device="cpu", probability_estimator=object())
    pred._optimize_loss(np.asarray(p_s), np.asarray(p_f), np.asarray(p_i), np.asarray(y), verbose=False)
    return float(pred.lambda_star), float(pred.delta_star)


def refit_run(run_dir, ws=WS) -> list[dict]:
    run = Path(run_dir)
    cfg = OmegaConf.load(run / "resolved_config.yaml")
    tcfg = OmegaConf.to_container(cfg.threshold, resolve=True)
    run_w = float(tcfg.get("w", 0.9))
    rule = str(cfg.decision_rule)
    rows = []
    for art_path in sorted(run.glob("epoch_*/artifacts_v2.json")):
        ep_dir = art_path.parent
        f = ep_dir / "predictions_thrfit.npz"
        art = json.loads(art_path.read_text())
        ld = (art.get("eval_metrics") or {}).get("lambda_delta")
        if not f.exists() or ld is None:
            continue
        z, pp = np.load(f), np.load(ep_dir / "full_roa_per_point.npz")
        val = ep_dir / "predictions_val.npz"
        p_val = y_val = None
        if val.exists():
            zv = np.load(val)
            p_val = (zv["p_success"] if "n_success" not in zv.files
                     else zv["n_success"] / float(zv["num_mc_samples"]))
            y_val = zv["true_labels"]
        for w in ws:
            lam, delta = refit(tcfg, w, z["p_success"], z["p_failure"], z["p_invalid"], z["true_labels"])
            row = score_epoch(pp["p_success"], pp["p_failure"], pp["p_invalid"], pp["true_labels"],
                              lam, delta, rule, None, p_val, y_val)
            row.update(run=run.name, epoch=int(art["epoch"]), train_trajectories=int(art["train_trajectories"]),
                       w=w, lambda_star=lam, delta_star=delta,
                       matches_run=(abs(lam - float(ld["lambda_star"])) < 1e-12
                                    and abs(delta - float(ld["delta"])) < 1e-12) if w == run_w else None)
            rows.append(row)
    return rows
