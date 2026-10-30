"""Eval-K check from stored counts (tracker H04).

The K endpoint labels at one state are exchangeable, so a K' < K subsample's
(success, failure, invalid) counts follow a multivariate hypergeometric law given
the stored counts. Drawing it twice, sequentially, is exact:
n_s' ~ HG(n_s, K - n_s, K'), then n_f' ~ HG(n_f, n_i, K' - n_s').
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from adaptive_roa.adaptive_v2.eval.store_score import score_epoch

KS = (5, 10, 20, 50)


def subsample_counts(n_s, n_f, n_i, k_new: int, rng):
    n_s, n_f, n_i = (np.asarray(a, dtype=np.int64) for a in (n_s, n_f, n_i))
    k = n_s + n_f + n_i
    if np.any(k < k_new):
        raise ValueError(f"cannot subsample K'={k_new} from K={int(k.min())}")
    s = rng.hypergeometric(n_s, k - n_s, k_new)
    f = rng.hypergeometric(n_f, n_i, k_new - s) if np.any(n_i) else k_new - s
    f = np.where(n_f + n_i == 0, 0, f)
    return s, f, k_new - s - f


def k_check_run(run_dir, ks=KS, seed: int = 0) -> list[dict]:
    run = Path(run_dir)
    cfg = OmegaConf.load(run / "resolved_config.yaml")
    rule = str(cfg.decision_rule)
    k_eval = int(cfg.eval.num_mc_samples_eval)
    rows = []
    for art_path in sorted(run.glob("epoch_*/artifacts_v2.json")):
        art = json.loads(art_path.read_text())
        ld = (art.get("eval_metrics") or {}).get("lambda_delta")
        if ld is None:
            continue
        pp = np.load(art_path.parent / "full_roa_per_point.npz")
        # p = counts / K exactly, so the counts come back by rounding.
        n_s = np.rint(pp["p_success"] * k_eval).astype(int)
        n_f = np.rint(pp["p_failure"] * k_eval).astype(int)
        n_i = k_eval - n_s - n_f
        if np.any(n_i < 0):
            raise ValueError(f"{art_path}: per-point probabilities are not counts out of K={k_eval}")
        rng = np.random.default_rng([seed, int(art["epoch"])])
        for k in (*ks, k_eval):
            if k == k_eval:
                s, f, i = n_s, n_f, n_i
            else:
                s, f, i = subsample_counts(n_s, n_f, n_i, k, rng)
            row = score_epoch(s / k, f / k, i / k, pp["true_labels"], float(ld["lambda_star"]),
                              float(ld["delta"]), rule, ld.get("invalid_threshold"))
            row.update(run=run.name, epoch=int(art["epoch"]), train_trajectories=int(art["train_trajectories"]), K=k)
            rows.append(row)
    return rows
