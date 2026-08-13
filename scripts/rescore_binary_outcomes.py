#!/usr/bin/env python
"""Replay stored per-point probabilities under ONE scoring rule.

WHY THIS EXISTS
---------------
Commit 17bf202 changed how stochastic systems score the "invalid" outcome: a
sampled endpoint landing near no attractor is a FAILURE, not a third class,
because the ground truth (successes/trials) has no third count. That flipped the
meaning of every threshold-dependent metric mid-campaign, so runs evaluated
before and after it cannot share a table. On the noisy-torque pendulum an `fm`
run carries ~70% mean p_invalid, so this is not a rounding difference.

Retraining is unnecessary: `full_roa_per_point.npz` stores the RAW per-point
p_success / p_failure / p_invalid and the true labels, and every affected metric
is a pure function of those. This replays them under a single rule.

WHAT IT GUARANTEES
------------------
* NON-DESTRUCTIVE. Writes `rescored_binary.json` beside each artifact and never
  touches `artifacts_v2.json`. The original scoring stays auditable.
* SAME CODE PATH. Metrics come from full_roa.py's own predicates
  (`_predict_lambda_delta`, `_threshold_free_metrics`, ...), not a
  reimplementation, so a replayed number equals what a fresh eval would emit.
* IDEMPOTENT. Folding an already-folded run is a no-op (p_invalid is zero), so
  re-running after more cells land is safe.
* PER-RUN PROVENANCE. Each output records the fold applied, the source npz, and
  the code SHA, so a later reader can tell which rule produced the number.

The binary flag is derived per run the same way the system property derives it --
`eval_success_prob.npz` existing beside the dataset -- rather than being passed
in, so a deterministic anchor cannot be silently rescored as if it were
stochastic.

Usage:
    ./env/bin/python scripts/rescore_binary_outcomes.py            # all campaign runs
    ./env/bin/python scripts/rescore_binary_outcomes.py --dry-run  # report, write nothing
    ./env/bin/python scripts/rescore_binary_outcomes.py --csv out.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_roa.adaptive_v2.eval.full_roa import (  # noqa: E402
    _classification_metrics_from_predictions,
    _predict_fixed_threshold,
    _predict_lambda_delta,
    _predict_lambda_only,
    _threshold_free_metrics,
)

DEFAULT_ROOTS = [
    "adaptive_pendulum_noisy_torque_tau_0.10",
    "adaptive_pendulum_noisy_torque_tau_0.15",
    "adaptive_pendulum_noisy_torque_tau_0.30",
    "adaptive_pendulum_noisy_torque_tau_0.50",
    "adaptive_pendulum_det_anchor_lqr",
]


def code_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def _register_resolvers() -> None:
    """Register the ${data_dir:}/${exp_dir:} resolvers the configs interpolate through."""
    from omegaconf import OmegaConf

    from adaptive_roa.utils.env_config import get_data_dir, get_exp_dir, get_net_id
    for name, fn in (("data_dir", get_data_dir), ("exp_dir", get_exp_dir),
                     ("net_id", get_net_id)):
        try:
            OmegaConf.register_new_resolver(name, lambda _f=fn: _f())
        except ValueError:
            pass  # already registered
    try:
        OmegaConf.register_new_resolver("now", lambda fmt="": "")
    except ValueError:
        pass


def read_hydra(run_dir: Path) -> dict:
    """decision_rule / dataset_root from the run's own config, INTERPOLATIONS RESOLVED.

    `.hydra/config.yaml` stores the composed config with interpolations intact
    -- `dataset_root: ${data_dir}/${noise_regime}/${dataset_name}` -- so reading
    the raw line yields a path containing literal `${...}` that can never exist.
    An earlier draft did exactly that, so `is_binary` returned False for every
    run, no fold was applied, and the script cheerfully reported "0/55 changed"
    while doing nothing at all. Resolve through OmegaConf instead.
    """
    from omegaconf import OmegaConf

    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return {}
    _register_resolvers()
    cfg = OmegaConf.load(cfg_path)
    out = {}
    for key in ("decision_rule", "dataset_root"):
        try:
            val = OmegaConf.select(cfg, key)
            out[key] = str(val) if val is not None else None
        except Exception:
            out[key] = None
    return out


def is_binary(dataset_root: str | None) -> bool:
    """Mirror DynamicalSystem.binary_outcomes: the npz beside the data IS the flag."""
    if not dataset_root:
        return False
    return (Path(dataset_root) / "eval_success_prob.npz").exists()


def rescore_epoch(epoch_dir: Path, run_dir: Path, sha: str | None) -> dict | None:
    npz_path = epoch_dir / "full_roa_per_point.npz"
    art_path = epoch_dir / "artifacts_v2.json"
    if not npz_path.exists() or not art_path.exists():
        return None

    d = np.load(str(npz_path))
    art = json.loads(art_path.read_text())
    em = art.get("eval_metrics", {}) or {}

    hy = read_hydra(run_dir)
    binary = is_binary(hy.get("dataset_root"))
    rule = hy.get("decision_rule", "one_sided")

    p_s = np.asarray(d["p_success"], dtype=np.float64)
    p_f = np.asarray(d["p_failure"], dtype=np.float64)
    p_i = np.asarray(d["p_invalid"], dtype=np.float64)
    y = np.asarray(d["true_labels"])
    lam = float(d["lambda_star"])
    dlt = float(d["delta"])
    K = em.get("num_mc_samples")

    invalid_mass_before = float(p_i.mean())
    if binary:
        p_f = p_f + p_i
        p_i = np.zeros_like(p_i)

    pred_ld, extras_ld = _predict_lambda_delta(
        p_s, p_f, p_i, lambda_star=lam, delta=dlt, decision_rule=rule,
        invalid_threshold=None, binary_outcomes=binary,
    )
    m_ld = _classification_metrics_from_predictions(pred_ld, y)
    m_ld.update(extras_ld)

    pred_fx, extras_fx = _predict_fixed_threshold(p_s, p_f, p_i, binary_outcomes=binary)
    m_fx = _classification_metrics_from_predictions(pred_fx, y)
    m_fx.update(extras_fx)

    pred_lo, extras_lo = _predict_lambda_only(
        p_s, p_f, p_i, lambda_star=lam, decision_rule=rule,
        invalid_threshold=None, binary_outcomes=binary,
    )
    m_lo = _classification_metrics_from_predictions(pred_lo, y)
    m_lo.update(extras_lo)

    tf = _threshold_free_metrics(p_s, y, num_mc_samples=K, p_invalid=p_i)

    return {
        "_doc": "Metrics replayed from full_roa_per_point.npz under a single "
                "scoring rule. Does not replace artifacts_v2.json.",
        "provenance": {
            "source_npz": str(npz_path),
            "binary_outcomes_applied": binary,
            "binary_inferred_from": "eval_success_prob.npz beside dataset_root",
            "dataset_root": hy.get("dataset_root"),
            "decision_rule": rule,
            "num_mc_samples": K,
            "lambda_star": lam,
            "delta": dlt,
            "rescored_by_commit": sha,
            "mean_p_invalid_before_fold": invalid_mass_before,
        },
        "threshold_free": tf,
        "lambda_delta": m_ld,
        "fixed_threshold": m_fx,
        "lambda_only": m_lo,
        "original": {
            "lambda_delta_f1": (em.get("lambda_delta") or {}).get("f1"),
            "lambda_delta_accuracy": (em.get("lambda_delta") or {}).get("accuracy"),
            "threshold_free_auc": (em.get("threshold_free") or {}).get("auc"),
            "n_invalid": (em.get("lambda_delta") or {}).get("n_invalid"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="/common/users/shared/pracsys/adaptive_roa_experiments")
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    sha = code_sha()
    rows, n_written, n_changed = [], 0, 0

    for root in args.roots:
        base = Path(args.exp_dir) / root
        if not base.exists():
            print(f"  (missing, skipped) {root}")
            continue
        for art in sorted(base.glob("outputs/*/train_*/epoch_*/artifacts_v2.json")):
            epoch_dir = art.parent
            run_dir = epoch_dir.parent
            res = rescore_epoch(epoch_dir, run_dir, sha)
            if res is None:
                continue
            if not args.dry_run:
                (epoch_dir / "rescored_binary.json").write_text(json.dumps(res, indent=2))
                n_written += 1

            arm = run_dir.parent.name
            new_f1 = res["lambda_delta"].get("f1")
            old_f1 = res["original"]["lambda_delta_f1"]
            changed = (old_f1 is not None and new_f1 is not None
                       and abs(float(new_f1) - float(old_f1)) > 1e-9)
            n_changed += int(changed)
            rows.append({
                "level": root.replace("adaptive_pendulum_noisy_torque_", "").replace("adaptive_pendulum_", ""),
                "arm": arm,
                "train_size": run_dir.name.split("_")[1],
                "binary": res["provenance"]["binary_outcomes_applied"],
                "mean_p_invalid_before": round(res["provenance"]["mean_p_invalid_before_fold"], 5),
                "f1_old": old_f1, "f1_new": new_f1,
                "auc_old": res["original"]["threshold_free_auc"],
                "auc_new": res["threshold_free"].get("auc"),
                "n_invalid_old": res["original"]["n_invalid"],
                "n_invalid_new": res["lambda_delta"].get("n_invalid"),
            })

    if args.csv and rows:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote csv: {args.csv}")

    print(f"\nrescored {len(rows)} epochs "
          f"({'dry run, nothing written' if args.dry_run else f'{n_written} rescored_binary.json written'})")
    print(f"metrics changed vs original: {n_changed}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
