"""Check infer.run_counts against each run's own stored epoch_010 D_test predictions.

    PYTHONPATH=<detv3>:<slices> python slices/lib/validate_infer.py [--n 5000] [--settings cp-lqr,...]

Per run: load D_test from the config's eval_states_file and confirm it is row-aligned with the
stored full_roa_per_point.npz (exact start_states and labels), rerun a seeded random subset, and
compare. The noise-only expectation draws two independent multinomial(K, stored p) count vectors
per state and compares their decisions, i.e. two K = 100 evaluations of a model whose endpoint
probabilities are the stored ones. Writes lib/_scratch/validation.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SLICES = Path(__file__).resolve().parents[1]
if str(SLICES) not in sys.path:
    sys.path.insert(0, str(SLICES))
from lib import infer  # noqa: E402


def noise_only(c: dict, thr: dict, reps: int, rng) -> tuple[float, float]:
    """(expected decision agreement, expected mean |dps|) between two K-draw evaluations."""
    K = c["K"]
    P = np.stack([c["n_success"], c["n_failure"], c["n_invalid"]], 1) / K
    agree, dps = [], []
    for _ in range(reps):
        a = rng.multinomial(K, P)
        b = rng.multinomial(K, P)
        da = infer.decide(dict(n_success=a[:, 0], n_failure=a[:, 1], n_invalid=a[:, 2], K=K), thr)
        db = infer.decide(dict(n_success=b[:, 0], n_failure=b[:, 1], n_invalid=b[:, 2], K=K), thr)
        agree.append(np.mean(da == db))
        dps.append(np.mean(np.abs(a[:, 0] - b[:, 0])) / K)
    return float(np.mean(agree)), float(np.mean(dps))


def gt_view(d, y01):
    S, F = d == 1, d == 0
    return dict(gt_agree=float(np.mean(np.where(y01 == 1, S, F))), abstain=float(np.mean(d == -1)),
                R_s=float((S & (y01 == 1)).sum() / max(1, (y01 == 1).sum())),
                R_f=float((F & (y01 == 0)).sum() / max(1, (y01 == 0).sum())))


def validate(name: str, n: int, device: str, seed: int) -> dict:
    """``name``: a setting (its README run) or any run name."""
    infer._import_detv3()
    from adaptive_roa.adaptive.data_source import load_eval_states

    name = infer.RUNS.get(name, name)
    rd = infer.run_dir(name)
    setting = Path(rd).parent.name
    thr = infer.thresholds(name)
    cfg, system, _, _, ckpt = infer.load_run(rd, device)      # cfg with PATH_REMAP applied
    D = int(system.state_dim)
    test_file = cfg.data_source.test_set_file
    assert cfg.data_source.eval_states_file == test_file, "eval_states_file != test_set_file"
    X, _, y = load_eval_states(test_file, state_dim=D)
    z = np.load(Path(rd) / infer.EPOCH / "full_roa_per_point.npz", allow_pickle=True)
    aligned = bool(np.array_equal(z["start_states"], X) and np.array_equal(z["true_labels"], y))
    if not aligned:
        raise RuntimeError(f"{setting}: {test_file} is not row-aligned with the stored per-point npz")
    stored = infer.counts_from_per_point(Path(rd) / infer.EPOCH / "full_roa_per_point.npz")
    idx = np.sort(np.random.default_rng(seed).choice(len(y), size=min(n, len(y)), replace=False))
    new = infer.run_counts(rd, X[idx], device=device, seed=seed)
    st = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in stored.items()}
    d_new, d_st = infer.decide(new, thr), infer.decide(st, thr)
    y01 = (y[idx] == 1).astype(int)
    exp_agree, exp_dps = noise_only(st, thr, reps=50, rng=np.random.default_rng(seed + 1))
    out = dict(setting=setting, run=name, run_dir=rd, ckpt=ckpt, test_file=test_file, n_test=int(len(y)),
               n=int(len(idx)), aligned=aligned, thresholds=thr,
               mean_abs_dps=float(np.mean(np.abs(new["n_success"] - st["n_success"])) / new["K"]),
               exp_mean_abs_dps=exp_dps,
               mean_ps_stored=float(st["n_success"].mean() / 100), mean_ps_new=float(new["n_success"].mean() / 100),
               mean_pinv_stored=float(st["n_invalid"].mean() / 100), mean_pinv_new=float(new["n_invalid"].mean() / 100),
               agree=float(np.mean(d_new == d_st)), exp_agree=exp_agree,
               stored=gt_view(d_st, y01), rerun=gt_view(d_new, y01),
               seconds=new["seconds"], states_per_s=len(idx) / new["seconds"])
    print(json.dumps(out), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--settings", default=",".join(infer.RUNS), help="settings or run names, comma separated")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "_scratch" / "validation.json"))
    a = p.parse_args()
    print("adaptive_roa:", infer.adaptive_roa_file(), flush=True)
    res = [validate(s, a.n, a.device, a.seed) for s in a.settings.split(",")]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("\n| setting | run | n | mean abs dps (noise) | agree (noise) | GT agree stored / rerun | "
          "abstain stored / rerun | states/s |")
    print("|---|---|---|---|---|---|---|---|")
    for r in res:
        print(f"| {r['setting']} | {r['run']} | {r['n']} | {r['mean_abs_dps']:.4f} ({r['exp_mean_abs_dps']:.4f}) | "
              f"{r['agree']:.4f} ({r['exp_agree']:.4f}) | {r['stored']['gt_agree']:.4f} / {r['rerun']['gt_agree']:.4f} | "
              f"{r['stored']['abstain']:.4f} / {r['rerun']['abstain']:.4f} | {r['states_per_s']:.0f} |")


if __name__ == "__main__":
    main()
