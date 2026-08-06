#!/usr/bin/env python
"""What does each acquisition strategy actually select?

Downstream metrics tell you whether an arm ended up better; they do not tell you
whether the acquisition did what it claims. Entropy acquisition is supposed to
concentrate on states whose outcome is genuinely uncertain, i.e. where the TRUE
success probability sits near 1/2. This reads the acquired pool indices straight
out of each epoch's artifacts, looks up the ground-truth probability of the
states actually taken, and reports how ambiguous they were.

The training pool is sampled uniformly at random over the state space, so the
non-adaptive arm is the reference distribution: whatever it shows is what
"selecting nothing in particular" looks like.

Ground-truth p for a pool state comes from the nearest eval-grid cell (the pool
is continuous, the grid is 158x315), so it is a close proxy rather than exact.

Usage:
    python scripts/acquisition_diagnostics.py --out docs/stoch_compare/acquisition.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from stoch_compare_report import ARM_LABEL, ARM_ORDER, DATA, LEVELS, discover

POOL = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum/lqr")


def pool_starts(level: str) -> np.ndarray:
    """Start states in POOL order.

    Acquisition records pool indices k, not npz rows: the loader maps k through
    the shuffle with npz_row = rollout_ids[k]. Indexing the npz directly with k
    silently returns unrelated states -- and because the shuffle is a random
    permutation, the result looks exactly like random selection, which is a very
    convincing wrong answer.
    """
    with np.load(POOL / level / "train.npz") as z:
        starts = z["starts"].astype(np.float64)
    rollout_ids = np.loadtxt(POOL / level / "train_test_splits" / "shuffled_indices_0.txt",
                             dtype=np.int64, ndmin=1)
    return starts[rollout_ids]


def grid_lookup(level: str):
    with np.load(POOL / level / "eval_success_prob.npz") as z:
        starts = z["starts"].astype(np.float64)
        p = z["p_success"].astype(np.float64)
    tree = cKDTree(starts)
    return lambda q: p[tree.query(q, k=1)[1]]


def summarise(p_true: np.ndarray) -> dict:
    if len(p_true) == 0:
        return {}
    return {
        "n": int(len(p_true)),
        "mean_|p-0.5|": float(np.mean(np.abs(p_true - 0.5))),
        "frac_ambiguous": float(np.mean((p_true > 0.2) & (p_true < 0.8))),
        "frac_decided": float(np.mean((p_true < 0.05) | (p_true > 0.95))),
        "mean_p": float(np.mean(p_true)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/stoch_compare/acquisition.md"))
    ap.add_argument("--max-epoch", type=int, default=None)
    args = ap.parse_args()

    arms = discover()
    rows = []
    for lvl in LEVELS:
        lvl_arms = [a for a in arms if a["level"] == lvl]
        if not lvl_arms:
            continue
        starts = pool_starts(lvl)
        lookup = grid_lookup(lvl)
        for a in lvl_arms:
            d1_all, d2_all = [], []
            for ed in sorted(a["run_dir"].glob("epoch_*")):
                f = ed / "artifacts_v2.json"
                if not f.exists():
                    continue
                ep = int(ed.name.split("_")[1])
                if args.max_epoch is not None and ep > args.max_epoch:
                    continue
                try:
                    acq = json.loads(f.read_text()).get("acquisition") or {}
                except json.JSONDecodeError:
                    continue
                d1_all += list(acq.get("d1_indices") or [])
                d2_all += list(acq.get("d2_indices") or [])
            if not d1_all and not d2_all:
                continue
            row = {"predictor": a["predictor"], "level": lvl, "arm": a["arm"]}
            for tag, idx in (("D1(random)", d1_all), ("D2(scored)", d2_all)):
                if idx:
                    row[tag] = summarise(lookup(starts[np.asarray(idx, dtype=int)]))
            rows.append(row)

    md = ["# What each acquisition strategy actually selected", "",
          "`mean |p-0.5|` near 0 means the acquired states are genuinely ambiguous "
          "(the model cannot know the outcome); near 0.5 means their outcome is "
          "essentially determined. `frac_ambiguous` is the share with 0.2 < p < 0.8, "
          "`frac_decided` the share with p < 0.05 or p > 0.95.", "",
          "D1 is the random half of the budget, D2 the entropy-scored half. The "
          "non-adaptive arm draws its whole budget from the uniform pool, so its D1 "
          "row is the reference for 'no selection at all'.", ""]
    for pred in ("fm", "clf"):
        sub = [r for r in rows if r["predictor"] == pred]
        if not sub:
            continue
        md += [f"## {'Flow matching' if pred == 'fm' else 'Classifier'}", "",
               "`mean p` is the average TRUE success probability of the acquired states. "
               "The random split shows what the pool actually looks like, so a scored "
               "split that drifts away from it is training on a sample that no longer "
               "represents the space the model is scored on.", "",
               "| level | arm | split | n | mean \\|p-0.5\\| | frac ambiguous | "
               "frac decided | mean p |",
               "|---|---|---|---|---|---|---|---|"]
        for lvl in LEVELS:
            for arm in ARM_ORDER:
                r = next((x for x in sub if x["level"] == lvl and x["arm"] == arm), None)
                if r is None:
                    continue
                for tag in ("D1(random)", "D2(scored)"):
                    s = r.get(tag)
                    if not s:
                        continue
                    md.append(f"| {lvl} | {ARM_LABEL[arm]} | {tag} | {s['n']} | "
                              f"{s['mean_|p-0.5|']:.4f} | {s['frac_ambiguous']:.4f} | "
                              f"{s['frac_decided']:.4f} | {s['mean_p']:.4f} |")
        md.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md))
    print(f"wrote {args.out} ({len(rows)} arm rows)")


if __name__ == "__main__":
    main()
