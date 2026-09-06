#!/usr/bin/env python
"""Final-epoch level-set curves, one figure per noise level.

For each level beta both the ground-truth p and the predicted p_hat are
thresholded at the same beta, and the confusion matrix's rates are drawn against
beta. Every panel carries the oracle ceiling (a model that knows the field
exactly, scored through the same M-rollout and K-sample noise) and the 3-seed
FM control band. All ten levels are always drawn; the TPR and TNR panels print
the fraction of the grid in the true set at each level, which is what decides
how far a thin level can be trusted.

The oracle bounds the BALANCED statistics. A one-sided rate can exceed it: an
arm that over-claims the level-beta set buys TPR with FPR, and the oracle is
unbiased. Read TPR next to FPR or volume ratio, or read F0.5 / balanced accuracy.

Style follows plot_stoch_all_levels: colour is the (predictor, arm) pair, line
style is the predictor family, and an arm is never read against the wrong
baseline. Like that script it refuses a level whose arms sit at ragged depths
unless --allow-partial is passed, and then stamps the depths on the figure.

Usage:
    python scripts/plot_stoch_levelsets.py                 # every campaign
    python scripts/plot_stoch_levelsets.py pendulum cartpole
"""
from __future__ import annotations
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_stoch_all_levels import (  # noqa: E402
    ARMS, BAND_A, BAND_C, CAMPAIGNS, DOCS, MEAN_C, MEAN_LW, SEEDS, within_budget,
)
from score_stoch_incremental import levelsets_path  # noqa: E402

PANELS = [
    ("tpr",       "TPR   recall of the true level-β set"),
    ("tnr",       "TNR"),
    ("fpr",       "FPR   unsafe cells declared safe"),
    ("fnr",       "FNR   safe cells given up"),
    ("acc",       "accuracy   (base-rate dominated: read balanced)"),
    ("bal_acc",   "balanced accuracy"),
    ("f1",        "F1"),
    ("f05",       "F0.5   precision-weighted"),
    ("realized",  "realized success of the claimed set"),
    ("vol_ratio", "volume ratio   claimed / true   (log)"),
]
EXACT_FAMILIES = {"clf", "gp", "bnn"}


def _num(v):
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def level_rows_within_budget(long_rows: list[dict], wide_rows: list[dict], c: dict) -> list[dict]:
    """Long-format rows whose (level, arm, epoch) sits at or below the campaign's
    reporting budget. The trajectory count lives on the wide CSV, so the two are
    joined on their shared key; with no budget configured every row passes."""
    if c.get("budget") is None:
        return long_rows
    ok = {(r["level"], r["arm"], int(r["epoch"])) for r in within_budget(wide_rows, c)}
    return [r for r in long_rows if (r["level"], r["arm"], int(r["epoch"])) in ok]


def final_epoch_rows(rows: list[dict]) -> tuple[dict[str, dict[float, dict]], dict[str, int]]:
    """Per arm: the deepest epoch's rows keyed by beta (numeric fields parsed),
    and that depth. Rows are the long-format CSV filtered to one level."""
    by: dict[str, dict[int, dict[float, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by[r["arm"]][int(r["epoch"])][float(r["beta"])] = {k: _num(v) for k, v in r.items()}
    out, depth = {}, {}
    for arm, eps in by.items():
        e = max(eps)
        depth[arm] = e
        out[arm] = dict(eps[e])
    return out, depth


def _stat(vals, fn):
    v = [x for x in vals if isinstance(x, float) and not math.isnan(x)]
    return fn(v) if v else float("nan")


DEFAULT_SUBTITLE = ("truth and prediction both thresholded at β · "
                    "solid = flow matching · dotted = classifier · dash-dot = Part-X GP · "
                    "dashed black = what a perfect model scores through the same sampling noise")


def draw_level(by_arm: dict, depth: dict, out_path: Path, title: str,
               arms: list | None = None, subtitle: str | None = None) -> Path:
    """`arms` defaults to the canonical ARMS table. Pass a subset to draw a
    restricted comparison; the caller is responsible for filtering by_arm and
    depth to match, since depth drives the ragged-depth stamp."""
    betas = sorted({b for arm in by_arm for b in by_arm[arm]})
    have = [s for s in SEEDS if s in by_arm]
    # The true set sizes are a property of the level, not the arm: any row will do.
    probe = next(iter(by_arm.values()))
    n_grid = probe[betas[0]]["n_pos_true"] + probe[betas[0]]["n_neg_true"]
    fam = {arm: rows[betas[0]]["predictor"] for arm, rows in by_arm.items()}
    fm_src = have[0] if have else next((a for a in by_arm if fam[a] == "fm"), None)
    ex_src = next((a for a in by_arm if fam[a] in EXACT_FAMILIES), None)

    fig, axes = plt.subplots(2, 5, figsize=(27, 10.4))
    for ax, (col, ylab) in zip(axes.flat, PANELS):
        # Every level is always drawn and always in the area. What varies is
        # how much of the grid the true set holds, which is what decides how
        # far a level can be trusted -- so print it on the rate it governs.
        if col in ("tpr", "tnr") and n_grid:
            side = "n_pos_true" if col == "tpr" else "n_neg_true"
            for b in betas:
                frac = probe[b][side] / n_grid
                ax.text(b, 1.035, f"{100 * frac:.3g}%" if frac >= 0.001 else "<0.1%",
                        ha="center", va="bottom", fontsize=6.4, color="#555",
                        transform=ax.get_xaxis_transform())
            ax.text(0.0, 1.035, "true set:", ha="right", va="bottom", fontsize=6.4,
                    color="#555", transform=ax.get_xaxis_transform())
        if len(have) == 3:
            vals = [[by_arm[s][b][col] for s in have] for b in betas]
            ax.fill_between(betas, [_stat(v, min) for v in vals], [_stat(v, max) for v in vals],
                            color=BAND_C, alpha=BAND_A, zorder=1,
                            label="FM non-adaptive (3-seed range)")
            ax.plot(betas, [_stat(v, lambda z: sum(z) / len(z)) for v in vals],
                    color=MEAN_C, lw=MEAN_LW, zorder=2, label="FM non-adaptive (mean)")
        for arm, label, colour, style, mk in (ARMS if arms is None else arms):
            if arm not in by_arm:
                continue
            ax.plot(betas, [by_arm[arm][b][col] for b in betas], style, color=colour,
                    lw=1.9, marker=mk, ms=4, zorder=3, label=label)
        if fm_src is not None:
            ax.plot(betas, [by_arm[fm_src][b][f"{col}_oracle"] for b in betas], "--",
                    color="k", lw=1.6, zorder=4, label="oracle ceiling · FM (K-sample + M-rollout noise)")
        if ex_src is not None:
            ax.plot(betas, [by_arm[ex_src][b][f"{col}_oracle"] for b in betas], "--",
                    color="0.45", lw=1.4, zorder=4, label="oracle ceiling · exact predictor (M-rollout noise)")
        if col == "realized":
            ax.plot([0, 1], [0, 1], ":", color="0.3", lw=1.2, zorder=4,
                    label="realized = β   (claim holds on or above)")
            ax.set_ylim(0, 1)
        elif col == "vol_ratio":
            ax.axhline(1.0, ls=":", color="0.3", lw=1.2, zorder=4)
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, 1)
        ax.set_xticks(betas)
        ax.set_xlabel("level β   (= 1 − risk tolerance α)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, which="both", lw=0.5)

    top = max(depth.values())
    short = {a: e for a, e in sorted(depth.items()) if e != top}
    stamp = (f"final epoch {top}" if not short else
             f"final epoch {top}, RAGGED, still short: "
             + ", ".join(f"{a}={e}" for a, e in short.items()))
    fig.suptitle(f"{title} — level-set metrics at the {stamp}\n"
                 + (DEFAULT_SUBTITLE if subtitle is None else subtitle),
                 fontsize=13, y=0.995)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, max(3, (len(labels) + 1) // 2)),
               fontsize=8.6, frameon=False, bbox_to_anchor=(0.5, 0.004))
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaigns", nargs="*", default=list(CAMPAIGNS))
    ap.add_argument("--allow-partial", action="store_true",
                    help="draw a level whose arms sit at ragged depths; the figure is stamped")
    a = ap.parse_args()
    for name in (a.campaigns or list(CAMPAIGNS)):
        c = CAMPAIGNS[name]
        src = levelsets_path(DOCS / c["csv"])
        if not src.exists():
            print(f"  SKIP {name}: {src.name} absent (run score_stoch_incremental first)")
            continue
        wide = list(csv.DictReader((DOCS / c["csv"]).open()))
        rows = level_rows_within_budget(list(csv.DictReader(src.open())), wide, c)
        cap = f"  ·  read to {c['budget']:,} training trajectories" if c.get("budget") else ""
        for lv, lab in c["panels"]:
            sub = [r for r in rows if r["level"] == lv]
            if not sub:
                print(f"  SKIP {name}/{lv}: no rows")
                continue
            by_arm, depth = final_epoch_rows(sub)
            if len(set(depth.values())) > 1 and not a.allow_partial:
                print(f"  REFUSING {name}/{lv}: ragged depths {dict(sorted(depth.items()))} "
                      "(pass --allow-partial to override)")
                continue
            out = (DOCS / c["out"]).with_name(f"levelsets_{lv}.png")
            draw_level(by_arm, depth, out, lab + cap)
            print(f"  wrote {out.relative_to(DOCS)}  ({len(by_arm)} arms, "
                  f"final epoch {max(depth.values())})")


if __name__ == "__main__":
    main()
