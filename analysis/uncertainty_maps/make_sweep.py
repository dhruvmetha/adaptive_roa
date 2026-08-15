#!/usr/bin/env python
"""One quantity, every arm, every epoch, on one fixed scale.

The per-epoch pages in `make_maps.py` answer "what does this epoch look like".
This answers the question the campaign's original line figure tried to and
could not: how does the uncertainty MOVE through state space as data arrives.
That figure plotted pool means, gave every panel its own y-axis, and stopped
each curve wherever preemption landed, so no two panels could be compared. Here
the epoch ladder is identical for every arm and the colour scale is identical
for every panel, so a difference on screen is a difference in the data.

Arms that never reached an epoch get an explicit empty tile rather than a
shifted ladder -- a short row is a fact about the run, not something to hide by
renumbering.

    python make_sweep.py --pred fm --level high --quantity h_epistemic --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.uncertainty_maps.common import (  # noqa: E402
    ARMS, ARM_LABEL, EXP, LEVELS, binary_entropy, decompose, epochs_with_eval,
    load_marginal, oracle_full_image, rasteriser, to_grid,
)
from analysis.uncertainty_maps.make_maps import (  # noqa: E402
    COL_CMAP, COL_TITLE, INK, LN2, MUTED, epistemic_ceilings, load_members, panel,
)

QUANTITIES = ("p", "h_total", "h_aleatoric", "h_epistemic",
              "var_total", "var_aleatoric", "var_epistemic")


def values_for(quantity, marg, dec):
    if quantity == "p":
        return marg["p_success"]
    if dec is not None and quantity in dec:
        return dec[quantity]
    if quantity == "h_total":
        return binary_entropy(marg["p_success"])
    if quantity == "var_total":
        p = np.clip(marg["p_success"], 0, 1)
        return p * (1 - p)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, choices=("clf", "fm"))
    ap.add_argument("--level", required=True, choices=LEVELS)
    ap.add_argument("--quantity", required=True, choices=QUANTITIES)
    ap.add_argument("--every", type=int, default=2, help="epoch stride")
    ap.add_argument("--members", default="/common/users/shared/pracsys/"
                                         "adaptive_roa_experiments/_uncertainty_maps/members")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pred, level, q = a.pred, a.level, a.quantity
    arms = [arm for arm in ARMS if (EXP / f"{pred}_{level}_{arm}").exists()]
    all_eps = sorted({e for arm in arms for e in epochs_with_eval(pred, level, arm)})
    eps = all_eps[::a.every]

    ref = load_marginal(pred, level, arms[0], all_eps[0])
    shape, flat, extent = rasteriser(level, ref["states"])
    member_root = Path(a.members)
    h_max, v_max = epistemic_ceilings(pred, level, member_root, all_eps, arms)
    vmax = {"p": 1.0, "h_total": LN2, "h_aleatoric": LN2, "h_epistemic": h_max,
            "var_total": 0.25, "var_aleatoric": 0.25, "var_epistemic": v_max}[q]
    if vmax is None:
        raise SystemExit(f"{pred}_{level}: {q} needs per-member data, which this "
                         "cell does not have (checkpoints were deleted)")

    fig = plt.figure(figsize=(1.35 * len(eps) + 1.6, 1.5 * len(arms) + 1.15))
    gs = fig.add_gridspec(len(arms) + 1, len(eps),
                          height_ratios=[1] * len(arms) + [0.09],
                          hspace=0.10, wspace=0.05)
    for r, arm in enumerate(arms):
        avail = set(epochs_with_eval(pred, level, arm))
        for c, ep in enumerate(eps):
            ax = fig.add_subplot(gs[r, c])
            if ep not in avail:
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_facecolor("#f4f4f2")
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.text(.5, .5, "not\nreached", ha="center", va="center",
                        transform=ax.transAxes, color=MUTED, fontsize=6)
            else:
                marg = load_marginal(pred, level, arm, ep)
                pm, k = load_members(member_root, f"{pred}_{level}_{arm}", ep)
                vals = values_for(q, marg, decompose(pm, k) if pm is not None else None)
                if vals is None:
                    ax.axis("off")
                    ax.text(.5, .5, "no members", ha="center", va="center",
                            transform=ax.transAxes, color=MUTED, fontsize=6)
                else:
                    panel(ax, to_grid(vals, shape, flat), extent, COL_CMAP[q],
                          0, vmax, "")
            if r == 0:
                ax.set_title(f"epoch {ep}", color=INK, fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(ARM_LABEL[arm], color=INK, fontsize=8)

    cax = fig.add_subplot(gs[len(arms), :])
    sm = plt.cm.ScalarMappable(cmap=COL_CMAP[q], norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(COL_TITLE[q].replace("\n", " "), color=INK, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, color=MUTED, labelcolor=MUTED, length=2)

    fig.suptitle(f"{'flow matching' if pred == 'fm' else 'classifier'}  ·  noise = {level}"
                 f"  ·  {COL_TITLE[q].splitlines()[0]}  ·  every arm, every {a.every} epochs",
                 fontsize=12, color=INK, y=0.995)
    fig.text(0.5, 0.012,
             "One colour scale for the whole grid, identical to the per-epoch pages, so "
             "panels are comparable down columns and across rows. Empty tiles mark epochs "
             "an arm never reached rather than renumbering its ladder.",
             ha="center", fontsize=7, color=MUTED)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.925, bottom=0.075)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = out / f"sweep_{pred}_{level}_{q}.png"
    fig.savefig(p, dpi=145)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
