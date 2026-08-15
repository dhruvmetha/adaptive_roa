#!/usr/bin/env python
"""State-space maps of the ensemble's uncertainty and its decomposition.

One multi-page PDF per (predictor, noise level). One PAGE per adaptive epoch.
One ROW per acquisition arm. Columns:

    ground truth | acquired this epoch | predicted p | [p_invalid] |
    H total | H aleatoric | H epistemic | Var total | Var aleatoric | Var epistemic

so a single page answers "at this noise level, at this epoch, what does each arm
believe, where did it just collect, and how does its uncertainty split".

EVERY PANEL IS ON A FIXED SCALE, shared across all arms, all epochs and both
predictors. That is the whole point of the layout: the previous version of this
figure gave each panel its own y-scale and stopped each curve at whatever epoch
preemption happened to reach, which made cross-panel reading meaningless. Here
probabilities are always [0,1], entropies always [0, ln 2], variances always
[0, 0.25]. Only the two epistemic columns get a fitted scale -- they are ~30x
smaller than the total and would otherwise render as flat black -- and their
actual ceiling is printed on the colorbar so the compression cannot be misread.

Colour follows the job, not decoration: probability is DIVERGING about 0.5
(coolwarm, neutral midpoint); every uncertainty magnitude is a SINGLE-hue
sequential ramp, keyed to the same three colours the campaign's line figure used
-- blue = total, orange = aleatoric, green = epistemic -- so the decomposition
reads the same way in both. The debiased epistemic variance can legitimately go
negative and therefore gets a diverging ramp about zero.

    python make_maps.py --pred fm --level high --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.uncertainty_maps.common import (  # noqa: E402
    ARMS, ARM_LABEL, EXP, LEVELS, binary_entropy, decompose, epochs_with_eval,
    load_marginal, oracle_full_image, rasteriser, to_grid,
)

LN2 = float(np.log(2.0))
C_SUCCESS, C_FAILURE = "#0173B2", "#DE8F05"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#dcdcdc"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 8, "axes.titlesize": 8, "legend.frameon": False,
})


def members_path(root: Path, run: str, epoch: int) -> Path:
    return root / run / f"members_epoch_{epoch:03d}.npz"


def load_members(root: Path, run: str, epoch: int):
    f = members_path(root, run, epoch)
    if not f.exists():
        return None, None
    with np.load(f) as z:
        return z["p_members"].astype(np.float64), float(z["k"])


def load_acquired(root: Path, run: str):
    f = root / f"{run}_acquired.npz"
    if not f.exists():
        return {}
    z = np.load(f)
    return {int(e): (z[f"states_{e:03d}"], z[f"labels_{e:03d}"]) for e in z["epochs"]}


def epistemic_ceilings(pred, level, member_root, epochs, arms):
    """One shared ceiling per epistemic column, fitted over the WHOLE cell.

    Fitted rather than theoretical because epistemic mass is ~30x smaller than
    total; fitted ONCE over every arm and epoch rather than per panel, so panels
    stay comparable. The 99.5th percentile clips the handful of extreme cells
    that would otherwise flatten everything else.
    """
    h, v = [], []
    for arm in arms:
        for ep in epochs:
            pm, k = load_members(member_root, f"{pred}_{level}_{arm}", ep)
            if pm is None:
                continue
            d = decompose(pm, k)
            h.append(np.percentile(d["h_epistemic"], 99.5))
            v.append(np.percentile(d["var_epistemic"], 99.5))
    if not h:
        return None, None
    return float(max(h)), float(max(v))


def panel(ax, img, extent, cmap, vmin, vmax, title, norm=None):
    ax.imshow(img, origin="lower", extent=extent, aspect="auto", cmap=cmap,
              vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm,
              interpolation="nearest")
    ax.set_title(title, color=INK, fontsize=7.5, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


COL_TITLE = {
    "gt": "ground truth p(success)",
    "acq": "acquired this epoch",
    "p": "predicted p(success)",
    "p_invalid": "p(invalid)",
    "h_total": "H total  $H(\\bar p)$",
    "h_aleatoric": "H aleatoric  $E_m[H(p_m)]$",
    "h_epistemic": "H epistemic (BALD)",
    "var_total": "Var total  $\\bar p(1-\\bar p)$",
    "var_aleatoric": "Var aleatoric  $E_m[p_m(1{-}p_m)]$",
    "var_epistemic": "Var epistemic  $\\mathrm{Var}_m(p_m)$",
    "var_epistemic_debiased": "Var epistemic, MC-debiased\n(the score epi_var ranked on)",
}
COL_CMAP = {
    "gt": "coolwarm", "p": "coolwarm", "p_invalid": "Purples",
    "h_total": "Blues", "h_aleatoric": "Oranges", "h_epistemic": "Greens",
    "var_total": "Blues", "var_aleatoric": "Oranges", "var_epistemic": "Greens",
}


def columns_for(pred: str, has_members: bool, has_acq: bool) -> list[str]:
    """Column set for a whole predictor x level cell.

    A column is dropped only when it is unavailable for EVERY arm and epoch in
    the cell -- never per page, so pages within one PDF stay aligned and
    comparable. What was dropped is stated in the caption instead of leaving a
    grid of placeholder tiles that crowds out the panels that do have data.
    """
    cols = ["gt"]
    if has_acq:
        cols.append("acq")
    cols.append("p")
    if pred == "fm":
        cols.append("p_invalid")
    if has_members:
        cols += ["h_total", "h_aleatoric", "h_epistemic",
                 "var_total", "var_aleatoric", "var_epistemic"]
        if pred == "fm":
            # Flow-matching p_m is a K-sample estimate, so raw between-member
            # variance is inflated by mean_m[p_m(1-p_m)]/(K-1). The debiased
            # version is what the epi_var arm actually ranked on; it is signed,
            # so it gets its own diverging scale.
            cols.append("var_epistemic_debiased")
    else:
        # Totals are exact functions of the stored marginal, so they survive
        # even where the members do not.
        cols += ["h_total", "var_total"]
    return cols


def _caption(gt_note: str, missing_note: str) -> str:
    return (
        "Axes: theta in [-pi, pi] horizontal, theta-dot in [-2pi, 2pi] vertical. "
        "Every colour scale is FIXED across all arms, epochs and levels — p in [0,1], "
        "H in [0, ln 2] nats, Var in [0, 0.25] — so any two panels anywhere in this "
        "campaign are directly comparable. The epistemic columns are the only fitted "
        "scales (they are ~30x smaller than the total); their ceiling is on the colorbar. "
        f"Ground truth: {gt_note}. "
        "White inside a model panel = grid cell with no evaluation point."
        + (f"  {missing_note}" if missing_note else "")
    )


def build_page(pdf, pred, level, epoch, arms, member_root, acq_root, ceilings,
               states, shape, flat, extent, gt, gt_note, cols, missing_note):
    h_epi_max, var_epi_max = ceilings
    vmax_of = {"gt": 1.0, "p": 1.0, "p_invalid": 1.0,
               "h_total": LN2, "h_aleatoric": LN2, "h_epistemic": h_epi_max,
               "var_total": 0.25, "var_aleatoric": 0.25, "var_epistemic": var_epi_max,
               "var_epistemic_debiased": var_epi_max}

    nr, nc = len(arms), len(cols)
    # The caption wraps to however many lines the page width allows, so a narrow
    # page (a cell with few available columns) needs a taller reserved band or
    # the text lands on top of the colorbars.
    fig_w = 1.72 * nc
    caption = _caption(gt_note, missing_note)
    cap_lines = max(2, int(np.ceil(len(caption) / (fig_w * 19.0))))
    band_in = 0.16 * cap_lines + 0.42
    fig_h = 1.95 * nr + band_in + 0.62
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nr + 1, nc, height_ratios=[1] * nr + [0.10],
                          hspace=0.13, wspace=0.06)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(nc)] for r in range(nr)]
    handles = None
    for r, arm in enumerate(arms):
        run = f"{pred}_{level}_{arm}"
        try:
            marg = load_marginal(pred, level, arm, epoch)
        except FileNotFoundError:
            for c in range(len(cols)):
                axes[r][c].axis("off")
            axes[r][0].text(.5, .5, f"{arm}\nno epoch {epoch}", ha="center",
                            va="center", transform=axes[r][0].transAxes, color=MUTED)
            continue
        pm, k = load_members(member_root, run, epoch)
        dec = decompose(pm, k) if pm is not None else None
        acq = load_acquired(acq_root, run).get(epoch)

        for c, key in enumerate(cols):
            ax = axes[r][c]
            if key == "gt":
                panel(ax, gt, extent, "coolwarm", 0, 1,
                      COL_TITLE["gt"] if r == 0 else "")
            elif key == "acq":
                ax.set_facecolor("#f7f7f5")
                ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
                if acq is not None:
                    s, lab = acq
                    for m, col, name in ((lab == 1, C_SUCCESS, "success"),
                                         (lab != 1, C_FAILURE, "failure")):
                        ax.scatter(s[m, 0], s[m, 1], s=1.6, c=col, linewidths=0,
                                   rasterized=True, label=name)
                    handles = [Line2D([], [], marker="o", ls="", ms=4, color=C_SUCCESS,
                                      label="acquired: success"),
                               Line2D([], [], marker="o", ls="", ms=4, color=C_FAILURE,
                                      label="acquired: failure")]
                    ax.set_title(f"{COL_TITLE['acq']} (n={len(s)})" if r == 0 else "",
                                 color=INK, fontsize=7.5, pad=2)
                else:
                    ax.text(.5, .5, "no acquisition\nrecord", ha="center", va="center",
                            transform=ax.transAxes, color=MUTED, fontsize=6.5)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            elif key in ("p", "p_invalid"):
                # Where members exist, show THEIR marginal, not the stored one:
                # every uncertainty column on this page is derived from p_m, so
                # showing a different p would leave H(p) on screen inconsistent
                # with the p on screen. For the classifier the two are
                # bit-identical (drift 0.0000); for flow matching the recompute
                # is the same quantity from 5x more samples, and the two agree
                # to sampling noise with no systematic offset.
                if key == "p":
                    src = dec["p_bar"] if dec is not None else marg["p_success"]
                else:
                    src = marg["p_invalid"]
                panel(ax, to_grid(src, shape, flat), extent, COL_CMAP[key], 0, 1,
                      COL_TITLE[key] if r == 0 else "")
            else:
                if dec is None and key in ("h_total", "var_total"):
                    # Both totals are exact functions of the stored marginal, so
                    # they do not need the members at all.
                    pb = np.clip(marg["p_success"], 0.0, 1.0)
                    vals = binary_entropy(pb) if key == "h_total" else pb * (1.0 - pb)
                    panel(ax, to_grid(vals, shape, flat), extent, COL_CMAP[key],
                          0, vmax_of[key], COL_TITLE[key] if r == 0 else "")
                    continue
                if dec is None or key not in dec:
                    ax.axis("off")
                    if r == 0:
                        ax.set_title(COL_TITLE[key], color=MUTED, fontsize=7.5, pad=2)
                    ax.text(.5, .5, "not recoverable\n(no checkpoints)", ha="center",
                            va="center", transform=ax.transAxes, color=MUTED, fontsize=6)
                    continue
                if key == "var_epistemic_debiased":
                    panel(ax, to_grid(dec[key], shape, flat), extent, "PRGn", None, None,
                          COL_TITLE[key] if r == 0 else "",
                          norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax_of[key],
                                            vmax=vmax_of[key]))
                    continue
                panel(ax, to_grid(dec[key], shape, flat), extent, COL_CMAP[key],
                      0, vmax_of[key], COL_TITLE[key] if r == 0 else "")

        axes[r][0].set_ylabel(ARM_LABEL[arm], color=INK, fontsize=8)

    # One colorbar per column: the scale is a property of the column, not of any
    # single panel, which is the whole reason the panels are comparable.
    for c, key in enumerate(cols):
        cax = fig.add_subplot(gs[nr, c])
        if key == "acq":
            cax.axis("off")
            if handles:
                cax.legend(handles=handles, loc="center", ncol=1, fontsize=6.5,
                           handletextpad=0.3, labelspacing=0.25)
            continue
        signed = key == "var_epistemic_debiased"
        lo = -vmax_of[key] if signed else 0.0
        sm = plt.cm.ScalarMappable(cmap="PRGn" if signed else COL_CMAP[key],
                                   norm=plt.Normalize(lo, vmax_of[key]))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([lo, vmax_of[key]])
        cb.ax.set_xticklabels([f"{lo:.3g}", f"{vmax_of[key]:.3g}"],
                              fontsize=6.5, color=MUTED)
        # Pull the end labels inside the bar: adjacent columns are only 6% of a
        # panel apart, so centred end ticks from neighbouring colorbars collide
        # and read as one garbled number.
        cb.ax.get_xticklabels()[0].set_horizontalalignment("left")
        cb.ax.get_xticklabels()[-1].set_horizontalalignment("right")
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=2, pad=1)

    n_train = None
    try:
        import json
        n_train = json.loads((EXP / f"{pred}_{level}_{arms[0]}" /
                              f"epoch_{epoch:03d}" / "artifacts_v2.json").read_text()
                             ).get("train_trajectories")
    except Exception:
        pass

    top_in = 0.62
    fig.suptitle(
        f"{'flow matching' if pred == 'fm' else 'classifier'}  ·  noise = {level}  ·  "
        f"adaptive epoch {epoch}"
        + (f"  ·  {n_train} training trajectories" if n_train else ""),
        fontsize=12, color=INK, y=1.0 - 0.35 * top_in / fig_h)
    fig.text(0.5, 0.30 * band_in / fig_h, caption,
             ha="center", va="center", fontsize=7, color=MUTED, wrap=True)
    fig.subplots_adjust(left=0.035 * 9.0 / max(fig_w, 6.0) + 0.012, right=0.995,
                        top=1.0 - top_in / fig_h, bottom=band_in / fig_h)
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, choices=("clf", "fm"))
    ap.add_argument("--level", required=True, choices=LEVELS)
    ap.add_argument("--members", default="/common/users/shared/pracsys/"
                                         "adaptive_roa_experiments/_uncertainty_maps/members")
    ap.add_argument("--acquired", default="/common/users/shared/pracsys/"
                                          "adaptive_roa_experiments/_uncertainty_maps/acquired")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", default="all")
    ap.add_argument("--png-epochs", default="", help="also write these epochs as PNG")
    a = ap.parse_args()

    pred, level = a.pred, a.level
    arms = [arm for arm in ARMS if (EXP / f"{pred}_{level}_{arm}").exists()]
    if not arms:
        raise SystemExit(f"no runs for {pred}_{level}")

    # Union of epochs across arms: an arm that stopped early leaves an explicit
    # "no epoch N" tile rather than silently shortening the page sequence.
    all_eps = sorted({e for arm in arms for e in epochs_with_eval(pred, level, arm)})
    eps = all_eps if a.epochs == "all" else [int(x) for x in a.epochs.split(",")]

    ref = load_marginal(pred, level, arms[0], eps[0])
    states = ref["states"]
    shape, flat, extent = rasteriser(level, states)
    gt, gt_extent = oracle_full_image(level)
    gt_note = "oracle, 90 rollouts per grid cell"
    if gt is None:
        # Deterministic system: no rollout grid file, so the oracle probability
        # is degenerate in {0,1} and equals the recorded binary label.
        gt = to_grid((ref["true_labels"] == 1).astype(float), shape, flat)
        gt_note = "binary label (the system is deterministic, so p in {0,1})"
    elif gt_extent != extent:
        raise RuntimeError(f"oracle grid {gt_extent} != eval grid {extent}")

    member_root, acq_root = Path(a.members), Path(a.acquired)
    ceilings = epistemic_ceilings(pred, level, member_root, eps, arms)
    has_members = ceilings[0] is not None
    if not has_members:
        ceilings = (LN2, 0.25)
    has_acq = any((acq_root / f"{pred}_{level}_{arm}_acquired.npz").exists() for arm in arms)

    missing = []
    if not has_members:
        missing.append("the aleatoric/epistemic split (per-epoch member checkpoints "
                       "were deleted for this cell, so it is unrecoverable)")
    if not has_acq:
        missing.append("the acquired-states column (this run's config did not survive, "
                       "so the pool index space cannot be reconstructed)")
    missing_note = ("OMITTED HERE: " + "; ".join(missing) + ".") if missing else ""

    cols = columns_for(pred, has_members, has_acq)
    print(f"{pred}_{level}: members={has_members} acquired={has_acq} "
          + (f"ceilings H={ceilings[0]:.4f} Var={ceilings[1]:.5f}" if has_members else ""))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pdf_path = out / f"uncertainty_maps_{pred}_{level}.pdf"
    png_eps = {int(x) for x in a.png_epochs.split(",") if x.strip()}
    with PdfPages(pdf_path) as pdf:
        for ep in eps:
            build_page(pdf, pred, level, ep, arms, member_root, acq_root, ceilings,
                       states, shape, flat, extent, gt, gt_note, cols, missing_note)
            print(f"  page epoch {ep}", flush=True)
    print(f"wrote {pdf_path} ({len(eps)} pages)")

    for ep in sorted(png_eps):
        sink = _PngSink(out / f"uncertainty_maps_{pred}_{level}_epoch{ep:03d}.png")
        build_page(sink, pred, level, ep, arms, member_root, acq_root, ceilings,
                   states, shape, flat, extent, gt, gt_note, cols, missing_note)
        print(f"wrote {sink.path}")


class _PngSink:
    """Same `savefig` surface as PdfPages, so `build_page` needs no branch."""

    def __init__(self, path):
        self.path = path

    def savefig(self, fig, dpi=140):
        fig.savefig(self.path, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    main()
