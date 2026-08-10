#!/usr/bin/env python
"""Two figures for the diagnostic report. READ-ONLY over the run tree.

FIG 1  uncertainty decomposition (total / aleatoric / epistemic) vs epoch,
       per level x predictor, with the FM Monte-Carlo floor drawn in.
FIG 2  predicted p_success over the FULL evaluation state space (all 39,770
       eval points, not the test subset), per arm, at a common late epoch.

Palette: #0173B2 / #DE8F05 / #029E73 — validated with the dataviz skill's
validate_palette.js (light surface): lightness band PASS, chroma PASS, CVD
separation PASS (worst adjacent dE 9.2 protan), normal-vision PASS. The orange
carries a contrast WARN vs surface, relieved here by a legend plus direct
labels, per the skill's rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path("/common/users/shared/pracsys/adaptive_roa_experiments/ensemble_epistemic")
OUT = Path(__file__).parent
LEVELS = ["det", "low", "med", "high", "xhigh"]
ARMS = ["dir00", "total", "epi_var", "epi_bald", "aleat"]
K_ACQ = 20

C_TOTAL, C_ALEA, C_EPI = "#0173B2", "#DE8F05", "#029E73"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#dcdcdc"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def series(run: str):
    """{epoch: diagnostics} for a run, epoch 0 excluded (identical pre-acquisition)."""
    out = {}
    for f in sorted((EXP / run).glob("epoch_*/artifacts_v2.json")):
        ep = int(f.parent.name.split("_")[1])
        if ep == 0:
            continue
        try:
            dg = (json.load(open(f)).get("acquisition") or {}).get("diagnostics") or {}
        except Exception:
            continue
        if "epistemic_mean" in dg:
            out[ep] = dg
    return out


# ------------------------------------------------------------------ FIG 1
def fig1():
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.4), sharex=True)
    for r, pred in enumerate(["clf", "fm"]):
        for c, lvl in enumerate(LEVELS):
            ax = axes[r, c]
            d = series(f"{pred}_{lvl}_total")          # reference model: the entropy arm
            if not d:
                ax.text(.5, .5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color=MUTED)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0: ax.set_title(lvl, color=INK)
                continue
            eps = sorted(d)
            alea = np.array([d[e]["aleatoric_mean"] for e in eps])
            epi = np.array([d[e]["epistemic_mean"] for e in eps])
            tot = alea + epi                            # H(p̄) = E[H] + I, exact
            ax.plot(eps, tot, color=C_TOTAL, lw=2, label="total H(p̄)")
            ax.plot(eps, alea, color=C_ALEA, lw=2, label="aleatoric E$_m$[H]")
            ax.plot(eps, epi, color=C_EPI, lw=2, label="epistemic I(y;m)")
            if pred == "fm":
                # MC floor: mean_m[p(1-p)/(K-1)], read empirically off the
                # epi_var arm's score_min (= -floor when some raw variance is 0)
                dv = series(f"{pred}_{lvl}_epi_var")
                if dv:
                    fl = np.nanmean([-dv[e].get("score_min", np.nan) for e in sorted(dv)])
                    if fl == fl and fl > 0:
                        ax.axhline(fl, color=C_EPI, lw=1.2, ls=(0, (4, 3)), alpha=.9)
                        ax.text(eps[-1], fl, "  MC floor", va="center", ha="left",
                                fontsize=7.5, color=C_EPI)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", color=GRID, lw=.6)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(lvl, color=INK)
            if c == 0:
                ax.set_ylabel(f"{'classifier' if pred=='clf' else 'flow matching'}\nnats",
                              color=INK)
            if r == 1:
                ax.set_xlabel("adaptive epoch")
            # direct labels on the last panel only (relieves the orange contrast WARN).
            # total and aleatoric coincide wherever epistemic ~ 0, so de-collide in
            # display space before drawing rather than stacking text on itself.
            if r == 0 and c == len(LEVELS) - 1:
                items = sorted(((tot[-1], "total", C_TOTAL),
                                (alea[-1], "aleatoric", C_ALEA),
                                (epi[-1], "epistemic", C_EPI)), key=lambda t: t[0])
                lo, hi = ax.get_ylim()
                gap = 0.075 * (hi - lo)          # min vertical separation, data units
                ys = []
                for y, _, _ in items:
                    ys.append(y if not ys else max(y, ys[-1] + gap))
                for (yv, lab, col), y in zip(items, ys):
                    ax.annotate(lab, (eps[-1], y), xytext=(5, 0),
                                textcoords="offset points", color=col,
                                fontsize=8, va="center", annotation_clip=False)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Uncertainty decomposition over the candidate pool, by adaptive epoch  "
                 "(model = the `total` arm; pool = 50,000 candidates/epoch at the noisy "
                 "levels, ~18-20k at det where the pool is the whole remaining dataset)",
                 y=.99, fontsize=11, color=INK)
    fig.text(.5, .005,
             "Epoch 0 excluded (all arms hold identical pre-acquisition data). "
             "Each panel carries its own y-scale — magnitudes differ ~10x across noise levels; "
             "the comparison of interest is within a panel. "
             "Dashed line = Monte-Carlo sampling floor mean$_m$[p(1-p)/(K-1)] at K=20; "
             "epistemic estimates below it are not resolvable.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, .03, 1, .96))
    fig.savefig(OUT / "fig1_uncertainty_vs_epoch.png", dpi=150)
    print("wrote fig1_uncertainty_vs_epoch.png")


# ------------------------------------------------------------------ ground truth
POOL = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum/lqr")
_gt_cache: dict = {}


def gt_prob(level: str, query: np.ndarray):
    """Oracle p(success) at `query`, by nearest-neighbour on the 158x315 rollout grid.

    `eval_success_prob.npz` holds successes/trials per grid cell (90 rollouts each);
    p_success is that ratio. `det` has no such file — the system is deterministic, so
    the oracle probability is degenerate in {0,1} and equals the binary label.
    """
    f = POOL / level / "eval_success_prob.npz"
    if not f.exists():
        return None
    if level not in _gt_cache:
        from scipy.spatial import cKDTree
        with np.load(f) as z:
            _gt_cache[level] = (cKDTree(z["starts"].astype(np.float64)),
                                z["p_success"].astype(np.float64))
    tree, p = _gt_cache[level]
    return p[tree.query(query, k=1)[1]]


def fig0():
    """Standalone reference: how the true ROA blurs as process noise rises."""
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(3.0 * len(LEVELS), 3.1), squeeze=False)
    for c, lvl in enumerate(LEVELS):
        ax = axes[0][c]
        f = EXP / f"fm_{lvl}_dir00"
        cand = sorted(f.glob("epoch_*/full_roa_per_point.npz"))
        if not cand:
            ax.text(.5, .5, "no data", ha="center", va="center", transform=ax.transAxes,
                    color=MUTED); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(lvl); continue
        z = np.load(cand[-1])
        S = z["start_states"]
        p = gt_prob(lvl, S)
        sub = "oracle, 90 rollouts/grid cell"
        if p is None:
            p = z["true_labels"].astype(float)
            sub = "deterministic: p ∈ {0,1} = label"
        sc = ax.scatter(S[:, 0], S[:, 1], c=p, cmap="coolwarm", s=.6, vmin=0, vmax=1,
                        linewidths=0, rasterized=True)
        ax.set_title(f"{lvl}\n{sub}", color=INK, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(r"$\theta$", color=MUTED)
        if c == 0:
            ax.set_ylabel(r"$\dot\theta$", color=INK)
    cb = fig.colorbar(sc, ax=axes, fraction=.015, pad=.02)
    cb.set_label("true p(success)", color=INK, fontsize=9)
    cb.outline.set_visible(False)
    fig.suptitle("Ground-truth p(success) over the full evaluation state space, by noise level",
                 fontsize=11, color=INK, y=1.14)
    fig.text(.5, -.04, "White speckle = grid cells with no evaluation point; the eval set is "
             "39,770 of the 49,770 grid cells. Nearest-neighbour lookup onto the 158x315 "
             "rollout grid.", ha="center", fontsize=8, color=MUTED)
    fig.savefig(OUT / "fig0_ground_truth_prob.png", dpi=150, bbox_inches="tight")
    print("wrote fig0_ground_truth_prob.png")


# ------------------------------------------------------------------ FIG 2
def arm_last_epoch(pred, lvl, arm):
    """Deepest epoch this arm actually completed (epoch 0 excluded)."""
    eps = [int(f.parent.name.split("_")[1])
           for f in (EXP / f"{pred}_{lvl}_{arm}").glob("epoch_*/full_roa_per_point.npz")]
    eps = [e for e in eps if e > 0]
    return max(eps) if eps else None


def last_common_epoch(pred, lvl):
    """Any arm with data -> the level is plottable. Each arm is drawn at its OWN
    final epoch (annotated per panel); arms reached different depths because of
    preemption, so a shared epoch would pin several levels to epoch 1."""
    eps = [arm_last_epoch(pred, lvl, a) for a in ARMS]
    return max([e for e in eps if e is not None], default=None)


def fig2(pred="fm"):
    rows = [l for l in LEVELS if last_common_epoch(pred, l) is not None]
    if not rows:
        print(f"fig2({pred}): no level has a common epoch across all arms"); return
    ncol = len(ARMS) + 2
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.55 * ncol, 2.5 * len(rows)),
                             squeeze=False)
    for r, lvl in enumerate(rows):
        ep0 = arm_last_epoch(pred, lvl, "dir00")
        z0 = np.load(EXP / f"{pred}_{lvl}_dir00" / f"epoch_{ep0:03d}" / "full_roa_per_point.npz")
        S, truth = z0["start_states"], z0["true_labels"].astype(float)
        # col 0 — oracle probability; col 1 — the binary label the model is scored against
        gp = gt_prob(lvl, S)
        gp_lab = "ground truth\np(success)"
        if gp is None:
            gp, gp_lab = truth, "ground truth\np(success)\n(det: = label)"
        axes[r][0].scatter(S[:, 0], S[:, 1], c=gp, cmap="coolwarm", s=.6,
                           vmin=0, vmax=1, linewidths=0, rasterized=True)
        axes[r][0].set_ylabel(f"{lvl}\n$\\dot\\theta$", color=INK)
        if r == 0:
            axes[r][0].set_title(gp_lab, color=INK, fontsize=9)
        axes[r][1].scatter(S[:, 0], S[:, 1], c=truth, cmap="coolwarm", s=.6,
                           vmin=0, vmax=1, linewidths=0, rasterized=True)
        if r == 0:
            axes[r][1].set_title("ground truth\nbinary label", color=INK, fontsize=9)
        for c, arm in enumerate(ARMS, start=2):
            axx = axes[r][c]
            ea = arm_last_epoch(pred, lvl, arm)
            f = (EXP / f"{pred}_{lvl}_{arm}" / f"epoch_{ea:03d}" / "full_roa_per_point.npz"
                 if ea is not None else None)
            if f is None or not f.exists():
                axx.text(.5, .5, "—", ha="center", va="center", transform=axx.transAxes,
                         color=MUTED)
            else:
                z = np.load(f)
                sc = axx.scatter(z["start_states"][:, 0], z["start_states"][:, 1],
                                 c=z["p_success"], cmap="coolwarm", s=.6,
                                 vmin=0, vmax=1, linewidths=0, rasterized=True)
                axx.text(.03, .04, f"ep {ea}", transform=axx.transAxes, fontsize=7.5,
                         color=INK, ha="left", va="bottom",
                         bbox=dict(fc="white", ec="none", alpha=.82, pad=1.4))
            if r == 0:
                axx.set_title(arm, color=INK)
        for c in range(ncol):
            a = axes[r][c]
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if r == len(rows) - 1:
                a.set_xlabel(r"$\theta$", color=MUTED)

    cb = fig.colorbar(sc, ax=axes, fraction=.012, pad=.03)
    cb.set_label("predicted p(success)   —   0.5 = maximally ambiguous", color=INK, fontsize=9)
    cb.outline.set_visible(False)
    name = "flow matching" if pred == "fm" else "classifier"
    fig.suptitle(f"Predicted p(success) over the FULL evaluation state space — {name} "
                 f"({len(S):,} eval points, not the test subset)", fontsize=11, color=INK)
    fig.text(.5, .085, "Each arm is shown at its OWN deepest completed epoch (labelled in "
             "each panel); arms reached different depths because of preemption, so epochs "
             "are not matched across a row. Ground-truth p(success) is the 90-rollout oracle; "
             "for det it is degenerate and equals the label.",
             ha="center", fontsize=8, color=MUTED)
    fig.savefig(OUT / f"fig2_state_space_{pred}.png", dpi=150, bbox_inches="tight")
    print(f"wrote fig2_state_space_{pred}.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["fig0", "fig1", "fig2fm", "fig2clf"]
    if "fig0" in which: fig0()
    if "fig1" in which: fig1()
    if "fig2fm" in which: fig2("fm")
    if "fig2clf" in which: fig2("clf")
