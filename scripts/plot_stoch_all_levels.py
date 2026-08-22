#!/usr/bin/env python
"""ONE generator for every stochastic all-levels figure: KL / debiased Brier / sAUROC.

Replaces the per-system scratchpad scripts (plot_cp_all.py, plot_gauss.py) that
produced cartpole/ and pendulum/ gaussian_all_levels.png. Those were duplicates and
drifted; this file is the single source of truth for the arm table, the floor
formula, and the panel/figure furniture, so a method looks the same in every figure.

STYLE CONTRACT -- do not vary per system. Colour identifies the (predictor, arm)
pair; line style identifies the predictor family, because the families are NOT
interchangeable and an arm must never be read against the wrong baseline:

    solid    = flow-matching ensemble   -> judged against the 3-seed FM uniform band
    dotted   = deep-ensemble classifier -> judged against clf_dir00, its own uniform run
    dash-dot = Part-X GP                -> a different model class again

FLOOR. 2*sqrt(mean over shared epochs of the 3-seed variance), epoch 0 excluded.
Epoch 0 is the pre-acquisition model, so its spread reflects initialisation rather
than acquisition and including it inflates the floor. Only the FM floor is drawn:
the classifier arms have a single seed each, so they have no floor.

COMPLETENESS GUARD. The script refuses to plot a level whose arms sit at ragged
depths, because a figure of half-length lines reads as "this method stopped
improving" when it actually means "this run has not finished". Pass --allow-partial
to override; each affected panel is then stamped with the per-arm depth so the
raggedness is visible on the figure itself rather than only in the caller's head.
"""
from __future__ import annotations
import argparse, csv, json, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ["dir00_s42", "dir00_s43", "dir00_s44"]

# (arm, label, colour, linestyle, marker) -- THE canonical table.
ARMS = [
    ("epi_var",          "FM  epistemic var",        "#1f77b4", "-",  "v"),
    ("epi_var_anch",     "FM  epi var anch (d2=.5)", "#17becf", "-",  "*"),
    ("epi_bald",         "FM  epistemic BALD",       "#2ca02c", "-",  "^"),
    ("yield_a1",         "FM  yield α=1",            "#d62728", "-",  "o"),
    ("yield_mlp",        "FM  yield (MLP len)",      "#ff7f0e", "-",  "s"),
    ("partx",            "Part-X  (GP)",             "#9467bd", "-.", "D"),
    ("clf_dir00",        "CLF  non-adaptive",        "#8c564b", ":",  "X"),
    ("clf_yield",        "CLF  yield",               "#e377c2", ":",  "P"),
    ("clf_epi_var",      "CLF  epistemic var",       "#7f7f7f", ":",  "v"),
    ("clf_epi_bald",     "CLF  epistemic BALD",      "#bcbd22", ":",  "^"),
    ("clf_epi_var_anch", "CLF  epi var anch",        "#aec7e8", ":",  "*"),
]
BAND_C, BAND_A, MEAN_C, MEAN_LW = "0.55", 0.35, "0.15", 3.0

# --clean: the four-arm read for slides and the paper. One representative per
# predictor family plus its own non-adaptive control, so the figure answers
# "does acquisition help, in each family" without eleven overlapping lines.
# The FM non-adaptive control is the 3-seed band, not an entry here.
CLEAN = ["epi_bald", "partx", "clf_dir00", "clf_epi_bald"]

METRICS = [("KL",             "KL",             "KL divergence  (log, lower better)",  True),
           ("brier_debiased", "debiased Brier", "debiased Brier  (log, lower better)", True),
           ("sAUROC",         "sAUROC",         "sAUROC  (higher better)",             False)]

CAMPAIGNS = {
  "cartpole": dict(
    title="CartPole stochastic gaussian_signal",
    csv="docs/experiments/stochastic/cartpole/gaussian_all_levels.csv",
    out="docs/experiments/stochastic/cartpole/gaussian_all_levels.png",
    panels=[("low","CartPole gaussian_signal — low"),
            ("med","CartPole gaussian_signal — med"),
            ("high","CartPole gaussian_signal — high")]),
  "pendulum": dict(
    title="Pendulum stochastic gaussian_signal",
    csv="docs/experiments/stochastic/pendulum/gaussian_all_levels.csv",
    out="docs/experiments/stochastic/pendulum/gaussian_all_levels.png",
    panels=[("low","Pendulum gaussian_signal — low"),
            ("med","Pendulum gaussian_signal — med"),
            ("high","Pendulum gaussian_signal — high")]),
  "quad2d_nd": dict(
    title="Quadrotor2D stochastic noisy_dynamics",
    csv="docs/experiments/stochastic/quadrotor2d/quad2d_noisy_dynamics_all_levels.csv",
    out="docs/experiments/stochastic/quadrotor2d/quad2d_noisy_dynamics_all_levels.png",
    panels=[("noisy_dynamics_f_0.150","Quadrotor2D noisy_dynamics — f_0.150")]),
  "quad2d_cs": dict(
    title="Quadrotor2D stochastic corridor_sine_ambient",
    csv="docs/experiments/stochastic/quadrotor2d/quad2d_corridor_sine_ambient_all_levels.csv",
    out="docs/experiments/stochastic/quadrotor2d/quad2d_corridor_sine_ambient_all_levels.png",
    panels=[("corridor_sine_ambient_smooth","Quadrotor2D corridor_sine_ambient — smooth")]),
  "quad3d_nd": dict(
    title="Quadrotor3D stochastic noisy_dynamics",
    csv="docs/experiments/stochastic/quadrotor3d/quad3d_noisy_dynamics_all_levels.csv",
    out="docs/experiments/stochastic/quadrotor3d/quad3d_noisy_dynamics_all_levels.png",
    panels=[("noisy_dynamics_f_0.048","Quadrotor3D noisy_dynamics — f_0.048"),
            ("noisy_dynamics_f_0.060","Quadrotor3D noisy_dynamics — f_0.060")]),
}


def load(csv_path, level, keep=None):
    m = defaultdict(dict)
    for r in csv.DictReader(open(csv_path)):
        if r["level"] != level or (keep is not None and r["arm"] not in keep):
            continue
        m[r["arm"]][int(str(r["epoch"]).split("_")[-1])] = r
    return m


def xs(m, arm):
    """x values for an arm: cumulative training trajectories, epoch order preserved.

    The x axis is training-set size, not epoch index. Epoch is only a proxy for
    budget, and it stops being a fair one as soon as an arm fails to spend its
    allowance: Part-X acquires far fewer than samples_per_epoch on quad2D and
    nothing at all on quad3D, so on an epoch axis its line runs the full width
    while carrying half the data. On this axis it stops where its data stops.
    """
    e = sorted(m[arm])
    return [int(m[arm][x]["train_trajectories"]) for x in e], e


def budget_short(m, frac=0.9):
    """Arms that RAN TO FULL DEPTH yet hold materially less data than the leader.

    On this axis a short line has two possible causes and they mean opposite
    things: the arm is still running (says nothing), or the arm finished every
    epoch and still acquired less (a result -- its acquisition rule declined to
    spend the budget). Only the second is stamped, so the caller must not read a
    short in-flight line as a finding. Depth is the discriminator: an arm at the
    campaign's deepest epoch has no epochs left to spend.
    """
    deep = max(max(v) for v in m.values())
    fin = {a_: int(m[a_][max(m[a_])]["train_trajectories"]) for a_ in m}
    top = max(fin.values())
    return ({a_: v for a_, v in sorted(fin.items())
             if max(m[a_]) == deep and v < frac * top}, top)


def floor(m, col):
    have = [s for s in SEEDS if s in m]
    if len(have) < 3:
        return None, None, len(have)
    shared = sorted(set.intersection(*[set(m[s]) for s in have]) - {0})
    if not shared:
        return None, None, 3
    var = [st.variance([float(m[s][e][col]) for s in have]) for e in shared]
    return 2 * (sum(var) / len(var)) ** 0.5, shared[-1], 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaigns", nargs="*", default=list(CAMPAIGNS),
                    help=f"one or more of: {' '.join(CAMPAIGNS)} (default: all)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="plot ragged depths anyway; panels get a per-arm depth stamp")
    ap.add_argument("--clean", action="store_true",
                    help=f"draw only {' '.join(CLEAN)} plus the FM 3-seed control; "
                         "writes alongside the full figure with a _clean suffix")
    a = ap.parse_args()
    keep = set(CLEAN + SEEDS) if a.clean else None

    for name in (a.campaigns or list(CAMPAIGNS)):
        c = CAMPAIGNS[name]
        src = ROOT / c["csv"]
        if not src.exists():
            print(f"  SKIP {name}: {c['csv']} absent"); continue

        loaded, ragged = [], {}
        for lv, lab in c["panels"]:
            m = load(src, lv, keep)
            if not m:
                print(f"  SKIP {name}/{lv}: no rows"); m = None
            else:
                depths = {arm: len(v) for arm, v in m.items()}
                if len(set(depths.values())) > 1:
                    ragged[lab] = depths
            loaded.append((lv, lab, m))
        loaded = [t for t in loaded if t[2]]
        if not loaded:
            continue

        if ragged and not a.allow_partial:
            print(f"  REFUSING to plot {name}: ragged depths (pass --allow-partial to override)")
            for lab, d in ragged.items():
                mx = max(d.values())
                short = {k: v for k, v in sorted(d.items()) if v != mx}
                print(f"    {lab}: full depth {mx}; short -> {short}")
            continue

        nrow, ncol = len(loaded), len(METRICS)
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.7*ncol, 4.7*nrow), squeeze=False)
        for ri, (lv, lab, m) in enumerate(loaded):
            for ci, (col, nm, ylab, logy) in enumerate(METRICS):
                ax = axes[ri][ci]
                f, depth, nseed = floor(m, col)
                have = [s for s in SEEDS if s in m]
                if len(have) == 3:
                    eps = sorted(set.intersection(*[set(m[s]) for s in have]))
                    vals = [[float(m[s][e][col]) for s in have] for e in eps]
                    # the three seeds share one budget schedule, so any of them
                    # gives the band's x; assert rather than assume
                    bx = [int(m[have[0]][e]["train_trajectories"]) for e in eps]
                    for sd in have[1:]:
                        assert [int(m[sd][e]["train_trajectories"]) for e in eps] == bx, \
                            "uniform seeds disagree on training-set size"
                    ax.fill_between(bx, [min(v) for v in vals], [max(v) for v in vals],
                                    color=BAND_C, alpha=BAND_A, zorder=1,
                                    label="FM non-adaptive (3-seed range)")
                    ax.plot(bx, [st.mean(v) for v in vals], color=MEAN_C, lw=MEAN_LW,
                            zorder=2, label="FM non-adaptive (mean)")
                for arm, label, colour, style, mk in ARMS:
                    if arm not in m:
                        continue
                    x, e = xs(m, arm)
                    ax.plot(x, [float(m[arm][k][col]) for k in e], style, color=colour,
                            lw=1.9, marker=mk, ms=3.8, zorder=3, label=label)
                if logy:
                    ax.set_yscale("log")
                else:
                    vv = [float(m[x][e][col]) for x in m for e in m[x] if e > 0]
                    if vv: ax.set_ylim(min(vv)-0.004, max(vv)+0.004)
                ax.set_ylabel(ylab); ax.set_xlabel("training trajectories")
                ax.xaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
                ftxt = (f"FM 2·SD floor = {f:.4f}   ·   depth {depth}" if f is not None
                        else f"NO FM floor ({nseed}/3 seeds)")
                ax.set_title(f"{lab} — {nm}   ·   {ftxt}", fontsize=10.5)
                ax.grid(alpha=0.25, which="both", lw=0.5)
                shortb, topb = budget_short(m)
                if shortb:
                    ax.text(.99, .98, "BUDGET NOT SPENT — full %s traj; %s" % (
                                f"{topb:,}",
                                ", ".join(f"{k}={v:,}" for k, v in shortb.items())),
                            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                            color="#7a4b00",
                            bbox=dict(fc="#fff8e6", ec="#7a4b00", alpha=.95, pad=2))
                if lab in ragged:
                    mx = max(ragged[lab].values())
                    sh = {k: v for k, v in sorted(ragged[lab].items()) if v != mx}
                    ax.text(.99, .02, "PARTIAL — full depth %d; short: %s" % (
                                mx, ", ".join(f"{k}={v}" for k, v in sh.items())),
                            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                            color="#b00", bbox=dict(fc="#fff0f0", ec="#b00", alpha=.9, pad=2))
                if ri == 0 and ci == 0:
                    ax.legend(fontsize=7.6, loc="lower left", framealpha=0.93, ncol=2)

        # Count the acquisition arms drawn as lines separately from the uniform
        # seeds, which are drawn once as the band. Pooling them reads as extra
        # methods -- the 4-arm clean figure would announce itself as 7 arms.
        table = {x[0] for x in ARMS}
        present = {a_ for _, _, m in loaded for a_ in m}
        narms = len(present & table)
        nseed = len(present & set(SEEDS))
        scope = ("one arm per predictor family vs non-adaptive"
                 if a.clean else "all acquisition methods vs non-adaptive")
        fig.suptitle(f"{c['title']} — {scope}   "
                     "(solid = flow matching · dotted = classifier · dash-dot = Part-X GP)\n"
                     f"{narms} acquisition arms vs a {nseed}-seed FM non-adaptive control "
                     "(shaded band)   ·   "
                     "gaps under the printed 2·SD floor are not claimable",
                     fontsize=13, y=0.988)
        fig.tight_layout(rect=[0, 0, 1, 0.955 if nrow > 1 else 0.90])
        out = ROOT / c["out"]
        if a.clean:
            out = out.with_name(out.stem + "_clean" + out.suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=145); plt.close(fig)
        print(f"  wrote {out.relative_to(ROOT)}  ({nrow} level(s), "
              f"{narms} arms + {nseed}-seed control)")


if __name__ == "__main__":
    main()
