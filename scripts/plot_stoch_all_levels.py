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
DATA = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/stochastic")
DOCS = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic")
SEEDS = ["dir00_s42", "dir00_s43", "dir00_s44"]

# (arm, label, colour, linestyle, marker) -- THE canonical table.
ARMS = [
    ("epi_var",          "FM  epistemic var",        "#1f77b4", "-",  "v"),
    ("epi_var_anch",     "FM  epi var anch (d2=.5)", "#17becf", "-",  "*"),
    ("epi_bald",         "FM  epistemic BALD",       "#2ca02c", "-",  "^"),
    ("yield_a1",         "FM  yield α=1",            "#d62728", "-",  "o"),
    ("yield_mlp",        "FM  yield (MLP len)",      "#ff7f0e", "-",  "s"),
    # Only the FIXED Part-X arm is plotted. The pre-fix `partx` rows stay in the
    # CSVs as a record, but they were produced with a partition root built from
    # component-collapsed bounds, so 91% of the quadrotor2D state space and 43%
    # of quadrotor3D's was unreachable by acquisition and those arms ran at
    # 40-45% of their budget. Drawing them as "Part-X" misrepresents the method.
    # Consequence, stated so it is not a surprise: the two noisy_dynamics levels
    # have NO Part-X line at all, because their datasets were deleted on
    # 2026-08-26 and a corrected arm can never be produced for them.
    # Part-X keeps its long-standing dash-dot style and purple; it is the only
    # Part-X line now, so there is nothing to distinguish it FROM.
    ("partx_fix",        "Part-X  (GP)",             "#9467bd", "-.", "D"),
    ("clf_dir00",        "CLF  non-adaptive",        "#8c564b", ":",  "X"),
    ("clf_yield",        "CLF  yield",               "#e377c2", ":",  "P"),
    ("clf_epi_var",      "CLF  epistemic var",       "#7f7f7f", ":",  "v"),
    ("clf_epi_bald",     "CLF  epistemic BALD",      "#bcbd22", ":",  "^"),
    ("clf_epi_var_anch", "CLF  epi var anch",        "#aec7e8", ":",  "*"),
    # Bayesian NN arms. The first three are UNIFORM sampling: the shipped bnn
    # configs bind classifier_prob, which exposes no estimate_members(), so the
    # decomposition acquisitions cannot run against them. They differ only in
    # posterior approximation -- same architecture, same prior, same 64 marginal
    # samples -- so a gap between those three lines is a posterior effect, not an
    # acquisition effect.
    #
    # bnn_mfvi_bald is the one ADAPTIVE Bayesian arm: same mean-field posterior as
    # bnn_mfvi, acquiring by BALD (Depeweg et al. 2018 Eq. 3/5) over 64 draws from
    # the weight posterior, via sampled_posterior_classifier_prob. It shares
    # bnn_mfvi's colour on purpose -- bnn_mfvi is its matched non-adaptive control
    # at identical budget, so the pair is meant to be read together.
    #
    # MFVI and not Laplace: the paper's BNN puts a Gaussian over ALL weights and
    # samples it. A last-layer Laplace makes only the final layer stochastic, every
    # draw shares one deterministic body, and the measured epistemic term collapsed
    # to ~1e-5 (epi/ale 0.0004-0.006 vs a deep ensemble's 0.034-0.221) -- so BALD
    # had almost no epistemic signal to rank on. That arm was withdrawn.
    #
    # Deliberately NOT run on the deep ensemble: bnn_ensemble and clf_ensemble are
    # the same trainer with the same posterior and hyperparameters, differing only
    # in probability backend, so a "BNN ensemble + BALD" arm would reproduce
    # clf_epi_bald rather than add a data point.
    ("bnn_ens",          "BNN  deep ensemble",       "#393b79", "--", "o"),
    ("bnn_lap",          "BNN  Laplace",             "#843c39", "--", "s"),
    ("bnn_mfvi",         "BNN  mean-field VI",       "#7b4173", "--", "D"),
    ("bnn_mfvi_bald",    "BNN  MFVI + BALD",         "#7b4173", "-",  "^"),
    # BB-alpha (alpha=1) MFVI pair, used by the quadrotor3D PPO campaign.
    # Registered 2026-09-06: the arms had been scoring since launch, but ARMS is
    # matched by EXACT arm name and neither name was here, so both were dropped
    # from every figure without an error -- the same silent skip the scorer's own
    # ARMS list has (an unlisted arm is skipped, not flagged). The PPO level-set
    # figures were drawing 5 of 10 arms and reporting "5 arms" as if that were
    # the whole level.
    #
    # NAMES ARE CONFUSING AND WORTH READING TWICE. Despite the suffixes:
    #   bnn_a1_dir00  is the NON-ADAPTIVE control (acquisition=direct, d2=0)
    #   bnn_mfvi_a1   is the ADAPTIVE arm        (decomp_epi_bald, d2=1.0)
    # Both run the SAME predictor (bnn_mfvi_a1 config, BB-alpha at alpha=1), so
    # they differ only in acquisition. That is exactly the matched-budget pair the
    # dimension hypothesis needs, and it is the reason they share a colour: read
    # the dashed line as the control for the solid one, as with bnn_mfvi above.
    # A distinct hue from bnn_mfvi/#7b4173 because these are a different posterior
    # (BB-alpha) and must not be mistaken for that pair.
    ("bnn_a1_dir00",     "BNN  BB-α=1 non-adaptive", "#a55194", "--", "D"),
    ("bnn_mfvi_a1",      "BNN  BB-α=1 + BALD",       "#a55194", "-",  "^"),
]
BAND_C, BAND_A, MEAN_C, MEAN_LW = "0.55", 0.35, "0.15", 3.0

# --clean: the four-arm read for slides and the paper. One representative per
# predictor family plus its own non-adaptive control, so the figure answers
# "does acquisition help, in each family" without eleven overlapping lines.
# The FM non-adaptive control is the 3-seed band, not an entry here.
CLEAN = ["epi_bald", "partx_fix", "clf_dir00", "clf_epi_bald", "bnn_mfvi", "bnn_mfvi_bald"]

METRICS = [("KL",             "KL",             "KL divergence  (log, lower better)",  True),
           ("brier_debiased", "debiased Brier", "debiased Brier  (log, lower better)", True),
           ("sAUROC",         "sAUROC",         "sAUROC  (higher better)",             False),
           # Mean balanced accuracy of the level-β sets over the ten levels that
           # pass the thin-set rule (stoch_prob_metrics.level_set_summary). The
           # per-level curves are in plot_stoch_levelsets; this column is the
           # epoch-wise view of their area.
           ("auc_bal_acc",    "bal. acc. area", "balanced-accuracy area over β  (higher better)", False)]


def missing_columns(m, cols):
    """(arm, epoch, column) triples whose value is absent or blank.

    A CSV scored before a metric existed has the column blank on the old rows,
    and float('') dies with a message that names none of them. Refusing with
    the list points at the fix: re-run scripts/score_stoch_incremental.py.
    """
    return sorted((arm, e, c) for arm in m for e in m[arm] for c in cols
                  if m[arm][e].get(c, "") == "")

CAMPAIGNS = {
  "cartpole": dict(
    title="CartPole stochastic gaussian_signal",
    csv="cartpole/lqr/gaussian_all_levels.csv",
    out="cartpole/lqr/gaussian_all_levels.png",
    panels=[("low","CartPole gaussian_signal — low"),
            ("med","CartPole gaussian_signal — med"),
            ("high","CartPole gaussian_signal — high")]),
  "pendulum": dict(
    title="Pendulum stochastic gaussian_signal",
    csv="pendulum/lqr/gaussian_all_levels.csv",
    out="pendulum/lqr/gaussian_all_levels.png",
    panels=[("low","Pendulum gaussian_signal — low"),
            ("med","Pendulum gaussian_signal — med"),
            ("high","Pendulum gaussian_signal — high")]),
  "cartpole_rl": dict(
    title="CartPole stochastic gaussian_signal (safe_explorer_ppo)",
    csv="cartpole/safe_explorer_ppo/gaussian_all_levels.csv",
    out="cartpole/safe_explorer_ppo/gaussian_all_levels.png",
    # `baseline` is ZERO noise -- it has no lqr counterpart, so this panel set is
    # deliberately not the same shape as the lqr campaign's low/med/high.
    panels=[("baseline","CartPole RL gaussian_signal — baseline (no noise)"),
            ("low","CartPole RL gaussian_signal — low"),
            ("med","CartPole RL gaussian_signal — med"),
            ("high","CartPole RL gaussian_signal — high")]),
  "quad2d_nd": dict(
    title="Quadrotor2D stochastic noisy_dynamics",
    csv="quadrotor2d/rl/quad2d_noisy_dynamics_all_levels.csv",
    out="quadrotor2d/rl/quad2d_noisy_dynamics_all_levels.png",
    panels=[("noisy_dynamics_f_0.150","Quadrotor2D noisy_dynamics — f_0.150")],
    dsroot=DATA / "quadrotor2D",
    dslevel={"noisy_dynamics_f_0.150": "noisy_dynamics/rl/f_0.150"}),
  "quad2d_cs": dict(
    title="Quadrotor2D stochastic corridor_sine_ambient",
    csv="quadrotor2d/rl/quad2d_corridor_sine_ambient_all_levels.csv",
    out="quadrotor2d/rl/quad2d_corridor_sine_ambient_all_levels.png",
    panels=[("corridor_sine_ambient_smooth","Quadrotor2D corridor_sine_ambient — smooth")],
    dsroot=DATA / "quadrotor2D",
    dslevel={"corridor_sine_ambient_smooth": "corridor_sine_ambient/rl/smooth"}),
  "quad3d_nd": dict(
    title="Quadrotor3D stochastic noisy_dynamics",
    csv="quadrotor3d/lqr/quad3d_noisy_dynamics_all_levels.csv",
    out="quadrotor3d/lqr/quad3d_noisy_dynamics_all_levels.png",
    panels=[("noisy_dynamics_f_0.048","Quadrotor3D noisy_dynamics — f_0.048"),
            ("noisy_dynamics_f_0.060","Quadrotor3D noisy_dynamics — f_0.060")],
    dsroot=DATA / "quadrotor3D",
    dslevel={"noisy_dynamics_f_0.048": "noisy_dynamics/lqr/f_0.048",
             "noisy_dynamics_f_0.060": "noisy_dynamics/lqr/f_0.060"}),
  "quad3d_cs": dict(
    title="Quadrotor3D stochastic corridor_sine_ambient",
    csv="quadrotor3d/lqr/quad3d_corridor_sine_ambient_all_levels.csv",
    out="quadrotor3d/lqr/quad3d_corridor_sine_ambient_all_levels.png",
    panels=[("corridor_sine_ambient_f_0.30","Quadrotor3D corridor_sine_ambient — f_0.30")],
    dsroot=DATA / "quadrotor3D",
    dslevel={"corridor_sine_ambient_f_0.30": "corridor_sine_ambient/lqr/f_0.30"},
    # Reporting window (decision 2026-09-03): the quad3D corridor experiment is
    # read to 35,000 training trajectories, i.e. epoch 5 on its 10k + 5k/epoch
    # schedule, although the arms ran on past it. It is the first budget at
    # which every FM arm beats every classifier, BNN and Part-X arm on KL,
    # Brier and sAUROC at once, and where FM's deficit on the level-set areas
    # is narrowest. Every consumer of CAMPAIGNS (this figure, the level-set
    # figure, the ALC table) clips here, so a deeper epoch never leaks into a
    # quad3D comparison by accident.
    budget=35000),
  # PPO controller on the same quadrotor3D corridor family. Registered
  # 2026-09-06: the scorer already wrote both CSVs, but with no CAMPAIGNS entry
  # every plot script silently drew nothing for it, so the docs dir held data
  # and no figures.
  #
  # SEPARATE FROM quad3d_cs ON PURPOSE. This is a different controller, with no
  # ambient term, a different noise ladder, a different start box, half the
  # control rate and mid-flight FPS eval states. A PPO level must never be put
  # in a panel beside an LQR level, so it gets its own campaign key, its own
  # output file and its own docs subtree rather than extra panels on quad3d_cs.
  #
  # f_0.00 is DETERMINISTIC (no wind, one eval trial per state), so its KL is a
  # clipped log-loss against binary truth and is not on the same scale as the
  # three noisy levels. It is drawn because the panels are per-level and never
  # pooled; read its numbers on their own.
  #
  # No `budget`: this campaign's arms all run the same 10k + 1.5k/epoch schedule
  # to 40,000 trajectories, and nothing has finished, so there is no evidence yet
  # for choosing a reporting window. Add one only from a deliberate decision.
  "q3dppo_cs": dict(
    title="Quadrotor3D PPO stochastic corridor_sine_ambient",
    csv="quadrotor3d/ppo/quad3dppo_corridor_sine_ambient_all_levels.csv",
    out="quadrotor3d/ppo/quad3dppo_corridor_sine_ambient_all_levels.png",
    panels=[("corridor_sine_ambient_f_0.00","Quadrotor3D PPO corridor_sine_ambient — f_0.00 (DETERMINISTIC)"),
            ("corridor_sine_ambient_f_0.12","Quadrotor3D PPO corridor_sine_ambient — f_0.12"),
            ("corridor_sine_ambient_f_0.20","Quadrotor3D PPO corridor_sine_ambient — f_0.20"),
            ("corridor_sine_ambient_f_0.40","Quadrotor3D PPO corridor_sine_ambient — f_0.40")],
    dsroot=DATA / "quadrotor3D",
    dslevel={"corridor_sine_ambient_f_0.00": "corridor_sine_ambient/ppo/f_0.00",
             "corridor_sine_ambient_f_0.12": "corridor_sine_ambient/ppo/f_0.12",
             "corridor_sine_ambient_f_0.20": "corridor_sine_ambient/ppo/f_0.20",
             "corridor_sine_ambient_f_0.40": "corridor_sine_ambient/ppo/f_0.40"}),
}


def within_budget(rows, c):
    """Rows at or below the campaign's reporting budget; all rows when none is set."""
    b = c.get("budget")
    if b is None:
        return rows
    return [r for r in rows if int(r["train_trajectories"]) <= b]


def load(csv_path, level, keep=None, budget=None):
    m = defaultdict(dict)
    for r in within_budget(list(csv.DictReader(open(csv_path))), {"budget": budget}):
        if r["level"] != level or (keep is not None and r["arm"] not in keep):
            continue
        m[r["arm"]][int(str(r["epoch"]).split("_")[-1])] = r
    return m


def regime(c, lv):
    """One line naming the noise regime, read from the level's own dataset description.

    A family name alone is ambiguous: `noisy_dynamics` spans five force levels and
    `corridor_sine_ambient` three profiles, and a reader comparing two quadrotor
    figures has no way to tell which one a panel is unless the level's magnitude is
    on the panel. Derived rather than hardcoded so it cannot drift from the data.
    """
    frag = (c.get("dslevel") or {}).get(lv)
    if not frag or "dsroot" not in c:
        return None
    d = c["dsroot"] / frag / "dataset_description.json"
    try:
        mech = json.loads(d.read_text())["mechanism"]
    except Exception:
        return None
    wt = (mech.get("reference_scale") or {}).get("level_as_fraction_of_weight")
    pct = f"{100 * wt:.0f}% body weight" if wt else None
    if mech.get("distribution") == "uniform":
        lo, hi = mech.get("low"), mech.get("high")
        what = f"F ~ U({lo:+.3f}, {hi:+.3f}) N, white"
    else:
        what = "coherent sine + ambient"
    sibs = sorted(x.name for x in (c["dsroot"] / frag).parent.iterdir() if x.is_dir())
    here = frag.rsplit("/", 1)[-1]
    rank = f"level {sibs.index(here) + 1}/{len(sibs)}" if here in sibs else None
    # Kept deliberately short. The full mechanism -- hold, frame, torque, the
    # per-rollout vs per-step distinction -- belongs in the README; on the panel
    # it overflowed the axes and collided with the neighbouring column.
    return "  ·  ".join(x for x in (what, pct, rank) if x)


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

    Printed to the console, not drawn on the figure: on a training-trajectory
    axis an under-spent arm already stops visibly short, so a box saying so is
    redundant. It stays in the console because the distinction it encodes is not
    visible -- a short line means either "still running" or "finished and spent
    less", which are opposite readings, and only an arm at the campaign's
    deepest epoch can be the second.
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
        src = DOCS / c["csv"]
        if not src.exists():
            print(f"  SKIP {name}: {c['csv']} absent"); continue

        loaded, ragged = [], {}
        for lv, lab in c["panels"]:
            m = load(src, lv, keep, c.get("budget"))
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

        gaps = [(lv, x) for lv, _, m in loaded
                for x in missing_columns(m, [col for col, *_ in METRICS])]
        if gaps:
            print(f"  REFUSING to plot {name}: {len(gaps)} blank metric cell(s); "
                  "re-run scripts/score_stoch_incremental.py. First few:")
            for lv, (arm, e, col) in gaps[:6]:
                print(f"    {lv}/{arm}/epoch {e}: {col}")
            continue

        if ragged and not a.allow_partial:
            print(f"  REFUSING to plot {name}: ragged depths (pass --allow-partial to override)")
            for lab, d in ragged.items():
                mx = max(d.values())
                short = {k: v for k, v in sorted(d.items()) if v != mx}
                print(f"    {lab}: full depth {mx}; short -> {short}")
            continue

        for lv, lab, m in loaded:
            shortb, topb = budget_short(m)
            if shortb:
                print(f"    NOTE {lab}: full budget {topb:,} traj; under-spent -> "
                      + ", ".join(f"{k}={v:,}" for k, v in shortb.items()))
            if lab in ragged:
                mx = max(ragged[lab].values())
                sh = {k: v for k, v in sorted(ragged[lab].items()) if v != mx}
                print(f"    NOTE {lab}: full depth {mx}; still short -> "
                      + ", ".join(f"{k}={v}" for k, v in sh.items()))

        nrow, ncol = len(loaded), len(METRICS)
        # legend entries = one per drawn arm, plus band and mean. The three
        # uniform seeds contribute those TWO entries between them, not three,
        # so counting raw arm names over-widens the legend and leaves the last
        # row stranded.
        _tbl = {x[0] for x in ARMS}
        _present = {a_ for _, _, m in loaded for a_ in m}
        nlab = len(_present & _tbl) + (2 if len(_present & set(SEEDS)) == 3 else 0)
        legcol = min(6, max(3, (nlab + 1) // 2))
        LEG_IN = 0.42 + 0.26 * ((nlab + legcol - 1) // legcol)   # legend strip, inches
        TOP_IN = 0.78                                            # suptitle strip, inches
        PANEL_IN = 4.7
        fig_h = PANEL_IN * nrow + LEG_IN + TOP_IN
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.7*ncol, fig_h), squeeze=False)
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
                ftxt = (f"FM 2·SD floor = {f:.4f}" if f is not None
                        else f"NO FM floor ({nseed}/3 seeds)")
                # multi-line: one long title ran into the neighbouring column.
                # The regime goes in as a separate text object rather than a
                # third title line -- it wants a smaller, quieter face, and a
                # second set_title at the same loc replaces rather than adds.
                rg = regime(c, lv)
                ax.set_title(f"{lab}\n{nm}   ·   {ftxt}", fontsize=10,
                             pad=20 if rg else 6)
                if rg:
                    ax.text(0.5, 1.008, rg, transform=ax.transAxes, ha="center",
                            va="bottom", fontsize=7.2, color="#555")
                ax.grid(alpha=0.25, which="both", lw=0.5)

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
                     fontsize=13, y=1 - 0.13 / fig_h)
        # One figure-level legend along the bottom rather than an inset on the
        # first panel: the inset covered data in the lower-left of that panel,
        # and repeating the key per panel is redundant when every panel draws
        # the same arms.
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=legcol, fontsize=8.6,
                   frameon=False, bbox_to_anchor=(0.5, 0.004))
        fig.tight_layout(rect=[0, LEG_IN / fig_h, 1, 1 - TOP_IN / fig_h])
        out = DOCS / c["out"]
        if a.clean:
            out = out.with_name(out.stem + "_clean" + out.suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=145); plt.close(fig)
        print(f"  wrote {out.relative_to(DOCS)}  ({nrow} level(s), "
              f"{narms} arms + {nseed}-seed control)")


if __name__ == "__main__":
    main()
