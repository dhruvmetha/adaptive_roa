#!/usr/bin/env python
"""Paper figure: learning curves (metric against training trajectories) and an
area-under-the-learning-curve (ALC) table.

One figure per (system, figure set). Rows are the metrics (KL on a log axis,
sAUROC, skill score, and the F0.5 level-set area when the areas file exists),
columns are the noise levels, x is the number of training trajectories. The
"main" set is FM-BALD against the baselines; the "ablation" set is FM-BALD
against its own variants plus the non-adaptive FM control drawn as a black
mean line with a grey min..max band over the three seeds.

ALC per (system, level, arm, metric) is the trapezoid area of the metric
against training trajectories, epoch 0 included, divided by the trajectory
span, so it reads in the metric's own units. The span is the common budget:
epoch 0 to the last epoch that EVERY arm of the paper set has reached at that
level, so arms are compared at matched trajectory budget.

Shared style (arm colours, figure sets, output locations, typography) is
imported from the sibling `plot_levelsets_paper.py`; the STYLE block below
holds only what is specific to this figure family.

Usage:
    python scripts/paper/plot_learning_curves_paper.py --paper
    python scripts/paper/plot_learning_curves_paper.py --paper --set main
    python scripts/paper/plot_learning_curves_paper.py --system quad3d_ppo800k
    python scripts/paper/plot_learning_curves_paper.py --paper --no-wait   # skip the areas-job wait
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

_TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz   # numpy >= 2 renamed it

# Shared style from the level-set figure script (scripts/paper is not a package).
_SIB = Path(__file__).resolve().parent / "plot_levelsets_paper.py"
_spec = importlib.util.spec_from_file_location("plot_levelsets_paper", _SIB)
_ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ls)

# BALD arms are the top-N (greedy) runs (decision 2026-09-10); greedy_diverse is out of the paper
ARM_STYLES = _ls.ARM_STYLES
OURS = _ls.OURS
FAMILY_LS = _ls.FAMILY_LS
FIGURE_SETS = _ls.FIGURE_SETS
SEED_LABEL = _ls.SEED_LABEL
SEED_MEAN_COLOR = _ls.SEED_MEAN_COLOR
SEED_MEAN_LW = _ls.SEED_MEAN_LW
SEED_BAND_COLOR = _ls.SEED_BAND_COLOR
SEED_BAND_ALPHA = _ls.SEED_BAND_ALPHA
MARKER_SIZE = _ls.MARKER_SIZE
LINE_ALPHA = _ls.LINE_ALPHA
level_title = _ls.level_title
SYSTEM_TITLES = _ls.SYSTEM_TITLES
LEVEL_ORDER = _ls.LEVEL_ORDER
PAPER_DIR = _ls.PAPER_DIR
FIG_DIR = _ls.FIG_DIR
FILE_MODE = _ls.FILE_MODE
FORMATS = _ls.FORMATS
DPI = _ls.DPI
CONTROL_ARMS = list(getattr(_ls, "ALL_SEED_ARMS", ["dir00_s42", "dir00_s43", "dir00_s44"]))
FONT_FAMILY = getattr(_ls, "FONT_FAMILY", "sans-serif")
FONT_SIZE = getattr(_ls, "FONT_SIZE", 8)
TITLE_SIZE = getattr(_ls, "TITLE_SIZE", 9)
SUPTITLE_SIZE = getattr(_ls, "SUPTITLE_SIZE", 10)
SUPTITLE_SUB_SIZE = getattr(_ls, "SUPTITLE_SUB_SIZE", 7)
LABEL_SIZE = getattr(_ls, "LABEL_SIZE", 8)
TICK_SIZE = getattr(_ls, "TICK_SIZE", 7)
LEGEND_SIZE = getattr(_ls, "LEGEND_SIZE", 7.5)
GRID = getattr(_ls, "GRID", dict(color="0.85", lw=0.5))
SPINES_OFF = getattr(_ls, "SPINES_OFF", ("top", "right"))

# =============================================================================
# STYLE: everything specific to this figure family. Nothing below is a choice.
# =============================================================================

# Wide probability-metric CSVs, one row per (level, arm, epoch).
INPUTS = {
    "pendulum_lqr": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/pendulum/lqr/gaussian_all_levels.csv",
    "cartpole_ppo": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/cartpole/safe_explorer_ppo/gaussian_all_levels.csv",
    "quad2d_rl": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/quadrotor2d/rl/quad2d_corridor_sine_ambient_all_levels.csv",
    # quad3d = 800k-pool campaign (decision 2026-09-10); 100k is obsolete
    "quad3d_ppo800k": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/quadrotor3d/ppo_800k/quad3d800k_corridor_sine_ambient_all_levels.csv",
}
# Levels kept per system (None -> every level in the file, in LEVEL_ORDER then file order).
LEVEL_FILTER = {
    "pendulum_lqr": ["low", "med", "high"],
    "cartpole_ppo": ["baseline", "low", "med", "high"],
    "quad2d_rl": ["corridor_sine_ambient_baseline", "corridor_sine_ambient_smooth", "corridor_sine_ambient_loud"],
    "quad3d_ppo800k": ["corridor_sine_ambient_f_0.00", "corridor_sine_ambient_f_0.12", "corridor_sine_ambient_f_0.12_a0.03",
                       "corridor_sine_ambient_f_0.20", "corridor_sine_ambient_f_0.20_a0.04",
                       "corridor_sine_ambient_f_0.40", "corridor_sine_ambient_f_0.40_a0.04"],
}
LEVEL_BUDGET = {
    "quad2d_rl": {"corridor_sine_ambient_baseline": 8000, "corridor_sine_ambient_loud": 8000},
    "quad3d_ppo800k": {"corridor_sine_ambient_f_0.12_a0.03": 14500,
                       "corridor_sine_ambient_f_0.20_a0.04": 14500,
                       "corridor_sine_ambient_f_0.40_a0.04": 14500},
}
# Trajectory schedule (initial, per-epoch step) used when `train_trajectories`
# is blank on a row: traj = initial + epoch * step. The run prints a line
# whenever it had to fall back.
SCHEDULE = {
    "pendulum_lqr": (100, 100),
    "cartpole_ppo": (300, 150),
    "quad2d_rl": (2000, 500),
    "quad3d_ppo800k": (10000, 1500),   # same schedule as the 100k campaign
}

# Optional level-set areas per (level, arm, epoch), written by a background
# job and possibly incomplete. If the file exists and holds a curve (>= 2
# epochs) for at least one arm, an F0.5-area row is added and missing arms
# are NaN-skipped; if it is absent the row is omitted.
AREAS_DIR = PAPER_DIR / "levelsets_greedy"
AREAS_FILE = "{system}_levelset_areas_b50.csv"
AREAS_COL = "auc_f05"            # column in the areas file
AREAS_KEY = "f05_area"           # name used here (figure row, ALC table)

# The areas files are being appended by a background scoring job. Before the
# final render the script waits for its log to carry a line starting with
# WAIT_DONE_PREFIX (the job's timing summary), polling every WAIT_POLL_S
# seconds for at most WAIT_MAX_S; `--no-wait` skips this, and a missing log
# is treated as nothing to wait for. Arms the job has not written are
# NaN-skipped either way.
WAIT_LOG = Path("/common/users/st1122/tmp/claude-1171/-common-home-st1122-Projects-adaptive-roa/"
                "43c5522d-25aa-48ac-b5a0-b707d94fec98/scratchpad/score_rest_all.log")
WAIT_DONE_PREFIX = "[time]"
WAIT_POLL_S = 30
WAIT_MAX_S = 40 * 60

# Rows: (column, y label, y scale). Order is the paper's order.
METRICS = [
    ("KL",          "KL ↓",          "log"),
    ("sAUROC",      "sAUROC ↑",      "linear"),
    ("skill_score", "skill score ↑", "linear"),
]
AREAS_METRIC = (AREAS_KEY, "F0.5 area ↑", "linear")
LOWER_BETTER = {"KL": True, "sAUROC": False, "skill_score": False, AREAS_KEY: False}

# Display-only y floor: early-epoch skill scores reach -1.2 and would squash
# the range the paper cares about. Lines leave the panel below the floor; the
# ALC uses the raw values.
Y_FLOOR = {"skill_score": -0.5}
Y_CEIL = {"sAUROC": 1.0, "skill_score": 1.0, AREAS_KEY: 1.0}
Y_PAD_FRAC = 0.04                # linear-axis padding as a fraction of the range
LOG_SUBS = (1.0, 2.0, 5.0)       # major ticks per decade on the KL axis
X_LABEL = "training trajectories"
X_PAD_FRAC = 0.03                # x padding as a fraction of the budget span
X_TICK_BINS = 4                  # MaxNLocator bins for the trajectory axis
X_TICK_ROTATION = 0
MARKEVERY = 1                    # marker on every n-th epoch

# Control (non-adaptive FM seeds): aggregated key used in the ALC outputs.
CONTROL_KEY = "dir00_s42"   # single-seed control (2026-09-11)

# ALC budget span: "common" integrates every arm from epoch 0 to the last
# epoch all paper arms (both sets plus the control seeds) have reached at that
# level, so the numbers are matched-budget; "per_arm" integrates each arm to
# its own last epoch.
ALC_SPAN = "common"

# Deterministic levels get an asterisk in the column title and one footnote.
DETERMINISTIC_LEVELS = {
    "cartpole_ppo": {"baseline"},
    "quad2d_rl": {"corridor_sine_ambient_baseline"},
    "quad3d_ppo800k": {"corridor_sine_ambient_f_0.00"},
}
FOOTNOTE = "* deterministic level (one rollout per eval cell): KL is a clipped log-loss"
SET_TITLES = {"main": "main evaluation", "ablation": "ablation"}

# Geometry (inches).
PANEL_W, PANEL_H = 2.4, 1.7      # per panel
SINGLE_COL_PANEL_W = 3.6         # a one-level system gets a wider panel
FIG_MIN_W = 7.0                  # figures are padded to at least this width
LEFT_MARGIN, RIGHT_MARGIN = 0.6, 0.1
TITLE_BLOCK_H = 0.7              # suptitle + footnote above the panels
SUPTITLE_Y_IN = 0.15             # suptitle baseline, inches from the top
FOOTNOTE_Y_IN = 0.4              # footnote top, inches from the top
XLABEL_BLOCK_H = 0.5             # tick labels + x label under the last row
LEGEND_ROW_H = 0.28              # per legend row
LEGEND_MAX_W_PER_ENTRY = 1.6     # legend wraps when entries * this > figure width
SUBPLOT_WSPACE, SUBPLOT_HSPACE = 0.12, 0.3
CONTROL_Z, ARM_Z, OURS_Z = 2, 3, 4

# Output.
TABLE_DIR = PAPER_DIR / "tables"
FIG_STEM = "learning_curves_{system}_{set}"
ALC_CSV = "alc.csv"
ALC_MD = "alc_{set}.md"
FMT_DECIMALS = 4
# =============================================================================


def _apply_rc() -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE, "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _paper_arms() -> list[str]:
    arms: list[str] = []
    for cfg in FIGURE_SETS.values():
        arms += [a for a in cfg["arms"] if a not in arms]
    return arms + [a for a in CONTROL_ARMS if a not in arms]


def _levels(df: pd.DataFrame, system: str) -> list[str]:
    file_order = list(dict.fromkeys(df["level"].tolist()))
    ordered = [l for l in LEVEL_ORDER if l in file_order] + \
              [l for l in file_order if l not in LEVEL_ORDER]
    keep = LEVEL_FILTER.get(system)
    return [l for l in ordered if keep is None or l in keep]


def _log_done(log: Path) -> bool:
    try:
        with log.open() as fh:
            return any(line.startswith(WAIT_DONE_PREFIX) for line in fh)
    except OSError:
        return False


def wait_for_areas(log: Path = WAIT_LOG, poll_s: float = WAIT_POLL_S,
                   max_s: float = WAIT_MAX_S) -> bool:
    """Block until the areas job's log carries its done line. Returns True if
    it did, False on timeout (the caller renders whatever is there)."""
    if not log.exists():
        print(f"[wait] {log} missing; nothing to wait for")
        return True
    t0 = time.monotonic()
    while not _log_done(log):
        waited = time.monotonic() - t0
        if waited >= max_s:
            print(f"[wait] no '{WAIT_DONE_PREFIX}' line after {waited / 60:.0f} min; "
                  "rendering with the areas written so far")
            return False
        print(f"[wait] areas job still running ({waited / 60:.1f} min); next check in {poll_s:.0f} s")
        time.sleep(poll_s)
    print(f"[wait] areas job done ({(time.monotonic() - t0) / 60:.1f} min waited)")
    return True


def load_system(system: str) -> tuple[pd.DataFrame, list[str], list[tuple[str, str, str]]]:
    """Wide CSV restricted to the paper arms, with a filled `traj` column and
    the optional areas column merged in. Returns (df, levels, metric rows)."""
    # Canonical arm keys first (q2d_csbase `_fast` runs, dead twins dropped),
    # so the arm filter and the exclusions below see the paper names.
    df = _ls.apply_aliases(pd.read_csv(INPUTS[system]), system)
    df = df[df["arm"].isin(_paper_arms())].copy()
    for (sys_, lvl), arms_ in getattr(_ls, "EXCLUDE_ARMS", {}).items():
        if sys_ == system:
            df = df[~((df["level"] == lvl) & df["arm"].isin(arms_))]
    df["epoch"] = df["epoch"].astype(int)
    if "train_trajectories" not in df.columns:
        df["train_trajectories"] = np.nan
    traj = pd.to_numeric(df["train_trajectories"], errors="coerce")
    blank = traj.isna()
    if blank.any():
        init, step = SCHEDULE[system]
        traj = traj.where(~blank, init + df["epoch"] * step)
        print(f"     {system}: train_trajectories blank on {int(blank.sum())} rows, "
              f"used schedule {init} + epoch*{step}")
    df["traj"] = traj.astype(float)
    # Budget caps: the loud and ambient campaigns were stopped at 8,000 (quad2d)
    # and 14,500 (quad3d) trajectories; a few arms that predate the cap ran past
    # it and are truncated here so no arm gets a head start (main-comp, 2026-09-11).
    for lvl, cap in LEVEL_BUDGET.get(system, {}).items():
        df = df[~((df["level"] == lvl) & (df["traj"] > cap))]
    levels = _levels(df, system)
    df = df[df["level"].isin(levels)].copy()

    metrics = list(METRICS)
    areas_path = AREAS_DIR / AREAS_FILE.format(system=system)
    if areas_path.exists():
        ar = pd.read_csv(areas_path)
        if AREAS_COL in ar.columns:
            ar = ar[["level", "arm", "epoch", AREAS_COL]].copy()
            ar["epoch"] = ar["epoch"].astype(int)
            ar = ar.rename(columns={AREAS_COL: AREAS_KEY})
            df = df.merge(ar, on=["level", "arm", "epoch"], how="left")
            n_curve = (df.dropna(subset=[AREAS_KEY]).groupby(["level", "arm"])["epoch"]
                       .nunique() >= 2).sum()
            if n_curve:
                metrics.append(AREAS_METRIC)
                print(f"     {system}: F0.5 area from {areas_path} ({n_curve} curves)")
            else:
                df = df.drop(columns=[AREAS_KEY])
                print(f"     {system}: {areas_path} has no curve with >= 2 epochs; row omitted")
        else:
            print(f"     {system}: {areas_path} lacks {AREAS_COL}; row omitted")
    return df, levels, metrics


def _curve(df: pd.DataFrame, level: str, arm: str, col: str) -> tuple[np.ndarray, np.ndarray]:
    if col not in df.columns:
        return np.array([]), np.array([])
    sub = df[(df["level"] == level) & (df["arm"] == arm)].sort_values("epoch")
    sub = sub.dropna(subset=[col])
    return sub["traj"].to_numpy(float), sub[col].to_numpy(float)


def _control_curves(df: pd.DataFrame, level: str, col: str):
    """Per-epoch mean / min / max over the control seeds, aligned on epoch."""
    per = {}
    for s in CONTROL_ARMS:
        sub = df[(df["level"] == level) & (df["arm"] == s)]
        if col in sub.columns and not sub.empty:
            per[s] = sub.set_index("epoch")[[col, "traj"]]
    if not per:
        return None
    epochs = sorted(set().union(*[set(p.index) for p in per.values()]))
    ys = np.array([[p[col].get(e, np.nan) for e in epochs] for p in per.values()], float)
    xs = np.array([[p["traj"].get(e, np.nan) for e in epochs] for p in per.values()], float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x = np.nanmedian(xs, axis=0)
        return x, np.nanmean(ys, axis=0), np.nanmin(ys, axis=0), np.nanmax(ys, axis=0)


# ----------------------------------------------------------------------------
# ALC
# ----------------------------------------------------------------------------
def _alc(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Trapezoid area of y(x) over [lo, hi] divided by (hi - lo). The curve is
    linearly interpolated at the ends if it extends past them; NaN when it
    does not span the interval or has fewer than two points."""
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or hi <= lo or x[0] > lo + 1e-9 or x[-1] < hi - 1e-9:
        return float("nan")
    inside = (x >= lo - 1e-9) & (x <= hi + 1e-9)
    xs = np.concatenate([[lo], x[inside], [hi]])
    ys = np.concatenate([[np.interp(lo, x, y)], y[inside], [np.interp(hi, x, y)]])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    return float(_TRAPEZOID(ys, xs) / (hi - lo))


def _span(df: pd.DataFrame, level: str, col: str, arms: list[str]) -> tuple[float, float, int, int]:
    """Common trajectory budget for `col` at `level`: (lo, hi, epoch_lo, epoch_hi)."""
    los, his, e_los, e_his = [], [], [], []
    for a in arms:
        sub = df[(df["level"] == level) & (df["arm"] == a)]
        if col in sub.columns:
            sub = sub.dropna(subset=[col])
        else:
            sub = sub.iloc[0:0]
        if sub.empty:
            continue
        los.append(sub["traj"].min()); his.append(sub["traj"].max())
        e_los.append(sub["epoch"].min()); e_his.append(sub["epoch"].max())
    if not his:
        return float("nan"), float("nan"), -1, -1
    return float(max(los)), float(min(his)), int(max(e_los)), int(min(e_his))


def compute_alc(df: pd.DataFrame, system: str, levels: list[str],
                metrics: list[tuple[str, str, str]]) -> list[dict]:
    rows: list[dict] = []
    arms = _paper_arms()
    for level in levels:
        for col, _, _ in metrics:
            lo, hi, e_lo, e_hi = _span(df, level, col, arms)
            per_arm: dict[str, float] = {}
            spans: dict[str, tuple] = {}
            for a in arms:
                x, y = _curve(df, level, a, col)
                if ALC_SPAN == "per_arm" and len(x):
                    a_lo, a_hi, a_elo, a_ehi = x[0], x[-1], None, None
                else:
                    a_lo, a_hi, a_elo, a_ehi = lo, hi, e_lo, e_hi
                per_arm[a] = _alc(x, y, a_lo, a_hi) if len(x) else float("nan")
                spans[a] = (a_lo, a_hi, a_elo, a_ehi, int(np.sum((x >= a_lo - 1e-9) & (x <= a_hi + 1e-9))))
            for a in arms:
                a_lo, a_hi, a_elo, a_ehi, n_pts = spans[a]
                label = ARM_STYLES[a][0] if a in ARM_STYLES else f"{SEED_LABEL} {a.split('_')[-1]}"
                rows.append(dict(system=system, level=level, arm=a, arm_label=label, metric=col,
                                 alc=per_arm[a], alc_2sd=np.nan, n_seeds=1,
                                 epoch_lo=a_elo, epoch_hi=a_ehi, traj_lo=a_lo, traj_hi=a_hi,
                                 n_points=n_pts))
            seeds = [per_arm[s] for s in CONTROL_ARMS if not math.isnan(per_arm.get(s, np.nan))]
            if seeds:
                sd = float(np.std(seeds, ddof=1)) if len(seeds) > 1 else float("nan")
                rows.append(dict(system=system, level=level, arm=CONTROL_KEY, arm_label=SEED_LABEL,
                                 metric=col, alc=float(np.mean(seeds)), alc_2sd=2 * sd,
                                 n_seeds=len(seeds), epoch_lo=e_lo, epoch_hi=e_hi,
                                 traj_lo=lo, traj_hi=hi, n_points=spans[CONTROL_ARMS[0]][4]))
    return rows


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def render(df: pd.DataFrame, system: str, levels: list[str],
           metrics: list[tuple[str, str, str]], set_name: str,
           suffix: str = "") -> tuple[list[Path], list[str]]:
    _apply_rc()
    cfg = FIGURE_SETS[set_name]
    arms = list(cfg["arms"])
    draw_band = bool(cfg["band"])
    skipped: list[str] = []

    n_rows, n_cols = len(metrics), len(levels)
    panel_w = PANEL_W if n_cols > 1 else SINGLE_COL_PANEL_W
    block_w = panel_w * n_cols
    fig_w = max(block_w + LEFT_MARGIN + RIGHT_MARGIN, FIG_MIN_W)
    n_entries = len(arms) + (1 if draw_band else 0)
    legend_ncol = max(1, min(n_entries, int(fig_w / LEGEND_MAX_W_PER_ENTRY)))
    legend_h = LEGEND_ROW_H * math.ceil(n_entries / legend_ncol)
    fig_h = PANEL_H * n_rows + TITLE_BLOCK_H + XLABEL_BLOCK_H + legend_h
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(fig_w, fig_h))
    handles: dict[str, object] = {}
    has_det = False
    thousands = FuncFormatter(lambda v, _: f"{int(round(v)):,}")
    plain = FuncFormatter(lambda v, _: f"{v:g}")

    for j, level in enumerate(levels):
        det = level in DETERMINISTIC_LEVELS.get(system, set())
        has_det |= det
        lvl = df[df["level"] == level]
        xs_all = lvl[lvl["arm"].isin(arms + (CONTROL_ARMS if draw_band else []))]["traj"]
        if xs_all.empty:
            skipped.append(f"{level}: no plotted arm present")
            for i in range(n_rows):
                axes[i][j].set_axis_off()
            continue
        x_lo, x_hi = float(xs_all.min()), float(xs_all.max())
        x_pad = X_PAD_FRAC * max(x_hi - x_lo, 1.0)

        for i, (col, ylab, scale) in enumerate(metrics):
            ax = axes[i][j]
            ax.grid(True, **GRID)
            for sp in SPINES_OFF:
                ax.spines[sp].set_visible(False)
            # Control band + mean.
            if draw_band:
                ctrl = _control_curves(lvl, level, col)
                if ctrl is not None:
                    x, mean, lo, hi = ctrl
                    ok = ~np.isnan(mean)
                    ax.fill_between(x[ok], lo[ok], hi[ok], color=SEED_BAND_COLOR,
                                    alpha=SEED_BAND_ALPHA, lw=0, zorder=CONTROL_Z - 1)
                    ax.plot(x[ok], mean[ok], color=SEED_MEAN_COLOR, lw=SEED_MEAN_LW,
                            zorder=CONTROL_Z)
                    handles.setdefault(SEED_LABEL, (
                        Line2D([], [], color=SEED_MEAN_COLOR, lw=SEED_MEAN_LW),
                        Patch(facecolor=SEED_BAND_COLOR, alpha=SEED_BAND_ALPHA, lw=0)))
                elif i == 0:
                    skipped.append(f"{level}: control seeds absent")
            # Adaptive arms.
            for arm in arms:
                label, fam, color, lw, mk = ARM_STYLES[arm]
                x, y = _curve(lvl, level, arm, col)
                if len(x) == 0:
                    if i == 0 or col == AREAS_KEY:
                        skipped.append(f"{level}: {arm} absent for {col}")
                    continue
                ax.plot(x, y, ls=FAMILY_LS[fam], color=color, lw=lw, marker=mk,
                        ms=MARKER_SIZE, markevery=MARKEVERY, alpha=LINE_ALPHA,
                        zorder=OURS_Z if arm == OURS else ARM_Z)
                handles.setdefault(label, Line2D(
                    [], [], ls=FAMILY_LS[fam], color=color, lw=lw, marker=mk, ms=MARKER_SIZE))
            # Axes cosmetics.
            if scale == "log":
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=LOG_SUBS))
                ax.yaxis.set_major_formatter(plain)
                ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=X_TICK_BINS, integer=True))
            ax.xaxis.set_major_formatter(thousands)
            ax.tick_params(axis="x", rotation=X_TICK_ROTATION)
            if i == 0:
                ax.set_title(level_title(level) + ("*" if det else ""))
            if i == n_rows - 1:
                ax.set_xlabel(X_LABEL)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(ylab)

    # Shared y range per row (linear rows honour the display floor / ceiling).
    for i, (col, _, scale) in enumerate(metrics):
        row = [axes[i][j] for j in range(n_cols) if axes[i][j].axison]
        if not row:
            continue
        ys = []
        for ax in row:
            for ln in ax.get_lines():
                ys.append(ln.get_ydata())
            for coll in ax.collections:
                ys.append(coll.get_paths()[0].vertices[:, 1] if coll.get_paths() else np.array([]))
        ys = np.concatenate([np.asarray(v, float) for v in ys]) if ys else np.array([])
        ys = ys[np.isfinite(ys)]
        if not len(ys):
            continue
        if scale == "log":
            ys = ys[ys > 0]
            lo, hi = float(ys.min()), float(ys.max())
            lo, hi = lo / 1.25, hi * 1.25
        else:
            lo, hi = float(ys.min()), float(ys.max())
            if col in Y_FLOOR:
                lo = max(lo, Y_FLOOR[col])
            if col in Y_CEIL:
                hi = min(hi, Y_CEIL[col])
            pad = Y_PAD_FRAC * max(hi - lo, 1e-6)
            lo, hi = lo - pad, hi + pad
        for k, ax in enumerate(row):
            ax.set_ylim(lo, hi)
            if k > 0:
                ax.tick_params(labelleft=False)

    # Legend, titles, layout.
    order = ([SEED_LABEL] if draw_band else []) + [ARM_STYLES[a][0] for a in arms]
    keys = [k for k in order if k in handles]
    fig.legend([handles[k] for k in keys], keys, loc="lower center",
               ncol=min(legend_ncol, len(keys)), frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.2)
    fig.suptitle(f"{SYSTEM_TITLES.get(system, system)} — {SET_TITLES[set_name]}",
                 fontsize=SUPTITLE_SIZE, y=1 - SUPTITLE_Y_IN / fig_h)
    if has_det:
        fig.text(0.5, 1 - FOOTNOTE_Y_IN / fig_h, FOOTNOTE, ha="center", va="top",
                 fontsize=SUPTITLE_SUB_SIZE, color="0.3")
    left_in = (fig_w - block_w - LEFT_MARGIN - RIGHT_MARGIN) / 2 + LEFT_MARGIN
    fig.subplots_adjust(left=left_in / fig_w, right=(left_in + block_w) / fig_w,
                        top=1 - TITLE_BLOCK_H / fig_h,
                        bottom=(legend_h + XLABEL_BLOCK_H) / fig_h,
                        wspace=SUBPLOT_WSPACE, hspace=SUBPLOT_HSPACE)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outs = []
    for fmt in FORMATS:
        p = FIG_DIR / f"{FIG_STEM.format(system=system, set=set_name)}{suffix}.{fmt}"
        fig.savefig(p, dpi=DPI)
        os.chmod(p, FILE_MODE)
        outs.append(p)
    plt.close(fig)
    return outs, skipped


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def _fmt(v: float) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{FMT_DECIMALS}f}"


def _md_table(system: str, level: str, rows: list[dict], set_name: str,
              cols: list[str]) -> str:
    cfg = FIGURE_SETS[set_name]
    keys = list(cfg["arms"]) + ([CONTROL_KEY] if cfg["band"] else [])
    by = {(r["arm"], r["metric"]): r for r in rows if r["level"] == level}
    body = [k for k in keys if any((k, m) in by for m in cols)]
    if not body:
        return ""
    best: dict[str, str] = {}
    for m in cols:
        cands = [(by[(k, m)]["alc"], k) for k in body
                 if (k, m) in by and not math.isnan(by[(k, m)]["alc"])]
        if cands:
            best[m] = (min(cands) if LOWER_BETTER.get(m, False) else max(cands))[1]
    base = next((by[(k, cols[0])] for k in body if (k, cols[0]) in by), None)
    span = ""
    if base is not None and not math.isnan(base["traj_hi"]):
        span = (f" (epochs {base['epoch_lo']} to {base['epoch_hi']}, "
                f"{int(base['traj_lo']):,} to {int(base['traj_hi']):,} trajectories)")
    lines = [f"### {system} / {level}{span}, {set_name}", "",
             "| Arm | n_traj | " + " | ".join(cols) + " |",
             "|---|---:|" + "---:|" * len(cols)]
    notes: list[str] = []
    for k in body:
        cells = []
        for m in cols:
            r = by.get((k, m))
            s = _fmt(r["alc"]) if r else ""
            if r and k == CONTROL_KEY and not math.isnan(r["alc_2sd"]):
                s = f"{s} ± {_fmt(r['alc_2sd'])}"
            if s and best.get(m) == k:
                s = f"**{s}**"
            cells.append(s)
        r0 = next((by[(k, m)] for m in cols if (k, m) in by), None)
        label = r0["arm_label"] if r0 else k
        nt = "" if r0 is None or math.isnan(r0["traj_hi"]) else f"{int(r0['traj_hi']):,}"
        lines.append(f"| {label} | {nt} | " + " | ".join(cells) + " |")
    for m in cols[1:]:
        rm = next((by[(k, m)] for k in body if (k, m) in by), None)
        if rm is not None and base is not None and not math.isnan(rm["traj_hi"]) \
                and abs(rm["traj_hi"] - base["traj_hi"]) > 0.5:
            notes.append(f"{m} is integrated to {int(rm['traj_hi']):,} trajectories "
                         f"(epoch {rm['epoch_hi']}), the deepest epoch it is scored for on every arm.")
    if level in DETERMINISTIC_LEVELS.get(system, set()):
        notes.append(FOOTNOTE)
    if notes:
        lines += [""] + [f"{n}  " for n in notes]
    lines.append("")
    return "\n".join(lines)


MD_HEADERS = {
    "main": [
        "# Main evaluation: area under the learning curve (ALC)", "",
        "ALC is the trapezoid area of a metric against training trajectories, from epoch 0 to "
        "the last epoch every paper arm has reached at that level, divided by the trajectory "
        "span. It reads in the metric's own units: a budget-weighted mean over the run. Arms "
        "are compared at matched trajectory budget; the adaptive arms train on more rows per "
        "trajectory than the non-adaptive control, so the budget is trajectories, not rows.", "",
        "FM-BALD (ours) against the baselines Part-X (GP) and CLF-BALD. Bold marks the best "
        "of the three per column (KL lower is better; sAUROC, skill_score and F0.5 area higher).", ""],
    "ablation": [
        "# Ablation: area under the learning curve (ALC)", "",
        "ALC is the trapezoid area of a metric against training trajectories, from epoch 0 to "
        "the last epoch every paper arm has reached at that level, divided by the trajectory "
        "span. It reads in the metric's own units: a budget-weighted mean over the run. Arms "
        "are compared at matched trajectory budget; the adaptive arms train on more rows per "
        "trajectory than the non-adaptive control, so the budget is trajectories, not rows.", "",
        "FM-BALD (ours) against its variants FM epi-var and BNN-BALD, and the non-adaptive FM "
        "control. The control row is the mean ± 2sd over seeds s42/s43/s44 of the per-seed ALC. "
        "Bold marks the best of the four per column (KL lower is better; sAUROC, skill_score "
        "and F0.5 area higher).", ""],
}


def write_tables(all_rows: list[dict], per_system: dict[str, tuple[list[str], list[str]]],
                 sets: list[str]) -> list[Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    outs = []
    csv_path = TABLE_DIR / ALC_CSV
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    os.chmod(csv_path, FILE_MODE)
    outs.append(csv_path)
    for set_name in sets:
        parts = list(MD_HEADERS[set_name])
        for system, (levels, cols) in per_system.items():
            rows = [r for r in all_rows if r["system"] == system]
            for level in levels:
                t = _md_table(system, level, rows, set_name, cols)
                if t:
                    parts.append(t)
        p = TABLE_DIR / ALC_MD.format(set=set_name)
        p.write_text("\n".join(parts))
        os.chmod(p, FILE_MODE)
        outs.append(p)
    return outs


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", action="store_true", help="all four systems")
    ap.add_argument("--system", choices=list(INPUTS), help="one system (instead of --paper)")
    ap.add_argument("--set", default="all", choices=["all"] + list(FIGURE_SETS),
                    help="figure set to render; 'all' renders each set as its own figure")
    ap.add_argument("--suffix", default="", help="figure filename suffix, e.g. _test")
    ap.add_argument("--no-tables", action="store_true", help="figures only")
    ap.add_argument("--no-wait", action="store_true",
                    help=f"do not wait for the areas job ({WAIT_LOG.name}) before reading the areas files")
    args = ap.parse_args()
    if args.paper:
        systems = list(INPUTS)
    elif args.system:
        systems = [args.system]
    else:
        ap.error("give --paper or --system")
    sets = list(FIGURE_SETS) if args.set == "all" else [args.set]
    if not args.no_wait:
        wait_for_areas()

    all_rows: list[dict] = []
    per_system: dict[str, tuple[list[str], list[str]]] = {}
    for system in systems:
        path = Path(INPUTS[system])
        if not path.exists():
            print(f"[skip] {system}: {path} missing")
            continue
        df, levels, metrics = load_system(system)
        for set_name in sets:
            outs, skipped = render(df, system, levels, metrics, set_name, args.suffix)
            print(f"[ok] {system} ({set_name}): " + ", ".join(str(o) for o in outs))
            for s in dict.fromkeys(skipped):
                print(f"     skipped {s}")
        if not args.no_tables:
            all_rows += compute_alc(df, system, levels, metrics)
            per_system[system] = (levels, [m[0] for m in metrics])
    if all_rows and not args.no_tables:
        for p in write_tables(all_rows, per_system, sets):
            print(f"[ok] table: {p}")


if __name__ == "__main__":
    main()
