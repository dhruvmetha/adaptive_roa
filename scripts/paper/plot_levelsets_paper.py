#!/usr/bin/env python
"""Paper figure: level-set metrics (F0.5, recall, precision) against the level β.

One figure per system. Rows are the three metrics, columns are the noise
levels in the file's natural order, x is β. Every design decision lives in the
STYLE block below; everything under it is plumbing.

Input is the long "levelsets" CSV with one row per (arm, epoch, beta). The
training-trajectory count is not on that file; it is joined from the wide CSV
(column `train_trajectories`) when one is given or found next to the input.

Usage:
    python scripts/paper/plot_levelsets_paper.py --paper --epoch common
    python scripts/paper/plot_levelsets_paper.py --system pendulum_lqr \
        --levelsets path/to/pendulum_lqr_levelsets_b50.csv --epoch 19
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import pandas as pd  # noqa: E402

# =============================================================================
# STYLE  -- every design choice is here. Nothing below this block is a choice.
# =============================================================================

# Okabe-Ito palette.
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermilion="#D55E00", purple="#CC79A7", black="#000000")

# Line style per predictor family.
FAMILY_LS = {"fm": "-", "clf": ":", "bnn": "-.", "gp": "--"}

# Style per arm: arm -> (label, family, color, linewidth, marker).
# BALD arms are the top-N (greedy) runs (decision 2026-09-10); greedy_diverse is out of the paper
OURS = "epi_bald_greedy"
# Validated categorical palette (dataviz reference instance; the main trio passes
# all-pairs, the ablation trio adjacent pairs). Shared by every figure family.
PAL = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
           purple="#7d4fc4")   # partx_faithful; distinct from partx_fix's orange so the two
                               # Part-X arms never read as one method drawn twice
ARM_STYLES = {
    "epi_bald_greedy":     ("FM-BALD (ours)", "fm",  PAL["blue"],    2.2, "o"),
    "epi_var_greedy":      ("FM epi-var",     "fm",  PAL["yellow"],  1.5, "s"),
    "clf_epi_bald_greedy": ("CLF-BALD",       "clf", PAL["aqua"],    1.5, "^"),
    "bnn_mfvi_a1_greedy":  ("BNN-BALD",       "bnn", PAL["magenta"], 1.5, "D"),
    "partx_fix":           ("Part-X (GP)",    "gp",  PAL["orange"],  1.5, "v"),
    # The faithful implementation of Algorithms 1-4 (arXiv 2110.10729), as opposed
    # to partx_fix above, which is a single global GP with straddle sampling and is
    # NOT Part-X. The label carries the readout because the two arms score
    # differently -- this one from a per-leaf local GP per region, every other arm
    # in these figures from one global model -- and that difference is currently
    # confounded with its sampling. Do not drop "per-leaf GP readout" from the
    # label until the global-GP rescore separates the two.
    "partx_faithful":      ("Part-X (faithful, per-leaf GP readout)",
                                               "gp",  PAL["purple"],  1.5, "P"),
}
MARKER_SIZE = 3.2
LINE_ALPHA = 0.95

# Figure sets. Each set is rendered as its OWN figure (file suffix `_<set>`):
# the main evaluation is the method against the baselines, the ablation is the
# method against its own variants. `arms` is the legend order; `band` draws the
# non-adaptive FM seeds (mean line + min..max band) in that figure.
FIGURE_SETS = {
    "main":     dict(arms=["epi_bald_greedy", "partx_fix", "clf_epi_bald_greedy"], band=False),
    "ablation": dict(arms=["epi_bald_greedy", "epi_var_greedy", "bnn_mfvi_a1_greedy"], band=True),
    # The timeout-fix campaign ran exactly four methods, so they fit one figure:
    # the three BALD arms as lines plus the uniform control as the black band arm.
    # It has no partx_fix or epi_var_greedy, so `main`/`ablation` would each draw
    # a panel with a missing curve.
    # partx_faithful joins the three BALD arms wherever it has run to full depth.
    # plot_timeout_fix.py drops it per-system where it has not: these figures are
    # drawn at the DEEPEST EPOCH ALL ARMS SHARE, so one arm at epoch 1 would drag
    # every other arm in that panel down to epoch 1 as well and quietly replace a
    # finished comparison with a first-epoch one.
    "timeoutfix": dict(arms=["epi_bald_greedy", "bnn_mfvi_a1_greedy", "clf_epi_bald_greedy",
                             "partx_faithful"],
                       band=True),
}

# Non-adaptive FM seeds: black mean line + light grey min..max band.
# Single-seed control everywhere (user decision 2026-09-11): seed 42 only, no
# 3-seed mean, band or pooled floor anywhere in the paper.
ALL_SEED_ARMS = ["dir00_s42"]
SEED_LABEL = "Uniform sampling"
# Where the control sits among the ablation arms (bar order and legend order):
# FM-BALD, FM epi-var, uniform, BNN-BALD (user, 2026-09-11).
CONTROL_POSITION = 2
SEED_MEAN_COLOR = "black"
SEED_MEAN_LW = 1.4
SEED_BAND_COLOR = "0.55"
SEED_BAND_ALPHA = 0.28

# Active set (filled in by _select_set before each render; do not edit here).
ARMS: dict = dict(ARM_STYLES)
SEED_ARMS: list = list(ALL_SEED_ARMS)
DRAW_BAND = False

# Oracle ceilings (from the `_oracle` columns). Off for the paper figures.
DRAW_ORACLE = False
FM_ORACLE_SRC = "dir00_s42"                      # row whose *_oracle is the FM ceiling
EXACT_ORACLE_SRCS = ["clf_epi_bald_greedy", "bnn_mfvi_a1_greedy", "partx_fix"]  # first present wins
FM_ORACLE_LABEL = "Oracle (FM)"
EXACT_ORACLE_LABEL = "Oracle (exact predictor)"
FM_ORACLE_STYLE = dict(color="black", ls="--", lw=0.9)
EXACT_ORACLE_STYLE = dict(color="0.45", ls=":", lw=0.9)

# Rows: metric column -> y label.
METRICS = [("f05", "$F_{0.5}$"), ("tpr", "Recall"), ("prec", "Precision")]   # draft; the final profile keeps F0.5 only
X_LABEL = "level β"
Y_LIM = (-0.02, 1.02)
Y_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]
BETA_MIN = 0.5           # rows with beta < BETA_MIN are dropped
X_PAD = 0.03             # x margin beyond first/last beta

# True-set fraction annotation along the top of the TPR row.
DRAW_TRUE_SET = False    # the per-beta percentage text (superseded by the shaded area)
TRUE_SET_ROW = "tpr"
# Shaded area: fraction of the eval set whose true p >= beta (the true level
# set), drawn from 0 up to that fraction in EVERY panel, behind the curves.
DRAW_TRUE_SET_FILL = True
TRUE_SET_FILL_COLOR = "0.6"
TRUE_SET_FILL_ALPHA = 0.18
TRUE_SET_FILL_LABEL = "true level-set mass"
TRUE_SET_FONT = 5.6
TRUE_SET_COLOR = "#555555"
TRUE_SET_Y = 1.03        # axes-fraction height of the text
TRUE_SET_PREFIX = "true set:"

# Column order: levels named here come first in this order, the rest follow in
# file order. Per-system level filter: None -> every level in the file.
LEVEL_ORDER = ["baseline", "low", "med", "high",
               # quad2d and quad3d corridor levels, base level then its ambient variant
               "corridor_sine_ambient_baseline",
               "corridor_sine_ambient_smooth", "corridor_sine_ambient_loud",
               "corridor_sine_ambient_f_0.00", "corridor_sine_ambient_f_0.12", "corridor_sine_ambient_f_0.12_a0.03",
               "corridor_sine_ambient_f_0.20", "corridor_sine_ambient_f_0.20_a0.04",
               "corridor_sine_ambient_f_0.40", "corridor_sine_ambient_f_0.40_a0.04"]
# The paper shows at most three noise levels per system, named deterministic /
# low / high (user decision 2026-09-11). Keys are the level names in the CSVs.
PAPER_LEVELS = {
    "pendulum_lqr":  [("low", "low"), ("med", "high")],
    "cartpole_ppo":  [("baseline", "deterministic"), ("low", "low"), ("med", "high")],
    "quad2d_rl":     [("corridor_sine_ambient_baseline", "deterministic"),
                      ("corridor_sine_ambient_smooth", "low"),
                      ("corridor_sine_ambient_loud", "high")],
    "quad3d_ppo800k": [("corridor_sine_ambient_f_0.00", "deterministic"),
                       ("corridor_sine_ambient_f_0.12_a0.03", "low"),
                       ("corridor_sine_ambient_f_0.40_a0.04", "high")],
}
PAPER_LEVEL_NAMES = {(s, k): n for s, lv in PAPER_LEVELS.items() for k, n in lv}
# Draft profile: every level in the file (quad2d restricted to its three paper
# levels because the file also carries single-arm stubs). Final profile: the
# PAPER_LEVELS above, with their deterministic / low / high names.
DRAFT_LEVEL_FILTER = {
    "quad2d_rl": ["corridor_sine_ambient_baseline", "corridor_sine_ambient_smooth",
                  "corridor_sine_ambient_loud"],
}
FINAL_LEVEL_FILTER = {s: [k for k, _ in lv] for s, lv in PAPER_LEVELS.items()}
LEVEL_FILTER = dict(DRAFT_LEVEL_FILTER)
USE_PAPER_NAMES = False
# Runs excluded from the paper at a given level (user instruction 2026-09-11: the
# quad2d loud epi-var run is still in flight). Keyed (system, level) -> arm keys.
EXCLUDE_ARMS = {
    # (quad2d loud epi-var finished 2026-09-11 13:07; its exclusion is lifted)
    # q2d_csbase has a 1-epoch bnn_a1_dir00 stub; not a paper arm at that level
    ("quad2d_rl", "corridor_sine_ambient_baseline"): {"bnn_a1_dir00"},
}
# Arm-name aliases per (system, level): CSV arm key -> paper arm key. On the
# quad2d no-noise campaign (q2d_csbase) the LIVE FM runs are the `_fast`
# directories (13 banked epochs, 0..12); the bare-name directories at that
# prefix (dir00_s42, epi_bald_greedy, epi_var_greedy) are CANCELLED twins with
# 2-3 epochs. The scorer writes both under their own names, so every reader
# drops the rows of the dead twins at that level and renames the `_fast` rows
# to the canonical keys (`apply_aliases`). Nothing in the shared tree is renamed.
ARM_ALIAS = {
    ("quad2d_rl", "corridor_sine_ambient_baseline"): {
        "dir00_s42_fast": "dir00_s42",
        "epi_bald_greedy_fast": "epi_bald_greedy",
        "epi_var_greedy_fast": "epi_var_greedy",
    },
}
# Budget caps per (system, level): rows with more training trajectories are
# dropped. The q2d_csbase campaign was capped at 8,000 trajectories = epoch 12
# (schedule 2000 + 500/epoch); clf_dir00 and partx_fix ran on to epoch 23.
LEVEL_BUDGET_CAP = {
    ("quad2d_rl", "corridor_sine_ambient_baseline"): 8000,
}

# Level name -> column title.
LEVEL_TITLES = {
    "baseline": "no noise", "low": "low noise", "med": "medium noise", "high": "high noise",
    "corridor_sine_ambient_smooth": "smooth",
    "corridor_sine_ambient_loud": "loud",
    "corridor_sine_ambient_f_0.12_a0.03": "f = 0.12, a = 0.03",
    "corridor_sine_ambient_f_0.20_a0.04": "f = 0.20, a = 0.04",
    "corridor_sine_ambient_f_0.40_a0.04": "f = 0.40, a = 0.04",
    "corridor_sine_ambient_sharp": "sharp",
    "corridor_sine_ambient_baseline": "no noise",
}
def level_title(level: str, system: str | None = None) -> str:
    if USE_PAPER_NAMES and system is not None and (system, level) in PAPER_LEVEL_NAMES:
        return PAPER_LEVEL_NAMES[(system, level)]
    if level in LEVEL_TITLES:
        return LEVEL_TITLES[level]
    if "_f_" in level:
        tail = level.split("_f_")[-1]
        if "_a" in tail:
            f, a = tail.split("_a", 1)
            return f"f = {f}, a = {a}"
        return f"f = {tail}"
    return level

# System name -> suptitle.
SYSTEM_TITLES = {
    "pendulum_lqr": "Pendulum (LQR)", "cartpole_ppo": "Cartpole (RL)",
    "quad2d_rl": "Quadrotor 2D (RL)",
    # quad3d = 800k-pool campaign (decision 2026-09-10); 100k is obsolete
    "quad3d_ppo800k": "Quadrotor 3D (RL)",
}

# Figure geometry and typography (inches unless stated).
PANEL_W, PANEL_H = 2.6, 1.9      # per panel
FIG_MIN_W = 7.0                  # a one-column figure is padded to this width
TITLE_BLOCK_H = 0.75             # suptitle + epoch line above the panels (draft)
TITLE_BLOCK_H_FINAL = 0.55       # suptitle only (final profile)
XLABEL_BLOCK_H = 0.45            # x label under the last row
LEGEND_ROW_H = 0.26              # per legend row
LEGEND_MAX_W_PER_ENTRY = 1.35    # legend wraps when entries * this > figure width
LEFT_MARGIN, RIGHT_MARGIN = 0.55, 0.08
DPI = 200
FONT_FAMILY = "sans-serif"
FONT_SIZE = 8
TITLE_SIZE = 9
SUPTITLE_SIZE = 10
SUPTITLE_SUB_SIZE = 7            # epoch / trajectory line under the suptitle
LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 7.5
LEGEND_NCOL = None               # None -> auto-wrap from LEGEND_MAX_W_PER_ENTRY
X_TICK_ROTATION = 0
X_TICK_LABEL_EVERY = 1           # 2 -> label every other beta (draft)
X_TICK_LABEL_EVERY_FINAL = 2     # final: 0.50, 0.60, ..., 0.90 labelled, the rest ticks only
PANEL_W_FINAL = 3.2              # final: wider panels for the single-row figure
GRID = dict(color="0.85", lw=0.5)
SPINES_OFF = ("top", "right")
SUBPLOT_WSPACE, SUBPLOT_HSPACE = 0.12, 0.28

# Wide CSVs carrying `train_trajectories` (level, arm, epoch), used for the
# suptitle when no sibling wide CSV sits next to the levelsets file.
WIDE_INPUTS = {
    "pendulum_lqr": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/pendulum/lqr/gaussian_all_levels.csv",
    "cartpole_ppo": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/cartpole/safe_explorer_ppo/gaussian_all_levels.csv",
    "quad2d_rl": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/quadrotor2d/rl/quad2d_corridor_sine_ambient_all_levels.csv",
    # quad3d = 800k-pool campaign (decision 2026-09-10); 100k is obsolete
    "quad3d_ppo800k": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/quadrotor3d/ppo_800k/quad3d800k_corridor_sine_ambient_all_levels.csv",
}

# Output. Two profiles: "draft" (everything, under paper_draft) and "final"
# (the paper's reduced view, figures under paper_final). The level-set stores
# and tables are only ever written under paper_draft; final reads them.
STOCH_DIR = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic")
PAPER_DIR = STOCH_DIR / "paper_draft"
LEVELSETS_DIR = PAPER_DIR / "levelsets"
FIG_DIR = PAPER_DIR / "figures"
PROFILE = "draft"
SHOW_EPOCH_SUBTITLE = True   # draft only; the final profile carries nothing a caption should say
# The paper's evaluation budget per system (user, 2026-09-11): every arm is read
# at this epoch. "common" = the deepest epoch every plotted arm reached.
#   cartpole  epoch 11 = 1,950 trajectories (the last epoch; 300 + 150/epoch)
#   quad2d    epoch 12 = 8,000 (2,000 + 500/epoch; the cap of the loud/baseline runs)
#   quad3d    epoch  3 = 14,500 (10,000 + 1,500/epoch; the cap of the ambient runs)
#   pendulum  final common epoch (19 = 2,000)
# The final profile reads the all-epoch store so these epochs exist for every level.
FINAL_EPOCHS = {"pendulum_lqr": "common", "cartpole_ppo": 11, "quad2d_rl": 12, "quad3d_ppo800k": 3}


def set_profile(name: str) -> None:
    """Switch between the draft view and the paper's final view."""
    global PROFILE, PAPER_DIR, FIG_DIR, METRICS, LEVEL_FILTER, USE_PAPER_NAMES, SHOW_EPOCH_SUBTITLE
    global X_TICK_LABEL_EVERY, PANEL_W, LEVELSETS_DIR
    PROFILE = name
    SHOW_EPOCH_SUBTITLE = name != "final"
    X_TICK_LABEL_EVERY = X_TICK_LABEL_EVERY_FINAL if name == "final" else 1
    PANEL_W = PANEL_W_FINAL if name == "final" else 2.6
    if name == "final":
        PAPER_DIR = STOCH_DIR / "paper_final"
        FIG_DIR = PAPER_DIR / "figures"
        LEVELSETS_DIR = STOCH_DIR / "paper_draft" / "levelsets_greedy"   # every epoch
        METRICS = [("f05", "$F_{0.5}$")]
        LEVEL_FILTER = dict(FINAL_LEVEL_FILTER)
        USE_PAPER_NAMES = True
    else:
        PAPER_DIR = STOCH_DIR / "paper_draft"
        FIG_DIR = PAPER_DIR / "figures"
        LEVELSETS_DIR = PAPER_DIR / "levelsets"
        METRICS = [("f05", "$F_{0.5}$"), ("tpr", "Recall"), ("prec", "Precision")]
        LEVEL_FILTER = dict(DRAFT_LEVEL_FILTER)
        USE_PAPER_NAMES = False
FILE_MODE = 0o660
FORMATS = ("png",)   # png only (2026-09-11)

# =============================================================================
# Plumbing
# =============================================================================


def apply_aliases(df: pd.DataFrame, system: str) -> pd.DataFrame:
    """Canonicalise arm keys on a frame with `level`, `arm` (and optionally
    `train_trajectories`) columns, per ARM_ALIAS and LEVEL_BUDGET_CAP.

    For each aliased (system, level) whose frame still carries the alias
    SOURCES (the raw wide CSV): rows whose arm is a bare alias TARGET are
    dropped (the dead twins), then the sources are renamed to their targets.
    A frame with no source at that level is already canonical (the level-set
    stores, written through the scorer's --arm-alias) and is left alone.
    For each capped (system, level): rows whose train_trajectories exceed the
    cap are dropped (blank counts are kept)."""
    df = df.copy()
    for (sys_, level), table in ARM_ALIAS.items():
        if sys_ != system:
            continue
        at = df["level"] == level
        if not df.loc[at, "arm"].isin(set(table)).any():
            continue
        df = df[~(at & df["arm"].isin(set(table.values())))]
        at = df["level"] == level
        df.loc[at, "arm"] = df.loc[at, "arm"].map(lambda a: table.get(a, a))
    if "train_trajectories" in df.columns:
        traj = pd.to_numeric(df["train_trajectories"], errors="coerce")
        for (sys_, level), cap in LEVEL_BUDGET_CAP.items():
            if sys_ != system:
                continue
            df = df[~((df["level"] == level) & (traj > cap))]
            traj = traj.loc[df.index]
    return df


def _apply_rc() -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE, "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _select_set(name: str) -> None:
    """Point the module-level ARMS / SEED_ARMS / DRAW_BAND at one figure set."""
    global ARMS, SEED_ARMS, DRAW_BAND
    cfg = FIGURE_SETS[name]
    ARMS = {a: ARM_STYLES[a] for a in cfg["arms"]}
    SEED_ARMS = list(ALL_SEED_ARMS) if cfg["band"] else []
    DRAW_BAND = bool(cfg["band"])


def _plotted_arms(lvl: pd.DataFrame) -> list[str]:
    present = set(lvl["arm"].unique())
    return [a for a in list(ARMS) + SEED_ARMS if a in present]


def paper_epoch(system: str, epoch: str) -> str:
    """Resolve the epoch argument for a system: in the final profile a plain
    "common" means the paper budget in FINAL_EPOCHS; an explicit number wins."""
    if PROFILE == "final" and epoch == "common":
        return str(FINAL_EPOCHS.get(system, "common"))
    return epoch


def _pick_epoch(lvl: pd.DataFrame, epoch: str) -> int | None:
    arms = _plotted_arms(lvl)
    if not arms:
        return None
    if epoch == "common":
        return int(min(lvl[lvl["arm"] == a]["epoch"].max() for a in arms))
    return int(epoch)


def _traj_count(wide: pd.DataFrame | None, level: str, ep: int) -> int | None:
    if wide is None or "train_trajectories" not in wide.columns:
        return None
    sub = wide[(wide["level"] == level) & (wide["epoch"].astype(int) == ep)]
    sub = sub[sub["arm"].isin(list(ARMS) + SEED_ARMS)]
    if sub.empty:
        return None
    return int(sub["train_trajectories"].astype(int).median())


def _series(rows: pd.DataFrame, col: str, betas: list[float]) -> list[float]:
    m = dict(zip(rows["beta"].round(6), rows[col]))
    return [float(m.get(round(b, 6), float("nan"))) for b in betas]


def _nanstat(cols: list[list[float]], fn) -> list[float]:
    out = []
    for vals in zip(*cols):
        v = [x for x in vals if not math.isnan(x)]
        out.append(fn(v) if v else float("nan"))
    return out


def _find_wide(levelsets: Path) -> Path | None:
    stem = levelsets.stem
    for suffix in ("_levelsets_b50", "_levelsets"):
        if stem.endswith(suffix):
            cand = levelsets.with_name(stem[: -len(suffix)] + ".csv")
            if cand.exists():
                return cand
    return None


def render(system: str, levelsets: Path, epoch: str, wide: Path | None,
           suffix: str = "") -> tuple[list[Path], list[str]]:
    _apply_rc()
    df = apply_aliases(pd.read_csv(levelsets), system)
    df = df[df["beta"] >= BETA_MIN - 1e-9].copy()
    wide_df = apply_aliases(pd.read_csv(wide), system) if wide is not None else None
    levels = list(dict.fromkeys(df["level"]))   # file order
    levels = ([l for l in LEVEL_ORDER if l in levels]
              + [l for l in levels if l not in LEVEL_ORDER])
    if LEVEL_FILTER.get(system):
        levels = [l for l in levels if l in LEVEL_FILTER[system]]
    betas = sorted(df["beta"].round(6).unique())
    skipped: list[str] = []

    n_rows, n_cols = len(METRICS), len(levels)
    block_w = PANEL_W * n_cols
    fig_w = max(block_w + LEFT_MARGIN + RIGHT_MARGIN, FIG_MIN_W)
    n_entries = len(SEED_ARMS and [1]) + len(ARMS) + 2
    legend_ncol = LEGEND_NCOL or max(1, min(n_entries, int(fig_w / LEGEND_MAX_W_PER_ENTRY)))
    legend_rows = math.ceil(n_entries / legend_ncol)
    legend_h = LEGEND_ROW_H * legend_rows
    title_h = TITLE_BLOCK_H if SHOW_EPOCH_SUBTITLE else TITLE_BLOCK_H_FINAL
    fig_h = PANEL_H * n_rows + title_h + XLABEL_BLOCK_H + legend_h
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(fig_w, fig_h))
    handles: dict[str, object] = {}
    sub_lines: list[str] = []

    for j, level in enumerate(levels):
        lvl = df[df["level"] == level]
        ep = _pick_epoch(lvl, paper_epoch(system, epoch))
        if ep is None:
            skipped.append(f"{level}: no plotted arm present")
            continue
        at = lvl[lvl["epoch"].astype(int) == ep]
        by_arm = {a: at[at["arm"] == a].sort_values("beta") for a in at["arm"].unique()}
        if not by_arm:
            # A level named by the filter that has no rows at all (a campaign
            # whose runs for it have not banked an epoch yet). The `probe` below
            # would raise StopIteration and kill the whole figure, so skip the
            # column the same way an epoch-less level is skipped above.
            skipped.append(f"{level}: no rows")
            continue
        for a in list(ARMS) + SEED_ARMS:
            if a in lvl["arm"].unique() and a not in by_arm:
                skipped.append(f"{level}: {a} has no epoch {ep}")
            elif a not in lvl["arm"].unique():
                skipped.append(f"{level}: {a} absent")
        seeds = [s for s in SEED_ARMS if s in by_arm]
        fm_src = FM_ORACLE_SRC if FM_ORACLE_SRC in by_arm else next(
            (s for s in seeds), None)
        ex_src = next((a for a in EXACT_ORACLE_SRCS if a in by_arm), None)
        probe = next(iter(by_arm.values()))
        n_pos = _series(probe, "n_pos_true", betas)
        n_neg = _series(probe, "n_neg_true", betas)
        n_traj = _traj_count(wide_df, level, ep)
        sub_lines.append(f"{level_title(level, system)}: epoch {ep}"
                         + (f", {n_traj:,} traj" if n_traj is not None else ""))

        for i, (col, ylab) in enumerate(METRICS):
            ax = axes[i][j]
            ax.grid(True, **GRID)
            for sp in SPINES_OFF:
                ax.spines[sp].set_visible(False)
            # True level-set fraction as a grey area under the curves.
            if DRAW_TRUE_SET_FILL:
                frac = [p / (p + n) if (p + n) and not math.isnan(p + n) else float("nan")
                        for p, n in zip(n_pos, n_neg)]
                ax.fill_between(betas, 0.0, frac, color=TRUE_SET_FILL_COLOR,
                                alpha=TRUE_SET_FILL_ALPHA, lw=0, zorder=0)
                handles.setdefault(TRUE_SET_FILL_LABEL,
                                   Patch(facecolor=TRUE_SET_FILL_COLOR, alpha=TRUE_SET_FILL_ALPHA, lw=0))
            # Seed band + mean.
            if seeds:
                cols = [_series(by_arm[s], col, betas) for s in seeds]
                if DRAW_BAND and len(seeds) > 1:
                    ax.fill_between(betas, _nanstat(cols, min), _nanstat(cols, max),
                                    color=SEED_BAND_COLOR, alpha=SEED_BAND_ALPHA,
                                    lw=0, zorder=1)
                ax.plot(betas, _nanstat(cols, lambda v: sum(v) / len(v)),
                        color=SEED_MEAN_COLOR, lw=SEED_MEAN_LW, zorder=2)
                handles.setdefault(SEED_LABEL, (
                    Line2D([], [], color=SEED_MEAN_COLOR, lw=SEED_MEAN_LW),
                    Patch(facecolor=SEED_BAND_COLOR, alpha=SEED_BAND_ALPHA, lw=0))
                    if DRAW_BAND and len(seeds) > 1 else Line2D([], [], color=SEED_MEAN_COLOR, lw=SEED_MEAN_LW))
            # Adaptive arms.
            for arm, (label, fam, color, lw, mk) in ARMS.items():
                if arm not in by_arm:
                    continue
                ax.plot(betas, _series(by_arm[arm], col, betas), ls=FAMILY_LS[fam],
                        color=color, lw=lw, marker=mk, ms=MARKER_SIZE,
                        alpha=LINE_ALPHA, zorder=3 if arm != OURS else 4)
                handles.setdefault(label, Line2D(
                    [], [], ls=FAMILY_LS[fam], color=color, lw=lw, marker=mk, ms=MARKER_SIZE))
            # Oracles.
            if DRAW_ORACLE and fm_src is not None:
                ax.plot(betas, _series(by_arm[fm_src], f"{col}_oracle", betas),
                        zorder=5, **FM_ORACLE_STYLE)
                handles.setdefault(FM_ORACLE_LABEL, Line2D([], [], **FM_ORACLE_STYLE))
            if DRAW_ORACLE and ex_src is not None:
                ax.plot(betas, _series(by_arm[ex_src], f"{col}_oracle", betas),
                        zorder=5, **EXACT_ORACLE_STYLE)
                handles.setdefault(EXACT_ORACLE_LABEL, Line2D([], [], **EXACT_ORACLE_STYLE))
            # True-set fraction.
            if DRAW_TRUE_SET and col == TRUE_SET_ROW:
                for b, p, n in zip(betas, n_pos, n_neg):
                    tot = p + n
                    if not tot or math.isnan(tot):
                        continue
                    frac = p / tot
                    txt = f"{100 * frac:.2g}%" if frac >= 0.001 else "<0.1%"
                    ax.text(b, TRUE_SET_Y, txt, ha="center", va="bottom",
                            fontsize=TRUE_SET_FONT, color=TRUE_SET_COLOR,
                            transform=ax.get_xaxis_transform())
                if j == 0:
                    ax.text(betas[0] - X_PAD, TRUE_SET_Y, TRUE_SET_PREFIX, ha="right",
                            va="bottom", fontsize=TRUE_SET_FONT, color=TRUE_SET_COLOR,
                            transform=ax.get_xaxis_transform())
            # Axes cosmetics.
            ax.set_xlim(betas[0] - X_PAD, betas[-1] + X_PAD)
            ax.set_xticks(betas)
            ax.set_xticklabels([f"{b:.2f}" if k % X_TICK_LABEL_EVERY == 0 else ""
                                for k, b in enumerate(betas)], rotation=X_TICK_ROTATION)
            ax.set_ylim(*Y_LIM)
            ax.set_yticks(Y_TICKS)
            if i == 0:
                ax.set_title(level_title(level, system))
            if i == n_rows - 1:
                ax.set_xlabel(X_LABEL)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(ylab)
            else:
                ax.tick_params(labelleft=False)

    # Legend (shared, bottom) and titles.
    order = [v[0] for v in ARMS.values()]
    order.insert(CONTROL_POSITION, SEED_LABEL)
    order += [FM_ORACLE_LABEL, EXACT_ORACLE_LABEL, TRUE_SET_FILL_LABEL]
    keys = [k for k in order if k in handles]
    if not keys:
        # Nothing was plotted: every level was skipped (no rows, or no arm at the
        # requested epoch). fig.legend divides by the column count and raises
        # "number sections must be larger than 0", which reads like a plotting
        # bug rather than "this campaign has no data at that epoch". Hand the
        # skip list back and let the caller report it.
        plt.close(fig)
        return [], skipped
    fig.legend([handles[k] for k in keys], keys, loc="lower center",
               ncol=min(legend_ncol, len(keys)), frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.2)
    title = SYSTEM_TITLES.get(system, system)
    fig.suptitle(title, fontsize=SUPTITLE_SIZE, y=1 - 0.12 / fig_h)
    if SHOW_EPOCH_SUBTITLE:
      fig.text(0.5, 1 - 0.45 / fig_h, "   ·   ".join(sub_lines), ha="center", va="top",
             fontsize=SUPTITLE_SUB_SIZE, color="0.35")
    left_in = (fig_w - block_w - LEFT_MARGIN - RIGHT_MARGIN) / 2 + LEFT_MARGIN
    fig.subplots_adjust(left=left_in / fig_w, right=(left_in + block_w) / fig_w,
                        top=1 - title_h / fig_h,
                        bottom=(legend_h + XLABEL_BLOCK_H) / fig_h,
                        wspace=SUBPLOT_WSPACE, hspace=SUBPLOT_HSPACE)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outs = []
    for fmt in FORMATS:
        p = FIG_DIR / f"levelsets_{system}{suffix}.{fmt}"
        fig.savefig(p, dpi=DPI)
        os.chmod(p, FILE_MODE)
        outs.append(p)
    plt.close(fig)
    return outs, skipped


# Development inputs (betas 0.05..0.95; filtered to BETA_MIN) used when the
# paper b50 files are not there yet. Output gets a `_test` suffix and is deleted
# by the --test-cleanup flag.
DEV_INPUTS = {
    "pendulum_lqr": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/pendulum/lqr/gaussian_all_levels_levelsets.csv",
    "cartpole_ppo": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/cartpole/safe_explorer_ppo/gaussian_all_levels_levelsets.csv",
    "quad2d_rl": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/quadrotor2d/rl/quad2d_corridor_sine_ambient_all_levels_levelsets.csv",
    # quad3d = 800k-pool campaign (decision 2026-09-10); 100k is obsolete
    "quad3d_ppo800k": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/quadrotor3d/ppo_800k/quad3d800k_corridor_sine_ambient_all_levels_levelsets.csv",
}


# Final-profile grid: rows = systems (in this order), columns = the paper levels.
GRID_SYSTEMS = ["pendulum_lqr", "cartpole_ppo", "quad2d_rl", "quad3d_ppo800k"]
GRID_COLUMNS = ["deterministic", "low", "high"]
GRID_PANEL_W, GRID_PANEL_H = 2.35, 1.45
GRID_ROW_LABEL_W = 0.55          # inches reserved left of the first column for the system names
GRID_WSPACE, GRID_HSPACE = 0.10, 0.16
GRID_ROW_LABEL_SIZE = 8.5
# A (system, column) cell with no dataset yet is drawn as an empty panel carrying
# this text, so the grid keeps its shape until the run exists (pendulum deterministic).
GRID_PLACEHOLDER_TEXT = "pending"
GRID_PLACEHOLDER_COLOR = "#898781"


def render_grid(set_name: str, epoch: str = "common", suffix: str = "") -> list[Path]:
    """One figure per set for the paper: every system as a row, the three
    paper levels as columns, F0.5 only. A system without a level leaves that
    cell empty. Reads the same level-set stores as the per-system figures."""
    _select_set(set_name)
    _apply_rc()
    col, ylab = METRICS[0]
    frames = {}
    for system in GRID_SYSTEMS:
        path = LEVELSETS_DIR / f"{system}_levelsets_b50.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["beta"] >= BETA_MIN - 1e-9].copy()
        df = apply_aliases(df, system)
        frames[system] = df
    systems = [s for s in GRID_SYSTEMS if s in frames]
    n_rows, n_cols = len(systems), len(GRID_COLUMNS)
    legend_h = LEGEND_ROW_H + 0.12
    fig_w = GRID_ROW_LABEL_W + LEFT_MARGIN + n_cols * GRID_PANEL_W + RIGHT_MARGIN
    fig_h = n_rows * GRID_PANEL_H + 0.35 + XLABEL_BLOCK_H + legend_h
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    handles: dict[str, object] = {}
    for i, system in enumerate(systems):
        df = frames[system]
        names = dict(PAPER_LEVELS[system])            # level key -> paper name
        by_name = {v: k for k, v in names.items()}
        for j, cname in enumerate(GRID_COLUMNS):
            ax = axes[i][j]
            level = by_name.get(cname)
            ax.grid(True, **GRID)
            for sp in SPINES_OFF:
                ax.spines[sp].set_visible(False)
            placeholder = level is None or level not in set(df["level"])
            if placeholder:
                betas = [round(0.5 + 0.05 * k, 2) for k in range(10)]
                by_arm = {}
                ax.text(0.5, 0.5, GRID_PLACEHOLDER_TEXT, transform=ax.transAxes, ha="center",
                        va="center", color=GRID_PLACEHOLDER_COLOR, fontsize=FONT_SIZE)
            else:
                lvl = df[df["level"] == level]
                ep = _pick_epoch(lvl, paper_epoch(system, epoch))
                at = lvl[lvl["epoch"].astype(int) == ep]
                by_arm = {a: at[at["arm"] == a].sort_values("beta") for a in at["arm"].unique()}
                betas = sorted(at["beta"].round(6).unique())
            probe = next(iter(by_arm.values()), None)
            if DRAW_TRUE_SET_FILL and probe is not None:
                n_pos = _series(probe, "n_pos_true", betas)
                n_neg = _series(probe, "n_neg_true", betas)
                frac = [p / (p + n) if (p + n) and not math.isnan(p + n) else float("nan")
                        for p, n in zip(n_pos, n_neg)]
                ax.fill_between(betas, 0.0, frac, color=TRUE_SET_FILL_COLOR,
                                alpha=TRUE_SET_FILL_ALPHA, lw=0, zorder=0)
                handles.setdefault(TRUE_SET_FILL_LABEL,
                                   Patch(facecolor=TRUE_SET_FILL_COLOR, alpha=TRUE_SET_FILL_ALPHA, lw=0))
            seeds = [s for s in SEED_ARMS if s in by_arm]
            if seeds:
                cols = [_series(by_arm[s], col, betas) for s in seeds]
                ax.plot(betas, _nanstat(cols, lambda v: sum(v) / len(v)), color=SEED_MEAN_COLOR,
                        lw=SEED_MEAN_LW, zorder=2)
                handles.setdefault(SEED_LABEL, Line2D([], [], color=SEED_MEAN_COLOR, lw=SEED_MEAN_LW))
            for arm, (label, fam, color, lw, mk) in ARMS.items():
                if arm not in by_arm:
                    continue
                ax.plot(betas, _series(by_arm[arm], col, betas), ls=FAMILY_LS[fam], color=color,
                        lw=lw, marker=mk, ms=MARKER_SIZE, alpha=LINE_ALPHA,
                        zorder=4 if arm == OURS else 3)
                handles.setdefault(label, Line2D([], [], ls=FAMILY_LS[fam], color=color, lw=lw,
                                                 marker=mk, ms=MARKER_SIZE))
            ax.set_xlim(betas[0] - X_PAD, betas[-1] + X_PAD)
            ax.set_xticks(betas)
            ax.set_xticklabels([f"{b:.2f}" if k % X_TICK_LABEL_EVERY == 0 else ""
                                for k, b in enumerate(betas)])
            ax.set_ylim(*Y_LIM)
            ax.set_yticks(Y_TICKS)
            if i == 0:
                ax.set_title(cname)
            if i == n_rows - 1:
                ax.set_xlabel(X_LABEL)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(ylab)
            else:
                ax.tick_params(labelleft=False)
        # row label: the system, left of the row
        axes[i][0].annotate(SYSTEM_TITLES.get(system, system), xy=(0, 0.5), xycoords="axes fraction",
                       xytext=(-GRID_ROW_LABEL_W * 72 - 8, 0), textcoords="offset points",
                       rotation=90, ha="center", va="center", fontsize=GRID_ROW_LABEL_SIZE)
    order = [v[0] for v in ARMS.values()]
    order.insert(CONTROL_POSITION, SEED_LABEL)
    order.append(TRUE_SET_FILL_LABEL)
    keys = [k for k in order if k in handles]
    fig.legend([handles[k] for k in keys], keys, loc="lower center", ncol=len(keys), frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.2)
    left = (GRID_ROW_LABEL_W + LEFT_MARGIN) / fig_w
    fig.subplots_adjust(left=left, right=1 - RIGHT_MARGIN / fig_w, top=1 - 0.35 / fig_h,
                        bottom=(legend_h + XLABEL_BLOCK_H) / fig_h,
                        wspace=GRID_WSPACE, hspace=GRID_HSPACE)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outs = []
    for fmt in FORMATS:
        p = FIG_DIR / f"levelsets_{set_name}{suffix}.{fmt}"
        fig.savefig(p, dpi=DPI)
        os.chmod(p, FILE_MODE)
        outs.append(p)
    plt.close(fig)
    return outs



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levelsets", type=Path, help="long levelsets CSV for one system")
    ap.add_argument("--system", choices=list(SYSTEM_TITLES), help="system name for --levelsets")
    ap.add_argument("--wide", type=Path, help="wide CSV with train_trajectories (optional; auto-found next to --levelsets)")
    ap.add_argument("--paper", action="store_true", help=f"render all four systems from {LEVELSETS_DIR}")
    ap.add_argument("--dev", action="store_true", help="render all four from the development CSVs with a _test suffix")
    ap.add_argument("--epoch", default="common", help="'common' (deepest epoch shared by all plotted arms, per level) or an int")
    ap.add_argument("--suffix", default="", help="output filename suffix, e.g. _test")
    ap.add_argument("--profile", default="draft", choices=["draft", "final"],
                    help="draft: all metrics and levels under paper_draft; final: F0.5 only, the three named levels, under paper_final")
    ap.add_argument("--grid", action="store_true",
                    help="final layout: one figure per set, rows = systems, columns = paper levels")
    ap.add_argument("--set", default="all", choices=["all"] + list(FIGURE_SETS),
                    help="which figure set to render; 'all' renders every set as a separate figure")
    args = ap.parse_args()
    set_profile(args.profile)
    sets = list(FIGURE_SETS) if args.set == "all" else [args.set]
    if args.grid:
        for fig_set in sets:
            outs = render_grid(fig_set, args.epoch, args.suffix)
            print(f"[ok] grid ({fig_set}): " + ", ".join(str(o) for o in outs))
        return

    jobs: list[tuple[str, Path, str]] = []
    if args.paper:
        for s in SYSTEM_TITLES:
            jobs.append((s, LEVELSETS_DIR / f"{s}_levelsets_b50.csv", args.suffix))
    elif args.dev:
        for s, p in DEV_INPUTS.items():
            jobs.append((s, Path(p), args.suffix or "_test"))
    elif args.levelsets and args.system:
        jobs.append((args.system, args.levelsets, args.suffix))
    else:
        ap.error("give --paper, --dev, or both --levelsets and --system")

    for system, path, suffix in jobs:
        if not path.exists():
            print(f"[skip] {system}: {path} missing")
            continue
        wide = args.wide or _find_wide(path)
        if wide is None and system in WIDE_INPUTS and Path(WIDE_INPUTS[system]).exists():
            wide = Path(WIDE_INPUTS[system])
        for fig_set in sets:
            _select_set(fig_set)
            outs, skipped = render(system, path, args.epoch, wide, f"_{fig_set}{suffix}")
            print(f"[ok] {system} ({fig_set}): " + ", ".join(str(o) for o in outs))
            for s in skipped:
                print(f"     skipped {s}")


if __name__ == "__main__":
    main()
