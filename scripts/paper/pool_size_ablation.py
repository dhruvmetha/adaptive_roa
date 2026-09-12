#!/usr/bin/env python
"""Pool-size ablation for the quadrotor 3D PPO system.

The paper's arms, trained with an 800,000-flight trainable pool (campaign
q3d800k, the paper's quad3d campaign since the 2026-09-10 decision), against
the same arms on a 100,000-flight pool (campaign q3dppo, now the ablation).
Same controller, success criterion, ground truth and 10,000 + 1,500/epoch
schedule; only the pool differs. Every number in the table is taken at the
DEEPEST epoch present in BOTH files for that (level, arm), and the table
carries each side's max epoch so the depth of the comparison is visible.

Outputs (all chmod 0o660):
  tables/pool_size_ablation.csv   one row per (level, arm)
  tables/pool_size_ablation.md    one table per level plus a verdict summary
  figures/pool_size_ablation.{png,pdf}
      rows = KL (log y), sAUROC, skill score; columns = the four levels;
      x = training trajectories; 800k and 100k curves per arm in one colour.

Verdicts use the 800k (paper) non-adaptive FM control as the noise floor: 2 x
sd of the metric across seeds s42/s43/s44 at that epoch. A delta (100k minus
800k) inside the floor is a tie; outside it the better side wins (KL lower is
better, sAUROC and skill score higher).

Shared style (arm colours, labels, control style, typography, output
locations) is imported from the sibling `plot_levelsets_paper.py`; the STYLE
block below holds only what is specific to this figure.

Usage:
    python scripts/paper/pool_size_ablation.py
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

# Shared style from the level-set figure script (scripts/paper is not a package).
_SIB = Path(__file__).resolve().parent / "plot_levelsets_paper.py"
_spec = importlib.util.spec_from_file_location("plot_levelsets_paper", _SIB)
_ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ls)

ARM_STYLES = _ls.ARM_STYLES
FAMILY_LS = _ls.FAMILY_LS
SEED_ARMS = list(_ls.ALL_SEED_ARMS)
SEED_LABEL = _ls.SEED_LABEL
SEED_MEAN_COLOR = _ls.SEED_MEAN_COLOR
SEED_MEAN_LW = _ls.SEED_MEAN_LW
MARKER_SIZE = _ls.MARKER_SIZE
LINE_ALPHA = _ls.LINE_ALPHA
level_title = _ls.level_title
SYSTEM_TITLES = _ls.SYSTEM_TITLES
PAPER_DIR = _ls.PAPER_DIR
FIG_DIR = _ls.FIG_DIR
FILE_MODE = _ls.FILE_MODE
FORMATS = _ls.FORMATS
DPI = _ls.DPI
FONT_FAMILY = _ls.FONT_FAMILY
FONT_SIZE = _ls.FONT_SIZE
TITLE_SIZE = _ls.TITLE_SIZE
SUPTITLE_SIZE = _ls.SUPTITLE_SIZE
SUPTITLE_SUB_SIZE = _ls.SUPTITLE_SUB_SIZE
LABEL_SIZE = _ls.LABEL_SIZE
TICK_SIZE = _ls.TICK_SIZE
LEGEND_SIZE = _ls.LEGEND_SIZE
GRID = _ls.GRID
SPINES_OFF = _ls.SPINES_OFF

# =============================================================================
# STYLE: everything specific to this ablation. Nothing below is a choice.
# =============================================================================

# quad3d = 800k-pool campaign (decision 2026-09-10); 100k is obsolete
SYSTEM = "quad3d_ppo800k"
SUPTITLE = f"{SYSTEM_TITLES[SYSTEM]} — trainable pool 800k (paper) vs 100k"

# The two campaigns: pool label -> wide CSV, one row per (level, arm, epoch).
# The paper pool comes first; the ablation pool is the 100k campaign.
INPUTS = {
    "800k": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/quadrotor3d/ppo_800k/"
            "quad3d800k_corridor_sine_ambient_all_levels.csv",
    "100k": "/common/users/shared/pracsys/genMoPlan/docs/stochastic/quadrotor3d/ppo/"
            "quad3dppo_corridor_sine_ambient_all_levels.csv",
}
PAPER_POOL, ABL_POOL = "800k", "100k"

LEVELS = ["corridor_sine_ambient_f_0.00", "corridor_sine_ambient_f_0.12",
          "corridor_sine_ambient_f_0.20", "corridor_sine_ambient_f_0.40"]
DETERMINISTIC_LEVELS = {"corridor_sine_ambient_f_0.00"}
FOOTNOTE = "* deterministic level (one rollout per eval cell): KL is a clipped log-loss"

# Trajectory schedule used when `train_trajectories` is blank on a row.
SCHEDULE = (10000, 1500)          # traj = initial + epoch * step

# Arms in the table, in table order. The control key aggregates the three
# non-adaptive FM seeds (mean; 2 x sd across seeds is the floor).
# BALD arms are the top-N (greedy) runs (decision 2026-09-10); greedy_diverse is out of the paper
CONTROL_KEY = "dir00_s42"   # single-seed control (2026-09-11)
TABLE_ARMS = [
    ("epi_bald_greedy",      ARM_STYLES["epi_bald_greedy"][0]),
    ("epi_var_greedy",       ARM_STYLES["epi_var_greedy"][0]),
    ("clf_epi_bald_greedy",  ARM_STYLES["clf_epi_bald_greedy"][0]),
    ("bnn_mfvi_a1_greedy",   ARM_STYLES["bnn_mfvi_a1_greedy"][0]),
    ("partx_fix",            ARM_STYLES["partx_fix"][0]),
    (CONTROL_KEY,            SEED_LABEL),
    ("clf_dir00",            "Non-adaptive (CLF)"),
    ("bnn_a1_dir00",         "Non-adaptive (BNN)"),
]
# Arms over which the verdict summary is counted.
SUMMARY_ARMS = ["epi_bald_greedy", "epi_var_greedy", "clf_epi_bald_greedy", "bnn_mfvi_a1_greedy",
                "partx_fix", CONTROL_KEY]
# Arms drawn in the figure (legend order).
FIGURE_ARMS = ["epi_bald_greedy", "clf_epi_bald_greedy", "bnn_mfvi_a1_greedy", "partx_fix", CONTROL_KEY]

# Metrics: (column, y label, y scale, lower is better).
METRICS = [
    ("KL",          "KL ↓",          "log",    True),
    ("sAUROC",      "sAUROC ↑",      "linear", False),
    ("skill_score", "skill score ↑", "linear", False),
]
FMT_DECIMALS = 4

# Pool encoding. The two pools share the arm colour; the pool is carried by the
# line style and the marker fill: 800k (paper) solid with filled markers, 100k
# (ablation) dashed with hollow markers. FAMILY_LS (predictor-family line styles in the sibling
# figures) is deliberately NOT applied here because the line style is taken.
POOL_STYLE = {
    "800k": dict(ls="-",  fill=True,  label="800k pool (paper): solid, filled markers"),
    "100k": dict(ls="--", fill=False, label="100k pool: dashed, hollow markers"),
}
POOL_LEGEND_COLOR = "0.4"
HOLLOW_FACE = "white"
HOLLOW_MEW = 0.9                  # marker edge width for hollow markers
CONTROL_MARKER = "h"              # control has no marker in the sibling; one is needed for the fill code
CONTROL_Z, ARM_Z, OURS_Z = 2, 3, 4
OURS = "epi_bald_greedy"

# Axes.
Y_FLOOR = {"skill_score": -0.5}   # display-only; the table uses raw values
Y_CEIL = {"sAUROC": 1.0, "skill_score": 1.0}
Y_PAD_FRAC = 0.04
LOG_SUBS = (1.0, 2.0, 5.0)
LOG_TICK_FMT = "{:g}"             # KL ticks as plain decimals (0.05, 0.1, 0.2), not 2x10^-1
X_LABEL = "training trajectories"
X_PAD_FRAC = 0.03
X_TICK_BINS = 4

# Geometry (inches).
PANEL_W, PANEL_H = 2.4, 1.7
LEFT_MARGIN, RIGHT_MARGIN = 0.7, 0.1
TITLE_BLOCK_H = 0.8
SUPTITLE_Y_IN = 0.15
FOOTNOTE_Y_IN = 0.4
XLABEL_BLOCK_H = 0.5
LEGEND_ROW_H = 0.28
LEGEND_MAX_W_PER_ENTRY = 1.75
SUBPLOT_WSPACE, SUBPLOT_HSPACE = 0.12, 0.3

# Output.
TABLE_DIR = PAPER_DIR / "tables"
STEM = "pool_size_ablation"
# =============================================================================


def _apply_rc() -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE, "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _chmod(path: Path) -> None:
    try:
        os.chmod(path, FILE_MODE)
    except OSError as exc:            # file owned by someone else; the write still landed
        print(f"[warn] chmod {path}: {exc}")


def load(pool: str) -> pd.DataFrame:
    df = pd.read_csv(INPUTS[pool])
    df = df[df["level"].isin(LEVELS)].copy()
    df["epoch"] = df["epoch"].astype(int)
    blank = df["train_trajectories"].isna()
    if blank.any():
        print(f"[{pool}] {int(blank.sum())} rows without train_trajectories; using the schedule")
        df.loc[blank, "train_trajectories"] = SCHEDULE[0] + df.loc[blank, "epoch"] * SCHEDULE[1]
    df["train_trajectories"] = df["train_trajectories"].astype(int)
    return df


def aggregate(df: pd.DataFrame, pool: str, skipped: list[str]) -> pd.DataFrame:
    """One row per (level, arm, epoch) with the metric columns; the control
    seeds are pooled into CONTROL_KEY (mean over seeds, `<m>_sd` across seeds)
    at the epochs where ALL three seeds are present on this side."""
    cols = [m for m, *_ in METRICS]
    out = []
    for level in LEVELS:
        lvl = df[df["level"] == level]
        for arm, _label in TABLE_ARMS:
            if arm == CONTROL_KEY:
                seeds = lvl[lvl["arm"].isin(SEED_ARMS)]
                by_ep = seeds.groupby("epoch")
                full = [e for e, g in by_ep if set(g["arm"]) == set(SEED_ARMS)]
                n_seed_max = int(seeds["epoch"].max()) if len(seeds) else -1
                if full and max(full) < n_seed_max:
                    skipped.append(f"[{pool}] {level_title(level)} control: seeds reach epoch "
                                   f"{n_seed_max} but all three only to {max(full)}; "
                                   f"control curve stops at {max(full)}")
                if not full:
                    skipped.append(f"[{pool}] {level_title(level)} control: no epoch with all three seeds")
                    continue
                for e in sorted(full):
                    g = by_ep.get_group(e)
                    row = {"level": level, "arm": CONTROL_KEY, "epoch": e,
                           "train_trajectories": int(g["train_trajectories"].iloc[0])}
                    for m in cols:
                        row[m] = float(g[m].mean())
                        row[f"{m}_sd"] = float(g[m].std(ddof=1))
                    out.append(row)
            else:
                sub = lvl[lvl["arm"] == arm].sort_values("epoch")
                if sub.empty:
                    skipped.append(f"[{pool}] {level_title(level)} {arm}: absent")
                    continue
                for _, r in sub.iterrows():
                    row = {"level": level, "arm": arm, "epoch": int(r["epoch"]),
                           "train_trajectories": int(r["train_trajectories"])}
                    for m in cols:
                        row[m] = float(r[m])
                        row[f"{m}_sd"] = np.nan
                    out.append(row)
    return pd.DataFrame(out)


def verdict(delta: float, floor: float, lower_better: bool) -> str:
    if math.isnan(delta) or math.isnan(floor):
        return "n/a"
    if abs(delta) <= floor:
        return "tie"
    better_abl = delta < 0 if lower_better else delta > 0
    return ABL_POOL if better_abl else PAPER_POOL


def build_table(agg: dict[str, pd.DataFrame], skipped: list[str]) -> pd.DataFrame:
    base, big = agg[PAPER_POOL], agg[ABL_POOL]
    ctrl_base = base[base["arm"] == CONTROL_KEY].set_index(["level", "epoch"])
    rows = []
    for level in LEVELS:
        for arm, label in TABLE_ARMS:
            b = base[(base["level"] == level) & (base["arm"] == arm)].set_index("epoch")
            g = big[(big["level"] == level) & (big["arm"] == arm)].set_index("epoch")
            shared = sorted(set(b.index) & set(g.index))
            if not shared:
                skipped.append(f"{level_title(level)} {label}: no shared epoch "
                               f"({PAPER_POOL} rows {len(b)}, {ABL_POOL} rows {len(g)})")
                continue
            e = shared[-1]
            if b.loc[e, "train_trajectories"] != g.loc[e, "train_trajectories"]:
                skipped.append(f"{level_title(level)} {label}: train_trajectories differ at epoch {e} "
                               f"({b.loc[e, 'train_trajectories']} vs {g.loc[e, 'train_trajectories']})")
            row = {"level": level, "arm": arm, "arm_label": label, "epoch": e,
                   f"max_epoch_{PAPER_POOL}": int(b.index.max()),
                   f"max_epoch_{ABL_POOL}": int(g.index.max()),
                   "train_trajectories": int(b.loc[e, "train_trajectories"])}
            for m, _lab, _sc, lower in METRICS:
                v0, v1 = float(b.loc[e, m]), float(g.loc[e, m])
                floor = 2.0 * float(ctrl_base.loc[(level, e), f"{m}_sd"]) \
                    if (level, e) in ctrl_base.index else np.nan
                d = v1 - v0
                row[f"{m}_{PAPER_POOL}"] = v0
                row[f"{m}_{ABL_POOL}"] = v1
                row[f"{m}_delta"] = d
                row[f"{m}_2sd"] = floor
                row[f"{m}_verdict"] = verdict(d, floor, lower)
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(v: float) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{FMT_DECIMALS}f}"


def md_level_table(level: str, tab: pd.DataFrame) -> str:
    sub = tab[tab["level"] == level]
    head = ["arm", "epoch", f"max ep {PAPER_POOL}/{ABL_POOL}", "traj"]
    for m, *_ in METRICS:
        head += [f"{m} {PAPER_POOL}", f"{m} {ABL_POOL}", "Δ", "2sd floor", "verdict"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in sub.iterrows():
        cells = [r["arm_label"], str(r["epoch"]),
                 f"{r[f'max_epoch_{PAPER_POOL}']} / {r[f'max_epoch_{ABL_POOL}']}",
                 f"{int(r['train_trajectories']):,}"]
        for m, *_ in METRICS:
            cells += [fmt(r[f"{m}_{PAPER_POOL}"]), fmt(r[f"{m}_{ABL_POOL}"]),
                      f"{r[f'{m}_delta']:+.{FMT_DECIMALS}f}", fmt(r[f"{m}_2sd"]),
                      f"**{r[f'{m}_verdict']}**" if r[f"{m}_verdict"] in (PAPER_POOL, ABL_POOL)
                      else r[f"{m}_verdict"]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary_counts(tab: pd.DataFrame) -> tuple[str, dict[str, dict[str, int]]]:
    sub = tab[tab["arm"].isin(SUMMARY_ARMS)]
    outcomes = [ABL_POOL, PAPER_POOL, "tie"]
    head = ["level", "metric"] + outcomes + ["n arms"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    totals: dict[str, dict[str, int]] = {m: {o: 0 for o in outcomes} for m, *_ in METRICS}
    for level in LEVELS:
        lv = sub[sub["level"] == level]
        for m, *_ in METRICS:
            vc = lv[f"{m}_verdict"].value_counts()
            counts = {o: int(vc.get(o, 0)) for o in outcomes}
            for o in outcomes:
                totals[m][o] += counts[o]
            lines.append("| " + " | ".join([level_title(level) + ("*" if level in DETERMINISTIC_LEVELS else ""),
                                            m] + [str(counts[o]) for o in outcomes] + [str(len(lv))]) + " |")
    for m, *_ in METRICS:
        n = sum(totals[m].values())
        lines.append("| " + " | ".join(["**all levels**", m] + [str(totals[m][o]) for o in outcomes] + [str(n)]) + " |")
    return "\n".join(lines), totals


def write_md(tab: pd.DataFrame, agg: dict[str, pd.DataFrame], skipped: list[str],
             path: Path) -> dict[str, dict[str, int]]:
    summ, totals = summary_counts(tab)
    n_ep = {p: agg[p].groupby(["level", "arm"])["epoch"].max() for p in INPUTS}
    parts = [
        f"# Pool-size ablation: {SYSTEM_TITLES[SYSTEM]}, trainable pool {PAPER_POOL} (paper) vs {ABL_POOL}",
        "",
        f"The {PAPER_POOL} campaign (q3d800k) is the paper's quadrotor 3D campaign (decision "
        f"2026-09-10); the {ABL_POOL} campaign (q3dppo) is the ablation. Every number below is taken "
        "at the deepest epoch present in BOTH campaigns for that (level, arm); the `max ep` column "
        "shows how deep each side goes.",
        "",
        f"Same controller, success criterion, ground truth and schedule ({SCHEDULE[0]:,} + "
        f"{SCHEDULE[1]:,}/epoch); only the trainable pool differs ({PAPER_POOL} flights, campaign q3d800k, "
        f"vs {ABL_POOL} flights, campaign q3dppo).",
        "",
        f"Δ = {ABL_POOL} minus {PAPER_POOL}. The floor is 2 x sd across the three non-adaptive FM seeds "
        f"(s42/s43/s44) of the {PAPER_POOL} campaign at the same epoch; |Δ| inside the floor is a tie, "
        "otherwise the better side wins (KL lower is better; sAUROC and skill score higher). "
        f"`{SEED_LABEL}` is the seed mean, at the deepest epoch where all three seeds are present on "
        "both sides.",
        "",
        FOOTNOTE,
        "",
        "## Verdict summary",
        "",
        f"Counted over {len(SUMMARY_ARMS)} arms: "
        + ", ".join(dict(TABLE_ARMS)[a] for a in SUMMARY_ARMS) + ".",
        "",
        summ,
        "",
    ]
    for level in LEVELS:
        star = "*" if level in DETERMINISTIC_LEVELS else ""
        parts += [f"## {level_title(level)}{star}", "", md_level_table(level, tab), ""]
    if skipped:
        parts += ["## Skipped / caveats", ""] + [f"- {s}" for s in skipped] + [""]
    parts += ["## Inputs", ""] + [f"- {p}: `{INPUTS[p]}`" for p in INPUTS] + [""]
    path.write_text("\n".join(parts))
    _chmod(path)
    return totals


def _fmt_traj(v: float, _pos=None) -> str:
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


def _marker_kw(color: str, marker: str, pool: str) -> dict:
    st = POOL_STYLE[pool]
    kw = dict(marker=marker, ms=MARKER_SIZE, ls=st["ls"], color=color)
    if st["fill"]:
        kw.update(mfc=color, mec=color)
    else:
        kw.update(mfc=HOLLOW_FACE, mec=color, mew=HOLLOW_MEW)
    return kw


def render(agg: dict[str, pd.DataFrame], out_stem: Path) -> list[Path]:
    _apply_rc()
    n_rows, n_cols = len(METRICS), len(LEVELS)
    block_w = PANEL_W * n_cols
    fig_w = LEFT_MARGIN + block_w + RIGHT_MARGIN
    n_entries = len(FIGURE_ARMS) + len(POOL_STYLE)
    ncol = max(1, min(n_entries, int(fig_w // LEGEND_MAX_W_PER_ENTRY)))
    legend_rows = math.ceil(n_entries / ncol)
    legend_h = LEGEND_ROW_H * legend_rows + 0.15
    fig_h = TITLE_BLOCK_H + PANEL_H * n_rows + XLABEL_BLOCK_H + legend_h
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(fig_w, fig_h))

    handles: dict[str, Line2D] = {}
    x_all = pd.concat(agg.values())["train_trajectories"]
    x_lo, x_hi = int(x_all.min()), int(x_all.max())
    x_pad = (x_hi - x_lo) * X_PAD_FRAC
    for j, level in enumerate(LEVELS):
        for i, (m, ylab, yscale, _lower) in enumerate(METRICS):
            ax = axes[i][j]
            ys = []
            for arm in FIGURE_ARMS:
                if arm == CONTROL_KEY:
                    label, color, lw, marker, z = SEED_LABEL, SEED_MEAN_COLOR, SEED_MEAN_LW, CONTROL_MARKER, CONTROL_Z
                else:
                    label, _fam, color, lw, marker = ARM_STYLES[arm]
                    z = OURS_Z if arm == OURS else ARM_Z
                for pool in INPUTS:
                    d = agg[pool]
                    s = d[(d["level"] == level) & (d["arm"] == arm)].sort_values("epoch")
                    if s.empty:
                        continue
                    y = s[m].to_numpy(dtype=float)
                    ys.append(y)
                    ax.plot(s["train_trajectories"], y, lw=lw, alpha=LINE_ALPHA, zorder=z,
                            **_marker_kw(color, marker, pool))
                handles.setdefault(label, Line2D([], [], lw=lw, **_marker_kw(color, marker, PAPER_POOL)))
            if i == 0:
                ax.set_title(level_title(level) + ("*" if level in DETERMINISTIC_LEVELS else ""))
            if j == 0:
                ax.set_ylabel(ylab)
            else:
                ax.tick_params(labelleft=False)
            if i == n_rows - 1:
                ax.xaxis.set_major_formatter(FuncFormatter(_fmt_traj))
            else:
                ax.tick_params(labelbottom=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=X_TICK_BINS, integer=True))
            ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
            ax.grid(True, **GRID)
            for sp in SPINES_OFF:
                ax.spines[sp].set_visible(False)
            if yscale == "log":
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=LOG_SUBS))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: LOG_TICK_FMT.format(v)))
                ax.yaxis.set_minor_formatter(NullFormatter())
            elif ys:
                allv = np.concatenate(ys)
                allv = allv[np.isfinite(allv)]
                lo = max(float(allv.min()), Y_FLOOR.get(m, -np.inf))
                hi = min(float(allv.max()), Y_CEIL.get(m, np.inf))
                pad = (hi - lo) * Y_PAD_FRAC or 0.01
                ax.set_ylim(lo - pad, hi + pad)
        # share y within a row so the four levels are comparable
    for i, (m, _ylab, yscale, _l) in enumerate(METRICS):
        row = axes[i]
        if yscale == "log":
            lo = min(a.get_ylim()[0] for a in row)
            hi = max(a.get_ylim()[1] for a in row)
        else:
            lo = min(a.get_ylim()[0] for a in row)
            hi = max(a.get_ylim()[1] for a in row)
        for a in row:
            a.set_ylim(lo, hi)

    for pool, st in POOL_STYLE.items():
        handles[st["label"]] = Line2D([], [], lw=1.4, **_marker_kw(POOL_LEGEND_COLOR, "o", pool))
    keys = [SEED_LABEL if a == CONTROL_KEY else ARM_STYLES[a][0] for a in FIGURE_ARMS]
    keys = [k for k in keys if k in handles] + [st["label"] for st in POOL_STYLE.values()]
    fig.legend([handles[k] for k in keys], keys, loc="lower center", ncol=ncol,
               frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=2.6)
    fig.suptitle(SUPTITLE, fontsize=SUPTITLE_SIZE, y=1 - SUPTITLE_Y_IN / fig_h)
    fig.text(0.5, 1 - FOOTNOTE_Y_IN / fig_h, FOOTNOTE, ha="center", va="top",
             fontsize=SUPTITLE_SUB_SIZE, color="0.35")
    fig.supxlabel(X_LABEL, y=(legend_h + 0.08) / fig_h, fontsize=LABEL_SIZE)
    fig.subplots_adjust(left=LEFT_MARGIN / fig_w, right=(LEFT_MARGIN + block_w) / fig_w,
                        top=1 - TITLE_BLOCK_H / fig_h,
                        bottom=(legend_h + XLABEL_BLOCK_H) / fig_h,
                        wspace=SUBPLOT_WSPACE, hspace=SUBPLOT_HSPACE)
    paths = []
    for ext in FORMATS:
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=DPI)
        _chmod(p)
        paths.append(p)
    plt.close(fig)
    return paths


def main() -> None:
    skipped: list[str] = []
    agg = {pool: aggregate(load(pool), pool, skipped) for pool in INPUTS}
    tab = build_table(agg, skipped)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / f"{STEM}.csv"
    cols = ["level", "arm", "arm_label", "epoch", f"max_epoch_{PAPER_POOL}", f"max_epoch_{ABL_POOL}",
            "train_trajectories"]
    for m, *_ in METRICS:
        cols += [f"{m}_{PAPER_POOL}", f"{m}_{ABL_POOL}", f"{m}_delta", f"{m}_2sd", f"{m}_verdict"]
    tab[cols].to_csv(csv_path, index=False, float_format=f"%.{FMT_DECIMALS + 2}f")
    _chmod(csv_path)
    md_path = TABLE_DIR / f"{STEM}.md"
    totals = write_md(tab, agg, skipped, md_path)
    fig_paths = render(agg, FIG_DIR / STEM)

    print(f"wrote {csv_path}\nwrote {md_path}")
    for p in fig_paths:
        print(f"wrote {p}")
    print("\nverdict counts over", ", ".join(dict(TABLE_ARMS)[a] for a in SUMMARY_ARMS))
    for m, c in totals.items():
        print(f"  {m:<12} " + "  ".join(f"{o}={n}" for o, n in c.items()))
    print(f"\nFM-BALD per level (shared epoch; KL {PAPER_POOL} -> {ABL_POOL}):")
    for _, r in tab[tab["arm"] == OURS].iterrows():
        print(f"  {level_title(r['level']):<9} epoch {r['epoch']:>2} "
              f"(max {r[f'max_epoch_{PAPER_POOL}']}/{r[f'max_epoch_{ABL_POOL}']}) "
              f"KL {r[f'KL_{PAPER_POOL}']:.4f} -> {r[f'KL_{ABL_POOL}']:.4f} "
              f"({r['KL_delta']:+.4f}, floor {r['KL_2sd']:.4f}) {r['KL_verdict']}")
    if skipped:
        print("\nskipped / caveats:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
