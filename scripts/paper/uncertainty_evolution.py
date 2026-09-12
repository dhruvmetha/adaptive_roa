#!/usr/bin/env python
"""Paper analysis: how the acquisition's epistemic and aleatoric estimates evolve
over the adaptive epochs, for the arms that score the candidate pool.

Every scored epoch of a run writes `epoch_XXX/artifacts_v2.json`. Its
`acquisition.diagnostics` dict holds the means of the epistemic and aleatoric
terms over the scored candidate pool (`epistemic_mean`, `aleatoric_mean`) and
over the chosen batch (`*_mean_selected`). Epoch 0 is the first acquisition,
scored after training on the initial uniform draw, and carries diagnostics like
every later epoch. Uniform arms (dir00_*) only write `skipped_reason` and Part-X
writes partition-tree counts, so neither is read here; an epoch file of a scored
arm that lacks the keys (a uniform fallback) is skipped and counted.

Outputs (under PAPER_DIR):
  tables/uncertainty_evolution.csv   one row per (system, level, arm, epoch)
  tables/uncertainty_evolution.md    first-to-last change of the pool means
  figures/uncertainty_<system>.{png,pdf}
      rows = pool means / selected-batch means, columns = noise levels,
      x = training trajectories, one colour per arm, epistemic solid and
      aleatoric dashed, log y.

No model inference: everything comes from the JSON artifacts on disk.

Usage:
    python scripts/paper/uncertainty_evolution.py
    python scripts/paper/uncertainty_evolution.py --systems pendulum_lqr cartpole_ppo
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    """Import a script that is not on a package path."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SSI = _load("score_stoch_incremental", "scripts/score_stoch_incremental.py")
PLP = _load("plot_levelsets_paper", "scripts/paper/plot_levelsets_paper.py")

# =============================================================================
# STYLE  -- every design choice is here. Nothing below this block is a choice.
# =============================================================================

# System -> campaigns (column order is the LEVEL_ORDER of the sibling figure,
# then campaign order for levels it does not name).
SYSTEMS = {
    "pendulum_lqr": ["pend_low", "pend_med", "pend_high"],
    "cartpole_ppo": ["cprl_base", "cprl_low", "cprl_med", "cprl_high"],
    "quad2d_rl":    ["q2d_csbase", "q2d_cs", "q2d_csloud"],
    # quad3d = 800k-pool campaign (decision 2026-09-10); 100k is obsolete
    "quad3d_ppo800k": ["q3d800k_cs000", "q3d800k_cs012", "q3d800k_cs020", "q3d800k_cs040",
                       "q3damb_f012a03", "q3damb_f020a04", "q3damb_f040a04"],
}
LEVEL_ORDER = PLP.LEVEL_ORDER
SYSTEM_TITLES = PLP.SYSTEM_TITLES
level_title = PLP.level_title

# Run-dir name overrides, (campaign, arm) -> directory arm suffix. On q2d_csbase
# the LIVE FM runs are the `_fast` directories; the bare-name ones are cancelled
# 2-3 epoch twins (see PLP.ARM_ALIAS). Nothing in the shared tree is renamed.
RUN_DIR_ARM = {
    ("q2d_csbase", "dir00_s42"): "dir00_s42_fast",
    ("q2d_csbase", "epi_bald_greedy"): "epi_bald_greedy_fast",
    ("q2d_csbase", "epi_var_greedy"): "epi_var_greedy_fast",
}
# Campaign -> last epoch kept (budget cap: q2d_csbase stopped at 8,000
# trajectories = epoch 12; some arms have epoch dirs past it).
EPOCH_CAP = {"q2d_csbase": 12}

# Arms that score the pool, in legend order. Label / colour / linewidth /
# marker come from the sibling paper figure so the arms look the same there.
# BALD arms are the top-N (greedy) runs (decision 2026-09-10); greedy_diverse is out of the paper
# Uncertainty figures show FM-BALD only (decision 2026-09-10).
ARMS = ["epi_bald_greedy"]
OURS = PLP.OURS
ARM_STYLES = {a: PLP.ARM_STYLES[a] for a in ARMS}   # (label, family, color, lw, marker)
MARKER_SIZE = 2.6
LINE_ALPHA = 0.95

# Quantity -> line style. Same colour per arm, the dash tells the two apart.
EPI_LS, ALE_LS = "-", (0, (3.2, 1.6))
EPI_LABEL, ALE_LABEL = "epistemic (solid)", "aleatoric (dashed)"
QUANTITY_LEGEND_COLOR = "0.4"

# Rows: (y label, epistemic key, aleatoric key).
ROWS = [
    ("pool mean",           "epistemic_mean",          "aleatoric_mean"),
    ("selected-batch mean", "epistemic_mean_selected", "aleatoric_mean_selected"),
]
SUBTITLE = ("top: mean over the scored candidate pool;  "
            "bottom: mean over the batch the acquisition picked")
X_LABEL = "training trajectories"
Y_SCALE = "log"
Y_FLOOR = 1e-5            # non-positive means are drawn at this floor (log axis)
Y_LIM = (1e-4, 1.0)       # shared by every panel of every system; None -> data range
Y_PAD = 1.4               # multiplicative pad when Y_LIM is None
X_PAD = 0.03              # fraction of the x range beyond first / last epoch

# Figure geometry and typography (inches unless stated).
PANEL_W, PANEL_H = 2.6, 1.9
FIG_MIN_W = 7.0
TITLE_BLOCK_H = 0.85         # suptitle + subtitle above the column titles
SUPTITLE_Y_IN = 0.12         # inches below the top edge
SUBTITLE_Y_IN = 0.40
XLABEL_BLOCK_H = 0.45
LEGEND_ROW_H = 0.26
LEGEND_MAX_W_PER_ENTRY = 1.35   # entries per row = fig width / this; rows are balanced
LEFT_MARGIN, RIGHT_MARGIN = 0.62, 0.08
DPI = 200
FONT_FAMILY = "sans-serif"
FONT_SIZE = 8
TITLE_SIZE = 9
SUPTITLE_SIZE = 10
SUPTITLE_SUB_SIZE = 7
LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 7.5
LEGEND_NCOL = None
GRID = dict(color="0.85", lw=0.5)
SPINES_OFF = ("top", "right")
SUBPLOT_WSPACE, SUBPLOT_HSPACE = 0.12, 0.28

# Output.
PAPER_DIR = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic/paper_draft")
TABLE_DIR = PAPER_DIR / "tables"
FIG_DIR = PAPER_DIR / "figures"
CSV_OUT = TABLE_DIR / "uncertainty_evolution.csv"
MD_OUT = TABLE_DIR / "uncertainty_evolution.md"
FILE_MODE = 0o660
FORMATS = ("png",)   # png only (2026-09-11)

CSV_FIELDS = ["system", "level", "campaign", "arm", "epoch", "train_trajectories",
              "epistemic_mean", "aleatoric_mean",
              "epistemic_mean_selected", "aleatoric_mean_selected",
              "score_mode", "n_candidates", "member_sample_size",
              "sel_ratio"]   # epistemic_mean_selected / epistemic_mean

# Caveats that close the markdown note. The BNN numbers are the cartpole
# medium-noise (cprl_med) pool epistemic mean at epochs 7..11, read from the
# same artifacts this script tabulates; update them if the run is redone.
FM_CAVEAT = ("FM arms (FM-BALD, FM epi-var) estimate each member's p_m by Monte "
             "Carlo with member_sample_size = 20 draws, so their BALD carries a "
             "finite-K bias: the sampling noise inflates the apparent disagreement "
             "between members and puts a floor under the epistemic term.")
BNN_CAVEAT = ("The BNN's estimates jump between adjacent epochs (cartpole medium "
              "noise, pool epistemic mean at epochs 7..11: 0.0039 -> 0.082 -> 0.081 "
              "-> 0.0089 -> 0.047), so a first-to-last delta for BNN-BALD says "
              "little about its trend.")

# =============================================================================
# Plumbing
# =============================================================================


def _f(v) -> float:
    return float("nan") if v is None else float(v)


def run_dir(campaign: str, arm: str) -> Path:
    c = SSI.CAMPAIGNS[campaign]
    return (c.get("exp") or SSI.EXP) / f"{c['prefix']}_{RUN_DIR_ARM.get((campaign, arm), arm)}"


def read_arm(system: str, campaign: str, arm: str) -> tuple[list[dict], int]:
    """Rows for every epoch that carries acquisition diagnostics, plus the
    number of epoch files that had none (epoch 0, or a uniform fallback)."""
    level = SSI.CAMPAIGNS[campaign]["key"]
    rows, no_diag = [], 0
    for p in sorted(run_dir(campaign, arm).glob("epoch_*/artifacts_v2.json")):
        j = json.loads(p.read_text())
        if int(j["epoch"]) > EPOCH_CAP.get(campaign, 1 << 30):
            continue
        d = (j.get("acquisition") or {}).get("diagnostics") or {}
        if "epistemic_mean" not in d:
            no_diag += 1
            continue
        epi, epi_sel = _f(d.get("epistemic_mean")), _f(d.get("epistemic_mean_selected"))
        ratio = epi_sel / epi if epi and not math.isnan(epi) else float("nan")
        rows.append(dict(
            system=system, level=level, campaign=campaign, arm=arm,
            epoch=int(j["epoch"]), train_trajectories=int(j["train_trajectories"]),
            epistemic_mean=epi, aleatoric_mean=_f(d.get("aleatoric_mean")),
            epistemic_mean_selected=epi_sel,
            aleatoric_mean_selected=_f(d.get("aleatoric_mean_selected")),
            score_mode=d.get("score_mode"),
            n_candidates=d.get("n_candidates_evaluated"),
            member_sample_size=d.get("member_sample_size"),
            sel_ratio=ratio,
        ))
    rows.sort(key=lambda r: r["epoch"])
    return rows, no_diag


def _chmod(p: Path) -> None:
    os.chmod(p, FILE_MODE)


def write_csv(rows: list[dict]) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in r.items()})
    _chmod(CSV_OUT)
    return CSV_OUT


def _ordered_levels(system: str) -> list[str]:
    camps = SYSTEMS[system]
    keyed = {SSI.CAMPAIGNS[c]["key"]: c for c in camps}
    levels = [l for l in LEVEL_ORDER if l in keyed] + [l for l in keyed if l not in LEVEL_ORDER]
    return [keyed[l] for l in levels]


def _fmt(v: float) -> str:
    return "nan" if math.isnan(v) else f"{v:.2e}" if abs(v) < 1e-2 else f"{v:.3f}"


def _delta(a: float, b: float) -> str:
    if math.isnan(a) or math.isnan(b) or a <= 0:
        return "n/a"
    return f"x{b / a:.2f}"


def write_md(by_key: dict[tuple, list[dict]], systems: list[str]) -> Path:
    lines = ["# Uncertainty evolution over adaptive epochs", "",
             "Pool means of the acquisition's epistemic and aleatoric terms at the first "
             "and last scored epoch of each run, read from `epoch_XXX/artifacts_v2.json` "
             "(`acquisition.diagnostics`). Epoch 0 is the first acquisition, scored "
             "after training on the initial uniform draw. `x` is last / first. Full "
             "per-epoch values, including the selected-batch means and the "
             "selected / pool epistemic ratio (`sel_ratio`), are in "
             "`uncertainty_evolution.csv`.", ""]
    for system in systems:
        lines += [f"## {SYSTEM_TITLES.get(system, system)}", ""]
        for camp in _ordered_levels(system):
            level = SSI.CAMPAIGNS[camp]["key"]
            lines += [f"### {level_title(level)} (`{camp}`)", "",
                      "| arm | epochs | trajectories | pool epistemic first -> last | change "
                      "| pool aleatoric first -> last | change |",
                      "|---|---|---|---|---|---|---|"]
            for arm in ARMS:
                rows = by_key.get((system, camp, arm), [])
                if not rows:
                    lines.append(f"| {ARM_STYLES[arm][0]} | 0 | | | | | |")
                    continue
                a, b = rows[0], rows[-1]
                lines.append(
                    f"| {ARM_STYLES[arm][0]} | {a['epoch']}..{b['epoch']} "
                    f"| {a['train_trajectories']:,} -> {b['train_trajectories']:,} "
                    f"| {_fmt(a['epistemic_mean'])} -> {_fmt(b['epistemic_mean'])} "
                    f"| {_delta(a['epistemic_mean'], b['epistemic_mean'])} "
                    f"| {_fmt(a['aleatoric_mean'])} -> {_fmt(b['aleatoric_mean'])} "
                    f"| {_delta(a['aleatoric_mean'], b['aleatoric_mean'])} |")
            lines.append("")
    lines += ["## Caveats", "", f"- {FM_CAVEAT}", f"- {BNN_CAVEAT}", ""]
    MD_OUT.write_text("\n".join(lines))
    _chmod(MD_OUT)
    return MD_OUT


def _apply_rc() -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE, "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _floor(vals: list[float]) -> list[float]:
    return [Y_FLOOR if (math.isnan(v) or v <= 0) else v for v in vals]


def render(system: str, by_key: dict[tuple, list[dict]]) -> tuple[list[Path], list[str]]:
    _apply_rc()
    camps = _ordered_levels(system)
    skipped: list[str] = []
    n_rows, n_cols = len(ROWS), len(camps)
    block_w = PANEL_W * n_cols
    fig_w = max(block_w + LEFT_MARGIN + RIGHT_MARGIN, FIG_MIN_W)
    n_entries = len(ARMS) + 2
    max_ncol = LEGEND_NCOL or max(1, min(n_entries, int(fig_w / LEGEND_MAX_W_PER_ENTRY)))
    legend_rows = math.ceil(n_entries / max_ncol)
    legend_ncol = math.ceil(n_entries / legend_rows)   # balanced rows, no orphan entry
    legend_h = LEGEND_ROW_H * legend_rows
    fig_h = PANEL_H * n_rows + TITLE_BLOCK_H + XLABEL_BLOCK_H + legend_h
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(fig_w, fig_h))

    # Shared y range across the whole figure.
    if Y_LIM is None:
        vals = [v for c in camps for a in ARMS for r in by_key.get((system, c, a), [])
                for _, ek, ak in ROWS for v in _floor([r[ek], r[ak]])]
        y_lim = (min(vals) / Y_PAD, max(vals) * Y_PAD) if vals else (Y_FLOOR, 1.0)
    else:
        y_lim = Y_LIM

    for j, camp in enumerate(camps):
        level = SSI.CAMPAIGNS[camp]["key"]
        xs_all = [r["train_trajectories"] for a in ARMS for r in by_key.get((system, camp, a), [])]
        if not xs_all:
            skipped.append(f"{system}/{camp}: no arm has diagnostics")
        for i, (ylab, ek, ak) in enumerate(ROWS):
            ax = axes[i][j]
            ax.grid(True, **GRID)
            for sp in SPINES_OFF:
                ax.spines[sp].set_visible(False)
            for arm in ARMS:
                rows = by_key.get((system, camp, arm), [])
                if not rows:
                    if i == 0:
                        skipped.append(f"{system}/{camp}: {arm} absent")
                    continue
                label, _fam, color, lw, mk = ARM_STYLES[arm]
                xs = [r["train_trajectories"] for r in rows]
                z = 4 if arm == OURS else 3
                ax.plot(xs, _floor([r[ek] for r in rows]), ls=EPI_LS, color=color, lw=lw,
                        marker=mk, ms=MARKER_SIZE, alpha=LINE_ALPHA, zorder=z)
                ax.plot(xs, _floor([r[ak] for r in rows]), ls=ALE_LS, color=color, lw=lw,
                        marker=mk, ms=MARKER_SIZE, alpha=LINE_ALPHA, zorder=z,
                        markerfacecolor="white")
            ax.set_yscale(Y_SCALE)
            ax.set_ylim(*y_lim)
            if xs_all:
                lo, hi = min(xs_all), max(xs_all)
                pad = X_PAD * max(hi - lo, 1)
                ax.set_xlim(lo - pad, hi + pad)
            if i == 0:
                ax.set_title(level_title(level))
            if i == n_rows - 1:
                ax.set_xlabel(X_LABEL)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(ylab)
            else:
                ax.tick_params(labelleft=False)

    handles = [Line2D([], [], ls=EPI_LS, color=c, lw=lw, marker=mk, ms=MARKER_SIZE)
               for (_l, _f_, c, lw, mk) in ARM_STYLES.values()]
    labels = [s[0] for s in ARM_STYLES.values()]
    handles += [Line2D([], [], ls=EPI_LS, color=QUANTITY_LEGEND_COLOR, lw=1.6),
                Line2D([], [], ls=ALE_LS, color=QUANTITY_LEGEND_COLOR, lw=1.6)]
    labels += [EPI_LABEL, ALE_LABEL]
    fig.legend(handles, labels, loc="lower center", ncol=legend_ncol,
               frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.2)
    fig.suptitle(SYSTEM_TITLES.get(system, system), fontsize=SUPTITLE_SIZE,
                 y=1 - SUPTITLE_Y_IN / fig_h)
    fig.text(0.5, 1 - SUBTITLE_Y_IN / fig_h, SUBTITLE, ha="center", va="top",
             fontsize=SUPTITLE_SUB_SIZE, color="0.35")
    left_in = (fig_w - block_w - LEFT_MARGIN - RIGHT_MARGIN) / 2 + LEFT_MARGIN
    fig.subplots_adjust(left=left_in / fig_w, right=(left_in + block_w) / fig_w,
                        top=1 - TITLE_BLOCK_H / fig_h,
                        bottom=(legend_h + XLABEL_BLOCK_H) / fig_h,
                        wspace=SUBPLOT_WSPACE, hspace=SUBPLOT_HSPACE)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outs = []
    for fmt in FORMATS:
        p = FIG_DIR / f"uncertainty_{system}.{fmt}"
        fig.savefig(p, dpi=DPI)
        _chmod(p)
        outs.append(p)
    plt.close(fig)
    return outs, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="*", choices=list(SYSTEMS), default=list(SYSTEMS))
    args = ap.parse_args()

    by_key: dict[tuple, list[dict]] = {}
    all_rows: list[dict] = []
    for system in args.systems:
        for camp in _ordered_levels(system):
            for arm in ARMS:
                d = run_dir(camp, arm)
                if not d.exists():
                    print(f"[skip] {camp:13s} {arm:13s} no run dir {d}")
                    continue
                rows, no_diag = read_arm(system, camp, arm)
                by_key[(system, camp, arm)] = rows
                all_rows += rows
                print(f"[read] {camp:13s} {arm:13s} {len(rows):3d} epochs with diagnostics"
                      f" ({no_diag} without)  {d}")

    print(f"[ok] {write_csv(all_rows)}  ({len(all_rows)} rows)")
    print(f"[ok] {write_md(by_key, args.systems)}")
    for system in args.systems:
        outs, skipped = render(system, by_key)
        print(f"[ok] {system}: " + ", ".join(str(o) for o in outs))
        for s in skipped:
            print(f"     skipped {s}")


if __name__ == "__main__":
    main()
