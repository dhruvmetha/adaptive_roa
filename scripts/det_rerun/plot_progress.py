#!/usr/bin/env python
"""Per-setting progress plots of the det-paper rerun, read from each cell's epoch artifacts.

One figure per setting (system x total trajectory budget): rows are predictors (and
Part-X), columns are metrics, lines are acquisition strategies, x is the training
trajectory count. Cells still running are drawn dashed. q3d30 and q3d60 have no
uniform cell of their own; their uniform line is the q3d110 uniform cell cut at the
setting's budget (same seed, initial batch and eval set).

Metrics, all on the held-out eval grid with the run's lambda*/delta* rule:
  KL      KL(true outcome || predicted p_success) in nats. Labels are deterministic,
          so this equals the log score (mean negative log-likelihood).
  F1      success F1 over committed (non-abstained) states
  Unc %   invalid + outcome-uncertain share of all states
  Invalid %, Outcome-unc %   the two parts of Unc %
  FPR, FNR   over committed states: fp / (fp + tn), fn / (fn + tp)

Writes progress_<setting>.png under OUT and prints the settings whose data changed
since the last run (state in OUT/.progress_state.json).  Usage: plot_progress.py [--all]
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path("/common/home/st1122/Projects/adaptive_roa/slurm_logs/det_rerun")
OUT = HERE / "plots"
SETTINGS = ["pen", "cp", "cpppo", "q2d", "q2dc", "q3d30", "q3d60", "q3d110", "q3drl40", "q3drl100"]
TITLES = {"pen": "Pendulum", "cp": "Cartpole", "q2d": "Quadrotor 2D", "q3d30": "Quadrotor 3D, 30k budget",
          "q3d60": "Quadrotor 3D, 60k budget", "q3d110": "Quadrotor 3D, 110k budget",
          "cpppo": "Cartpole RL, 3k budget", "q2dc": "Quadrotor 2D corridor, 15k budget", "q3drl40": "Quadrotor 3D RL, 40k budget", "q3drl100": "Quadrotor 3D RL, 100k budget"}
ROWS = [("fm_det", "FM (ours)"), ("det_mlp_clf", "MLP classifier"), ("det_bnn_clf", "BNN classifier"),
        ("det_mlp_reg", "MLP final-state"), ("det_bnn_reg", "BNN final-state"), ("partx_faithful", "Part-X")]
ACQ = [("uniform", "uniform", "#2a78d6", "o"), ("direct", "direct (uniform top-up)", "#eb6834", "s"),
       ("ranked", "ranked", "#1baf7a", "^"), ("entropy", "entropy", "#eda100", "D"),
       ("builtin", "Part-X built-in", "#6250d6", "v")]
METRICS = ["KL (nats)", "F1", "Unc %", "Invalid %", "Outcome-unc %", "FPR", "FNR"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _rate(a, b):
    return a / (a + b) if (a + b) else math.nan


def metrics(art: dict) -> list[float]:
    m = art.get("eval_metrics") or {}
    ld, tf = m.get("lambda_delta") or {}, m.get("threshold_free") or {}
    tp, tn, fp, fn = (ld.get(k, 0) for k in ("tp", "tn", "fp", "fn"))
    inv, unc = ld.get("invalid_pct", math.nan), ld.get("uncertain_pct", math.nan)
    return [tf.get("log_score", math.nan), ld.get("f1", math.nan), 100 * (inv + unc), 100 * inv, 100 * unc,
            _rate(fp, tn), _rate(fn, tp)]


def load() -> dict:
    """{setting: {(arm, mode): (xs, [metric series], finished)}}"""
    rows = [r for r in csv.DictReader(open(HERE / "manifest.tsv"), delimiter="\t")
            if r["name"].startswith("dr_") and "startpool" not in r["name"]]
    data = {s: {} for s in SETTINGS}
    for r in rows:
        out = re.search(r"output_dir=(\S+)", r["overrides"]).group(1)
        m = re.search(r"(?:^| )n_epochs=(\d+)", r["overrides"])
        n_ep = int(m.group(1)) if m else 11
        arts = []
        for p in sorted(glob.glob(f"{out}/epoch_*/artifacts_v2.json")):
            try:
                arts.append(json.load(open(p)))
            except (OSError, json.JSONDecodeError):
                break      # an artifact mid-write: plot up to the last complete one
        if not arts or r["system"] not in data:
            continue
        xs = [a["train_trajectories"] / 1000 for a in arts]
        ys = list(zip(*[metrics(a) for a in arts]))
        data[r["system"]][(r["arm"], r["mode"])] = (xs, ys, len(arts) >= n_ep)
    return fill_shared_uniform(data)


def fill_shared_uniform(data: dict) -> dict:
    """q3d30 / q3d60 uniform from the q3d110 uniform run cut at the budget, only where the
    schedule's own uniform cell has not banked an epoch yet (D24 withdrawn 2026-11-01)."""
    for s, cap in (("q3d30", 30), ("q3d60", 60)):
        u = {k: v for k, v in data["q3d110"].items() if k[1] == "uniform"}
        for (arm, mode), (xs, ys, fin) in u.items():
            if (arm, mode) in data[s]:
                continue
            keep = [i for i, x in enumerate(xs) if x <= cap + 1e-9]
            if keep:
                data[s][(arm, mode)] = ([xs[i] for i in keep], [[y[i] for i in keep] for y in ys],
                                        fin or xs[-1] >= cap)
    return data


def plot(setting: str, cells: dict, path: Path) -> None:
    fig, axes = plt.subplots(len(ROWS), len(METRICS), figsize=(22, 15), sharex=True, squeeze=False)
    n_done = sum(1 for v in cells.values() if v[2])
    for i, (arm, rlabel) in enumerate(ROWS):
        for j, mlabel in enumerate(METRICS):
            ax = axes[i][j]
            for mode, _, color, marker in ACQ:
                if (arm, mode) not in cells:
                    continue
                xs, ys, fin = cells[(arm, mode)]
                ax.plot(xs, ys[j], color=color, lw=1.5, ls="-" if fin else "--", marker=marker, ms=3.5,
                        mfc=color if fin else "white", mew=1.0)
            ax.grid(True, color=GRID, lw=0.6)
            ax.tick_params(colors=MUTED, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if i == 0:
                ax.set_title(mlabel, fontsize=11, color=INK)
            if j == 0:
                ax.set_ylabel(rlabel, fontsize=10, color=INK)
            if i == len(ROWS) - 1:
                ax.set_xlabel("training trajectories (k)", fontsize=9, color=MUTED)
    handles = [Line2D([], [], color=c, marker=mk, lw=1.5, ms=5, label=lab) for _, lab, c, mk in ACQ]
    handles += [Line2D([], [], color=MUTED, lw=1.5, ls="--", marker="o", mfc="white", ms=5, label="still running")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.965))
    note = ("uniform lines without their own cell yet are the q3d110 uniform cell cut at this budget; "
            if setting in ("q3d30", "q3d60") else "")
    fig.suptitle(f"{TITLES[setting]}: {n_done} of {len(cells)} cells finished (seed 99512)", fontsize=14,
                 color=INK, y=0.995)
    fig.text(0.5, 0.005, note + "KL = KL(true outcome || predicted p_success), equal to the log score for "
             "deterministic labels; FPR and FNR over committed states", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.015, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


# One adaptive regime for every method: FM's best (user ruling 2026-10-31). Chosen by
# mean KL over adaptive iterations, ranked within each setting and averaged over the six
# settings on 2026-10-31 14:5x: ranked 1.83, entropy 2.00, direct 2.17 (a narrow margin).
COMPARE_REGIME = "ranked"
COMPARE_LINES = [("fm_det", COMPARE_REGIME, "FM (ours), ranked", "#2a78d6", "-", 2.6),
                 ("fm_det", "uniform", "FM (ours), uniform", "#2a78d6", "--", 1.8),
                 ("det_mlp_clf", COMPARE_REGIME, "MLP classifier, ranked", "#eb6834", "-", 1.4),
                 ("det_bnn_clf", COMPARE_REGIME, "BNN classifier, ranked", "#1baf7a", "-", 1.4),
                 ("det_mlp_reg", COMPARE_REGIME, "MLP final-state, ranked", "#eda100", "-", 1.4),
                 ("det_bnn_reg", COMPARE_REGIME, "BNN final-state, ranked", "#e87ba4", "-", 1.4),
                 ("partx_faithful", "builtin", "Part-X", "#6250d6", "-", 1.4)]
COMPARE_TITLE = ("FM (ours) against each baseline on one adaptive regime (ranked, FM's best by KL); "
                 "FM uniform for reference (seed 99512)")
# FM alone across acquisition regimes, colours as in the per-setting progress figures.
FM_REGIME_LINES = [("fm_det", "uniform", "FM, uniform (non-adaptive)", "#2a78d6", "--", 1.8),
                   ("fm_det", "direct", "FM, direct (uniform top-up)", "#eb6834", "-", 1.6),
                   ("fm_det", "ranked", "FM, ranked", "#1baf7a", "-", 2.2),
                   ("fm_det", "entropy", "FM, entropy", "#eda100", "-", 1.6)]
FM_REGIME_TITLE = "FM (ours): adaptive acquisition regimes against non-adaptive uniform (seed 99512)"
# --compare renders both; the user wants the pair on every update (2026-11-01).
COMPARE_FIGURES = [("compare", f"compare_{COMPARE_REGIME}.png", COMPARE_LINES, COMPARE_TITLE),
                   ("compare_fm_regimes", "compare_fm_regimes.png", FM_REGIME_LINES, FM_REGIME_TITLE)]


def plot_compare(data: dict, path: Path, lines=COMPARE_LINES, title=COMPARE_TITLE) -> None:
    """All settings in one figure: rows are settings, columns metrics, one line per entry of `lines`."""
    fig, axes = plt.subplots(len(SETTINGS), len(METRICS), figsize=(24, 3.2 * len(SETTINGS)), squeeze=False)
    for i, s in enumerate(SETTINGS):
        for j, mlabel in enumerate(METRICS):
            ax = axes[i][j]
            inner, px = [], None
            for arm, mode, _, color, ls, lw in lines:
                if (arm, mode) not in data[s]:
                    continue
                xs, ys, fin = data[s][(arm, mode)]
                ax.plot(xs, ys[j], color=color, ls=ls, lw=lw, marker="o", ms=3.2,
                        mfc=color if fin else "white", mew=1.0, zorder=3 if arm == "fm_det" else 2)
                vals = [v for v in ys[j] if v is not None and math.isfinite(v)]
                if arm == "partx_faithful":
                    px = vals
                else:
                    inner += vals
            if j == 0:
                ax.set_yscale("log")
            # Scale to the learned methods; Part-X is drawn but labelled when it falls outside.
            if inner:
                lo, hi = min(inner), max(inner)
                if j == 0:
                    lo, hi = lo / 1.15, hi * 1.15
                else:
                    pad = 0.06 * (hi - lo or abs(hi) or 1.0)
                    lo, hi = lo - pad, hi + pad
                ax.set_ylim(lo, hi)
                if px and (max(px) > hi or min(px) < lo):
                    med = sorted(px)[len(px) // 2]
                    ax.text(0.98, 0.95 if med > hi else 0.05, f"Part-X off scale (median {med:.3g})",
                            transform=ax.transAxes, ha="right", va="top" if med > hi else "bottom",
                            fontsize=7.5, color="#6250d6")
            ax.grid(True, color=GRID, lw=0.6)
            ax.tick_params(colors=MUTED, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if i == 0:
                ax.set_title(mlabel + (" (log scale)" if j == 0 else ""), fontsize=12, color=INK)
            if j == 0:
                ax.set_ylabel(TITLES[s], fontsize=10, color=INK)
            if i == len(SETTINGS) - 1 or j == 0:
                ax.set_xlabel("training trajectories (k)", fontsize=8, color=MUTED)
    handles = [Line2D([], [], color=c, ls=ls, lw=lw, marker="o", ms=4, label=lab)
               for _, _, lab, c, ls, lw in lines]
    handles += [Line2D([], [], color=MUTED, lw=1.2, marker="o", mfc="white", ms=5, label="hollow: still running")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(title, fontsize=14, color=INK, y=0.997)
    fig.text(0.5, 0.004, "q3d30 and q3d60 uniform: their own cells once banked, until then the q3d110 uniform cell cut at the budget. KL = KL(true outcome || "
             "predicted p_success), the log score for deterministic labels. FPR and FNR over committed states.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.012, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="re-render every figure")
    ap.add_argument("--compare", action="store_true",
                    help="only the two comparison figures (methods on one regime; FM across regimes)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    state_f = OUT / ".progress_state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    data = load()
    if a.compare:
        hashes, stale = {}, False
        for key, name, lines, _ in COMPARE_FIGURES:
            keys = {(arm, mode) for arm, mode, *_ in lines}
            sub = {s: {k: v for k, v in data[s].items() if k in keys} for s in SETTINGS}
            hashes[key] = hashlib.sha1(json.dumps(sorted((s, str(k), v) for s in SETTINGS for k, v in sub[s].items()),
                                                  default=str).encode()).hexdigest()
            stale |= state.get(key) != hashes[key] or not (OUT / name).exists()
        if a.all or stale:       # either changed: re-render and print both, so they go out as a pair
            for key, name, lines, title in COMPARE_FIGURES:
                plot_compare(data, OUT / name, lines, title)
                state[key] = hashes[key]
            state_f.write_text(json.dumps(state))
            for _, name, _, _ in COMPARE_FIGURES:
                print(OUT / name)
        return
    changed = []
    for s in SETTINGS:
        h = hashlib.sha1(json.dumps(sorted((str(k), v) for k, v in data[s].items()), default=str).encode()).hexdigest()
        path = OUT / f"progress_{s}.png"
        if a.all or state.get(s) != h or not path.exists():
            plot(s, data[s], path)
            state[s] = h
            changed.append(str(path))
    state_f.write_text(json.dumps(state))
    for p in changed:
        print(p)


if __name__ == "__main__":
    main()
