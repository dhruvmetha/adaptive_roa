#!/usr/bin/env python
"""Ground-truth vs predicted p(success) maps for the stochastic pendulum runs.

For each noise level, reads the newest epoch's full_roa_per_point.npz from the
adaptive_pendulum_stoch_{level} experiment and the corresponding test_set.txt
(3-col: theta, theta_dot, p_success), and renders a 3-panel figure:
ground truth | predicted | difference.

Usage: python docs/plots/plot_stoch_prob_maps.py [--out DIR] [--levels low med ...]
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/common/users/shared/pracsys/adaptive_roa_experiments"
DATA = "/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum/lqr"
LEVELS = ["low", "med", "high", "xhigh"]


def plot_level(level: str, out_dir: Path) -> Path:
    files = sorted(glob.glob(
        f"{EXP}/adaptive_pendulum_stoch_{level}/outputs/*/*/epoch_*/full_roa_per_point.npz"))
    f = files[-1]
    epoch = int(re.search(r"epoch_(\d+)", f).group(1))
    z = np.load(f)
    X, p_pred = z["start_states"], z["p_success"]
    gt = np.loadtxt(f"{DATA}/{level}/test_set.txt", delimiter=",")
    assert np.allclose(gt[:, :2], X, atol=1e-5), f"{level}: test_set order mismatch"
    p_true = gt[:, 2]
    diff = p_pred - p_true

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), constrained_layout=True)
    common = dict(s=2.5, marker="s", linewidths=0)
    panels = [
        (axes[0], p_true, "Blues", 0, 1, "Ground truth  p(success)"),
        (axes[1], p_pred, "Blues", 0, 1, "Predicted  p(success)"),
        (axes[2], diff, "RdBu_r", -1, 1, "Predicted − ground truth"),
    ]
    for ax, c, cmap, vmin, vmax, title in panels:
        sc = ax.scatter(X[:, 0], X[:, 1], c=c, cmap=cmap, vmin=vmin, vmax=vmax, **common)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("θ (rad)")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-2 * np.pi, 2 * np.pi)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(colors="0.35")
        ax.grid(alpha=0.15)
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.outline.set_visible(False)
    axes[0].set_ylabel("θ̇ (rad/s)")
    mae = np.abs(diff).mean()
    fig.suptitle(
        f"Stochastic pendulum ({level} noise) — test set, epoch {epoch}   |   MAE = {mae:.3f}",
        fontsize=12.5)
    out = out_dir / f"prob_success_{level}_epoch{epoch:03d}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out, f"MAE={mae:.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/plots/stoch_prob_maps")
    ap.add_argument("--levels", nargs="*", default=LEVELS)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for level in args.levels:
        plot_level(level, out_dir)


if __name__ == "__main__":
    main()
