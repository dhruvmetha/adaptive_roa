#!/usr/bin/env python3
"""Plot p_success heatmaps for the eval grid of each noise-level dataset.

Each dataset directory holds ``eval_success_prob.npz`` with a dense grid of
per-cell success probabilities over (theta, theta_dot).  This renders one panel
per noise level on a shared 0-1 color scale, plus a standalone figure per level.

Usage:
    python scripts/plot_eval_prob_heatmaps.py \
        --root /common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum/lqr
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_ROOT = Path(
    "/common/users/shared/pracsys/genMoPlan/data_trajectories/noisy/pendulum/lqr"
)
DEFAULT_ORDER = ["low", "med", "high", "xhigh"]

# Matches the model-side ROA heatmaps in adaptive_v2/eval/full_roa.py so the
# ground truth can be read side by side with predictions: diverging about the
# p = 0.5 decision boundary, red = failure, blue = success.
CMAP = "RdYlBu"


def load_grid(dataset_dir: Path):
    """Return (p_grid, theta_edges, theta_dot_edges, meta) for one dataset."""
    z = np.load(dataset_dir / "eval_success_prob.npz")
    shape = tuple(int(v) for v in z["grid_shape"])
    theta = z["grid_theta"]
    theta_dot = z["grid_theta_dot"]

    # Sanity-check the flat ordering before trusting the reshape.
    starts = z["starts"].reshape(*shape, 2)
    if not np.allclose(starts[:, :, 0], theta[:, None]) or not np.allclose(
        starts[:, :, 1], theta_dot[None, :]
    ):
        raise ValueError(f"{dataset_dir}: starts do not match grid_theta x grid_theta_dot")

    p_grid = z["p_success"].reshape(shape)

    meta = {"n_batches": int(z["n_batches"]), "trials": int(np.median(z["trials"]))}
    desc_path = dataset_dir / "eval_description.json"
    if desc_path.exists():
        desc = json.loads(desc_path.read_text())
        meta.update(
            noise=desc.get("noise"),
            success_rate=desc.get("success_rate"),
            mean_se=desc.get("mean_se"),
            converged=desc.get("converged"),
        )

    # Cell centers -> edges, so pcolormesh/imshow covers the true sampled area.
    def edges(centers):
        step = float(np.median(np.diff(centers)))
        return np.concatenate([centers - step / 2, [centers[-1] + step / 2]])

    return p_grid, edges(theta), edges(theta_dot), meta


def style_axes(ax, theta_edges, theta_dot_edges, xlabel=True, ylabel=True):
    ax.set_xlim(theta_edges[0], theta_edges[-1])
    ax.set_ylim(theta_dot_edges[0], theta_dot_edges[-1])
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
    ax.set_yticklabels([r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])
    ax.tick_params(labelsize=8, length=3, width=0.6, color="#9a9a9a")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#c8c8c8")
    if xlabel:
        ax.set_xlabel(r"$\theta$", fontsize=10)
    if ylabel:
        ax.set_ylabel(r"$\dot{\theta}$", fontsize=10)


def draw(ax, p_grid, theta_edges, theta_dot_edges):
    return ax.pcolormesh(
        theta_edges,
        theta_dot_edges,
        p_grid.T,  # (theta, theta_dot) -> rows are theta_dot for pcolormesh
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        rasterized=True,
    )


def panel_title(name, meta):
    bits = []
    rate = meta.get("success_rate")
    if rate is not None:
        bits.append(f"mean p = {rate:.3f}")
    se = meta.get("mean_se")
    if se is not None:
        bits.append(f"SE = {se:.3f}")
    if meta.get("converged") is False:
        bits.append("not converged")
    return name, "   ".join(bits)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="override output dir; by default each figure is written next to its dataset",
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="subdirectory names (default: low med high xhigh, then any others found)",
    )
    args = ap.parse_args()

    root = args.root
    if args.datasets:
        names = args.datasets
    else:
        found = {d.name for d in root.iterdir() if (d / "eval_success_prob.npz").exists()}
        names = [n for n in DEFAULT_ORDER if n in found]
        names += sorted(found - set(names))
    if not names:
        raise SystemExit(f"no datasets with eval_success_prob.npz under {root}")

    # By default each dataset's figure lands in its own directory and the
    # combined panel in the controller root; --out redirects everything.
    combined_dir = args.out or root
    combined_dir.mkdir(parents=True, exist_ok=True)

    def dataset_out(name):
        return args.out or (root / name)

    loaded = []
    for name in names:
        p_grid, te, tde, meta = load_grid(root / name)
        loaded.append((name, p_grid, te, tde, meta))
        print(
            f"{name:>6}: grid {p_grid.shape}  mean p={p_grid.mean():.4f}  "
            f"p=0 {np.mean(p_grid == 0):.1%}  p=1 {np.mean(p_grid == 1):.1%}  "
            f"interior {np.mean((p_grid > 0) & (p_grid < 1)):.1%}  "
            f"batches={meta['n_batches']}"
        )

    # ── Combined panel figure ────────────────────────────────────────────────
    n = len(loaded)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n + 0.9, 3.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    mesh = None
    for i, (name, p_grid, te, tde, meta) in enumerate(loaded):
        ax = axes[i]
        mesh = draw(ax, p_grid, te, tde)
        style_axes(ax, te, tde, ylabel=(i == 0))
        title, sub = panel_title(name, meta)
        ax.set_title(title, fontsize=11, pad=20)
        ax.text(
            0.5, 1.012, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.5, color="#6b6b6b",
        )

    cbar = fig.colorbar(mesh, ax=axes.tolist(), fraction=0.03, pad=0.015)
    cbar.set_label("p(success)", fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=3, width=0.6)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_color("#c8c8c8")
    fig.suptitle(
        f"Eval-grid success probability — {root.parent.name}/{root.name}",
        fontsize=12,
    )

    for ext in ("png", "pdf"):
        path = combined_dir / f"eval_prob_heatmaps.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)

    # ── One standalone figure per dataset ────────────────────────────────────
    for name, p_grid, te, tde, meta in loaded:
        fig, ax = plt.subplots(figsize=(4.4, 3.6), constrained_layout=True)
        mesh = draw(ax, p_grid, te, tde)
        style_axes(ax, te, tde)
        title, sub = panel_title(name, meta)
        ax.set_title(f"{title}   ({sub})", fontsize=10)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("p(success)", fontsize=9)
        cbar.ax.tick_params(labelsize=8, length=3, width=0.6)
        cbar.outline.set_linewidth(0.6)
        cbar.outline.set_color("#c8c8c8")
        target = dataset_out(name)
        target.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            path = target / f"eval_prob_heatmap{'_' + name if args.out else ''}.{ext}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Saved {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
