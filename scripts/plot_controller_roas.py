"""ROA visualization: ground-truth vs predicted for each controller + compositions.

Figure 1 (controllers_roa.png): rows = GT / predicted, cols = rl, rl-weak, lqr.
Figure 2 (compositions_roa.png): rows = GT / predicted, cols = rl->lqr, rl-weak->lqr.
Metrics (roa_metrics.csv): one row per model, threshold lambda* = max-F1 on cal set.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from adaptive_roa.utils.env_config import get_data_dir

PEND = Path(get_data_dir()) / "deterministic" / "pendulum"
OUT = Path(__file__).resolve().parent.parent / "docs" / "pendulum_roa_eval_2026-07-24"
RUNS = {
    "rl": Path("/common/users/shared/pracsys/adaptive_roa_experiments/adaptive_pendulum_rl_dhruv/outputs/training_index_0_warm_start_False_adapt_iter_19/2026-07-23_14-49-19"),
    "rl-weak": Path("/common/users/shared/pracsys/adaptive_roa_experiments/adaptive_pendulum_rl-weak_dhruv/outputs/training_index_0_warm_start_False_adapt_iter_19/2026-07-23_14-49-19"),
}

C_SUCCESS, C_FAILURE = "#4269d0", "#d4740c"  # validated colorblind-safe pair
C_UNCERTAIN = "#b3b1aa"  # neutral gray for the lambda+/-delta band
INK = "#3a3a38"


def pick_lambda(p, y):
    best_lam, best_f1 = 0.5, -1.0
    for lam in np.linspace(0.0, 1.0, 201):
        pred = p >= lam
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_f1, best_lam = f1, lam
    return best_lam


def metrics(p, y, lam):
    pred = (p >= lam).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) if n_pos and n_neg else np.nan
    acc05 = float(((p >= 0.5).astype(int) == y).mean())
    return dict(lambda_star=lam, n=len(y), base_success_rate=float(y.mean()),
                accuracy=(tp + tn) / len(y), f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
                auc=float(auc), accuracy_at_0p5=acc05, tp=tp, fp=fp, tn=tn, fn=fn)


def grid_labels(dataset, col):
    ev = np.loadtxt(PEND / dataset / "eval_states.txt", delimiter=",")
    return ev[:, :2], ev[:, col].astype(int)


def load_model_points(name):
    """Return (X, p, y) over the full grid (cal + test) and the cal-only (p, y)."""
    if name in ("rl", "rl-weak"):  # G1 task: rl_success labels
        ev = np.loadtxt(PEND / name / "eval_states.txt", delimiter=",")
        state2y = {(round(float(a), 4), round(float(b), 4)): int(v) for a, b, v in zip(ev[:, 0], ev[:, 1], ev[:, 4])}
        z_test = np.load(RUNS[name] / "epoch_018" / "full_roa_per_point.npz")
        X_test, p_test = z_test["start_states"], z_test["p_success"]
        y_test = np.array([state2y[(round(float(a), 4), round(float(b), 4))] for a, b in X_test])
        z_cal = np.load(OUT / f"{name}_cal_p_success.npz")
        X_cal, p_cal = z_cal["X"], z_cal["p_success"]
        y_cal = np.array([state2y[(round(float(a), 4), round(float(b), 4))] for a, b in X_cal])
    elif name == "lqr":
        z_cal = np.load(OUT / "lqr_cal_set_p_success.npz")
        z_test = np.load(OUT / "lqr_test_set_p_success.npz")
        X_cal, p_cal, y_cal = z_cal["X"], z_cal["p_success"], z_cal["y"]
        X_test, p_test, y_test = z_test["X"], z_test["p_success"], z_test["y"]
    else:  # compositions: rl_to_lqr / rl-weak_to_lqr
        base = name.replace("_to_lqr", "").replace("rl-weak", "rl-weak")
        ctrl = "rl" if name == "rl_to_lqr" else "rl-weak"
        z = np.load(RUNS[ctrl] / "composition_eval" / "per_point.npz")
        X_cal, p_cal, y_cal = z["X_cal"], z["p_cal"], z["y_cal"]
        X_test, p_test, y_test = z["X_test"], z["p_success"], z["y_test"]
    X = np.vstack([X_cal, X_test]); p = np.concatenate([p_cal, p_test]); y = np.concatenate([y_cal, y_test])
    return X, p, y, p_cal, y_cal, p_test, y_test


def style_axis(ax):
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-2 * np.pi, 2 * np.pi)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
    ax.set_yticklabels([r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])
    ax.tick_params(colors=INK, labelsize=9)
    for s in ax.spines.values():
        s.set_color("#c9c7c2"); s.set_linewidth(0.8)


def to_grid(X, values):
    # The eval states form a regular 158x315 grid (0.04 rad spacing).
    th = np.round(X[:, 0], 4); thd = np.round(X[:, 1], 4)
    th_u, thd_u = np.unique(th), np.unique(thd)
    img = np.full((len(thd_u), len(th_u)), np.nan)
    img[np.searchsorted(thd_u, thd), np.searchsorted(th_u, th)] = values
    return img, [th_u[0], th_u[-1], thd_u[0], thd_u[-1]]


def panel(ax, X, labels, title, three_class=False):
    from matplotlib.colors import ListedColormap
    img, extent = to_grid(X, labels)
    if three_class:  # 0=failure, 1=success, 2=uncertain
        cmap = ListedColormap([C_FAILURE, C_SUCCESS, C_UNCERTAIN]); vmax = 2
    else:
        cmap = ListedColormap([C_FAILURE, C_SUCCESS]); vmax = 1
    ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
              extent=extent, interpolation="nearest")
    ax.set_title(title, fontsize=10, color=INK)
    style_axis(ax)


def prob_panel(ax, X, p, title):
    img, extent = to_grid(X, p)
    im = ax.imshow(img, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1,
                   extent=extent, interpolation="nearest")
    ax.set_title(title, fontsize=10, color=INK)
    style_axis(ax)
    return im


def nominal_labels(p, lam=0.5, delta=0.1):
    """one_sided lambda-delta rule: success p>lam+delta, failure p<lam-delta, else uncertain."""
    out = np.full(len(p), 2)
    out[p > lam + delta] = 1
    out[p < lam - delta] = 0
    return out


def nominal_metrics(p, y, lam=0.5, delta=0.1):
    lab = nominal_labels(p, lam, delta)
    decided = lab != 2
    correct = (lab == y) & decided
    return dict(lambda_=lam, delta=delta, n=len(y),
                uncertain_pct=float((~decided).mean()),
                accuracy_decided=float(correct.sum() / max(decided.sum(), 1)),
                accuracy_all=float(correct.sum() / len(y)),
                tp=int(((lab == 1) & (y == 1)).sum()), fp=int(((lab == 1) & (y == 0)).sum()),
                tn=int(((lab == 0) & (y == 0)).sum()), fn=int(((lab == 0) & (y == 1)).sum()),
                n_uncertain=int((~decided).sum()))


def make_figure(models, titles, fname, suptitle, results, nominal=False, nominal_results=None):
    n = len(models)
    fig, axes = plt.subplots(3, n, figsize=(4.0 * n, 11.2), dpi=200)
    axes = np.atleast_2d(axes)
    im = None
    for j, name in enumerate(models):
        X, p, y, p_cal, y_cal, p_test, y_test = load_model_points(name)
        lam = pick_lambda(p_cal, y_cal)
        results[name] = metrics(p_test, y_test, lam)
        panel(axes[0, j], X, y, f"{titles[j]}\nground truth")
        if nominal:
            if nominal_results is not None:
                nominal_results[name] = nominal_metrics(p_test, y_test)
            panel(axes[1, j], X, nominal_labels(p),
                  "predicted ($\\lambda$=0.5, $\\delta$=0.1)", three_class=True)
        else:
            panel(axes[1, j], X, (p >= lam).astype(int),
                  f"predicted ($\\lambda^*$={lam:.2f})")
        im = prob_panel(axes[2, j], X, p, "predicted $p(\\mathrm{success})$")
    for i in range(3):
        axes[i, 0].set_ylabel(r"$\dot\theta$ (rad/s)", fontsize=10, color=INK)
    for j in range(n):
        axes[2, j].set_xlabel(r"$\theta$ (rad)", fontsize=10, color=INK)
    cbar = fig.colorbar(im, ax=axes[2, :].tolist(), location="bottom",
                        fraction=0.02, pad=0.16, aspect=50)
    cbar.set_label("$p(\\mathrm{success})$", fontsize=9, color=INK)
    cbar.ax.tick_params(labelsize=8, colors=INK)
    handles = [Line2D([], [], marker="s", linestyle="", markersize=8, color=C_SUCCESS, label="success"),
               Line2D([], [], marker="s", linestyle="", markersize=8, color=C_FAILURE, label="failure")]
    if nominal:
        handles.append(Line2D([], [], marker="s", linestyle="", markersize=8, color=C_UNCERTAIN, label="uncertain"))
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=10,
               bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(suptitle, fontsize=13, color=INK, x=0.02, ha="left")
    fig.savefig(OUT / fname, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / fname}")


def main():
    results: dict[str, dict] = {}
    make_figure(["rl", "rl-weak", "lqr"],
                ["RL swing-up (strong)", "RL swing-up (weak)", "LQR"],
                "controllers_roa.png",
                "Individual controller ROA — ground truth vs predicted", results)
    make_figure(["rl_to_lqr", "rl-weak_to_lqr"],
                ["RL → LQR composition", "RL-weak → LQR composition"],
                "compositions_roa.png",
                "Composition ROA — ground truth vs predicted (chained RL-FM → LQR-FM)", results)

    nom: dict[str, dict] = {}
    make_figure(["rl", "rl-weak", "lqr"],
                ["RL swing-up (strong)", "RL swing-up (weak)", "LQR"],
                "controllers_roa_nominal.png",
                "Individual controller ROA — nominal thresholds ($\\lambda$=0.5, $\\delta$=0.1)",
                results, nominal=True, nominal_results=nom)
    make_figure(["rl_to_lqr", "rl-weak_to_lqr"],
                ["RL → LQR composition", "RL-weak → LQR composition"],
                "compositions_roa_nominal.png",
                "Composition ROA — nominal thresholds ($\\lambda$=0.5, $\\delta$=0.1)",
                results, nominal=True, nominal_results=nom)
    with open(OUT / "roa_metrics_nominal.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "lambda", "delta", "n_test", "uncertain_pct", "accuracy_decided",
                    "accuracy_all", "tp", "fp", "tn", "fn", "n_uncertain"])
        for name in ["rl", "rl-weak", "lqr", "rl_to_lqr", "rl-weak_to_lqr"]:
            r = nom[name]
            w.writerow([name, "0.5", "0.1", r["n"], f"{r['uncertain_pct']:.4f}",
                        f"{r['accuracy_decided']:.4f}", f"{r['accuracy_all']:.4f}",
                        r["tp"], r["fp"], r["tn"], r["fn"], r["n_uncertain"]])
    print(f"wrote {OUT / 'roa_metrics_nominal.csv'}")

    task = {"rl": "reach G1 (rl_success)", "rl-weak": "reach G1 (rl_success)",
            "lqr": "upright (roa label)", "rl_to_lqr": "composition (lqr_success)",
            "rl-weak_to_lqr": "composition (lqr_success)"}
    with open(OUT / "roa_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["model", "task", "lambda_star", "n_test", "base_success_rate",
                "accuracy", "f1", "auc", "accuracy_at_0p5", "tp", "fp", "tn", "fn"]
        w.writerow(cols)
        for name in ["rl", "rl-weak", "lqr", "rl_to_lqr", "rl-weak_to_lqr"]:
            r = results[name]
            w.writerow([name, task[name], f"{r['lambda_star']:.3f}", r["n"],
                        f"{r['base_success_rate']:.4f}", f"{r['accuracy']:.4f}",
                        f"{r['f1']:.4f}", f"{r['auc']:.4f}", f"{r['accuracy_at_0p5']:.4f}",
                        r["tp"], r["fp"], r["tn"], r["fn"]])
    print(f"wrote {OUT / 'roa_metrics.csv'}")


if __name__ == "__main__":
    main()
