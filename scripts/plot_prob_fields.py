#!/usr/bin/env python
"""Predicted vs ground-truth success-probability fields, one figure per level.

Pure re-projection of files already on disk -- no model inference. Ground truth is
the eval grid's Monte-Carlo p_success; each arm's p_hat is the full_roa_per_point.npz
its deepest completed epoch already wrote. Both are joined on the shared eval
states, so every panel shows the SAME cells and differences are model differences.

LEVEL IDENTITY COMES FROM THE RUN CONFIG, NOT FROM THE STATES. Every noise level
of a family reuses one eval lattice -- only p_success differs -- so matching start
states identifies the grid geometry and cannot separate `smooth` from `sharp`.
Runs are therefore assigned by the resolved `dataset_name` in .hydra/config.yaml,
and the state join is kept only as a compatibility check.

Field constructions, chosen by state dimension:

  fullstate  pendulum is 2-D: the eval grid IS the plane, nothing to pin.
  slice      cartpole (4-D) and quad2D (6-D) eval sets are product lattices, so an
             exact slice exists: pin every non-plotted dim to the lattice value
             nearest the goal.
  slab       quad3D's eval set is a 1M-point random sample in 13-D, so no exact
             slice exists -- pinning 11 coordinates selects nothing. Keep the
             points nearest the goal in the non-plotted dims instead. This is a
             near-goal conditional average, NOT a slice, and is labelled as such.
  marginal   average over every non-plotted dim. Uses all the data; answers "how
             does p vary with these two on average" rather than "at the goal".
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.spatial import cKDTree

EXP = Path("/common/users/shared/pracsys/adaptive_roa_experiments")
DATA = Path("/common/users/shared/pracsys/genMoPlan/data_trajectories/stochastic")
DOCS = Path("/common/users/shared/pracsys/genMoPlan/docs/stochastic")

CLASS_NAMES = ["FM ensemble", "Deep-ensemble classifier", "Bayesian NN", "Part-X GP"]
# Doubles as the arm registry: a run dir is split by matching the LONGEST suffix
# here, so `bnn_mfvi_bald` must be listed for `<prefix>_bnn_mfvi_bald` to resolve to
# it rather than to `bnn_mfvi` with a stray `_bald`. The pre-fix `partx` is absent
# on purpose -- only `partx_fix` is a valid Part-X result.
ORDER = ["dir00", "dir00_s42", "dir00_s43", "dir00_s44", "epi_var", "epi_var_anch",
         "epi_bald", "aleat", "total", "yield_a1", "yield_mlp", "partx_fix",
         "clf_dir00", "clf_yield", "clf_epi_var", "clf_epi_bald", "clf_epi_var_anch",
         "bnn_ens", "bnn_lap", "bnn_mfvi", "bnn_mfvi_bald"]

# system dim -> (axis labels, keys, goal, per-dim scale for the near-goal ball)
SYSTEMS = {
    2: (["$\\theta$", "$\\dot{\\theta}$"], ["th", "w"],
        np.array([2.1, 0.0]), None),
    4: (["x", "$\\theta$", "$\\dot{x}$", "$\\dot{\\theta}$"], ["x", "th", "vx", "w"],
        np.array([0.0, 0.0, 0.0, 0.0]), None),
    6: (["x", "z", "$\\theta$", "$\\dot{x}$", "$\\dot{z}$", "$\\dot{\\theta}$"],
        ["x", "z", "th", "vx", "vz", "w"],
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), None),
    13: (["x", "y", "z", "$q_w$", "$q_x$", "$q_y$", "$q_z$",
          "$v_x$", "$v_y$", "$v_z$", "$\\omega_x$", "$\\omega_y$", "$\\omega_z$"],
         ["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"],
         np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], float),
         np.array([1.8, 1.8, 1.45, 1, 1, 1, 1, 3, 3, 3, 24, 24, 24], float)),
}

# Which planes to draw per system, and in which mode. The pendulum has no choice;
# elsewhere the pair was picked by screening every free-dim pair for contrast.
PLANES = {
    2:  [("fullstate", "th,w")],
    4:  [("slice", "th,w"), ("marginal", "th,w")],
    6:  [("slice", "vx,w"), ("marginal", "vx,w")],
    13: [("slab", "vx,wx"), ("marginal", "vx,wx")],
}
# Quaternion components are unusable as axes: unit norm confines the sample to an
# arc and the panel comes out almost entirely empty.


def resolve_dataset(cfg_path: Path) -> str:
    if not cfg_path.exists():
        return ""
    txt = cfg_path.read_text()
    flat = {}
    for m in re.finditer(r"^(\w+):[ \t]*(\S.*?)[ \t]*$", txt, re.M):
        flat.setdefault(m.group(1), m.group(2))
    ds = flat.get("dataset_name", "")
    for _ in range(5):
        new = re.sub(r"\$\{(\w+)\}", lambda m: flat.get(m.group(1), m.group(0)), ds)
        if new == ds:
            break
        ds = new
    return ds


def last_epoch(run: Path):
    eps = sorted(d for d in run.glob("epoch_*")
                 if (d / "artifacts_v2.json").exists()
                 and (d / "full_roa_per_point.npz").exists())
    return eps[-1] if eps else None


def discover():
    """All runs grouped by (dataset, campaign, prefix).

    A run belongs to the LONGEST registered prefix that matches its name, so
    `cp_med_v2_epi_var` lands under `cp_med_v2` rather than `cp_med`, and
    `pen_low_dir00_s42` under `pen_low` rather than `pen`.
    """
    found = []
    for camp in sorted(EXP.iterdir()):
        if not camp.is_dir() or camp.name.startswith(("_", "archived", "dhruv")):
            continue
        for run in sorted(camp.iterdir()):
            if not run.is_dir():
                continue
            ed = last_epoch(run)
            if ed is None:
                continue
            ds = resolve_dataset(run / ".hydra" / "config.yaml")
            if not ds or not (DATA / ds / "eval_success_prob.npz").exists():
                continue
            found.append((ds, camp.name, run.name, run))

    # Split each run name into prefix + arm by matching the LONGEST arm suffix.
    # Matching the longest prefix instead would decompose `q2d_cs_clf_dir00` as
    # prefix `q2d_cs_clf` + arm `dir00`, splitting the five classifier arms into
    # their own cohort and dropping them from the level's figure.
    by_len = sorted(ORDER, key=len, reverse=True)
    groups = {}
    for ds, camp, name, run in found:
        for a in by_len:
            if name.endswith("_" + a):
                groups.setdefault((ds, camp, name[: -len(a) - 1]), []).append((a, run))
                break
    return groups


class Field:
    def __init__(self, states, dims, mode, goal, scale, bins, slab_frac, min_count):
        self.dims, self.mode, self.min_count = dims, mode, min_count
        a, b = dims
        n, D = states.shape
        if mode in ("slice", "fullstate"):
            axes = [np.unique(np.round(states[:, d], 6)) for d in range(D)]
            self.pin = {d: axes[d][np.argmin(np.abs(axes[d] - goal[d]))]
                        for d in range(D) if d not in dims}
            m = np.ones(n, bool)
            for d, v in self.pin.items():
                m &= np.isclose(states[:, d], v, atol=1e-6)
            self.mask = m
            self.xs, self.ys = axes[a], axes[b]
            self.ia = np.searchsorted(self.xs, np.round(states[m, a], 6))
            self.ib = np.searchsorted(self.ys, np.round(states[m, b], 6))
            self.shape = (len(self.xs), len(self.ys))
            sx = (self.xs[-1] - self.xs[0]) / max(len(self.xs) - 1, 1)
            sy = (self.ys[-1] - self.ys[0]) / max(len(self.ys) - 1, 1)
            self._ext = [self.xs[0] - sx / 2, self.xs[-1] + sx / 2,
                         self.ys[0] - sy / 2, self.ys[-1] + sy / 2]
            self.radius = None
        else:
            keep = [d for d in range(D) if d not in dims]
            if mode == "slab":
                dist = np.sqrt((((states[:, keep] - goal[keep]) / scale[keep]) ** 2).sum(1))
                k = max(1, int(n * slab_frac))
                self.radius = float(np.partition(dist, k)[k])
                m = dist <= self.radius
            else:
                self.radius = None
                m = np.ones(n, bool)
            self.pin = None
            self.mask = m
            self._ext = [0.0, 0.0, 0.0, 0.0]
            self.ia, na = self._axis(states, a, m, bins, 0)
            self.ib, nb = self._axis(states, b, m, bins, 1)
            self.shape = (na, nb)
        self.counts = np.zeros(self.shape)
        np.add.at(self.counts, (self.ia, self.ib), 1.0)
        self.valid = self.counts >= min_count

    def _axis(self, states, d, m, bins, which):
        # A plotted dim that is itself discrete keeps its own levels as the axis;
        # uniform bins over 12 distinct values leave most cells empty and the
        # panel then reads as missing data rather than as a coarse grid.
        levels = np.unique(np.round(states[:, d], 6))
        if len(levels) <= 64:
            setattr(self, "xs" if which == 0 else "ys", levels)
            step = (levels[-1] - levels[0]) / max(len(levels) - 1, 1)
            self._ext[2 * which] = levels[0] - step / 2
            self._ext[2 * which + 1] = levels[-1] + step / 2
            return np.searchsorted(levels, np.round(states[m, d], 6)), len(levels)
        # range from the RETAINED points: a near-goal ball spans only part of each
        # axis, and the full range would pad the panel with empty cells.
        edges = np.linspace(states[m, d].min(), states[m, d].max(), bins + 1)
        setattr(self, "xs" if which == 0 else "ys", edges)
        self._ext[2 * which] = edges[0]
        self._ext[2 * which + 1] = edges[-1]
        return np.clip(np.searchsorted(edges, states[m, d], "right") - 1, 0, bins - 1), bins

    def grid(self, values):
        s = np.zeros(self.shape)
        np.add.at(s, (self.ia, self.ib), values[self.mask])
        out = np.full(self.shape, np.nan)
        out[self.valid] = s[self.valid] / self.counts[self.valid]
        return out

    def extent(self):
        return list(self._ext)


def class_of(arm):
    if arm.startswith("partx"):
        return "Part-X GP"
    if arm.startswith("bnn"):
        return "Bayesian NN"
    return "Deep-ensemble classifier" if arm.startswith("clf") else "FM ensemble"


def flagged_epoch(ed: Path) -> bool:
    """True if this epoch fails the campaign's >1.0 / non-finite magnitude screen."""
    try:
        ee = (json.loads((ed / "artifacts_v2.json").read_text())
              .get("endpoint_error") or {})
    except (ValueError, OSError):
        return False
    vals = [v for v in (ee.get("success_mae"), ee.get("failure_mae"),
                        ee.get("overall_mae")) if v is not None]
    if not vals:
        return False
    mx = max(vals)
    return int(ed.name.split("_")[1]) > 0 and (mx != mx or mx > 1.0)


def build(ds, arms_runs, dims, mode, bins, slab_frac, min_count):
    with np.load(DATA / ds / "eval_success_prob.npz") as z:
        gt_starts = z["starts"].astype(np.float64)
        gt_p = z["p_success"].astype(np.float64)

    panels, states, idx, skipped = [], None, None, []
    rank = {a: i for i, a in enumerate(ORDER)}
    for arm, run in sorted(arms_runs, key=lambda t: (rank.get(t[0], 99), t[0])):
        ed = last_epoch(run)
        with np.load(ed / "full_roa_per_point.npz") as z:
            s = z["start_states"].astype(np.float64)
            p = z["p_success"].astype(np.float64)
        if states is None:
            states = s
            d, idx = cKDTree(gt_starts).query(states, k=1)
            if d.max() > 1e-3:
                raise SystemExit(f"{ds}: eval states off grid (max {d.max():.3g})")
        elif not np.array_equal(s, states):
            # a different eval grid cannot share panels with the rest
            skipped.append(arm)
            continue
        panels.append(dict(cls=class_of(arm), arm=arm, ep=int(ed.name.split("_")[1]),
                           p=p, flagged=flagged_epoch(ed)))
    if not panels:
        return None

    D = states.shape[1]
    _, _, goal, scale = SYSTEMS[D]
    f = Field(states, dims, mode, goal, scale, bins, slab_frac, min_count)
    truth = f.grid(gt_p[idx])
    for pan in panels:
        g = f.grid(pan["p"])
        ok = np.isfinite(g) & np.isfinite(truth)
        pan["grid"] = g
        pan["mae"] = float(np.mean(np.abs(g[ok] - truth[ok]))) if ok.any() else float("nan")
    return dict(field=f, truth=truth, panels=panels, dim=D, skipped=skipped,
                n_cells=int(f.valid.sum()), n_points=int(f.mask.sum()),
                n_total=len(states))


def draw(res, title, blurb, out):
    f = res["field"]
    names, keys, _, _ = SYSTEMS[res["dim"]]
    a, b = f.dims
    groups = {c: [p["arm"] for p in res["panels"] if p["cls"] == c] for c in CLASS_NAMES}
    groups = {c: v for c, v in groups.items() if v}
    ncol = max(max(len(v) for v in groups.values()), 3)
    rows = [("Ground truth", None)] + list(groups.items())

    fig = plt.figure(figsize=(1.78 * ncol + 1.3, 2.05 * len(rows) + 0.95))
    gs = gridspec.GridSpec(len(rows), ncol + 1, figure=fig,
                           width_ratios=[1] * ncol + [0.075],
                           hspace=0.62, wspace=0.13,
                           left=0.062, right=0.94, top=0.872, bottom=0.062)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.82")
    vmax = res["vmax"]
    ext = f.extent()
    kw = dict(origin="lower", aspect="auto", vmin=0, vmax=vmax,
              cmap=cmap, interpolation="nearest")
    by_arm = {p["arm"]: p for p in res["panels"]}

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(res["truth"].T, extent=ext, **kw)
    ax.set_title("GROUND TRUTH  (MC $p_{success}$)", fontsize=9, fontweight="bold")
    ax.set_xlabel(names[a], fontsize=8); ax.set_ylabel(names[b], fontsize=8)
    ax.tick_params(labelsize=7)
    ax2 = fig.add_subplot(gs[0, 1:ncol]); ax2.axis("off")
    ax2.text(0.01, 0.52, blurb, fontsize=8.2, va="center", family="monospace")

    for r, (cls, arms) in enumerate(rows[1:], start=1):
        for c in range(ncol):
            ax = fig.add_subplot(gs[r, c])
            if c >= len(arms):
                ax.axis("off"); continue
            p = by_arm[arms[c]]
            ax.imshow(p["grid"].T, extent=ext, **kw)
            tag = "  [FLAGGED]" if p["flagged"] else ""
            ax.set_title(f"{p['arm']}\nep{p['ep']}{tag}   MAE {p['mae']:.3f}",
                         fontsize=7.6, color="#b00020" if p["flagged"] else "black")
            if p["flagged"]:
                for sp in ax.spines.values():
                    sp.set_color("#b00020"); sp.set_linewidth(1.8)
            ax.tick_params(labelsize=6)
            ax.set_ylabel(f"{cls}\n{names[b]}" if c == 0 else "", fontsize=8)
            if c: ax.set_yticklabels([])
            ax.set_xlabel(names[a], fontsize=7, labelpad=1)

    cax = fig.add_subplot(gs[:, ncol])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$p_{success}$", fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.978)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def outdir_for(ds):
    """<docs>/stochastic/<system>/<controller>/slices/<family>/<level>/"""
    sysname, family, ctrl, level = ds.split("/")
    return (DOCS / sysname.lower() / ctrl / "slices" / family / level,
            sysname, family, level)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="print the discovered level/arm inventory and exit")
    ap.add_argument("--only", default=None,
                    help="substring filter on the dataset path")
    ap.add_argument("--bins", type=int, default=28)
    ap.add_argument("--slab-frac", type=float, default=0.05)
    ap.add_argument("--min-count", type=int, default=1)
    args = ap.parse_args()

    groups = discover()
    # one cohort per dataset: the prefix carrying the most arms. A superseded
    # cohort on the same level would duplicate arm names in one figure.
    best = {}
    for (ds, camp, pref), arms in groups.items():
        if ds not in best or len(arms) > len(best[ds][2]):
            best[ds] = (camp, pref, arms)

    if args.list:
        print("### levels with runs (cohort = prefix with the most arms)\n")
        for ds in sorted(best):
            camp, pref, arms = best[ds]
            others = [(c, p, len(a)) for (d, c, p), a in groups.items()
                      if d == ds and p != pref]
            print("%-46s  %s/%s  %d arms" % (ds, camp, pref, len(arms)))
            print("      " + ", ".join(sorted(a for a, _ in arms)))
            for c, p, n in others:
                print("      (superseded cohort %s/%s, %d arms)" % (c, p, n))
            print()
        return

    made = []
    for ds in sorted(best):
        if args.only and args.only not in ds:
            continue
        camp, pref, arms = best[ds]
        outdir, sysname, family, level = outdir_for(ds)
        with np.load(DATA / ds / "eval_success_prob.npz") as z:
            D = z["starts"].shape[1]
        _, keys, _, _ = SYSTEMS[D]
        for mode, dimspec in PLANES[D]:
            dims = tuple(keys.index(k) for k in dimspec.split(","))
            res = build(ds, arms, dims, mode, args.bins, args.slab_frac, args.min_count)
            if res is None:
                print("  SKIP %s (%s): no usable arm" % (ds, mode)); continue
            hi = max([np.nanmax(res["truth"])] + [np.nanmax(p["grid"]) for p in res["panels"]])
            res["vmax"] = float(min(1.0, np.ceil(hi * 20) / 20))
            ka, kb = keys[dims[0]], keys[dims[1]]
            if mode == "fullstate":
                head = ("full state space, %s x %s\n(2-D system -- nothing to pin;\n"
                        " this is the whole eval grid)\n" % (ka, kb))
            elif mode == "slice":
                pins = ", ".join("%s=%+.3f" % (keys[d], v) for d, v in sorted(res["field"].pin.items()))
                head = ("exact lattice slice, %s x %s\nothers pinned at the grid value\n"
                        "nearest the goal:\n  %s\n" % (ka, kb, pins))
            elif mode == "slab":
                head = ("near-goal conditional average, %s x %s\n"
                        "(NOT an exact slice: this eval set is a\n"
                        " random 13-D sample, so pinning 11\n coordinates selects nothing)\n"
                        "others within normalized r <= %.3f of goal\n" % (ka, kb, res["field"].radius))
            else:
                head = ("marginal over all other dims, %s x %s\n"
                        "no conditioning -- every eval state contributes\n" % (ka, kb))
            blurb = head + "%d cells from %s of %s eval states" % (
                res["n_cells"], f"{res['n_points']:,}", f"{res['n_total']:,}")
            if res["vmax"] < 1.0:
                blurb += "\ncolour ceiling fitted to the data: %.2f" % res["vmax"]
            blurb += "\n\ngrey cells hold no eval state"
            blurb += "\nred frame = epoch fails the instability screen"
            if res["skipped"]:
                blurb += "\nomitted (different eval grid): " + ", ".join(res["skipped"])

            title = "%s  %s %s  —  predicted vs ground-truth success probability" % (
                sysname, family, level)
            out = outdir / ("%s_%s_%s.png" % (level, mode, dimspec.replace(",", "_")))
            draw(res, title, blurb, out)
            made.append((ds, mode, out, res))
            print("  wrote %s" % out.relative_to(DOCS))
            print("      %d arms, %d cells, vmax %.2f%s" % (
                len(res["panels"]), res["n_cells"], res["vmax"],
                "  SKIPPED " + ",".join(res["skipped"]) if res["skipped"] else ""))

    print("\n### per-panel MAE (lower = closer to ground truth)\n")
    for ds, mode, out, res in made:
        print("%s  [%s]" % (ds, mode))
        for p in sorted(res["panels"], key=lambda p: p["mae"]):
            print("    %-24s %-18s ep%-3d MAE %.4f%s" % (
                p["cls"], p["arm"], p["ep"], p["mae"],
                "  [FLAGGED]" if p["flagged"] else ""))
        print()


if __name__ == "__main__":
    main()
