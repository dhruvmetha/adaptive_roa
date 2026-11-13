"""Paper Figure 1: three systems, one slice each, at 6.0 in text width.

    render_triplet(entries, out_stem, layout="2x3")

``entries``: three dicts {"slice": data/<setting>/<plane>.npz, "pred": pred npz with ``decision``
(ny, nx) and ``run``, "label": optional column title}. The pred may be a strided sub-grid of the
slice (``labels[::s, ::s]``, s = 2 in practice); the slice is then cut to the same cells.

``layout``: "2x3" (columns = systems; top row ground truth, bottom row TERA-FM certified) or
"1x3" (TERA-FM panels only, GT boundary as the black line).

Writes ``<out_stem>.png`` (300 dpi), ``<out_stem>.pdf`` (images rasterized, text and lines
vector) and ``<out_stem>.txt`` (per panel: fixed values, run, seeds, thresholds, in-box and
full-slice R_s, R_f, abstain, FP, FN, FDR, FOR, rendering notes).

Wrong calls, Success as the positive class: FP (Success declared on a GT fail) and FN (Fail declared
on a GT success) are filled cells in style.FP_COLOR and style.FN_COLOR, drawn into the TERA image
itself. Nothing is dilated, so each coloured area is the true FP or FN area.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from . import metrics as M
from . import style as S

WIDTH = 6.0                  # in, paper text width
DPI = 300
RC = {"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
      "legend.fontsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .6,
      "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
      "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42}

# Layout constants, inches.
ROW_LABEL_W = 0.20           # rotated row labels at the far left
COL_GAP = 0.10               # between a panel's right edge and the next column's y label
RIGHT = 0.04
TOP = 0.24                   # column titles
ROW_GAP = 0.07               # between the GT and TERA rows (no tick labels in between)
LEGEND_ROW = 0.19           # two legend rows: classes and lines, then FP and FN
FIXED_LINE = 0.11


# --- inputs --------------------------------------------------------------------------------------

def _sub_slice(sl: dict, s: int) -> dict:
    """The slice cut to cells [::s, ::s] (labels, axes, model_states rows); box keys kept."""
    if s == 1:
        return sl
    ny, nx = sl["ny"], sl["nx"]
    rows = (np.arange(ny)[::s, None] * nx + np.arange(nx)[None, ::s]).ravel()
    out = dict(sl)
    out.update(xs=sl["xs"][::s], ys=sl["ys"][::s], labels=sl["labels"][::s, ::s],
               model_states=sl["model_states"][rows])
    out["ny"], out["nx"] = out["labels"].shape
    return out


def _scalar(z, k, default=None):
    if k not in z.files:
        return default
    a = np.asarray(z[k])
    return a.item() if a.ndim == 0 else a


def load_entry(e: dict) -> dict:
    """Slice + pred for one panel, aligned to the pred's grid, with metrics and provenance."""
    sl = S.load_slice(e["slice"])
    with np.load(e["pred"], allow_pickle=False) as z:
        dec = np.array(z["decision"]).astype(np.int8)
        meta = {k: _scalar(z, k) for k in ("run", "seed", "stride", "K", "t_s", "t_f", "u_s", "u_f", "batch_size")}
    L = sl["labels"]
    stride = None
    for s in range(1, 9):
        if L[::s, ::s].shape == dec.shape:
            stride = s
            break
    if stride is None:
        raise ValueError(f"{e['pred']}: decision {dec.shape} is not labels[::s, ::s] of {L.shape} for any s <= 8")
    if meta["stride"] is not None and int(meta["stride"]) != stride:
        raise ValueError(f"{e['pred']}: stored stride {meta['stride']} but the decision shape implies {stride}")
    if not np.isin(dec, (-1, 0, 1)).all():
        raise ValueError(f"{e['pred']}: decision values outside {{-1, 0, 1}}")
    sub = _sub_slice(sl, stride)
    run = str(meta["run"])
    thr_pred = {k: meta[k] for k in ("t_s", "t_f", "u_s", "u_f")}
    from . import infer           # thresholds() reads the scored index only; no torch import
    thr = infer.thresholds(run)
    if all(v is not None for v in thr_pred.values()) and any(abs(float(thr_pred[k]) - thr[k]) > 1e-12 for k in thr):
        raise ValueError(f"{e['pred']}: stored thresholds {thr_pred} differ from {run}'s c6c {thr}")
    seeds = [json.loads(l).get("seed") for l in infer.MATRIX_ROWS.read_text().splitlines()
             if l.strip() and json.loads(l).get("name") == run]
    gt_m, tera_m = M.slice_metrics(sub, dec)
    label = e.get("label") or f"{sl['system_label']} ({sl['controller']})"
    return dict(sl=sl, sub=sub, dec=dec, stride=stride, run=run, thr=thr, train_seed=seeds[0] if seeds else None,
                infer_seed=meta["seed"], K=meta["K"], gt_m=gt_m, tera_m=tera_m, label=label,
                slice_path=str(e["slice"]), pred_path=str(e["pred"]))


# --- drawing -------------------------------------------------------------------------------------

def _cell_pt(ax, sl) -> float:
    bb = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    return 72 * min(bb.width * fw / sl["nx"], bb.height * fh / sl["ny"])


def _image(ax, rgb, sl):
    ax.imshow(rgb, origin="lower", interpolation="nearest", extent=S._extent(sl), aspect="auto",
              rasterized=True, zorder=1)


def _boundary(ax, sl, lw: float = 0.55):
    L = sl["labels"]
    if len(np.unique(L)) > 1:
        ax.contour(sl["xs"], sl["ys"], L.astype(float), levels=[0.5], colors="black", linewidths=lw, zorder=3)
        return True
    return False


def _box(ax, sl):
    # style._draw_box with a thinner line for the small panels
    drew = S._draw_box(ax, sl)
    for ln in ax.lines:
        if ln.get_linestyle() not in ("-", "solid"):
            ln.set_linewidth(0.75)
    return drew


def _axes_style(ax, sl, xlabels: bool, yprune=None):
    ax.set_xlim(*S._extent(sl)[:2])
    ax.set_ylim(*S._extent(sl)[2:])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], prune=yprune))
    ax.set_ylabel(S._axis_label(sl, 1), labelpad=1.5)
    if xlabels:
        ax.set_xlabel(S._axis_label(sl, 0), labelpad=1.5)
    else:
        ax.tick_params(labelbottom=False)


def _decorations(fig, axes):
    """Per column: inches of y label + tick labels left of the axes; global inches below the bottom row."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    left = []
    for col in axes:
        x0 = min(ax.get_tightbbox(r).x0 for ax in col) / fig.dpi
        left.append(col[0].get_position().x0 * W - x0)
    bottom = max(col[-1].get_position().y0 * H - col[-1].get_tightbbox(r).y0 / fig.dpi for col in axes)
    last = axes[-1]
    right = max(0.0, max(ax.get_tightbbox(r).x1 / fig.dpi for ax in last) - last[0].get_position().x1 * W)
    return left, bottom, right


def render_triplet(entries, out_stem, layout: str = "2x3", aspect: float | None = None,
                   fixed_text: bool = False, row_labels: bool = True, palette: dict | None = None,
                   show_box: bool = True, method_label: str = "TERA-FM", prune_gap_ticks: bool = False):
    """Render the Figure 1 composite. Returns (png, pdf, txt) paths.

    ``palette``: optional colours {success, fail, abstain, fp, fn}; None keeps style.py's defaults.
    ``show_box``: False draws no dashed D_test start-box lines and leaves them out of the legend.
    ``method_label``: the bottom-row label (Figure 1 passes method_names()['name']).
    ``prune_gap_ticks``: in the 2x3 layout, drop the top row's lowest and the bottom row's highest y tick so the
    two labels at the narrow row gap cannot collide (for example -1.0 over 1.0).

    ``aspect``: panel height / width; None takes clip(mean ny/nx of the three panels, 0.6, 1.0).
    ``fixed_text``: also print each column's fixed values under it (default off: they go in the
    sidecar for the caption, since a spatial-quadrotor slice holds ten or more dims fixed).
    """
    if layout not in ("2x3", "1x3"):
        raise ValueError("layout must be '2x3' or '1x3'")
    if len(entries) != 3:
        raise ValueError("exactly three entries")
    P = [load_entry(e) for e in entries]
    nrows = 2 if layout == "2x3" else 1
    if aspect is None:
        aspect = float(np.clip(np.mean([p["sub"]["ny"] / p["sub"]["nx"] for p in P]), 0.6, 1.0))
    fixed_lines = [S._wrap([S.fixed_line(p["sub"])], 1.9) if fixed_text else [] for p in P]
    n_fixed = max(len(f) for f in fixed_lines)

    with plt.rc_context(RC):
        # Pass 1: provisional geometry, to measure tick-label and axis-label extents.
        fig = plt.figure(figsize=(WIDTH, 4.0))
        axes = [[fig.add_axes([0.1 + c * 0.3, 0.55 - r * 0.4, 0.2, 0.3]) for r in range(nrows)] for c in range(3)]
        any_box, any_boundary = False, False
        for c, p in enumerate(P):
            sub = p["sub"]
            for r, ax in enumerate(axes[c]):
                yprune = None
                if prune_gap_ticks and nrows == 2:
                    yprune = "lower" if r == 0 else "upper"
                _axes_style(ax, sub, xlabels=(r == nrows - 1), yprune=yprune)
        left, bottom_dec, right_over = _decorations(fig, axes)

        # Final geometry.
        rl = ROW_LABEL_W if row_labels else 0.02
        pw = (WIDTH - rl - sum(left) - 2 * COL_GAP - RIGHT - right_over) / 3
        ph = pw * aspect
        legend_y0 = n_fixed * FIXED_LINE + (0.04 if n_fixed else 0)
        bottom = legend_y0 + 2 * LEGEND_ROW + bottom_dec + 0.02
        H = TOP + nrows * ph + (nrows - 1) * ROW_GAP + bottom
        fig.set_size_inches(WIDTH, H)
        x = rl
        for c in range(3):
            x += left[c]
            for r, ax in enumerate(axes[c]):
                y = bottom + (nrows - 1 - r) * (ph + ROW_GAP)
                ax.set_position([x / WIDTH, y / H, pw / WIDTH, ph / H])
            x += pw + COL_GAP

        cell_pts = [_cell_pt(axes[c][0], P[c]["sub"]) for c in range(3)]
        for c, p in enumerate(P):
            sub, dec = p["sub"], p["dec"]
            pal = S.palette_or_default(palette)
            if nrows == 2:
                _image(axes[c][0], S._rgb(sub["labels"], {1: pal["success"], 0: pal["fail"]}), sub)
            _image(axes[c][-1], S.decision_rgb(dec, sub["labels"], palette), sub)
            for ax in axes[c]:
                any_boundary |= _boundary(ax, sub)
                if show_box:
                    any_box |= _box(ax, sub)
            axes[c][0].set_title(p["label"], pad=3, fontsize=9)

        if row_labels:
            names = ["Ground truth", method_label] if nrows == 2 else [method_label]
            for r, name in enumerate(names):
                y = bottom + (nrows - 1 - r) * (ph + ROW_GAP) + ph / 2
                fig.text(0.05 / WIDTH, y / H, name, rotation=90, ha="left", va="center", fontsize=9)

        rows = S.legend_rows(True, any_boundary, any_box, box_label="start box (D_test)", palette=palette)
        for k, handles in enumerate(rows):             # top row first
            y = legend_y0 + (len(rows) - 1 - k) * LEGEND_ROW
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y / H), ncol=len(handles),
                       frameon=False, handlelength=1.3, handletextpad=0.35, columnspacing=0.9, borderaxespad=0.05)
        if fixed_text:
            for c, lines in enumerate(fixed_lines):
                bb = axes[c][-1].get_position()
                xc = bb.x0 + bb.width / 2
                for k, s in enumerate(lines):
                    fig.text(xc, (n_fixed - k) * FIXED_LINE / H, s, ha="center", va="top", fontsize=5.5,
                             color="#333333")

        out_stem = str(out_stem)
        png, pdf, txt = Path(f"{out_stem}.png"), Path(f"{out_stem}.pdf"), Path(f"{out_stem}.txt")
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=DPI)
        fig.savefig(pdf, dpi=DPI, metadata={"CreationDate": None})
        plt.close(fig)

    txt.write_text(_sidecar(P, layout, cell_pts, (WIDTH, H), pw, ph, S.palette_or_default(palette), show_box))
    return str(png), str(pdf), str(txt)


def _fmt(v, f=".3f"):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if math.isnan(v) else format(v, f)


def _box_txt(b):
    return "none" if b is None else "[" + ", ".join("unbounded" if math.isnan(v) else f"{v:.4g}" for v in b) + "]"


def _sidecar(P, layout, cell_pts, size, pw, ph, pal=None, show_box=True) -> str:
    pal = pal or S.palette_or_default()
    out = [f"Figure 1 composite, layout {layout}, {size[0]:.2f} x {size[1]:.2f} in, panels {pw:.2f} x {ph:.2f} in.",
           f"Colours: Success {pal['success']}, Fail {pal['fail']}, Abstain {pal['abstain']}. "
           f"Start-box lines {'drawn' if show_box else 'not drawn'}.",
           f"Wrong calls as filled cells, Success the positive class: FP {pal['fp']} ({S.FP_LABEL}), "
           f"FN {pal['fn']} ({S.FN_LABEL}); FDR = FP/(TP+FP), FOR = FN/(FN+TN). Smallest cell "
           f"{min(cell_pts):.2f} pt. Metrics are on in-box cells unless marked full.", ""]
    for c, p in enumerate(P):
        sub, t, g = p["sub"], p["tera_m"], p["gt_m"]
        nw = int(t["n_wrong_s"]) + int(t["n_wrong_f"])
        nwf = int(t["n_wrong_s_full"]) + int(t["n_wrong_f_full"])
        out += [
            f"[{c + 1}] {p['label']}: {sub['dim_names'][0]} vs {sub['dim_names'][1]}  (setting {sub['setting']}, plane {sub['plane']})",
            f"    slice {p['slice_path']}",
            f"    pred  {p['pred_path']}",
            f"    grid: GT {p['sl']['nx']}x{p['sl']['ny']}, shown {sub['nx']}x{sub['ny']} (stride {p['stride']}); "
            f"cell {cell_pts[c]:.2f} pt",
            f"    fixed: {sub['fixed_json']}",
            f"    box_x {_box_txt(sub.get('box_x'))}, box_y {_box_txt(sub.get('box_y'))}; "
            f"in-box {_fmt(g['in_box_frac'], '.1%')} of cells",
            f"    run {p['run']}, training seed {p['train_seed']}, inference seed {p['infer_seed']}, K {p['K']}",
            f"    thresholds (c6c): t_s {p['thr']['t_s']}, t_f {p['thr']['t_f']}, u_s {p['thr']['u_s']}, u_f {p['thr']['u_f']}",
            f"    in-box: R_s {_fmt(t['R_s'])}, R_f {_fmt(t['R_f'])}, abstain {_fmt(t['abstain_frac'], '.1%')}, "
            f"FP {t['n_wrong_s']}, FN {t['n_wrong_f']} (total {nw}), "
            f"FDR {_fmt(t['FDR'])}, FOR {_fmt(t['FOR'])}, gt_clarity {_fmt(g['gt_clarity'])}, rank_score {_fmt(t['rank_score'])}",
            f"    full:   R_s {_fmt(t['R_s_full'])}, R_f {_fmt(t['R_f_full'])}, abstain {_fmt(t['abstain_frac_full'], '.1%')}, "
            f"FP {t['n_wrong_s_full']}, FN {t['n_wrong_f_full']} (total {nwf}), FDR {_fmt(t['FDR_full'])}, "
            f"FOR {_fmt(t['FOR_full'])}, rank_score {_fmt(t['rank_score_full'])}",
            "",
        ]
    return "\n".join(out)



# --- Figure 1 in the reference figure's style ------------------------------------------------------

def render_flagship(entries, out_stem, layout: str = "3x2", width: float = 2.7, aspect: float = 0.82,
                    palette: dict | None = None, method_label: str = "TERA"):
    """Figure 1 in the look of dm1487's reference figure: flat colour fields with thin white gaps and no frame,
    bold rotated row labels, a small slice tag per system, y label and ticks on the left of each system only,
    no start box, no boundary line, and one legend line TP, TN, FP, FN, Abstention (paper terms,
    06_experiments.tex). Ground-truth panels use the TP and TN colours for true success and failure.

    ``entries``: three dicts as for render_triplet plus "system" (row/column title), "tag" (slice tag, mathtext)
    and "axes" (x and y axis labels in the paper's notation).
    ``layout``: "3x2" (rows = systems, columns = ground truth | method; for a wrapfigure of ``width`` in) or
    "2x3" (rows = ground truth | method, columns = systems; full text width).
    ``palette``: keys success (= TP), fail (= TN), abstain, fp, fn. Returns (png, pdf, txt)."""
    from matplotlib.patches import Patch
    if layout not in ("3x2", "2x3") or len(entries) != 3:
        raise ValueError("layout 3x2 or 2x3, three entries")
    P = [load_entry(e) for e in entries]
    for p, e in zip(P, entries):
        p["system"], p["tag"], p["axes"] = e["system"], e["tag"], e["axes"]
    pal = S.palette_or_default(palette)
    FS_TICK, FS_LAB, FS_TAG, FS_TITLE, FS_ROW = 6.5, 7.5, 6.5, 8, 8
    rc = dict(RC)
    rc.update({"xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "axes.labelsize": FS_LAB,
               "xtick.major.size": 1.8, "ytick.major.size": 1.8, "xtick.major.pad": 1.2, "ytick.major.pad": 1.2})
    GAP = 0.04                                   # white gap between panels, in
    ROWLAB, YAX, XAX, LEG = 0.30, 0.38, 0.30, 0.22
    if layout == "3x2":
        TITLE, TAG = 0.16, 0.155
        pw = (width - ROWLAB - YAX - GAP - 0.02) / 2
        ph = pw * aspect
        H = TITLE + 3 * (TAG + ph + XAX + GAP) + LEG
    else:
        TITLE, TAG = 0.15, 0.13
        pw = (width - ROWLAB - 3 * YAX - 2 * GAP - 0.02) / 3
        ph = pw * aspect
        H = TITLE + TAG + 2 * ph + GAP + XAX + LEG

    def panel(fig, x0, y0, sub, rgb, ylabel, xlabel, show_y):
        ax = fig.add_axes([x0 / width, y0 / H, pw / width, ph / H])
        _image(ax, rgb, sub)
        ax.set_xlim(*S._extent(sub)[:2]); ax.set_ylim(*S._extent(sub)[2:])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.tick_params(length=1.8, width=0.5, labelleft=show_y, labelbottom=xlabel is not None)
        if show_y:
            ax.set_ylabel(ylabel, labelpad=1)
        if xlabel:
            ax.set_xlabel(xlabel, labelpad=1)
        return ax

    gt = lambda sub: S._rgb(sub["labels"], {1: pal["success"], 0: pal["fail"]})
    dec = lambda p: S.decision_rgb(p["dec"], p["sub"]["labels"], palette)
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(width, H))
        axes, cell_pts = [], []
        if layout == "3x2":
            x_left = ROWLAB + YAX
            for c, name in enumerate(["Ground truth", method_label]):
                fig.text((x_left + c * (pw + GAP) + pw / 2) / width, (H - 0.02) / H, name, ha="center", va="top",
                         fontsize=FS_TITLE)
            for r, p in enumerate(P):
                sub = p["sub"]
                top = H - TITLE - r * (TAG + ph + XAX + GAP)
                y0 = top - TAG - ph
                fig.text((x_left + pw + GAP / 2) / width, (y0 + ph + 0.025) / H, p["tag"], ha="center", va="bottom",
                         fontsize=FS_TAG, color="#333333")
                a0 = panel(fig, x_left, y0, sub, gt(sub), p["axes"][1], p["axes"][0], True)
                a1 = panel(fig, x_left + pw + GAP, y0, sub, dec(p), None, p["axes"][0], False)
                a1.set_xlabel("")
                a0.set_xlabel("")
                fig.text((x_left + pw + GAP / 2) / width, (y0 - XAX + 0.015) / H, p["axes"][0], ha="center",
                         va="bottom", fontsize=FS_LAB)
                fig.text(0.02 / width, (y0 + ph / 2) / H, p["system"].replace(" (", "\n("), rotation=90, ha="left",
                         va="center", fontsize=FS_ROW, fontweight="bold", multialignment="center")   # two lines
                axes += [a0, a1]
                cell_pts.append(_cell_pt(a0, sub))
        else:
            for c, p in enumerate(P):
                sub = p["sub"]
                x0 = ROWLAB + YAX + c * (pw + YAX + GAP)
                fig.text((x0 + pw / 2) / width, (H - 0.02) / H, p["system"], ha="center", va="top", fontsize=FS_TITLE,
                         fontweight="bold")
                fig.text((x0 + pw / 2) / width, (H - TITLE - 0.01) / H, p["tag"], ha="center", va="top",
                         fontsize=FS_TAG, color="#333333")
                for r in range(2):
                    y0 = H - TITLE - TAG - (r + 1) * ph - r * GAP
                    rgb = gt(sub) if r == 0 else dec(p)
                    ax = panel(fig, x0, y0, sub, rgb, p["axes"][1], p["axes"][0] if r == 1 else None, True)
                    axes.append(ax)
                cell_pts.append(_cell_pt(axes[-1], sub))
            for r, name in enumerate(["Ground truth", method_label]):
                y0 = H - TITLE - TAG - (r + 1) * ph - r * GAP
                fig.text(0.02 / width, (y0 + ph / 2) / H, name, rotation=90, ha="left", va="center", fontsize=FS_ROW,
                         fontweight="bold")
        flat = lambda col, lab: Patch(facecolor=col, edgecolor="none", label=lab)
        handles = [flat(pal["success"], "TP"), flat(pal["fail"], "TN"), flat(pal["fp"], "FP"),
                   flat(pal["fn"], "FN"), flat(pal["abstain"], "Abstention")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=False,
                   fontsize=FS_LAB, handlelength=1.1, handleheight=0.9, handletextpad=0.3, columnspacing=0.8,
                   borderaxespad=0.0)
        # Fail loudly on overlapping text, text outside the page, or text over a panel.
        fig.canvas.draw()
        rdr = fig.canvas.get_renderer()
        boxes = [(t.get_text(), t.get_window_extent(rdr)) for t in fig.texts]
        def drawn_ticks(axis, lim):
            lo, hi = sorted(lim)
            return [t for t, v in zip(axis.get_ticklabels(), axis.get_majorticklocs())
                    if t.get_text() and t.get_visible() and lo - 1e-9 <= v <= hi + 1e-9]   # matplotlib skips the rest

        for ax in axes:
            boxes += [(t.get_text(), t.get_window_extent(rdr))
                      for t in drawn_ticks(ax.xaxis, ax.get_xlim()) + drawn_ticks(ax.yaxis, ax.get_ylim())]
            for lab in (ax.xaxis.label, ax.yaxis.label):
                if lab.get_text():
                    boxes.append((lab.get_text(), lab.get_window_extent(rdr)))
        leg = fig.legends[0].get_window_extent(rdr)
        boxes.append(("legend", leg))
        for i, (ta, a) in enumerate(boxes):
            for tb, b in boxes[i + 1:]:
                assert not a.overlaps(b), ("text overlap", ta, tb)
            assert fig.bbox.x0 - 0.5 <= a.x0 and a.x1 <= fig.bbox.x1 + 0.5 and a.y0 >= -0.5 and a.y1 <= fig.bbox.y1 + 0.5, \
                ("text outside the page", ta)
        panels = [ax.get_window_extent(rdr) for ax in axes]
        for ta, a in boxes:
            assert not any(a.overlaps(pb) for pb in panels), ("text over a panel", ta)
        out_stem = str(out_stem)
        png, pdf, txt = Path(f"{out_stem}.png"), Path(f"{out_stem}.pdf"), Path(f"{out_stem}.txt")
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=DPI)
        fig.savefig(pdf, dpi=DPI, metadata={"CreationDate": None})
        plt.close(fig)
    head = (f"Legend (paper terms, 06_experiments.tex): TP {pal['success']}, TN {pal['fail']}, FP {pal['fp']}, "
            f"FN {pal['fn']}, Abstention {pal['abstain']}; ground truth uses the TP / TN colours. No start box, no "
            "boundary line.\n" + "\n".join(f"[{c + 1}] {p['system']}: tag {p['tag']!r}; axes {p['axes']}"
                                             for c, p in enumerate(P)) + "\n")
    txt.write_text(head + _sidecar(P, layout, cell_pts, (width, H), pw, ph, pal, False))
    return str(png), str(pdf), str(txt)


# --- Figure 1, tight vertical layout ---------------------------------------------------------------

def _bold_math_rc() -> dict:
    """Bold italic axis names in mathtext (q, q-dot, q_z ...): DejaVu Sans Bold Oblique for variables and bold for
    the rest, so the dot over q is as heavy as the letter. Only the axis names use math in Figure 1."""
    return {"mathtext.fontset": "custom", "mathtext.it": "DejaVu Sans:italic:bold", "mathtext.rm": "DejaVu Sans:bold",
            "mathtext.bf": "DejaVu Sans:bold", "mathtext.sf": "DejaVu Sans:bold"}


def _grid_legend(fig, entries, fontsize, ncol):
    """A legend drawn as an aligned grid: each column as wide as its widest entry, entries left-aligned, filled row by row.
    ``entries``: (kind, colour, label), kind "patch" or "line". Returns (texts, handles, place, height_in);
    place(x_center_in, y_top_in, W, H) positions it and returns its (x0, y0, x1, y1) in inches."""
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    PT = 1 / 72
    hw, hh, pad, colsp, rowh = (0.9 * fontsize * PT, 0.8 * fontsize * PT, 0.3 * fontsize * PT, 0.8 * fontsize * PT,
                                1.35 * fontsize * PT)
    texts = [fig.text(0, 0, lab, fontsize=fontsize, ha="left", va="center") for _, _, lab in entries]
    handles = []
    for kind, col, _ in entries:
        a = (mlines.Line2D([0, 0], [0, 0], color=col, lw=0.8, transform=fig.transFigure) if kind == "line" else
             mpatches.Rectangle((0, 0), 0, 0, facecolor=col, edgecolor="none", transform=fig.transFigure))
        fig.add_artist(a)
        handles.append(a)
    nrow = -(-len(entries) // ncol)

    def place(xc, ytop, W, H):
        r_ = fig.canvas.get_renderer()
        tws = [t.get_window_extent(r_).width / fig.dpi for t in texts]
        # each column as wide as its widest entry; entries left-aligned, so both rows line up column by column
        cws = [hw + pad + max(tws[k] for k in range(c, len(texts), ncol)) for c in range(ncol)]
        total = sum(cws) + (ncol - 1) * colsp
        x0 = xc - total / 2
        lefts = [x0 + sum(cws[:c]) + c * colsp for c in range(ncol)]
        for k, (t, a, (kind, _, _)) in enumerate(zip(texts, handles, entries)):
            r, c = divmod(k, ncol)
            yc = ytop - (r + 0.5) * rowh
            cx = lefts[c]
            if kind == "line":
                a.set_data([cx / W, (cx + hw) / W], [yc / H, yc / H])
            else:
                a.set_xy((cx / W, (yc - hh / 2) / H)); a.set_width(hw / W); a.set_height(hh / H)
            t.set_position(((cx + hw + pad) / W, yc / H))
        return x0, ytop - nrow * rowh, x0 + total, ytop

    return texts, handles, place, nrow * rowh


def render_flagship_tight(entries, out_stem, width: float = 2.70, aspect: float = 0.86, palette: dict | None = None,
                          method_label: str = "TERA", col_gap: float = 0.04, margin: float = 0.01,
                          boundary: bool = True, row_gap_pt: float = 1.0, boundary_legend: bool = False,
                          boundary_label: str = "Boundary", fonts: dict | None = None,
                          legend_rows: int = 1, ybins: int = 4, xbins: int = 3, row_label_sep_pt: float = 2.0,
                          ticks: bool = True, ranges: list | None = None, bold_axis_names: bool = False,
                          legend_grid: bool = False, legend_top: bool = False, legend_kw: dict | None = None):
    """Figure 1, 3 rows (systems) x 2 columns (ground truth | method), packed as tightly as the text allows.

    Every gap is measured, not guessed: a first draw measures tick-label and axis-label extents, then panels,
    labels and the legend are placed in inches. One-line rotated bold row labels sit flush against the y labels;
    each row has only its x tick labels and one compact x label under the pair; the legend (TP, TN, FP, FN,
    Abstention) is one line centred on the panel block, just under the last x label. Outer whitespace is
    ``margin`` in. ``entries``: dicts with slice, pred, system, axes (x, y labels); "tag" is ignored (the fixed
    values go in the sidecar). Fails on any text overlap, text off the page or text over a panel."""
    from matplotlib.patches import Patch
    if len(entries) != 3:
        raise ValueError("three entries")
    P = [load_entry(e) for e in entries]
    for p, e in zip(P, entries):
        p["system"], p["axes"], p["tag"] = e["system"], e["axes"], e.get("tag", "")
    pal = S.palette_or_default(palette)
    fs = dict(tick=6.5, label=7.5, title=8, row=7.5, legend=7.5)
    fs.update(fonts or {})
    FS_TICK, FS_LAB, FS_TITLE, FS_ROW, FS_LEG = fs["tick"], fs["label"], fs["title"], fs["row"], fs["legend"]
    PT = 1 / 72
    TICK, PAD = 2.0, 1.0                                   # tick length and pads, pt
    rc = dict(RC)
    rc.update({"xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "axes.labelsize": FS_LAB,
               "xtick.major.size": TICK, "ytick.major.size": TICK, "xtick.major.pad": PAD, "ytick.major.pad": PAD,
               "xtick.major.width": .5, "ytick.major.width": .5})
    if bold_axis_names:
        rc.update(_bold_math_rc())
    gt = lambda sub: S._rgb(sub["labels"], {1: pal["success"], 0: pal["fail"]})
    dec = lambda p: S.decision_rgb(p["dec"], p["sub"]["labels"], palette)
    flat = lambda col, lab: Patch(facecolor=col, edgecolor="none", label=lab)
    handles = [flat(pal["success"], "TP"), flat(pal["fail"], "TN"), flat(pal["fp"], "FP"), flat(pal["fn"], "FN"),
               flat(pal["abstain"], "Abstention")]
    if boundary and boundary_legend:
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], color="black", lw=0.8, label=boundary_label))   # true success/failure boundary

    with plt.rc_context(rc):
        H0 = 4.0
        fig = plt.figure(figsize=(width, H0))
        axes = []
        for r, p in enumerate(P):
            sub = p["sub"]
            row = []
            for c in range(2):
                ax = fig.add_axes([0.3 + 0.35 * c, 0.7 - 0.3 * r, 0.3, 0.2])
                _image(ax, gt(sub) if c == 0 else dec(p), sub)
                if boundary:
                    _boundary(ax, sub)                     # black GT boundary, as in the earlier composites
                ax.set_xlim(*S._extent(sub)[:2]); ax.set_ylim(*S._extent(sub)[2:])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins, steps=[1, 2, 2.5, 5, 10]))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins, steps=[1, 2, 2.5, 5, 10]))
                ax.tick_params(labelleft=(c == 0))
                if not ticks:                              # dm1487: no tick marks or numbers; ranges go in the caption
                    ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(p["axes"][1], labelpad=PAD)
                row.append(ax)
            axes.append(row)
        xl = [fig.text(0, 0, p["axes"][0], ha="center", va="top", fontsize=FS_LAB) for p in P]
        rl = [fig.text(0, 0, p["system"], rotation=90, ha="left", va="center", fontsize=FS_ROW, fontweight="bold")
              for p in P]
        tl = [fig.text(0, 0, name, ha="center", va="bottom", fontsize=FS_TITLE) for name in ("Ground truth", method_label)]
        if legend_rows == 2 and not legend_grid:
            # two centred lines of three, read across: matplotlib fills legend columns first, so interleave
            half = (len(handles) + 1) // 2
            first, second = handles[:half], handles[half:]
            handles = [h for k in range(half) for h in ([first[k]] + ([second[k]] if k < len(second) else []))]
        grid_texts, grid_handles = [], []
        if legend_grid:                                    # aligned grid, filled row by row
            spec = [("patch", pal["success"], "TP"), ("patch", pal["fail"], "TN"), ("patch", pal["fp"], "FP"),
                    ("patch", pal["fn"], "FN"), ("patch", pal["abstain"], "Abstention")]
            if boundary and boundary_legend:
                spec.append(("line", "black", boundary_label))
            grid_texts, grid_handles, grid_place, grid_h = _grid_legend(fig, spec, FS_LEG, (len(spec) + 1) // 2)
            leg = None
        elif legend_kw is not None:                        # the paper figures' shared legend style
            leg = fig.legend(handles=handles, ncol=len(handles) if legend_rows == 1 else (len(handles) + 1) // 2,
                             **legend_kw)
        else:
            leg = fig.legend(handles=handles, loc="upper center", ncol=len(handles) if legend_rows == 1 else (len(handles) + 1) // 2, frameon=False, fontsize=FS_LEG, handlelength=0.9,
                             handleheight=0.8, handletextpad=0.25, columnspacing=0.45, borderaxespad=0.0, borderpad=0.0)

        # Pass 1: measure decorations in inches.
        fig.canvas.draw()
        r_ = fig.canvas.get_renderer()
        inch = lambda bb: (bb.width / fig.dpi, bb.height / fig.dpi)

        def dec_left(ax):
            return ax.get_position().x0 * width - ax.get_tightbbox(r_).x0 / fig.dpi

        def dec_below(ax):
            return ax.get_position().y0 * H0 - ax.get_tightbbox(r_).y0 / fig.dpi

        left_dec = max(dec_left(row[0]) for row in axes)
        below = [max(dec_below(a) for a in row) for row in axes]
        xl_h = [inch(t.get_window_extent(r_))[1] for t in xl]
        rl_w = max(inch(t.get_window_extent(r_))[0] for t in rl)
        rl_len = [inch(t.get_window_extent(r_))[1] for t in rl]
        tl_h = max(inch(t.get_window_extent(r_))[1] for t in tl)
        leg_h = inch(leg.get_window_extent(r_))[1] if leg is not None else grid_h

        # Geometry, inches. Panels share one x range across rows so the columns line up.
        x_panel = margin + rl_w + PAD * PT + left_dec
        pw = (width - margin - x_panel - col_gap) / 2
        strips = [below[i] + PAD * PT + xl_h[i] for i in range(3)]        # x tick labels, pad, x label
        ph = pw * aspect
        # A one-line rotated label is centred on its panel; keep neighbouring labels apart.
        for i in range(2):
            pitch = ph + strips[i] + row_gap_pt * PT
            need = (rl_len[i] + rl_len[i + 1]) / 2 + row_label_sep_pt * PT
            ph = max(ph, ph + need - pitch)
        LEG_GAP = (3 if legend_top else 2) * PT            # legend to column titles (top) or last x label (bottom)
        H = margin + tl_h + PAD * PT + 3 * ph + sum(strips) + 2 * row_gap_pt * PT + LEG_GAP + leg_h + margin
        fig.set_size_inches(width, H)
        y_top = H - margin - tl_h - PAD * PT - (leg_h + LEG_GAP if legend_top else 0)
        for c, t in enumerate(tl):
            t.set_position(((x_panel + c * (pw + col_gap) + pw / 2) / width, y_top / H + PAD * PT / H))
        y = y_top
        mid_x = (x_panel + pw + col_gap / 2) / width
        for i, row in enumerate(axes):
            y0 = y - ph
            for c, ax in enumerate(row):
                ax.set_position([(x_panel + c * (pw + col_gap)) / width, y0 / H, pw / width, ph / H])
            xl[i].set_position((mid_x, (y0 - below[i] - PAD * PT) / H))
            rl[i].set_position((margin / width, (y0 + ph / 2) / H))
            y = y0 - strips[i] - row_gap_pt * PT
        block_mid = (x_panel + (x_panel + 2 * pw + col_gap)) / 2 / width
        if leg is not None and legend_top:                 # top of the figure, centred on the figure
            leg.set_bbox_to_anchor((0.5, (H - margin) / H), transform=fig.transFigure)
        elif leg is not None:
            leg.set_bbox_to_anchor((block_mid, (y + row_gap_pt * PT - 2 * PT) / H), transform=fig.transFigure)
        else:
            fig.canvas.draw()
            grid_bb = grid_place(block_mid * width, y + row_gap_pt * PT - 2 * PT, width, H)

        # Checks: overlaps, page edges, text over panels, outer whitespace.
        fig.canvas.draw()
        r_ = fig.canvas.get_renderer()

        def drawn_ticks(axis, lim):
            lo, hi = sorted(lim)
            return [t for t, v in zip(axis.get_ticklabels(), axis.get_majorticklocs())
                    if t.get_text() and t.get_visible() and lo - 1e-9 <= v <= hi + 1e-9]

        flat_axes = [a for row in axes for a in row]
        boxes = [(t.get_text(), t.get_window_extent(r_)) for t in fig.texts]
        for ax in flat_axes:
            boxes += [(t.get_text(), t.get_window_extent(r_))
                      for t in drawn_ticks(ax.xaxis, ax.get_xlim()) + drawn_ticks(ax.yaxis, ax.get_ylim())]
            if ax.yaxis.label.get_text():
                boxes.append((ax.yaxis.label.get_text(), ax.yaxis.label.get_window_extent(r_)))
        if leg is not None:
            boxes.append(("legend", leg.get_window_extent(r_)))
        else:                                              # grid texts are in fig.texts already; add the swatches
            boxes += [("legend swatch", a.get_window_extent(r_)) for a in grid_handles]
        for i, (ta, a) in enumerate(boxes):
            for tb, b in boxes[i + 1:]:
                assert not a.overlaps(b), ("text overlap", ta, tb)
            assert a.x0 >= -0.5 and a.x1 <= fig.bbox.x1 + 0.5 and a.y0 >= -0.5 and a.y1 <= fig.bbox.y1 + 0.5, \
                ("text off the page", ta)
        panels = [ax.get_window_extent(r_) for ax in flat_axes]
        for ta, a in boxes:
            assert not any(a.overlaps(pb) for pb in panels), ("text over a panel", ta)
        ink = [b for _, b in boxes] + panels
        x0 = min(b.x0 for b in ink) / fig.dpi; x1 = max(b.x1 for b in ink) / fig.dpi
        y0_ = min(b.y0 for b in ink) / fig.dpi; y1 = max(b.y1 for b in ink) / fig.dpi
        edges = dict(left=x0, right=width - x1, bottom=y0_, top=H - y1)
        assert all(-0.005 <= v <= 0.03 for v in edges.values()), ("outer whitespace", edges)
        if leg is not None:
            leg_bb = leg.get_window_extent(r_)
            leg_mid = (leg_bb.x0 + leg_bb.x1) / 2 / fig.dpi
        else:
            leg_mid = (grid_bb[0] + grid_bb[2]) / 2
        want_mid = width / 2 if legend_top else block_mid * width
        assert abs(leg_mid - want_mid) < 0.005, ("legend not centred", leg_mid, want_mid)

        out_stem = str(out_stem)
        png, pdf, txt = Path(f"{out_stem}.png"), Path(f"{out_stem}.pdf"), Path(f"{out_stem}.txt")
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=DPI)
        fig.savefig(pdf, dpi=DPI, metadata={"CreationDate": None})
        plt.close(fig)
    cell_pts = [72 * min(pw / p["sub"]["nx"], ph / p["sub"]["ny"]) for p in P]
    rng = "" if not ranges else "Axis ranges (for the caption, checked against the slices): " + "; ".join(ranges) + ".\n"
    head = (f"Fonts (pt): {fs}. Ticks {'shown' if ticks else 'none (no marks, no numbers)'}.\n" + rng + f"Tight layout: {width:.2f} x {H:.2f} in, panels {pw:.3f} x {ph:.3f} in, column gap {col_gap} in, "
            f"outer whitespace (in) {', '.join(f'{k} {v:.3f}' for k, v in edges.items())}; legend centred on the "
            f"{'figure, at the top' if legend_top else 'panel block'}.\nLegend (paper terms, 06_experiments.tex): TP {pal['success']}, TN {pal['fail']}, FP {pal['fp']}, "
            f"FN {pal['fn']}, Abstention {pal['abstain']}; ground truth uses the TP / TN colours. No start box; "
            f"GT boundary {'drawn (black) in every panel' if boundary else 'not drawn'}"
            f"{' and in the legend' if boundary and boundary_legend else ' (not in the legend; name it in the caption)' if boundary else ''}; no slice tags on the figure (fixed values below and in the caption).\n"
            + "\n".join(f"[{c + 1}] {p['system']}: slice {p['tag']!r}; axes {p['axes']}" for c, p in enumerate(P)) + "\n")
    txt.write_text(head + _sidecar(P, "3x2 tight", cell_pts, (width, H), pw, ph, pal, False))
    return str(png), str(pdf), str(txt), H


def render_flagship_tight_h(entries, out_stem, width: float = 3.30, aspect: float = 0.86, palette: dict | None = None,
                            method_label: str = "TERA", col_gap: float = 0.08, row_gap: float = 1 / 72,
                            margin: float = 0.01, boundary: bool = True, boundary_legend: bool = True,
                            boundary_label: str = "GT boundary", fonts: dict | None = None, ticks: bool = True,
                            ranges: list | None = None, bold_axis_names: bool = False, legend_top: bool = False,
                            legend_kw: dict | None = None, base_rc: dict | None = None,
                            boundary_lw: float | None = None, inspect=None):
    """Figure 1, horizontal: 2 rows (ground truth | method) x 3 columns (systems), packed from measured text
    extents like render_flagship_tight. GT and method share x within a column, so x tick labels and the x label sit
    under the bottom row only; each column has its own y label and ticks on its left; rotated bold row labels at the
    far left; system names as column titles; one legend line centred on the panel block. Same checks, plus a check
    that no panel image or boundary line is clipped.

    Restyle hooks (dm1487, 2026-11-13; all default to the old behaviour, so the old outputs stay byte-identical):
    ``base_rc`` replaces this module's DejaVu RC as the base rcParams (the restyle passes common.style_rc(), usetex
    Times bold); ``boundary_lw`` sets the GT boundary line width in the panels and in the legend (default .55 and .8);
    ``inspect(fig, cols)`` is called after the checks and before saving, so a caller can read the drawn artists."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    P = [load_entry(e) for e in entries]
    for p, e in zip(P, entries):
        p["system"], p["axes"], p["tag"] = e["system"], e["axes"], e.get("tag", "")
    pal = S.palette_or_default(palette)
    fs = dict(tick=6.5, label=7.5, title=8, row=7.5, legend=7.5)
    fs.update(fonts or {})
    PT, TICK, PAD = 1 / 72, 2.0, 1.0
    rc = dict(RC if base_rc is None else base_rc)
    rc.update({"xtick.labelsize": fs["tick"], "ytick.labelsize": fs["tick"], "axes.labelsize": fs["label"],
               "xtick.major.size": TICK, "ytick.major.size": TICK, "xtick.major.pad": PAD, "ytick.major.pad": PAD,
               "xtick.major.width": .5, "ytick.major.width": .5})
    if bold_axis_names:
        rc.update(_bold_math_rc())
    gt = lambda sub: S._rgb(sub["labels"], {1: pal["success"], 0: pal["fail"]})
    dec = lambda p: S.decision_rgb(p["dec"], p["sub"]["labels"], palette)
    flat = lambda col, lab: Patch(facecolor=col, edgecolor="none", label=lab)
    handles = [flat(pal["success"], "TP"), flat(pal["fail"], "TN"), flat(pal["fp"], "FP"), flat(pal["fn"], "FN"),
               flat(pal["abstain"], "Abstention")]
    if boundary and boundary_legend:
        handles.append(Line2D([], [], color="black", lw=0.8 if boundary_lw is None else boundary_lw,
                              label=boundary_label))
    with plt.rc_context(rc):
        H0 = 2.5
        fig = plt.figure(figsize=(width, H0))
        cols = []
        for c, p in enumerate(P):
            sub = p["sub"]
            col = []
            for r in range(2):
                ax = fig.add_axes([0.15 + 0.3 * c, 0.55 - 0.35 * r, 0.2, 0.3])
                _image(ax, gt(sub) if r == 0 else dec(p), sub)
                if boundary:
                    _boundary(ax, sub) if boundary_lw is None else _boundary(ax, sub, lw=boundary_lw)
                ax.set_xlim(*S._extent(sub)[:2]); ax.set_ylim(*S._extent(sub)[2:])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
                ax.set_ylabel(p["axes"][1], labelpad=PAD)
                ax.tick_params(labelbottom=(r == 1))
                if not ticks:                              # dm1487: no tick marks or numbers; ranges go in the caption
                    ax.set_xticks([]); ax.set_yticks([])
                if r == 1:
                    ax.set_xlabel(p["axes"][0], labelpad=PAD)
                col.append(ax)
            cols.append(col)
        tl = [fig.text(0, 0, p["system"], ha="center", va="bottom", fontsize=fs["title"]) for p in P]
        rl = [fig.text(0, 0, name, rotation=90, ha="left", va="center", fontsize=fs["row"], fontweight="bold")
              for name in ("Ground truth", method_label)]
        if legend_kw is not None:                          # the paper figures' shared legend style
            leg = fig.legend(handles=handles, ncol=len(handles), **legend_kw)
        else:
            leg = fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=fs["legend"],
                             handlelength=0.9, handleheight=0.8, handletextpad=0.25, columnspacing=0.45,
                             borderaxespad=0.0, borderpad=0.0)
        fig.canvas.draw()

        def inward_edge_labels():
            # Every tick label stays (dm1487), but the two that face the narrow gap between the rows are anchored
            # inside their panel (top row's lowest: bottom-aligned; bottom row's highest: top-aligned).
            for col in cols:
                for r, ax in enumerate(col):
                    lo_v, hi_v = sorted(ax.get_ylim())
                    drawn = [(v, t) for v, t in zip(ax.yaxis.get_majorticklocs(), ax.yaxis.get_major_ticks())
                             if lo_v - 1e-9 <= v <= hi_v + 1e-9]
                    if not drawn:
                        continue
                    v, t = min(drawn, key=lambda d: d[0]) if r == 0 else max(drawn, key=lambda d: d[0])
                    edge = lo_v if r == 0 else hi_v
                    if abs(v - edge) <= 0.08 * (hi_v - lo_v):          # only a label sitting at the gap edge
                        t.label1.set_verticalalignment("bottom" if r == 0 else "top")

        inward_edge_labels()
        fig.canvas.draw()
        r_ = fig.canvas.get_renderer()
        left_dec = [max(a.get_position().x0 * width - a.get_tightbbox(r_).x0 / fig.dpi for a in col) for col in cols]
        below = max(col[1].get_position().y0 * H0 - col[1].get_tightbbox(r_).y0 / fig.dpi for col in cols)
        # top y tick labels stick out above the top panels; lift the column titles by that much
        above = max(0.0, max(col[0].get_tightbbox(r_).y1 / fig.dpi - (col[0].get_position().y0 + col[0].get_position().height) * H0
                             for col in cols))
        # y tick labels overhang each panel's top and bottom edge; the gap between the rows must clear both, since
        # every tick label stays (dm1487: no ticks removed)
        def overhang(ax):
            pos = ax.get_position()
            lo_v, hi_v = sorted(ax.get_ylim())
            lab = [t.get_window_extent(r_) for t, v in zip(ax.get_yticklabels(), ax.yaxis.get_majorticklocs())
                   if t.get_text() and t.get_visible() and lo_v - 1e-9 <= v <= hi_v + 1e-9]   # drawn ticks only
            lo = min(b.y0 for b in lab) / fig.dpi if lab else pos.y0 * H0
            hi = max(b.y1 for b in lab) / fig.dpi if lab else (pos.y0 + pos.height) * H0
            return max(0.0, pos.y0 * H0 - lo), max(0.0, hi - (pos.y0 + pos.height) * H0)
        need_gap = max(overhang(col[0])[0] + overhang(col[1])[1] for col in cols) + PAD * PT
        row_gap = max(row_gap, need_gap)
        rl_w = max(t.get_window_extent(r_).width for t in rl) / fig.dpi
        tl_h = max(t.get_window_extent(r_).height for t in tl) / fig.dpi
        tl_w = [t.get_window_extent(r_).width / fig.dpi for t in tl]
        leg_h = leg.get_window_extent(r_).height / fig.dpi
        x0 = margin + rl_w + PAD * PT
        pw = (width - margin - x0 - sum(left_dec) - 2 * col_gap) / 3
        ph = pw * aspect
        # The rotated row labels are centred on their panels; with a 1 pt row gap they need ph >= their mean length.
        rl_len = [t.get_window_extent(r_).height / fig.dpi for t in rl]
        ph = max(ph, (rl_len[0] + rl_len[1]) / 2 + 6 * PT - row_gap)     # 6 pt between the two labels
        LEG_GAP = (3 if legend_top else 2) * PT
        H = margin + tl_h + PAD * PT + above + 2 * ph + row_gap + below + LEG_GAP + leg_h + margin
        fig.set_size_inches(width, H)
        y_top = H - margin - tl_h - PAD * PT - above - (leg_h + LEG_GAP if legend_top else 0)
        x = x0
        panel_x = []
        for c, col in enumerate(cols):
            x += left_dec[c]
            for r, ax in enumerate(col):
                ax.set_position([x / width, (y_top - (r + 1) * ph - r * row_gap) / H, pw / width, ph / H])
            # centred on the panel, but kept on the page (a long title over the last panel shifts left)
            xc = min(max(x + pw / 2, margin + tl_w[c] / 2), width - margin - tl_w[c] / 2)
            tl[c].set_position((xc / width, (y_top + above + PAD * PT) / H))
            panel_x.append(x)
            x += pw + col_gap
        for r, t in enumerate(rl):
            t.set_position((margin / width, (y_top - r * (ph + row_gap) - ph / 2) / H))
        block_mid = (panel_x[0] + panel_x[-1] + pw) / 2 / width
        if legend_top:                                     # top of the figure, centred on the figure
            leg.set_bbox_to_anchor((0.5, (H - margin) / H), transform=fig.transFigure)
        else:
            y_leg = y_top - 2 * ph - row_gap - below - LEG_GAP
            leg.set_bbox_to_anchor((block_mid, y_leg / H), transform=fig.transFigure)

        fig.canvas.draw()
        inward_edge_labels()
        fig.canvas.draw()
        r_ = fig.canvas.get_renderer()

        def drawn_ticks(axis, lim):
            lo, hi = sorted(lim)
            return [t for t, v in zip(axis.get_ticklabels(), axis.get_majorticklocs())
                    if t.get_text() and t.get_visible() and lo - 1e-9 <= v <= hi + 1e-9]

        axes = [a for col in cols for a in col]
        boxes = [(t.get_text(), t.get_window_extent(r_)) for t in fig.texts]
        for ax in axes:
            boxes += [(t.get_text(), t.get_window_extent(r_))
                      for t in drawn_ticks(ax.xaxis, ax.get_xlim()) + drawn_ticks(ax.yaxis, ax.get_ylim())]
            for lab in (ax.xaxis.label, ax.yaxis.label):
                if lab.get_text():
                    boxes.append((lab.get_text(), lab.get_window_extent(r_)))
        boxes.append(("legend", leg.get_window_extent(r_)))
        for i, (ta, a) in enumerate(boxes):
            for tb, b in boxes[i + 1:]:
                assert not a.overlaps(b), ("text overlap", ta, tb)
            assert a.x0 >= -0.5 and a.x1 <= fig.bbox.x1 + 0.5 and a.y0 >= -0.5 and a.y1 <= fig.bbox.y1 + 0.5, \
                ("text off the page", ta)
        panels = [ax.get_window_extent(r_) for ax in axes]
        for ta, a in boxes:
            assert not any(a.overlaps(pb) for pb in panels), ("text over a panel", ta)
        ink = [b for _, b in boxes] + panels
        edges = dict(left=min(b.x0 for b in ink) / fig.dpi, right=width - max(b.x1 for b in ink) / fig.dpi,
                     bottom=min(b.y0 for b in ink) / fig.dpi, top=H - max(b.y1 for b in ink) / fig.dpi)
        assert all(-0.005 <= v <= 0.03 for v in edges.values()), ("outer whitespace", edges)
        lb = leg.get_window_extent(r_)
        want_mid = width / 2 if legend_top else block_mid * width
        assert abs((lb.x0 + lb.x1) / 2 / fig.dpi - want_mid) < 0.005, "legend not centred"
        # Clipped data: every image shows its whole extent (axes limits = image extent), every boundary vertex lies
        # inside its panel, and every panel lies on the page.
        for ax in axes:
            xl_, yl_ = sorted(ax.get_xlim()), sorted(ax.get_ylim())
            for im in ax.get_images():
                e = im.get_extent()
                assert np.allclose([min(e[:2]), max(e[:2]), min(e[2:]), max(e[2:])], xl_ + yl_), ("image clipped", e)
            for coll in ax.collections:
                for path in coll.get_paths():
                    v = path.vertices
                    if len(v):
                        assert (v[:, 0] >= xl_[0] - 1e-9).all() and (v[:, 0] <= xl_[1] + 1e-9).all() and \
                               (v[:, 1] >= yl_[0] - 1e-9).all() and (v[:, 1] <= yl_[1] + 1e-9).all(), "boundary clipped"
            pb = ax.get_window_extent(r_)
            assert pb.x0 >= -0.5 and pb.y0 >= -0.5 and pb.x1 <= fig.bbox.x1 + 0.5 and pb.y1 <= fig.bbox.y1 + 0.5, \
                "panel off the page"
        if inspect is not None:
            inspect(fig, cols)
        out_stem = str(out_stem)
        png, pdf, txt = Path(f"{out_stem}.png"), Path(f"{out_stem}.pdf"), Path(f"{out_stem}.txt")
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=DPI)
        fig.savefig(pdf, dpi=DPI, metadata={"CreationDate": None})
        plt.close(fig)
    cell_pts = [72 * min(pw / p["sub"]["nx"], ph / p["sub"]["ny"]) for p in P]
    rng = "" if not ranges else "Axis ranges (for the caption, checked against the slices): " + "; ".join(ranges) + ".\n"
    head = (f"Fonts (pt): {fs}. Ticks {'shown' if ticks else 'none (no marks, no numbers)'}.\n" + rng + f"Tight horizontal layout: {width:.2f} x {H:.2f} in, panels {pw:.3f} x {ph:.3f} in, column gap "
            f"{col_gap} in, row gap {row_gap} in, outer whitespace (in) {', '.join(f'{k} {v:.3f}' for k, v in edges.items())}; "
            f"legend centred on the {'figure, at the top' if legend_top else 'panel block'}.\nLegend: TP {pal['success']}, TN {pal['fail']}, FP {pal['fp']}, FN {pal['fn']}, "
            f"Abstention {pal['abstain']}" + (f", {boundary_label} (black line)" if boundary and boundary_legend else "") + ".\n"
            + "\n".join(f"[{c + 1}] {p['system']}: slice {p['tag']!r}; axes {p['axes']}" for c, p in enumerate(P)) + "\n")
    txt.write_text(head + _sidecar(P, "2x3 tight horizontal", cell_pts, (width, H), pw, ph, pal, False))
    return str(png), str(pdf), str(txt), H
