"""Slice loading and the ground-truth / TERA-FM figure style.

``load_slice`` reads the standardized npz (schema in slices/README.md). ``render_gt`` draws the
ground truth alone; ``render_pair`` draws ground truth left and TERA-FM's certified decisions
right. Both write ``<out_stem>.png`` (200 dpi) and ``<out_stem>.pdf`` (image layers rasterized,
text and lines vector). The layout is fixed in inches, so every figure of a kind has the same
size and tiles cleanly in a contact sheet.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Okabe-Ito blue/orange as in dv_l4dc_plot/scripts/common.py (BLUE, ORANGE); abstain a medium gray.
SUCCESS, FAIL, ABSTAIN = "#0072B2", "#D55E00", "#A8A8A8"
# Wrong calls, Success as the positive class (paper: FDR = FP/(TP+FP), FOR = FN/(FN+TN)). Filled cells.
# FP sits in GT-fail territory beside orange, FN in GT-success territory beside blue. Grey levels
# (Rec. 601 luma): FN yellow .84 > abstain .66 > fail orange .47 > success blue .34 > FP crimson .20,
# so the five classes stay apart in grayscale, and each wrong colour differs most from its neighbours.
FP_COLOR, FN_COLOR = "#A3001E", "#F0E442"
FP_LABEL, FN_LABEL = "FP (Success declared, was Fail)", "FN (Fail declared, was Success)"
WRONG = FP_COLOR           # kept for old callers
BOUNDARY, BOX = "black", "#3C3C3C"
DPI = 200
# Fonts as dv_l4dc_plot/scripts/common.py RC, one step larger so a thumbnail stays legible.
RC = {"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
      "legend.fontsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .65,
      "pdf.fonttype": 42, "ps.fonttype": 42}

REQUIRED = ["setting", "system_label", "controller", "plane", "dim_names", "dim_tex", "dim_units", "xs", "ys",
            "labels", "model_states", "fixed_json", "source", "source_note", "gt_check_json"]


# --- loading -------------------------------------------------------------------------------------

def _scalar_str(a) -> str:
    a = np.asarray(a)
    if a.ndim != 0:
        raise ValueError(f"expected a scalar string, got shape {a.shape}")
    return str(a.item())


def load_slice(path) -> dict:
    """Read a slice npz into a dict with python strings, lists and numpy arrays.

    Adds ``fixed`` (the parsed fixed_json), ``gt_check`` (parsed gt_check_json), ``nx``, ``ny``,
    ``path``, and ``box_x`` / ``box_y`` (float[2] or None when the keys are absent). Raises on a
    missing key, a shape mismatch, or non-ascending axes.
    """
    z = np.load(path, allow_pickle=False)
    missing = [k for k in REQUIRED if k not in z.files]
    if missing:
        raise KeyError(f"{path}: missing keys {missing}")
    sl = {"path": str(path)}
    for k in ("setting", "system_label", "controller", "plane", "fixed_json", "source", "source_note",
              "gt_check_json"):
        sl[k] = _scalar_str(z[k])
    for k in ("dim_names", "dim_tex", "dim_units"):
        v = [str(s) for s in np.asarray(z[k]).ravel()]
        if len(v) != 2:
            raise ValueError(f"{path}: {k} must have 2 entries, got {v}")
        sl[k] = v
    xs, ys = np.asarray(z["xs"], float), np.asarray(z["ys"], float)
    labels = np.asarray(z["labels"])
    ms = np.asarray(z["model_states"])
    if labels.shape != (len(ys), len(xs)):
        raise ValueError(f"{path}: labels {labels.shape} != (len(ys), len(xs)) = {(len(ys), len(xs))}")
    if ms.ndim != 2 or ms.shape[0] != labels.size:
        raise ValueError(f"{path}: model_states {ms.shape} does not have ny*nx = {labels.size} rows")
    if not set(np.unique(labels)) <= {0, 1}:
        raise ValueError(f"{path}: labels must be 0/1, got {np.unique(labels)}")
    for name, c in (("xs", xs), ("ys", ys)):
        if len(c) > 1 and not np.all(np.diff(c) > 0):
            raise ValueError(f"{path}: {name} is not strictly ascending")
    sl.update(xs=xs, ys=ys, labels=labels.astype(np.int8), model_states=ms, nx=len(xs), ny=len(ys))
    sl["fixed"] = json.loads(sl["fixed_json"]) if sl["fixed_json"].strip() else {}
    sl["gt_check"] = json.loads(sl["gt_check_json"]) if sl["gt_check_json"].strip() else {}
    for k in ("box_x", "box_y"):
        if k in z.files:
            b = np.asarray(z[k], float).ravel()
            if b.shape != (2,):
                raise ValueError(f"{path}: {k} must be float[2], got {b.shape}")
            sl[k] = b
        else:
            sl[k] = None
    return sl


# --- geometry helpers ----------------------------------------------------------------------------

def cell_edges(c) -> np.ndarray:
    """Edges of cells centred on ascending ``c`` (midpoints, end cells mirrored)."""
    c = np.asarray(c, float)
    if len(c) == 1:
        return np.array([c[0] - .5, c[0] + .5])
    mid = (c[1:] + c[:-1]) / 2
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def _extent(sl):
    ex, ey = cell_edges(sl["xs"]), cell_edges(sl["ys"])
    return [ex[0], ex[-1], ey[0], ey[-1]]


def _rgb(idx: np.ndarray, palette: dict) -> np.ndarray:
    out = np.zeros(idx.shape + (3,))
    for v, col in palette.items():
        out[idx == v] = matplotlib.colors.to_rgb(col)
    return out


def _axis_label(sl, i):
    u = sl["dim_units"][i].strip()
    return f"{sl['dim_tex'][i]} [{u}]" if u else sl["dim_tex"][i]


def _fmt_val(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:.3g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_val(x) for x in v) + "]"
    return str(v)


def fixed_line(sl) -> str:
    parts = [f"{k} = {_fmt_val(v)}" for k, v in sl["fixed"].items()]
    s = "Fixed: " + (", ".join(parts) if parts else "none")
    if len(np.unique(sl["labels"])) == 1:
        s += "  (single class)"
    return s


def _draw_boundary(ax, sl):
    L = sl["labels"]
    if len(np.unique(L)) < 2:
        return False
    ax.contour(sl["xs"], sl["ys"], L.astype(float), levels=[0.5], colors=BOUNDARY, linewidths=0.6)
    return True


def _draw_box(ax, sl) -> bool:
    """Dashed lines on the finite sides of the start box that fall inside the plotted extent."""
    bx, by = sl.get("box_x"), sl.get("box_y")
    if bx is None and by is None:
        return False
    x0, x1, y0, y1 = _extent(sl)
    nan2 = np.array([np.nan, np.nan])
    bx = nan2 if bx is None else bx
    by = nan2 if by is None else by
    # Clip the box to the plot so an unbounded or out-of-view side becomes the plot edge.
    cx0 = x0 if np.isnan(bx[0]) else max(x0, bx[0])
    cx1 = x1 if np.isnan(bx[1]) else min(x1, bx[1])
    cy0 = y0 if np.isnan(by[0]) else max(y0, by[0])
    cy1 = y1 if np.isnan(by[1]) else min(y1, by[1])
    if cx0 >= cx1 or cy0 >= cy1:
        return False
    kw = dict(color=BOX, lw=0.9, ls=(0, (3.5, 2)), zorder=4)
    drew = False
    for v, lo_side in ((bx[0], True), (bx[1], False)):
        if not np.isnan(v) and x0 < v < x1:
            ax.plot([v, v], [cy0, cy1], **kw)
            drew = True
    for v, lo_side in ((by[0], True), (by[1], False)):
        if not np.isnan(v) and y0 < v < y1:
            ax.plot([cx0, cx1], [v, v], **kw)
            drew = True
    return drew


def _style_axes(ax, sl, ylabels=True):
    ax.set_xlim(*_extent(sl)[:2])
    ax.set_ylim(*_extent(sl)[2:])
    ax.set_xlabel(_axis_label(sl, 0))
    if ylabels:
        ax.set_ylabel(_axis_label(sl, 1))
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(length=2.5, width=.6, pad=2)


# --- layout --------------------------------------------------------------------------------------
# Fixed in inches and saved without a tight bbox, so every figure of a kind (and every slice of a
# setting, whose fixed-values line wraps the same way) has the same size and tiles cleanly.

PANEL_W = 2.75     # in
LEFT, GAP, RIGHT = 0.62, 0.34, 0.20
TOP = 0.50                            # figure title + panel titles
XLAB, LEG_ROW, LINE = 0.40, 0.19, 0.135   # below the panels: x labels, each legend row, each text line
LINE_FS = 6.8                          # pt, footer text


def _panel_aspect(sl) -> float:
    """Panel height / width: the grid's ny/nx, kept within [0.6, 1.0]."""
    return float(np.clip(sl["ny"] / sl["nx"], 0.6, 1.0))


def _width(n_panels):
    return LEFT + n_panels * PANEL_W + (n_panels - 1) * GAP + RIGHT


def _wrap(lines, W):
    """Wrap footer lines to the figure width (DejaVu Sans averages ~0.55 em per character)."""
    import textwrap
    n = max(20, int((W - 0.2) * 72 / (0.56 * LINE_FS)))
    out = []
    for s in lines:
        out += textwrap.wrap(s, n, break_long_words=False) or [""]
    return out


def _figure(sl, n_panels, lines, rows):
    ph = PANEL_W * _panel_aspect(sl)
    W = _width(n_panels)
    bottom = XLAB + rows * LEG_ROW + 0.05 + len(lines) * LINE + 0.06
    H = TOP + ph + bottom
    fig = plt.figure(figsize=(W, H))
    axes = []
    for k in range(n_panels):
        x = LEFT + k * (PANEL_W + GAP)
        axes.append(fig.add_axes([x / W, bottom / H, PANEL_W / W, ph / H]))
    return fig, axes, W, H, bottom


def _title(sl):
    return f"{sl['system_label']} {sl['controller']}: {sl['dim_tex'][0]} vs {sl['dim_tex'][1]}"


def _cell_points(ax, sl) -> float:
    """Side of one cell in points (the smaller of width and height)."""
    bb = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    return 72 * min(bb.width * fw / sl["nx"], bb.height * fh / sl["ny"])


def _footer(fig, W, H, bottom, handle_rows, lines, x_center):
    """Legend rows (one centred row per list of handles) under the x labels, then the text lines."""
    y = bottom - XLAB
    for handles in handle_rows:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(x_center, y / H), ncol=len(handles),
                   frameon=False, handlelength=1.4, handletextpad=0.4, columnspacing=1.0, borderaxespad=0)
        y -= LEG_ROW
    y -= 0.05
    for s in lines:
        fig.text(x_center, y / H, s, ha="center", va="top", fontsize=LINE_FS, color="#333333")
        y -= LINE


def _patch(color, label):
    return Patch(facecolor=color, edgecolor="#555555", linewidth=0.4, label=label)


def palette_or_default(palette=None) -> dict:
    """Class and wrong-call colours: the module defaults, overridden key by key by ``palette``
    (keys success, fail, abstain, fp, fn). Callers that pass nothing keep the defaults."""
    pal = dict(success=SUCCESS, fail=FAIL, abstain=ABSTAIN, fp=FP_COLOR, fn=FN_COLOR)
    pal.update(palette or {})
    return pal


def legend_rows(pair: bool, boundary: bool, box: bool, box_label="dashed: D_test start box", palette=None):
    """Legend handles in rows: classes and lines first, then FP and FN (pair figures only)."""
    pal = palette_or_default(palette)
    row = [_patch(pal["success"], "Success"), _patch(pal["fail"], "Fail")]
    if pair:
        row.append(_patch(pal["abstain"], "Abstain"))
    if boundary:
        row.append(Line2D([], [], color=BOUNDARY, lw=0.9, label="GT boundary"))
    if box:
        row.append(Line2D([], [], color=BOX, lw=0.9, ls=(0, (3.5, 2)), label=box_label))
    if not pair:
        return [row[:2], row[2:]] if len(row) > 2 else [row]
    return [row, [_patch(pal["fp"], FP_LABEL), _patch(pal["fn"], FN_LABEL)]]


def decision_rgb(decision, labels, palette=None) -> np.ndarray:
    """RGB image of the decisions with FP and FN cells filled in their own colours."""
    pal = palette_or_default(palette)
    d, L = np.asarray(decision), np.asarray(labels)
    rgb = _rgb(d, {1: pal["success"], 0: pal["fail"], -1: pal["abstain"]})
    rgb[(d == 1) & (L == 0)] = matplotlib.colors.to_rgb(pal["fp"])
    rgb[(d == 0) & (L == 1)] = matplotlib.colors.to_rgb(pal["fn"])
    return rgb


def _save(fig, out_stem):
    # Append, never with_suffix: plane ids like "..._at_x1.4" contain dots.
    png, pdf = Path(f"{out_stem}.png"), Path(f"{out_stem}.pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf, dpi=300, metadata={"CreationDate": None})
    plt.close(fig)
    return str(png), str(pdf)


def _metrics_line(m: dict, boxed: bool) -> str:
    def f(k, fmt):
        v = m.get(k)
        return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, fmt)
    s = (f"R_s {f('R_s', '.3f')}   R_f {f('R_f', '.3f')}   abstain {f('abstain_frac', '.1%')}   "
         f"FP {int(m.get('n_wrong_s', 0))} (FDR {f('FDR', '.3f')})   FN {int(m.get('n_wrong_f', 0))} (FOR {f('FOR', '.3f')})")
    if "rank_score" in m:
        s += f"   rank {f('rank_score', '.3f')}"
    if boxed:
        s += "   (in-box cells)"
    return s


# --- figures -------------------------------------------------------------------------------------

def _show(ax, idx, palette, sl):
    ax.imshow(_rgb(idx, palette), origin="lower", interpolation="nearest", extent=_extent(sl), aspect="auto",
              rasterized=True, zorder=1)


def render_gt(sl: dict, out_stem):
    """Ground truth alone. Returns (png path, pdf path)."""
    with plt.rc_context(RC):
        has_b = len(np.unique(sl["labels"])) > 1
        has_box = sl.get("box_x") is not None or sl.get("box_y") is not None
        hrows = legend_rows(False, has_b, has_box)
        lines = _wrap([fixed_line(sl)], _width(1))
        fig, (ax,), W, H, bottom = _figure(sl, 1, lines, len(hrows))
        _show(ax, sl["labels"], {1: SUCCESS, 0: FAIL}, sl)
        b = _draw_boundary(ax, sl)
        box = _draw_box(ax, sl)
        _style_axes(ax, sl)
        ax.set_title("Ground truth", pad=3)
        xc = (LEFT + PANEL_W / 2) / W
        fig.suptitle(_title(sl), x=xc, y=1 - 0.04 / H, va="top", fontsize=9)
        _footer(fig, W, H, bottom, legend_rows(False, b, box), lines, xc)
        return _save(fig, out_stem)


def render_pair(sl: dict, decision, out_stem, metrics=None):
    """Ground truth (left) and TERA-FM certified decisions (right). Returns (png path, pdf path).

    ``decision``: int8 (ny, nx), 1 Success / 0 Fail / -1 Abstain. Wrong calls are filled cells:
    FP (Success on a GT fail) in FP_COLOR, FN (Fail on a GT success) in FN_COLOR. ``metrics``:
    optional tera metrics dict (``metrics.slice_metrics`` output, in-box numbers) printed under the
    legend with FP/FN counts, FDR and FOR.
    """
    d = np.asarray(decision)
    L = sl["labels"]
    if d.shape != L.shape:
        raise ValueError(f"decision {d.shape} != labels {L.shape}")
    if not np.isin(d, (-1, 0, 1)).all():
        raise ValueError("decision values must be in {-1, 0, 1}")
    with plt.rc_context(RC):
        has_box = sl.get("box_x") is not None or sl.get("box_y") is not None
        lines = [fixed_line(sl)] + ([] if metrics is None else [_metrics_line(metrics, has_box)])
        lines = _wrap(lines, _width(2))
        fig, (axL, axR), W, H, bottom = _figure(sl, 2, lines, 2)
        _show(axL, L, {1: SUCCESS, 0: FAIL}, sl)
        axR.imshow(decision_rgb(d, L), origin="lower", interpolation="nearest", extent=_extent(sl), aspect="auto",
                   rasterized=True, zorder=1)
        b = _draw_boundary(axL, sl)
        _draw_boundary(axR, sl)
        box = _draw_box(axL, sl)
        _draw_box(axR, sl)
        _style_axes(axL, sl)
        _style_axes(axR, sl, ylabels=False)
        axL.set_title("Ground truth", pad=3)
        axR.set_title("TERA-FM (certified)", pad=3)
        fig.suptitle(_title(sl), y=1 - 0.04 / H, va="top", fontsize=9)
        _footer(fig, W, H, bottom, legend_rows(True, b, box), lines, 0.5)
        return _save(fig, out_stem)
