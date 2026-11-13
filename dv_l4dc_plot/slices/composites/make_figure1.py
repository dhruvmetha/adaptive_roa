"""Paper Figure 1 (flagship slices), dm1487's chosen triplet, rendered for review. Not installed in the paper.

    python composites/make_figure1.py

Panels, left to right: cp-ppo k100_x_x_dot r1 (seed 99512), q2d-ppo viz_z_vs_zdot r2 (seed 33407),
q3d-ppo k100_x_x_dot r3 (seed 24475). Layout 2x3: ground truth over the certified decisions (two-sided c6c
thresholds), 6.0 in wide. Palette sampled from dm1487's reference figure; no start-box lines; the bottom row is
labelled with the paper's \\name from method-names.tex. Two variants, because the planar crop is still open:
  figure1_uncropped  the planar slice as stored (z_dot -1 to 1)
  figure1_cropped    the planar slice cut to its D_test start box (z 0.1-1.48, z_dot -0.72-0.96) by
                     composites/crop_to_box.py
Each writes .png (300 dpi), .pdf and .txt (per-panel numbers) in composites/.

    python composites/make_figure1.py --restyle

renders only the paper's Figure 1 (figure1_H_2x3_noticks in the paper style of scripts/common.py),
plots/flagship_slices.{pdf,png,txt}, the paper's file since 2026-11-13. A plain run renders the composites above and
then the paper figure; --paper also copies plots/flagship_slices.pdf into the paper clone.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "scripts"))
from lib import composite  # noqa: E402
from common import method_names, ours, paper_legend_kw  # noqa: E402

PALETTE = dict(success="#FFE33D", fail="#3E0349", abstain="#CDCDCD", fp="#00AE70", fn="#215D81")
PANELS = [("data/cp-ppo/k100_x_x_dot.npz", "pred/cp-ppo/k100_x_x_dot__r1.npz"),
          ("data/q2d-ppo/viz_z_vs_zdot.npz", "pred/q2d-ppo/viz_z_vs_zdot__r2.npz"),
          ("data/q3d-ppo/k100_x_x_dot.npz", "pred/q3d-ppo/k100_x_x_dot__r3.npz")]
CROPPED_PLANAR = ("composites/_crops/data/q2d-ppo/viz_z_vs_zdot.npz",
                  "composites/_crops/pred/q2d-ppo/viz_z_vs_zdot__r2.npz")


def entries(planar):
    rows = [PANELS[0], planar, PANELS[2]]
    return [{"slice": str(ROOT / s), "pred": str(ROOT / p)} for s, p in rows]


# Paper naming (dm1487, 2026-11-13): systems as in the results tables, state symbols from
# appendix_experiment_details.tex (cartpole (q, theta, q_dot, theta_dot); planar quadrotor
# (q_x, q_z, phi, q_dot_x, q_dot_z, phi_dot); spatial quadrotor (q, q_dot, q_quat, omega)), controller in the tag.
# No "(4D)" style dimension in these titles: with "quadrotor" spelled out, "Planar quadrotor (6D)" and "Spatial
# quadrotor (12D)" overlap at 3.30 in and the restyle falls back to text one point smaller (tried 2026-11-13).
META = [dict(system="Cartpole", tag=r"RL, $(q,\ \theta{=}0,\ \dot q,\ \dot\theta{=}0)$",
             axes=(r"$q$ [m]", r"$\dot q$ [m/s]")),
        dict(system="Planar quadrotor", tag=r"RL, $(q_x{=}0,\ q_z,\ \phi{=}0,\ \dot q_x{=}0,\ \dot q_z,\ \dot\phi{=}0)$",
             axes=(r"$q_z$ [m]", r"$\dot q_z$ [m/s]")),
        dict(system="Spatial quadrotor", tag=r"RL, $(q_x,\ \dot q_x)$, others at hover, $q_z{=}1$",
             axes=(r"$q_x$ [m]", r"$\dot q_x$ [m/s]"))]


# dm1487 (2026-11-13): compact V and H with no ticks; axis names only (no units), ranges in the caption.
NAMES_ONLY = [(r"$q$", r"$\dot q$"), (r"$q_z$", r"$\dot q_z$"), (r"$q_x$", r"$\dot q_x$")]
RANGES = [r"Cartpole q, q_dot in [-3, 3] (cells tile this square)",
          r"Planar Quad q_z in [0.10, 1.48], q_dot_z in [-0.72, 0.96] (the D_test box; drawn cells end there)",
          r"Spatial Quad q_x in [-1.8, 1.8], q_dot_x in [-2, 2] (grid points; cells extend half a cell further)"]


def restyle(inspect=None, out_stem=None):
    """--restyle: figure1_H_2x3_noticks (the paper's Figure 1) in the paper style of scripts/common.py, written to
    plots/flagship_slices.{pdf,png,txt} (the paper's file since 2026-11-13). Same panels, data, layout rules and checks as the old path;
    changed: usetex Times bold text and CM bold math (\\boldmath gives the bold italic axis names, replacing
    _bold_math_rc), FILL colours, the GT boundary at LW["boundary"], fonts per FS, and the legend in paper_legend_kw
    at the legend size. Fonts are FS's 7 / 8 pt if every check passes at 3.30 in, else one point smaller (allowed for
    Figure 1). Returns render_flagship_tight_h's result and the fonts used."""
    from common import FILL, FS, LW, PLOTS, PRINT_WIDTH, style_rc
    stem = "flagship_slices"
    out = out_stem or str(PLOTS / stem)
    label = method_names()["name"]
    ents = [dict(e, **m) for e, m in zip(entries(CROPPED_PLANAR), META)]
    ents_nt = [dict(e, axes=a) for e, a in zip(ents, NAMES_ONLY)]
    pal = {k: FILL[k] for k in ("success", "fail", "abstain", "fp", "fn")}
    full = dict(tick=FS["tick"], label=FS["label"], title=FS["title"], row=FS["label"], legend=FS["legend"])
    smaller = {k: v - 1 for k, v in full.items()}
    for fonts in (full, smaller):
        try:
            res = composite.render_flagship_tight_h(
                ents_nt, out, width=PRINT_WIDTH[stem], palette=pal, method_label=ours(label), boundary=True,
                boundary_legend=True, boundary_label="GT boundary", fonts=fonts, ticks=False, ranges=RANGES,
                bold_axis_names=False, row_gap=0.05, legend_top=True, legend_kw=paper_legend_kw(fonts["legend"]),
                base_rc=style_rc(), boundary_lw=LW["boundary"], inspect=inspect)
            return res, fonts
        except AssertionError as err:
            if fonts is smaller:
                raise
            print(f"restyle at {full} fails a check ({err}); trying one point smaller")


def restyle_main():
    """The paper's Figure 1: render plots/flagship_slices.{pdf,png,txt}, check the print width, note the fonts."""
    (png, pdf, txt, H), fonts = restyle()
    import re as _re
    m = _re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", Path(pdf).read_bytes())
    w_in = (float(m.group(3)) - float(m.group(1))) / 72
    from common import PRINT_WIDTH
    assert abs(w_in - PRINT_WIDTH["flagship_slices"]) <= .01, ("PDF width", w_in)
    Path(txt).write_text("Restyle (scripts/common.py paper style, usetex Times bold, CM bold math); fonts used "
                         f"{fonts} pt; PDF {w_in:.3f} x {H:.3f} in.\n" + Path(txt).read_text())
    print(f"flagship_slices (paper figure): {w_in:.3f} x {H:.3f} in, fonts {fonts} -> {pdf}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Paper Figure 1 (flagship slices).")
    ap.add_argument("--restyle", action="store_true",
                    help="render only the paper's Figure 1, plots/flagship_slices.{pdf,png,txt}, not the composites")
    ap.add_argument("--paper", action="store_true",
                    help="also copy plots/flagship_slices.pdf into the paper clone")
    args = ap.parse_args()
    if args.restyle:
        restyle_main()
        if args.paper:
            from common import install_paper
            install_paper("flagship_slices")
        raise SystemExit(0)
    # dm1487 (2026-11-13): planar panel cropped to its start box, reference-figure styling, paper naming.
    label = method_names()["name"]
    ents = [dict(e, **m) for e, m in zip(entries(CROPPED_PLANAR), META)]
    # dm1487 (2026-11-13): as tight as possible; no slice tags, one-line row labels, legend centred on the panels.
    # dm1487: GT boundary back in black; rows packed closer, the two columns 0.11 in apart. Then: panels dominate,
    # text smaller (ticks 5, axis labels 5.5, row labels 6.5 bold, column titles 7, legend 6 pt), "TERA (ours)",
    # and "GT boundary" as a sixth legend entry if one centred line still fits (otherwise left out).
    kw = dict(width=2.70, palette=PALETTE, method_label=ours(label), col_gap=0.11, boundary=True,
              row_gap_pt=1.0, boundary_label="GT boundary",
              fonts=dict(tick=5, label=5.5, row=6.5, title=7, legend=6))
    out = str(ROOT / "composites" / "figure1_A_3x2")
    try:
        print(composite.render_flagship_tight(ents, out, boundary_legend=True, **kw), "| GT boundary in the legend")
    except AssertionError as err:
        if "legend" not in str(err):
            raise
        print("six-entry legend does not fit:", err)
        print(composite.render_flagship_tight(ents, out, boundary_legend=False, **kw), "| GT boundary NOT in the legend")
    # Layout B (2x3, full width) was dropped by dm1487 on 2026-11-13; render_flagship(layout="2x3") still draws it.
    # Option for dm1487 (2026-11-13), not installed: horizontal 2 x 3 for a 0.55\linewidth wrapfigure (3.30 in).
    kw_h = {k: v for k, v in kw.items() if k not in ("width", "col_gap", "row_gap_pt")}
    try:
        print(composite.render_flagship_tight_h(ents, str(ROOT / "composites" / "figure1_H_2x3"), width=3.30,
                                                boundary_legend=True, **kw_h), "| GT boundary in the legend")
    except AssertionError as err:
        if "legend" not in str(err):
            raise
        print("six-entry legend does not fit (horizontal):", err)
        print(composite.render_flagship_tight_h(ents, str(ROOT / "composites" / "figure1_H_2x3"), width=3.30,
                                                boundary_legend=False, **kw_h), "| GT boundary NOT in the legend")
    # Compact vertical for a 0.36\linewidth wrap (2.16 in), dm1487 2026-11-13; not installed. One legend line if it
    # fits, else two centred lines of three; fewer y ticks only if the default collides.
    kw_v = dict(kw, width=2.16, col_gap=0.10, row_label_sep_pt=6.0)   # 6 pt between neighbouring row labels
    out_v = str(ROOT / "composites" / "figure1_V_3x2_compact")
    done = False
    for rows_ in (1, 2):
        for yb in (4, 3):
            try:
                print(composite.render_flagship_tight(ents, out_v, boundary_legend=True, legend_rows=rows_, ybins=yb,
                                                      **{k: v for k, v in kw_v.items()}),
                      f"| legend lines {rows_}, ybins {yb}")
                done = True
                break
            except AssertionError as err:
                print(f"compact V with legend lines {rows_}, ybins {yb} failed:", err)
        if done:
            break
    if not done:
        raise SystemExit("compact vertical did not fit")

    # dm1487 (2026-11-13): compact V and H with no ticks; axis names only (no units), ranges in the caption
    # (NAMES_ONLY and RANGES, defined above main so the --restyle path shares them).
    ents_nt = [dict(e, axes=a) for e, a in zip(ents, NAMES_ONLY)]
    for rows_ in (1, 2):
        try:
            print(composite.render_flagship_tight(ents_nt, str(ROOT / "composites" / "figure1_V_3x2_noticks"),
                                                  boundary_legend=True, legend_rows=rows_, ticks=False, ranges=RANGES,
                                                  bold_axis_names=True, legend_top=True,
                                                  legend_kw=paper_legend_kw(6), **kw_v),
                  f"| V no ticks, legend lines {rows_}")
            break
        except AssertionError as err:
            if rows_ == 2 or "legend" not in str(err):
                raise
            print("V no ticks: one legend line does not fit:", err)
    # dm1487: bold italic axis names; a visible 0.05 in white gap between the GT and TERA rows; legend at the top in
    # the paper figures' shared style (common.paper_legend_kw).
    print(composite.render_flagship_tight_h(ents_nt, str(ROOT / "composites" / "figure1_H_2x3_noticks"), width=3.30,
                                            boundary_legend=True, ticks=False, ranges=RANGES, bold_axis_names=True,
                                            row_gap=0.05, legend_top=True, legend_kw=paper_legend_kw(6), **kw_h),
          "| H no ticks")
    # The paper's Figure 1 (paper style, since 2026-11-13); the composites above are previews.
    restyle_main()
    if args.paper:
        from common import install_paper
        install_paper("flagship_slices")
