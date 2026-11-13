"""Paths and the save step shared by every plot script in dv_l4dc_plot.

Every script writes plots/<stem>.{pdf,png} here. Only with --paper does it also copy the PDF into
paper_writer's paper clone (deterministic-verification-pw) at figures/results/<paper stem>.pdf,
byte-identical to the copy in plots/.
"""
import argparse, json, os, re, shutil
import statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]               # this dv_l4dc_plot folder
REPO = HERE.parent                                       # the adaptive_roa checkout it sits in (branch feat/det-v3)
# The paper: a local clone of the Overleaf repo. Every --paper write and every read of paper content
# (method-names.tex, the committed Table 2) goes here. Set DV_PAPER to your own clone; the default is
# dm1487's paper_writer clone.
PAPER = Path(os.environ.get('DV_PAPER', '/common/home/dm1487/robotics_research/tripods/deterministic-verification-pw'))
# Inputs only: copies of the files the scripts read from results-discussion-assets/ in the paper's original
# checkout, where they are gitignored. Copied 2026-11-13; refresh by copying them again. DV_ASSETS overrides.
ASSETS = Path(os.environ.get('DV_ASSETS', HERE / 'inputs' / 'results-discussion-assets'))
assert PAPER != REPO and REPO not in PAPER.parents, f'DV_PAPER={PAPER} is inside the adaptive_roa checkout'
PAPER_FIGS = PAPER / 'figures/results'
PAPER_TABLES = PAPER / 'tables'
SCORED = Path('/common/users/shared/pracsys/adaptive_roa_experiments/det_v3_seeds/_index/scored')
PLOTS, VALUES = HERE / 'plots', HERE / 'values'
# The paper's own method-name macros (paper commit 811e036), \input by main.tex and supplementary.tex.
METHOD_NAMES_TEX = PAPER / 'method-names.tex'


def method_names():
    """Resolve the restricted nested name macros shared with the manuscript.

    The plot labels intentionally come from method-names.tex rather than a
    second Python spelling.  That file contains only no-argument
    ``\\newcommand`` definitions whose bodies are literal text and references
    to another name macro; this is deliberately not a general TeX parser.
    """
    if not METHOD_NAMES_TEX.is_file():
        raise FileNotFoundError(f'method-name source is missing: {METHOD_NAMES_TEX}')
    definitions = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}', METHOD_NAMES_TEX.read_text()))
    required = {'name', 'nameBNN', 'nameMLP', 'nameMLPLabel', 'nameBNNLabel',
                'nameUnif', 'nameUnifBNN', 'nameUnifMLP', 'nameUnifMLPLabel', 'nameUnifBNNLabel'}
    missing = required - definitions.keys()
    if missing:
        raise RuntimeError(f'method-name source lacks: {", ".join(sorted(missing))}')

    resolved = {}
    def resolve(key, visiting=frozenset()):
        if key in resolved:
            return resolved[key]
        if key in visiting:
            raise RuntimeError(f'cyclic method-name macro: {key}')
        try:
            body = definitions[key]
        except KeyError as exc:
            raise RuntimeError(f'unknown method-name macro: {key}') from exc
        value = re.sub(r'\\([A-Za-z]+)', lambda m: resolve(m.group(1), visiting | {key}), body)
        resolved[key] = value
        return value

    return {key: resolve(key) for key in required}

SETTINGS = ['pen-lqr', 'cp-lqr', 'q3d-lqr', 'cp-ppo', 'q2d-ppo', 'q3d-ppo']
TITLES = ['Pendulum (2D), LQR', 'Cartpole (4D), LQR', 'Spatial Quad (12D), LQR',
          'Cartpole (4D), RL', 'Planar Quad (6D), RL', 'Spatial Quad (12D), RL']
BLUE, ORANGE, GRAY = '#0072B2', '#D55E00', '#777777'
REF = '#666666'                                          # 5% reference line
RECALL = [('rec', BLUE), ('fail_rec', ORANGE)]           # success recall, failure recall
ERROR = [('fdr', BLUE), ('for', ORANGE)]                 # success error pairs with success recall, failure with failure
# Metric notation of the paper's method section (04_proposed_method.tex; Sumanth, 2026-11-13): results report the
# empirical recall and prediction error of outcome l on the test states, \widehat{Rec}_l and \widehat{Err}_l. They
# replace R_s, R_f, FDR and FOR. Every figure and table label takes its symbols from here; the data keys stay
# rec, fail_rec, fdr and for.
REC_S, REC_F = r'$\widehat{\mathrm{Rec}}_{\mathrm{s}}$', r'$\widehat{\mathrm{Rec}}_{\mathrm{f}}$'
ERR_S, ERR_F = r'$\widehat{\mathrm{Err}}_{\mathrm{s}}$', r'$\widehat{\mathrm{Err}}_{\mathrm{f}}$'
ABST = r'$\widehat{\mathrm{Abst}}$'                      # empirical abstention rate on the test states
# Figure 6 shades each arm's abstention rate (Sumanth, 2026-11-13). The arms need their own colours there, outside the
# success blue and failure orange: a soft green for the method and a soft violet for uniform sampling (same
# lightness and chroma as LINE; the pair passes the colour-vision check).
ARM_FILL = dict(adaptive='#56a76a', uniform='#9882d4')
RC = {'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
      'legend.fontsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .65,
      'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}
SOFT_FOUR = ASSETS / 'soft_four_wsweep.jsonl'            # final-budget fits for every run and w


def load_final():
    """FM final-budget runs with the soft rule fit on validation at w=0.2 (108 runs)."""
    final = [r for l in SOFT_FOUR.read_text().splitlines() if (r := json.loads(l))['w'] == .2 and r['model'] == 'FM']
    assert len(final) == 108
    for r in final:
        t = r['test']
        assert (t['fdr'] is None) == (t['S'] == 0) and (t['for'] is None) == (t['F'] == 0)
    return final


def spread(v):
    """The only spread in the paper, figures and tables alike: (mean, min, max) over seeds.

    Dhruv, 2026-11-12: "Even in the tables, even in the plots (everywhere)". Never add a sample SD.
    Returns (None, None, None) for no values.
    """
    v = [float(x) for x in v]
    if not v:
        return None, None, None
    return st.mean(v), min(v), max(v)


def stats(setting, arm, rho, metric, v):
    """One plotted point in %: mean and the min/max band over the contributing seeds v."""
    mean, lo, hi = spread(v)
    return dict(setting=setting, arm=arm, rho=rho, metric=metric, n=len(v), mean=mean, min=lo, max=hi)


def cell(final, setting, arm, rho, metric):
    """Three-seed test mean, min and max in %. FDR/FOR skip seeds with an empty predicted class."""
    rr = [r for r in final if (r['setting'], r['acq'].replace('τ', 'tau'), r['d2_ratio']) == (setting, arm, rho)]
    assert len(rr) == 3 and len({r['seed'] for r in rr}) == 3, (setting, arm, rho)
    v = [100 * r['test'][metric] for r in rr if r['test'][metric] is not None]
    return stats(setting, arm, rho, metric, v)


def parse_args(doc):
    p = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    p.add_argument('--paper', action='store_true',
                   help='also copy the paper output(s) into deterministic-verification-pw (figures/results/ or tables/)')
    return p.parse_args()


def page_size(pdf):
    m = re.search(rb'/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', Path(pdf).read_bytes())
    x0, y0, x1, y1 = map(float, m.groups())
    return round((x1 - x0) / 72, 2), round((y1 - y0) / 72, 2)


# Paper figures whose paper copy is the paper-style render (paper commit 34eb36f, 2026-11-13). plots/<stem>.pdf is
# that render, byte for byte the paper's file. The old-style draw of the same stem still runs where it writes values/,
# but save() sends it to plots/legacy/ and never copies it to the paper; --paper goes through install_paper().
PROMOTED = {'flagship_slices', 'pendulum_uncertainty_signals', 'adaptive_learning_a1d',
            'acquisition_strategy_certified', 'adaptive_fraction_certified', 'predictor_comparison_uniform'}
LEGACY = PLOTS / 'legacy'


def save(fig, stem, paper_stem=None, paper=False):
    """Write plots/<stem>.{pdf,png}; with paper=True also copy the PDF to figures/results/<paper_stem>.pdf.
    A PROMOTED stem is an old-style draw: it goes to plots/legacy/<stem>.{pdf,png} and is never copied to the paper."""
    legacy = stem in PROMOTED
    out = LEGACY if legacy else PLOTS
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f'{stem}.pdf'
    fig.savefig(pdf, bbox_inches='tight', pad_inches=.04, metadata={'CreationDate': None})
    fig.savefig(out / f'{stem}.png', dpi=220, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    w, h = page_size(pdf)
    where = ' (old style, plots/legacy/)' if legacy else ''
    if paper and paper_stem and legacy:
        where += '; not copied to the paper, whose copy is the paper-style render'
    elif paper and paper_stem:
        shutil.copyfile(pdf, PAPER_FIGS / f'{paper_stem}.pdf')
        where = f' -> paper figures/results/{paper_stem}.pdf'
    print(f'{stem}: {w} x {h} in{where}')
    return dict(width_in=w, height_in=h, paper_stem=paper_stem)


def install_paper(stem):
    """--paper for a PROMOTED figure: copy plots/<stem>.pdf, the paper-style render, to the paper clone's
    figures/results/<stem>.pdf when the bytes differ."""
    assert stem in PROMOTED, stem
    src, dst = PLOTS / f'{stem}.pdf', PAPER_FIGS / f'{stem}.pdf'
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        print(f'{stem}: paper copy already identical, not copied')
        return
    shutil.copyfile(src, dst)
    print(f'{stem} -> paper figures/results/{stem}.pdf')


def paper_legend_kw(fontsize: float) -> dict:
    """Legend style shared by the paper figures (Figures 5, 6 and S3 use these settings): one legend centred at the
    top of the figure above the panels and their titles, no frame, no border padding, square patch swatches and short
    line handles (handlelength = handleheight), the text pad and column spacing of Figure 6. Figure 1 calls this;
    the older scripts still pass the same values inline and are unchanged byte for byte."""
    return dict(loc="upper center", frameon=False, borderpad=0, borderaxespad=0, fontsize=fontsize,
                handlelength=0.8, handleheight=0.8, handletextpad=0.5, columnspacing=1.3)


# ==== Paper style (dm1487, 2026-11-13): one typeface, all text bold, one colour per metric ====================
# Every restyled figure imports this block. Nothing above it changes, so scripts that have not moved over yet keep
# their exact output.
#
# Typeface: the paper is \usepackage{times} with math in Computer Modern. Figures render through usetex with the same
# packages, all text \bfseries and all math \boldmath; tick numbers go through \text so they set in Times bold like
# the labels. matplotlib's template asks for type1ec (cm-super, not installed here) whenever a font list falls back
# to Computer Modern, so the three family lists point at bare keys that add no package.
# Regular weight since 2026-11-13 (Sumanth: the all-bold text looked gaudy); it was \bfseries\boldmath before.
TEX_PREAMBLE = (r"\usepackage{times}\usepackage{amsmath,bm}"
                r"\renewcommand{\mathdefault}[1]{\text{#1}}")
FS = dict(tick=7, legend=7, label=8, title=8, annotation=7)       # points on the printed page
STYLE_RC = {"text.usetex": True, "text.latex.preamble": TEX_PREAMBLE,
            "font.family": "serif", "font.serif": ["serif"], "font.sans-serif": ["sans-serif"],
            "font.monospace": ["courier"],
            "font.size": FS["label"], "axes.titlesize": FS["title"], "axes.labelsize": FS["label"],
            "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "legend.fontsize": FS["legend"],
            "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
            "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
            "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
            "savefig.dpi": 300}

# Print widths (in): render at the width the figure is printed at, so the scale on the page is 1.00.
PRINT_WIDTH = {"flagship_slices": 3.30, "pendulum_uncertainty_signals": 3.00, "adaptive_learning_a1d": 4.20,
               "acquisition_strategy_certified": 6.00, "adaptive_fraction_certified": 6.00,
               "predictor_comparison_uniform": 6.00}

# Region fills (maps): approved with Figure 1.
FILL = dict(success="#FFE33D", fail="#3E0349", fp="#00AE70", fn="#215D81", abstain="#CDCDCD", boundary="black")
# Lines and markers on white: one hue per side. Success side (R_s, FDR, success trajectories) is a dark gold from the
# TP yellow, darkened until it reads as a line (3.55:1 on white, 4.5:1 against the fail purple); the fail side (R_f,
# FOR, failure trajectories) is the TN purple. Abstention gray; reference lines thin gray.
# Since 2026-11-13 (Sumanth): a soft blue and a soft orange at one lightness and chroma (OKLCH L .66, C .12). The gold
# and near-black purple above read as muddy; this pair passes the lightness, chroma, colour-vision and 3:1 contrast
# checks. The map fills (FILL) are unchanged, so lines no longer share the maps' yellow and purple.
LINE = dict(success="#5497d9", fail="#ca7d44", abstain="#8C8C8C", ref="#666666")
# Line style encodes the metric kind: recall solid, error rate dash-dot (the paper's existing dash-dot).
LS = dict(recall="-", error=(0, (3.2, 1.3, 0.9, 1.3)), ref="-")
LW = dict(curve=1.1, ref=.6, boundary=.6)
# Methods differ by marker shape and fill, never by colour, so colour always names the metric.
METHOD_MARKER = {
    "TERA": dict(marker="o", filled=True), "Unif-TERA": dict(marker="s", filled=False),
    "TERA-MLP": dict(marker="^", filled=True), "TERA-BNN": dict(marker="D", filled=True),
    "MLP-Label": dict(marker="v", filled=False), "BNN-Label": dict(marker="d", filled=False),
    "Unif-TERA-MLP": dict(marker="^", filled=False), "Unif-TERA-BNN": dict(marker="D", filled=False),
    "Unif-MLP-Label": dict(marker="v", filled=False), "Unif-BNN-Label": dict(marker="d", filled=False),
    "DeepReach": dict(marker="X", filled=True), "Lyapunov NN": dict(marker="P", filled=True),
    "MORALS": dict(marker="*", filled=True), "Part-X GP": dict(marker="h", filled=True),
}
# Sequential map for continuous signals (Figure 5 I(x), D(x)): no class hue appears in it.
SIGNAL_CMAP = "Greys"
# Figure 5 since 2026-11-13 afternoon (Sumanth: I and D in slightly different colours): one ramp per signal, white to
# dark, in hues no class or line uses (Figure 1's yellow / purple / green / blue fills, the soft blue and orange lines):
# I is the ColorBrewer Reds ramp, D the teal half of BrBG. Both run from white, so zero reads as empty. Abstention
# cells sit on white there, so they take the darkest abstention gray rather than the light FILL abstain.
SIGNAL_RAMP = dict(I=["#ffffff", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"],
                   D=["#ffffff", "#e0f3f0", "#c7eae5", "#80cdc1", "#35978f", "#01665e", "#003c30"])
ABSTAIN_ON_WHITE = "#5C5C5C"
# The few colours one figure needs beyond FILL and LINE, kept here so the scheme lives in one module.
# Figure 5 abstention kinds as one gray family, darkest = invalid; weak is the common abstain gray.
ABSTAIN_KIND = dict(invalid="#5C5C5C", weak=FILL["abstain"], outcome_uncertain="#9B9B9B")
# Dark ink for whiskers and count labels (Figure S1), one step lighter than black.
INK = "#333333"
# A point from 1-2 seeds keeps its method's marker, with the side colour faded this far toward white (Figures 6, S2).
PARTIAL_TINT = .5


def style_rc(**overrides) -> dict:
    """The paper style rcParams, optionally with font sizes etc. overridden (Figure 1 runs a point smaller)."""
    rc = dict(STYLE_RC)
    rc.update(overrides)
    return rc


def marker_kw(method: str, color: str, size: float = 3.0, edge: float = .6) -> dict:
    """Marker keywords for one method in one metric colour: filled markers take the colour, open markers keep a
    white face and a coloured edge."""
    m = METHOD_MARKER[method]
    return dict(marker=m["marker"], markersize=size, markeredgewidth=edge, markeredgecolor=color,
                markerfacecolor=color if m["filled"] else "white")
