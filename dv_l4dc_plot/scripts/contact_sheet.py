"""Print-size contact sheet of every paper figure: plots/style/contact_sheet.pdf (+ one PNG per page).

Each figure is placed at its natural size, which is its print size: the matplotlib figures are rendered at the width
\\linewidth has where the paper includes them, and the TikZ figures are included at natural size by the paper. The
heading line gives the PDF size, the paper's \\linewidth there and the resulting scale, and the PDF's md5.

LINEWIDTH_PT was measured on 2026-11-13 by compiling a scratch copy of the paper clone (main.tex and supplementary.tex)
with \\typeout{\\the\\linewidth} before each \\includegraphics. Natural-size includes (no width=) list None.
Compiles in a temporary directory; writes plots/style/contact_sheet.pdf and contact_sheet-<page>.png only.
"""
import hashlib, re, shutil, subprocess, tempfile
from pathlib import Path
from common import PLOTS, PAPER

R = PLOTS                                  # the paper-style figures
OUT = PLOTS / 'style'
FIG = PAPER / 'figures'
# (paper number, label, source pdf, paper \linewidth at the include in TeX pt, or None for a natural-size include)
ITEMS = [
    ('Figure 1', 'fig:intro', R / 'flagship_slices.pdf', 238.49231),
    ('Figure 2', 'fig:uncertainty_sources', FIG / 'uncertainty_sources/uncertainty_sources.pdf', None),
    ('Figure 3', 'fig:pipeline', FIG / 'pipeline/pipeline.pdf', None),
    ('Figure 5', 'fig:pendulum_uncertainty_signals', R / 'pendulum_uncertainty_signals.pdf', 303.53267),
    ('Figure 5, preview variant', 'abstention, FP, FN cells at true size (--rare-pt 0)',
     R / 'variants/pendulum_uncertainty_signals_truesize.pdf', 303.53267),
    ('Figure 6', 'fig:adaptive_learning_a1d', R / 'adaptive_learning_a1d.pdf', 303.53267),
    ('Figure S1', 'fig:acquisition_strategy', R / 'acquisition_strategy_certified.pdf', 433.62),
    ('Figure S2', 'fig:adaptive_fraction', R / 'adaptive_fraction_certified.pdf', 433.62),
    ('Figure S3', 'fig:predictor_comparison_uniform_full', R / 'predictor_comparison_uniform.pdf', 433.62),
    ('Figure S4', 'fig:abstention_triangle', FIG / 'abstention_triangle/abstention_triangle.pdf', 433.62),
]
TEXTWIDTH_PT = 433.62          # l4dc2027 \textwidth (TeX pt)


def media_box(pdf):
    out = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True, check=True).stdout
    w, h = map(float, re.search(r'Page size:\s+([\d.]+) x ([\d.]+) pts', out).groups())
    return w, h                                   # PostScript points (bp)


def tex_escape(s):
    return s.replace('_', r'\_\allowbreak{}').replace('/', r'/\allowbreak{}').replace('#', r'\#').replace('%', r'\%')


def main():
    blocks = []
    for num, label, pdf, lw in ITEMS:
        w_bp, h_bp = media_box(pdf)
        w_in, h_in = w_bp / 72, h_bp / 72
        md5 = hashlib.md5(pdf.read_bytes()).hexdigest()
        if lw is None:
            scale = 'natural size (no width), fits the text width' if w_bp / 72 * 72.27 <= TEXTWIDTH_PT else 'OVERFLOWS'
        else:
            scale = f'paper linewidth {lw / 72.27:.3f} in, scale {lw / 72.27 / w_in:.3f}'
        where = 'plots/' + str(pdf.relative_to(R)) if R in pdf.parents else 'paper clone figures/' + \
            str(pdf.relative_to(FIG))
        head = (f'\\textbf{{{tex_escape(num)}}} ({tex_escape(label)}): {tex_escape(where)}\\\\'
                f'{w_in:.3f} $\\times$ {h_in:.3f} in; {scale}; md5 \\texttt{{{md5[:12]}}}')
        blocks.append(f'\\begin{{minipage}}{{\\textwidth}}\\raggedright\\footnotesize {head}\\par\\smallskip\n'
                      f'{{\\setlength{{\\fboxsep}}{{0pt}}\\setlength{{\\fboxrule}}{{.1pt}}'
                      f'\\fbox{{\\includegraphics{{{pdf}}}}}}}\\end{{minipage}}\\par\\bigskip')
    tex = ('\\documentclass[10pt]{article}\n\\usepackage{times}\\usepackage{graphicx}\n'
           f'\\usepackage[letterpaper,textwidth={TEXTWIDTH_PT}pt,top=.6in,bottom=.6in]{{geometry}}\n'
           '\\pagestyle{plain}\\setlength{\\parindent}{0pt}\\hfuzz=.5pt  % the .1 pt frame on a full-width figure\n\\begin{document}\n'
           '{\\large\\textbf{Restyled paper figures at print size}}\\par\\smallskip\n'
           '\\footnotesize Print at 100\\%: each figure is the size it appears in the paper. Hairline frames mark the '
           'PDF page edge. Figure 4 (system renders) is unchanged and not shown.\\par\\bigskip\n'
           + '\n'.join(blocks) + '\n\\end{document}\n')
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'contact_sheet.tex').write_text(tex)
        for _ in range(2):
            r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'contact_sheet.tex'],
                               cwd=tmp, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(r.stdout[-3000:])
        log = (Path(tmp) / 'contact_sheet.log').read_text()
        over = re.findall(r'Overfull \\hbox \(([\d.]+)pt too wide\)', log)
        assert not over, f'a figure is wider than the text width by {over} pt'
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(tmp) / 'contact_sheet.pdf', OUT / 'contact_sheet.pdf')
    for old in OUT.glob('contact_sheet-*.png'):
        old.unlink()                                  # this script's own page renders, rebuilt below
    subprocess.run(['pdftoppm', '-r', '100', '-png', str(OUT / 'contact_sheet.pdf'), str(OUT / 'contact_sheet')], check=True)
    pages = sorted(OUT.glob('contact_sheet-*.png'))
    print(OUT / 'contact_sheet.pdf', len(pages), 'pages')


if __name__ == '__main__':
    main()
