"""Print-size contact sheet of the Table 2 figure options (2026-11-13, for choosing; nothing installed):
plots/options_uncertified/contact_sheet_options.pdf (+ one PNG per page).

Each option is placed at natural size, which is its print size (6.00 in wide, the paper's text width), with its
size, md5 and a one-line note on what it shows well and what it hides. Notes come from looking at the renders;
edit NOTES when an option changes. Compiles in a temporary directory; writes only the files above.
"""
import hashlib, re, shutil, subprocess, tempfile
from pathlib import Path
from common import PLOTS
from contact_sheet import media_box, tex_escape, TEXTWIDTH_PT

D = PLOTS / 'options_uncertified'
# (stem, title, note)
NOTES = [
    ('optA_scatter', 'Option A: recall-error scatter, one panel per setting',
     'Shows each method\'s success and failure trade-off in one panel against the 5\\% reference. Hides the '
     'failure side, which crowds into the top 15\\% of the y axis.'),
    ('optB_rows', 'Option B: S3-style rows, success and failure rows per controller block',
     'Shows the $R_{\\mathrm{f}}$ cluster spread out on zoomed rows, so nearly every method separates. Hides the '
     'pairing of a method\'s two points, now in different panels, and is 1.5 in taller than A.'),
    ('optC1_bars_numbers', 'Option C1: bar rows, errors printed at the bar ends (preferred over C2)',
     'Shows each recall with its exact error beside it, in the table\'s method order. Hides the error\'s spread '
     'across seeds, and leaves out Abst.'),
    ('optC2_bars_erroraxis', 'Option C2: bar rows, errors as markers on a small log axis',
     'Shows where every error falls against the 5\\% reference, with its min--max spread. Hides exact values, and '
     'fits three decades of log scale into a 0.62 in axis.'),
    ('optD_heatmap', 'Option D: visual table, every cell printed and shaded',
     'Shows all 180 numbers with colour cues in the least height (2.52 in). Hides the spread across seeds, and '
     '$R_{\\mathrm{f}}$ values from 88 to 100 all shade near-dark purple.'),
]


def main(notes=NOTES):
    blocks = []
    for stem, title, note in notes:
        pdf = D / f'{stem}.pdf'
        w_bp, h_bp = media_box(pdf)
        w_in, h_in = w_bp / 72, h_bp / 72
        assert abs(w_in - 6.0) <= .01, (stem, w_in)
        md5 = hashlib.md5(pdf.read_bytes()).hexdigest()
        head = (f'\\textbf{{{title}}}\\\\{w_in:.3f} $\\times$ {h_in:.3f} in at 100\\%; '
                f'{tex_escape("plots/options_uncertified/" + pdf.name)}; md5 \\texttt{{{md5[:12]}}}\\\\{note}')
        blocks.append(f'\\begin{{minipage}}{{\\textwidth}}\\raggedright\\footnotesize {head}\\par\\smallskip\n'
                      f'{{\\setlength{{\\fboxsep}}{{0pt}}\\setlength{{\\fboxrule}}{{.1pt}}'
                      f'\\fbox{{\\includegraphics{{{pdf}}}}}}}\\end{{minipage}}\\par\\bigskip')
    tex = ('\\documentclass[10pt]{article}\n\\usepackage{times}\\usepackage{graphicx}\n'
           f'\\usepackage[letterpaper,textwidth={TEXTWIDTH_PT}pt,top=.6in,bottom=.6in]{{geometry}}\n'
           '\\pagestyle{plain}\\setlength{\\parindent}{0pt}\\hfuzz=.5pt\n\\begin{document}\n'
           '{\\large\\textbf{Table 2 as a figure: options at print size}}\\par\\smallskip\n'
           '\\footnotesize Empirical comparison at the validation-fitted thresholds ($w=0.2$), not certified. Six '
           'methods (BNN-Label and TERA-BNN left out), six settings; three-seed means with min--max whiskers over '
           'seeds; every plotted mean asserted against ' + tex_escape('values/table_matrix_uncertified_values.json') + ', which equals '
           'the paper\'s Table 2 body cell for cell. Print at 100\\%; hairline frames mark the PDF page edge.'
           '\\par\\bigskip\n' + '\n'.join(blocks) + '\n\\end{document}\n')
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / 'sheet.tex').write_text(tex)
        for _ in range(2):
            r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'sheet.tex'],
                               cwd=tmp, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(r.stdout[-3000:])
        over = re.findall(r'Overfull \\hbox \(([\d.]+)pt too wide\)', (Path(tmp) / 'sheet.log').read_text())
        assert not over, f'wider than the text width by {over} pt'
        shutil.copyfile(Path(tmp) / 'sheet.pdf', D / 'contact_sheet_options.pdf')
    for old in D.glob('contact_sheet_options-*.png'):
        old.unlink()                                  # this script's own page renders, rebuilt below
    subprocess.run(['pdftoppm', '-r', '100', '-png', str(D / 'contact_sheet_options.pdf'),
                    str(D / 'contact_sheet_options')], check=True)
    print(D / 'contact_sheet_options.pdf', len(list(D.glob('contact_sheet_options-*.png'))), 'pages')


if __name__ == '__main__':
    main()
