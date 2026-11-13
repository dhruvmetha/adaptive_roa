"""Colour-vision grids for every paper figure: plots/style/cvd/<stem>_cvd.png.

One method for every figure, so the grids compare: Machado, Oliveira and Fernandes (2009) full-severity matrices for
protanopia, deuteranopia and tritanopia, applied in linear RGB, plus CIE luminance grayscale for black-and-white print.
Each grid has five panels (normal, grayscale, protan, deutan, tritan) from one 300 dpi raster at print size.

Sources: the six matplotlib figures from plots/<stem>.png (300 dpi, the paper-style renders), and the three TikZ
figures rasterized at 300 dpi from the paper clone's figures/ (read only). Writes plots/style/cvd/ only.
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import PLOTS, PAPER

OUT = PLOTS / 'style' / 'cvd'
MATPLOTLIB = ['flagship_slices', 'pendulum_uncertainty_signals', 'adaptive_learning_a1d',
              'acquisition_strategy_certified', 'adaptive_fraction_certified', 'predictor_comparison_uniform']
TIKZ = ['uncertainty_sources/uncertainty_sources', 'pipeline/pipeline', 'abstention_triangle/abstention_triangle']
MACHADO = {
    'protanopia': np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    'deuteranopia': np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    'tritanopia': np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}


def to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(rgb, kind):
    """rgb: float array in [0, 1], shape (..., 3), sRGB. Returns the simulated sRGB array."""
    lin = to_linear(rgb)
    if kind == 'grayscale':
        y = lin @ np.array([0.2126, 0.7152, 0.0722])
        return to_srgb(np.repeat(y[..., None], 3, axis=-1))
    return to_srgb(lin @ MACHADO[kind].T)


def grid(img, title):
    rgb = np.asarray(img.convert('RGB'), float) / 255
    views = [('normal', rgb)] + [(k, simulate(rgb, k)) for k in ('grayscale', 'protanopia', 'deuteranopia', 'tritanopia')]
    h, w = rgb.shape[:2]
    pad, head = 24, 44
    cols = 2
    rows = 3
    sheet = Image.new('RGB', (cols * w + (cols + 1) * pad, rows * (h + head) + (rows + 1) * pad), 'white')
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 30)
    except OSError:
        font = ImageFont.load_default()
    cells = [('', None)] + views
    for i, (name, arr) in enumerate(cells):
        r, c = divmod(i, cols)
        x0, y0 = pad + c * (w + pad), pad + r * (h + head + pad)
        if arr is None:
            d.text((x0, y0 + head), title + '\n\nMachado 2009, full severity\nCIE luminance grayscale\n300 dpi at print size',
                   fill='black', font=font)
            continue
        d.text((x0, y0), name, fill='black', font=font)
        sheet.paste(Image.fromarray((arr * 255 + .5).astype(np.uint8)), (x0, y0 + head))
        d.rectangle([x0 - 1, y0 + head - 1, x0 + w, y0 + head + h], outline='#999999')
    return sheet


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for stem in MATPLOTLIB:
        src = PLOTS / f'{stem}.png'
        img = Image.open(src)
        dpi = img.info.get('dpi', (300, 300))[0]
        assert abs(dpi - 300) < 1, (stem, dpi)
        grid(img, stem).save(OUT / f'{stem}_cvd.png')
        made.append(stem)
    with tempfile.TemporaryDirectory() as tmp:
        for rel in TIKZ:
            stem = rel.split('/')[-1]
            subprocess.run(['pdftoppm', '-r', '300', '-png', '-singlefile', str(PAPER / 'figures' / f'{rel}.pdf'),
                            f'{tmp}/{stem}'], check=True)
            grid(Image.open(f'{tmp}/{stem}.png'), stem + ' (TikZ, paper clone)').save(OUT / f'{stem}_cvd.png')
            made.append(stem)
    for s in made:
        print(OUT / f'{s}_cvd.png')


if __name__ == '__main__':
    sys.exit(main())
