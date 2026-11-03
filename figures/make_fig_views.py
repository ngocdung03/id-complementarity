#!/usr/bin/env python
"""make_fig_views.py -- the controlled-view figure of the paper. Single column.

  Top row     one DDR test image in the four views (rdr_data.py views).
  Bottom row  colour = view: (a) ID mAUC and (b) net correction on DDR test, (c) OOD
              contribution on APTOS, Messidor-2 and DeepDRiD; diamonds and bars are seed
              means, dots the three seeds.

Every number is read from the output of `rdr_predict.py viewstats` (no literals). Colours are
Okabe-Ito; in the bottom row view order is fixed left to right.

Usage (artifacts under $RDR_ROOT/artifacts; the top row needs the cached DDR image):
    python figures/make_fig_views.py [--views artifacts/mechanism/views_retina-tier2.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'pdf.fonttype': 42,
    'font.size': 7.0,
    'axes.labelsize': 7.0,
    'axes.titlesize': 7.5,
    'xtick.labelsize': 6.3,
    'ytick.labelsize': 6.3,
    'legend.fontsize': 6.2,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get('RDR_ROOT', REPO)).resolve() / 'artifacts'
MECH = ART / 'mechanism'
HERE = Path(__file__).resolve().parent

VIEWS = ['rgb', 'green', 'vessel', 'crop']
VIEW_LABEL = {'rgb': 'RGB', 'green': 'CLAHE green', 'vessel': 'Frangi vessel', 'crop': 'central crop'}
SEEDS = (0, 1, 2)
SITES = [('aptos', 'APTOS'), ('messidor2', 'Messidor-2'), ('deepdrid', 'DeepDRiD')]
EXAMPLE = ('ddr', '007-5953-300.jpg')     # DDR test, grade 2

# Okabe-Ito, one colour per view
C_VIEW = {'rgb': '#E69F00', 'green': '#009E73', 'vessel': '#0072B2', 'crop': '#CC79A7'}
INK = '#333333'


def load(path):
    rows = json.loads(Path(path).read_text())['rows']
    ddr, nc, contrib = {}, {}, {}
    for v in VIEWS:
        tags = [f'retina-whole-{v}-t2s{s}' for s in SEEDS]
        ddr[v] = [rows[t]['ddr_test_mauc'] for t in tags]
        nc[v] = [rows[t]['net_correction'] for t in tags]
        contrib[v] = {s: [rows[t]['per_site'][s]['geo_contribution'] for t in tags] for s, _ in SITES}
    return ddr, nc, contrib


def view_images():
    sys.path.insert(0, str(REPO / 'retina_dr'))
    import cv2
    import rdr_data as RV
    import rdr_data as PRE
    site, image_id = EXAMPLE
    bgr = cv2.imread(PRE.cache_path(site, image_id), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f'cached image missing: {PRE.cache_path(site, image_id)}')
    rgb = bgr[:, :, ::-1].copy()
    out = {}
    for v in VIEWS:
        img = RV.apply(v, rgb, site, image_id)
        h, w = img.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        out[v] = cv2.resize(img[y0:y0 + side, x0:x0 + side], (256, 256), interpolation=cv2.INTER_AREA)
    return out


def _style(ax, title):
    ax.set_title(title, fontsize=6.6, pad=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=2, pad=1)
    ax.tick_params(axis='x', length=0, pad=2)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.35)
    ax.set_axisbelow(True)


def _dots(ax, vals_by_view):
    """Seed-mean diamond per view; no bar, as the axis does not start at zero."""
    for k, v in enumerate(VIEWS):
        ax.scatter([k], [float(np.mean(vals_by_view[v]))], s=22, marker='D', color=C_VIEW[v],
                   zorder=4, linewidths=0)
        ax.scatter([k] * len(vals_by_view[v]), vals_by_view[v], s=3.5, color=INK, zorder=5,
                   linewidths=0)
    ax.set_xlim(-0.6, len(VIEWS) - 0.4)
    ax.set_xticks([])


def figure(ddr, nc, contrib, out):
    imgs = view_images()
    fig = plt.figure(figsize=(3.35, 2.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.22,
                             left=0.09, right=0.995, top=0.985, bottom=0.08)
    top = outer[0].subgridspec(1, 4, wspace=0.08)
    for j, v in enumerate(VIEWS):
        ax = fig.add_subplot(top[0, j])
        ax.imshow(imgs[v])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.6); sp.set_edgecolor(C_VIEW[v])
        ax.set_title(VIEW_LABEL[v], fontsize=6.6, pad=2, color=INK)

    bot = outer[1].subgridspec(1, 3, width_ratios=[0.9, 0.9, 2.9], wspace=0.40)
    ax = fig.add_subplot(bot[0, 0])
    _dots(ax, ddr)
    ax.set_ylim(0.895, 0.950); ax.set_yticks([0.90, 0.92, 0.94])
    _style(ax, '(a) ID mAUC')

    ax = fig.add_subplot(bot[0, 1])
    _dots(ax, nc)
    ax.set_ylim(0.42, 0.66); ax.set_yticks([0.45, 0.55, 0.65])
    _style(ax, '(b) Net corr.')

    ax = fig.add_subplot(bot[0, 2])
    xs, width = np.arange(len(SITES)), 0.2
    for k, v in enumerate(VIEWS):
        x = xs + (k - 1.5) * width
        ax.bar(x, [float(np.mean(contrib[v][s])) for s, _ in SITES], width=width * 0.92,
               color=C_VIEW[v], zorder=3)
        for xi, (s, _) in zip(x, SITES):
            ax.scatter([xi] * len(contrib[v][s]), contrib[v][s], s=3.5, color=INK, zorder=5,
                       linewidths=0)
    ax.axhline(0.0, color=INK, linewidth=0.5, zorder=1)
    ax.set_xticks(xs); ax.set_xticklabels([n for _, n in SITES], fontsize=5.6)
    ax.set_ylabel(r'$\Delta$mAUC', fontsize=6.4, labelpad=1)
    _style(ax, '(c) OOD contribution')

    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out.with_suffix('.png'), dpi=250, bbox_inches='tight', pad_inches=0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', default=str(MECH / 'views_retina-tier2.json'),
                    help='output of rdr_predict.py viewstats')
    ap.add_argument('--out', default=str(HERE / 'fig_views.pdf'))
    args = ap.parse_args()
    ddr, nc, contrib = load(args.views)
    for v in VIEWS:
        print(f'{v:7s} DDR mAUC {np.mean(ddr[v]):.4f}  ' +
              '  '.join(f'{s} {np.mean(contrib[v][s]):+.4f}' for s, _ in SITES))
    out = Path(args.out)
    figure(ddr, nc, contrib, out)
    print('wrote', out)


if __name__ == '__main__':
    main()
