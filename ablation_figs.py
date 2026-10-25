"""Figures for Sec. VII from ablation_svr.json (see cayley_ablation_plots.py).

  fig_rate() -> ablation_rate.png   certified rate vs noise, one curve per constraint set
  fig_svr()  -> ablation_svr.png    log-SVR boxplots vs noise, one panel per constraint set
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from cayley_ablation_plots import STAGES, ADDED, SIGMAS, THRESH, OUT_JSON

SURF = '#fcfcfb'; GRID = '#e1e0d9'; AXIS = '#b9b8ad'; MUTED = '#8a8880'; INK = '#0b0b0b'
RED = '#d1382f'; BLUE = '#2a5db0'; BAND = '#e8e7e0'

# Figure text is set in the paper's own face (Nimbus Roman is the URW Times that IEEEtran
# uses) at the size it will be read at, so nothing is rescaled by \includegraphics.
RC = {
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'axes.linewidth': .6,
    'xtick.major.width': .6,
    'ytick.major.width': .6,
    'xtick.minor.width': .4,
    'ytick.minor.width': .4,
    'xtick.major.size': 2.6,
    'ytick.major.size': 2.6,
    'xtick.minor.size': 1.4,
    'ytick.minor.size': 1.4,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
}
# Okabe-Ito, which stays distinguishable in greyscale and to colour-blind readers.
MARK = ['o', 's', '^', 'D', 'v']
MS = [6.8, 4.3, 2.2, 3.6, 3.6]   # nested sizes: stages 0-2 coincide where they crowd
LS = ['-', '-', '-', '-', '-']
COL = ['#9a9a9a', '#56b4e9', '#0072b2', '#e69f00', '#009e73']


def _load(src=OUT_JSON):
    rows = [r for r in json.load(open(src)) if r['log_svr'] is not None]
    if not rows:
        raise SystemExit('no solved rows in %s' % src)
    return rows


def _grid(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def _style(ax):
    """Plain white panel, hairline box, faint dotted grid behind the data."""
    ax.grid(True, which='major', color='#cccccc', lw=.4, ls=':', zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color('#4d4d4d')
    ax.tick_params(colors='#4d4d4d')
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_color('black')


def fig_rate(png='ablation_rate.png', src=OUT_JSON):
    """Certified rate vs noise, authored at the \columnwidth it is included at.

    Three of the five curves sit identically at zero for every sigma, so they are drawn with
    nested open markers (large to small) instead of being hidden underneath one another.
    """
    rows = _load(src)
    sig = sorted({r['sigma'] for r in rows})
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.62))
        for si, (lab, _) in enumerate(STAGES):
            rate = []
            for s in sig:
                g = [r for r in rows if r['stage'] == si and r['sigma'] == s]
                rate.append(np.mean([r['log_svr'] > THRESH for r in g]) if g else np.nan)
            ax.plot(sig, rate, marker=MARK[si], color=COL[si], mfc='none', mew=.9,
                    ms=MS[si], lw=1.2, ls=LS[si], label=lab.replace('$+$', '+'),
                    clip_on=False, zorder=3 + si)
        ax.set_xscale('log')
        ax.set_xlabel(r'injected noise $\sigma$')
        ax.set_ylabel(r'certified rate,  $\mathrm{rank}\,\hat{Z} = 1$')
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, .25, .5, .75, 1.])
        ax.set_xlim(min(sig) / 1.6, max(sig) * 1.6)
        _style(ax)
        # the full-catalog curve runs along the top and B2 sweeps the middle, so no
        # interior region is free: the key goes above the axes in three columns
        ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.005), ncol=3,
                  handlelength=1.8, handletextpad=.45, columnspacing=1.1, borderpad=0)
        fig.tight_layout(pad=0.25)
        fig.savefig(png, dpi=600)
    print('wrote', png)


def fig_svr(png='ablation_svr.png', src=OUT_JSON):
    rows = _load(src)
    sig = sorted({r['sigma'] for r in rows})
    ns = len(STAGES)
    fig, axes = plt.subplots(1, ns, figsize=(7.16, 3.25), sharey=True, facecolor=SURF)
    # explicit margins: the panels take the top half, the identities and sweep arrow the rest
    fig.subplots_adjust(left=.085, right=.985, top=.905, bottom=.455, wspace=.10)
    allv = [r['log_svr'] for r in rows]
    lo, hi = min(allv) - 0.8, max(allv) + 0.8
    for si, ax in enumerate(axes):
        data = [[r['log_svr'] for r in rows if r['stage'] == si and r['sigma'] == s] for s in sig]
        ax.axhspan(THRESH, hi, color=BAND, zorder=0)          # certified region
        ax.axhline(THRESH, color=AXIS, lw=.8, zorder=1)
        bp = ax.boxplot(data, widths=.62, showfliers=True, whis=(0, 100),
                        patch_artist=True, zorder=3)
        for b in bp['boxes']:
            b.set(facecolor='none', edgecolor=BLUE, lw=.9)
        for w in bp['whiskers'] + bp['caps']:
            w.set(color=MUTED, lw=.7, ls='--')
        for m in bp['medians']:
            m.set(color=RED, lw=1.4)
        for f in bp['fliers']:
            f.set(marker='+', mec=RED, ms=3.5, mew=.7)
        ax.set_xticks(range(1, len(sig) + 1))
        ax.set_xticklabels([('%g' % s) for s in sig], rotation=90, fontsize=6.2)
        ax.set_title(STAGES[si][0], color=INK, fontsize=8.5, pad=3)
        _grid(ax)
    axes[0].set_ylim(lo, hi)
    axes[0].set_ylabel(r'$\log_{10}(\lambda_1/\lambda_2)$ of $\hat Z$', color=INK, fontsize=8.5)
    axes[0].text(.05, .955, 'certified', transform=axes[0].transAxes, ha='left', va='top',
                 fontsize=6.4, color=MUTED, style='italic')
    # one shared x label, then the identities each stage adds, at figure coordinates
    cx = [(ax.get_position().x0 + ax.get_position().x1) / 2 for ax in axes]
    fig.text(sum(cx) / len(cx), .300, r'noise, $\sigma$', ha='center', va='top',
             fontsize=8.5, color=INK)
    for si, x in enumerate(cx):
        if ADDED[si]:
            fig.text(x, .235, ADDED[si], ha='center', va='top', fontsize=6.4,
                     color=INK, linespacing=1.8)
    y = .075
    fig.patches.append(FancyArrowPatch((.10, y), (.955, y), transform=fig.transFigure,
                                       arrowstyle='-|>', mutation_scale=9,
                                       color=MUTED, lw=.8, zorder=5))
    fig.text(.53, y + .022, 'increasing the number of redundant constraints',
             ha='center', va='bottom', fontsize=7, color=MUTED)
    fig.savefig(png, dpi=200, facecolor=SURF)
    print('wrote', png)




def fig_ablation(png='ablation.png', src=OUT_JSON):
    """Table II and the certified-rate curve merged into one figure.

    Two panels share the noise axis: the rank of the SDP optimum, and the fraction of trials
    that reach a certificate. Rank is what the old Table II reported after averaging over sigma;
    showing it against sigma instead exposes where each family stops working.
    """
    rows = _load(src)
    sig = sorted({r['sigma'] for r in rows})
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.95), sharex=True)
        arank, arate = axes
        for si, (lab, _) in enumerate(STAGES):
            med_r, lo_r, hi_r, med_e, rate = [], [], [], [], []
            for s in sig:
                g = [r for r in rows if r['stage'] == si and r['sigma'] == s]
                med_r.append(np.median([r['rank'] for r in g]))
                lo_r.append(min(r['rank'] for r in g))
                hi_r.append(max(r['rank'] for r in g))
                med_e.append(np.median([r['eta'] for r in g]))
                rate.append(np.mean([r['rank'] == 1 for r in g]))
            lab = lab.replace('$+$', '+')
            # the three pre-Gram stages crowd together in every panel -- at rank ~100, at
            # eta ~1, and exactly at rate 0 -- so the nested marker sizes are used throughout
            kw = dict(color=COL[si], marker=MARK[si], mfc='none', mew=.9, lw=1.1,
                      ms=MS[si], clip_on=False, zorder=3 + si)
            arank.plot(sig, med_r, **kw)
            arate.plot(sig, rate, label=lab, **kw)

        for ax in axes:
            ax.set_xscale('log')
            _style(ax)
        arank.set_yscale('log')
        arank.set_ylabel(r'$\mathrm{rank}\,\hat{Z}$')
        arank.set_ylim(.75, 320)
        arate.set_ylabel(r'certified rate')
        arate.set_ylim(-.04, 1.04)
        arate.set_yticks([0, .5, 1.])
        arate.set_xlabel(r'injected noise $\sigma$')
        arate.set_xlim(min(sig) / 1.6, max(sig) * 1.6)

        h, l = arate.get_legend_handles_labels()      # curves are labelled in the rate panel
        arank.legend(h, l, frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3,
                     handlelength=1.8, handletextpad=.45, columnspacing=1.0, borderpad=0)
        fig.align_ylabels(axes)
        fig.tight_layout(pad=.25, h_pad=.55)
        fig.savefig(png, dpi=600)
    print('wrote', png)


if __name__ == '__main__':
    # fig_ablation is the paper's Fig. 3; the other two are the supporting panels.
    fig_ablation(); fig_rate(); fig_svr()
