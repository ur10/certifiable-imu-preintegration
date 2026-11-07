"""demos/out/sweep_*_carry.json -> demos/figures/sweep.png, plus a markdown summary table."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _common import OUT, FIG, BLUE, ORANGE, GREY, INK, INK2, certified
from plot_basins import style

SEQS = ['MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult']


def load(bias='carry'):
    rows = []
    for s in SEQS:
        fn = os.path.join(OUT, 'sweep_%s_%s.json' % (s, bias))
        if os.path.exists(fn):
            rows += [r for r in json.load(open(fn)) if 'failed' not in r]
    for r in rows:
        r['certified'] = certified(r)      # recomputed, so the criterion lives in one place
    return rows


def table(rows):
    lines = ['| sequence | windows | certified | attitude err. SDP (median / max) | LM, gravity-aligned start: same optimum | LM, identity start: same optimum |',
             '|---|---:|---:|---:|---:|---:|']
    for s in SEQS + ['all']:
        g = [r for r in rows if s == 'all' or r['seq'] == s]
        if not g:
            continue
        e = np.array([r['rot_err'] for r in g])
        lines.append('| %s | %d | %d | %.2f° / %.2f° | %d / %d | %d / %d |' % (
            '**all**' if s == 'all' else s.replace('_', '\\_'), len(g), sum(r['certified'] for r in g),
            np.median(e), e.max(),
            sum(r['lm']['gravity']['global_'] for r in g), len(g),
            sum(r['lm']['gnss']['global_'] for r in g), len(g)))
    return '\n'.join(lines)


def main(bias='carry'):
    rows = load(bias)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(11, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.25)

    # (a) one cell per window: certified or not, with its optimality gap
    a = fig.add_subplot(gs[0])
    a.set_title('Certificate, window by window', color=INK, fontsize=10.5, loc='left')
    for yi, s in enumerate(SEQS):
        g = sorted([r for r in rows if r['seq'] == s], key=lambda r: r['t0'])
        for r in g:
            x = r['t0'] / 16.0
            c = BLUE if r['certified'] else ORANGE
            a.add_patch(FancyBboxPatch((x + 0.06, -yi - 0.36), 0.88, 0.72,
                                       boxstyle='round,pad=0,rounding_size=0.12', fc=c, ec='white', lw=2))
            a.text(x + 0.5, -yi, '%d' % np.floor(np.log10(max(abs(r['eta_dual']), 1e-12))),
                   ha='center', va='center', fontsize=7.5, color='white')
    a.set_xlim(0, 11.2); a.set_ylim(-len(SEQS) + 0.45, 0.55)
    a.set_yticks([-i for i in range(len(SEQS))])
    a.set_yticklabels([s.replace('_', ' ') for s in SEQS], fontsize=8.5, color=INK2)
    a.set_xticks(np.arange(0, 11, 2) + 0.5)
    a.set_xticklabels(['%d' % (16 * k) for k in range(0, 11, 2)], fontsize=8.5, color=INK2)
    a.set_xlabel('window start (s)', color=INK2, fontsize=9.5)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.tick_params(length=0)
    n_c = sum(r['certified'] for r in rows)
    a.text(0.0, -0.2, 'blue = certified: rank 1 and dual gap ≤ 1e-6 (%d of %d windows).  '
           'Cell: log₁₀ dual gap.' % (n_c, len(rows)), transform=a.transAxes, fontsize=8,
           color=INK2, va='top')

    # (b) SDP vs LM attitude error: on the diagonal = the same answer
    b = fig.add_subplot(gs[1]); style(b)
    sd = np.array([r['rot_err'] for r in rows]); lm = np.array([r['lm']['gnss']['rot_err'] for r in rows])
    bad = np.array([not r['lm']['gnss']['global_'] for r in rows])
    cert = np.array([r['certified'] for r in rows])
    lo, hi = 0.05, 400
    b.plot([lo, hi], [lo, hi], color=GREY, lw=1, zorder=1)
    kw = dict(s=34, linewidths=1.2, edgecolors='white', zorder=3)
    b.scatter(sd[cert], lm[cert], c=BLUE, **kw)
    b.scatter(sd[~cert], lm[~cert], c=ORANGE, **kw)
    b.set_xscale('log'); b.set_yscale('log'); b.set_xlim(lo, hi); b.set_ylim(lo, hi)
    b.set_xlabel('Cayley SDP attitude error (deg)', color=INK2, fontsize=9.5)
    b.set_ylabel('LM from identity attitude (deg)', color=INK2, fontsize=9.5)
    b.set_title('Same windows, local solver', color=INK, fontsize=10.5, loc='left')
    if bad.any():
        b.annotate('%d windows: LM stops ~%.0f° off;\nthe SDP certified all of them'
                   % (bad.sum(), np.median(lm[bad])), xy=(np.max(sd[bad]), np.min(lm[bad])),
                   xytext=(1.2, 60), fontsize=8, color=INK, ha='left', va='top',
                   arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    b.text(0.97, 0.04, 'on the diagonal: LM found the same answer\n(from a gravity-aligned start: '
           '%d of %d)' % (sum(r['lm']['gravity']['global_'] for r in rows), len(rows)),
           transform=b.transAxes, ha='right', va='bottom', fontsize=7.5, color=INK2)

    fig.suptitle('EuRoC Machine Hall, back-to-back 16 s windows, GNSS (1 cm) + IMU, attitude '
                 'unknown, bias carried window to window', x=0.01, ha='left', fontsize=9,
                 color=INK2, y=0.995)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.2, top=0.86)
    os.makedirs(FIG, exist_ok=True)
    fn = os.path.join(FIG, 'sweep.png')
    fig.savefig(fn, dpi=160, facecolor='white')
    md = table(rows)
    open(os.path.join(OUT, 'sweep_%s.md' % bias), 'w').write(md + '\n')
    print(md); print('wrote', fn)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'carry')
