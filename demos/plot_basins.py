"""demos/out/basins.json -> demos/figures/basins.png"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _common import OUT, FIG, BLUE, ORANGE, GREY, INK, INK2


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color='#e6e5e1', lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def main():
    d = json.load(open(os.path.join(OUT, 'basins.json')))
    rnd = [q for q in d['lm'] if q['init'] == 'haar_shared']
    good = [q for q in rnd if q['global_']]
    bad = [q for q in rnd if not q['global_']]
    sdp = d['sdp']
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # (a) where each LM run started vs where it ended
    style(a)
    kw = dict(s=34, linewidths=1.2, edgecolors='white', zorder=3)
    a.scatter([q['init_err'] for q in good], [q['rot_err'] for q in good], c=GREY, label=
              'LM, reached the global optimum (%d)' % len(good), **kw)
    a.scatter([q['init_err'] for q in bad], [q['rot_err'] for q in bad], c=ORANGE, label=
              'LM, stopped in a local minimum (%d)' % len(bad), **kw)
    a.axhline(sdp['rot_err'], color=BLUE, lw=2, zorder=4)
    a.text(2, sdp['rot_err'] * 1.35, 'Cayley SDP: one solve, no initial guess, certified '
           '(%.2f°)' % sdp['rot_err'], color=INK, fontsize=9, va='bottom')
    a.set_yscale('log')
    a.set_xlim(0, 180); a.set_ylim(0.05, 400)
    a.set_xlabel('initial attitude error (deg)', color=INK2, fontsize=9.5)
    a.set_ylabel('final attitude error (deg)', color=INK2, fontsize=9.5)
    a.set_title('Levenberg-Marquardt from %d random attitudes' % len(rnd), color=INK,
                fontsize=10.5, loc='left')
    a.legend(loc='center left', frameon=False, fontsize=8.5, labelcolor=INK2,
             bbox_to_anchor=(0.0, 0.62))

    # (b) final cost relative to the optimum: the local minima are converged, stationary points
    style(b)
    ref = d['lm_ref']['cost']
    ratio = np.array([q['cost'] / ref for q in rnd])
    bins = np.logspace(-0.05, np.log10(max(ratio.max(), 10)) + 0.1, 28)
    b.hist(ratio[[q['global_'] for q in rnd]], bins=bins, color=GREY, rwidth=0.85, zorder=3)
    b.hist(ratio[[not q['global_'] for q in rnd]], bins=bins, color=ORANGE, rwidth=0.85, zorder=3)
    b.set_xscale('log')
    b.set_xlabel('final cost / global optimum', color=INK2, fontsize=9.5)
    b.set_ylabel('LM runs', color=INK2, fontsize=9.5)
    b.set_title('Every run converges to a stationary point', color=INK, fontsize=10.5, loc='left')
    b.text(0.97, 0.95, 'LM never sees the optimum, so it cannot\ntell a %.0fx cost from the best one.\n'
           'The SDP\'s dual bound is that missing\nlower bound: its certificate.' % np.median(ratio[[not q['global_'] for q in rnd]]),
           transform=b.transAxes, ha='right', va='top', fontsize=8.5, color=INK2)

    fig.suptitle('EuRoC %s, %g s window at t = %g s, GNSS (1 cm) + IMU, attitude unknown'
                 % (d['seq'].replace('_', ' '), d['T'], d['t_start']),
                 x=0.01, ha='left', fontsize=9, color=INK2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(FIG, exist_ok=True)
    fn = os.path.join(FIG, 'basins.png')
    fig.savefig(fn, dpi=160, facecolor='white')
    print('wrote', fn)


if __name__ == '__main__':
    main()
