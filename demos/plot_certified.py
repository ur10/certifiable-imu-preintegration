"""Certified windows only: the mission timeline and the two UrbanNav drives, as static figures for the README.

Reads the shipped JSON (no datasets, no MOSEK needed):

    python demos/plot_certified.py      # -> demos/figures/mission_certified.png, urbannav_certified.png

Every window drawn in colour is certified (rank one, dual gap <= 1e-6). Windows that are not
certified are left out of the statistics and drawn only as a recessive grey marker, so the reader
can see where they were.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIG = os.path.join(HERE, 'out'), os.path.join(HERE, 'figures')
BLUE, ORANGE, INK, INK2, MUTED, GRID = '#2a78d6', '#eb6834', '#1f1e1c', '#55544f', '#9a9994', '#e6e5e0'
WEAK = -4.0
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False})


def certified(r):
    return r.get('rank') == 1 and r.get('eta_dual') is not None and abs(r['eta_dual']) <= 1e-6


def mission():
    d = json.load(open(os.path.join(OUT, 'sim_mission.json')))
    W = d['windows']; T = d['T']
    C = [w for w in W if certified(w)]
    tc = np.array([w['t0'] + T / 2 for w in C])
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)
    short = {'figure-8, GNSS jammed': '', 'hover + slow yaw': 'hover + yaw', 'zig-zag dashes': 'dashes'}
    for s, e, lab in d['phases']:
        ax.text((s + e) / 2, 400, short.get(lab, lab), ha='center', va='bottom', fontsize=8.5, color=INK2)
    for s, e, lab in d['phases'][::2]:
        ax.axvspan(s, e, color='#f3f2ee', zorder=0, lw=0)
    # windows that are not certified: the jammed stretch, left out
    U = [w for w in W if not certified(w)]
    if U:
        lo, hi = min(w['t0'] for w in U) + T / 2, max(w['t0'] for w in U) + T / 2
        ax.axvspan(lo - 2, hi + 2, color='white', zorder=1, lw=0)
        ax.axvspan(lo - 2, hi + 2, facecolor='none', hatch='////', edgecolor=GRID, zorder=1, lw=0)
        ax.text((lo + hi) / 2, 0.02, 'GNSS jammed:\nnot certified,\nnot shown (%d windows)' % len(U),
                ha='center', va='bottom', fontsize=8.5, color=MUTED, zorder=2)
    ax.plot(tc, [w['lm']['rot_err'] for w in C], 'o', ms=8, mfc='white', mec=ORANGE, mew=1.5, zorder=3,
            label='LM sliding window (warm-started)')
    ax.plot(tc, [w['rot_err'] for w in C], 'o', ms=4.5, color=BLUE, zorder=4,
            label='Cayley SDP, certified (no initial guess)')
    tr = [w for w in C if not w['lm_cold']['global_']]
    ax.plot([w['t0'] + T / 2 for w in tr], [w['lm_cold']['rot_err'] for w in tr], 'X', ms=9, color=ORANGE,
            mec='white', mew=1, zorder=5, label='LM restarted from a random attitude: trapped')
    for w in tr:
        ax.annotate('%.0f°' % w['lm_cold']['rot_err'], (w['t0'] + T / 2, w['lm_cold']['rot_err']),
                    xytext=(0, -14), textcoords='offset points', ha='center', fontsize=8, color=INK2)
    ax.set_yscale('log'); ax.set_ylim(0.02, 900)
    ax.set_xlim(0, max(w['t0'] for w in W) + T)
    ax.set_xlabel('mission time (s), window centre'); ax.set_ylabel('attitude error (deg)')
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.2), ncol=3, frameon=False, fontsize=8.5,
              handletextpad=0.4, columnspacing=1.6)
    e = np.array([w['rot_err'] for w in C])
    ax.set_title('Simulated mission, certified windows: %d of %d.  SDP attitude error median %.2f°, worst %.2f°'
                 % (len(C), len(W), np.median(e), e.max()), loc='left', fontsize=10.5, color=INK, pad=22)
    fig.tight_layout()
    p = os.path.join(FIG, 'mission_certified.png'); fig.savefig(p, facecolor='white'); plt.close(fig)
    print('wrote', p)


def enu(lat, lon, lat0, lon0):
    r = np.pi / 180
    return (np.asarray(lon) - lon0) * r * 6378137 * np.cos(lat0 * r), (np.asarray(lat) - lat0) * r * 6378137


def urbannav():
    D = json.load(open(os.path.join(OUT, 'urbannav_map.json')))
    cmap = LinearSegmentedColormap.from_list('b', ['#bcd6f5', BLUE, '#123f7a'])
    norm = Normalize(0, 2.0)
    fig = plt.figure(figsize=(12, 5.6), dpi=150)
    axes = [fig.add_axes([0.06, 0.13, 0.36, 0.7]), fig.add_axes([0.49, 0.13, 0.40, 0.7])]
    cax = fig.add_axes([0.92, 0.2, 0.012, 0.56])
    names = dict(medium='Medium urban, Tsim Sha Tsui', deep='Deep urban, Whampoa')
    for ax, seq in zip(axes, ['medium', 'deep']):
        W = [w for w in D if w['seq'] == seq]
        lat0, lon0 = np.mean(W[0]['lat']), np.mean(W[0]['lon'])
        C = [w for w in W if certified(w)]
        segs, cols, dash = [], [], []
        for w in C:
            x, y = enu(w['lat'], w['lon'], lat0, lon0)
            segs.append(np.c_[x, y]); cols.append(w['heading_err']); dash.append(w['excitation'] < WEAK)
        for wk in (False, True):
            S = [s for s, d_ in zip(segs, dash) if d_ == wk]
            if S:
                lc = LineCollection(S, cmap=cmap, norm=norm, linewidths=3.2, capstyle='round',
                                    linestyles=(0, (1, 2.5)) if wk else 'solid')
                lc.set_array(np.array([c for c, d_ in zip(cols, dash) if d_ == wk])); ax.add_collection(lc)
        U = [w for w in W if not certified(w)]
        ux, uy = zip(*[enu(w['lat'][0], w['lon'][0], lat0, lon0) for w in U]) if U else ([], [])
        ax.plot(ux, uy, 'o', ms=4, color='#c8c7c2', zorder=3)
        allx = np.concatenate([s[:, 0] for s in segs]); ally = np.concatenate([s[:, 1] for s in segs])
        pad = 40
        ax.set_xlim(allx.min() - pad, allx.max() + pad); ax.set_ylim(ally.min() - pad, ally.max() + pad)
        ax.set_aspect('equal'); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel('east (m)'); ax.set_ylabel('north (m)')
        e = np.array([w['rot_err'] for w in C])
        ax.set_title('%s\n%d of %d windows certified, attitude error\nmedian %.2f°, worst %.1f°'
                     % (names[seq], len(C), len(W), np.median(e), e.max()), loc='left', fontsize=10, color=INK)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend='max')
    cb.set_label('heading error of the certified window (deg)'); cb.outline.set_visible(False)
    fig.text(0.01, 0.015, 'Each coloured segment is one certified 16 s window. Dotted: heading weakly observable '
             '(log₁₀ σ₂/σ₁ < −4). Grey dots: the car at a stop, not certified, left out.',
             fontsize=8.5, color=INK2)
    p = os.path.join(FIG, 'urbannav_certified.png'); fig.savefig(p, facecolor='white'); plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    mission(); urbannav()
