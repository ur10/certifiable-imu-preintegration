"""First-page teaser: 3D tracks for both solutions plus ground truth, with a 3D magnified inset.

EuRoC MH_01 at t=90s, K=33 keyframes, M=2 -- the window of Table III where Levenberg-Marquardt,
started from the GNSS measurements, converges to a gravity-flipped minimum.

Left: the whole window in 3D. Ground truth is the light grey track (the convention of Holmes
and Barfoot, RAL 2023, Fig. 1); each converged solution is a line. A dashed 3D box marks the
magnified segment. Right: that segment at the same viewpoint, with a body frame on every
keyframe, so the ~180 deg flip can be read off. Leader lines join the two.

Two things about mplot3d drove the layout. It does not clip 3D lines to the axes rectangle, so
the view zoom has to be calibrated or the left panel bleeds over the inset; and its inset
helpers are 2D only, so the leader lines are drawn by projecting the box corners to figure
coordinates by hand.
"""
import itertools
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import proj3d

from ablation_figs import RC

# Vector output for the paper. matplotlib defaults to Type 3 fonts in PDF/EPS, which IEEE
# PDF eXpress rejects; 42 is TrueType. Nimbus Roman is an OTF/CFF and embeds as CID Type 0C,
# which trips a font-type mismatch warning under fonttype 42, so the figure uses Liberation
# Serif instead -- metrically compatible with Times, and genuinely TrueType.
RC = dict(RC, **{'pdf.fonttype': 42, 'ps.fonttype': 42,
                 'font.serif': ['Liberation Serif', 'DejaVu Serif']})

GT = '#8a8a8a'          # darker than the original #c2c2c2, still clear of OURS
OURS = '#0b0b0b'
LOCAL = '#e69f00'       # Okabe-Ito orange; the old #d55e00 sat too close to AX[0] red
AX = ['#d62728', '#1a9850', '#1f6fb4']      # body x, y, z
BOX = '#e08a1e'
DASH = (0, (2.4, 1.7))
ELEV, AZIM = 35, 95


def _frames(ax, P, R, idx, L, lw, alpha=1.0, z0=10):
    for i in idx:
        o, M = np.asarray(P[i]), np.asarray(R[i])
        for k in range(3):
            ax.plot([o[0], o[0] + L * M[0, k]], [o[1], o[1] + L * M[1, k]],
                    [o[2], o[2] + L * M[2, k]], color=AX[k], lw=lw, alpha=alpha,
                    zorder=z0 - k)


PANE = (0.82, 0.82, 0.82, 1.0)   # box-wall shade, 1.0 = white
GRID = '#ffffff'                 # white grid; a grey one stops reading on panes this dark


def set_pane(shade, grid=None):
    """Pane (box wall) shade, 1.0 = white. Gridlines go white once the panes are dark
    enough that a grey grid stops reading against them."""
    global PANE, GRID
    PANE = (shade, shade, shade, 1.0)
    GRID = grid if grid is not None else ('#ffffff' if shade < .95 else '#cfcfcf')


def _boxed(ax, nticks=3, labels=True, ticklabels=True, tick_fs=7.5, lab_fs=9.0,
           lpad=(-3.0, -2.0, -3.5)):
    """Bounded 3D axes in the style of Dumbgen et al., RAL 2023, Fig. 1."""
    for a, name in ((ax.xaxis, 'x'), (ax.yaxis, 'y'), (ax.zaxis, 'z')):
        a.set_pane_color(PANE)
        a.pane.set_edgecolor(GRID)
        a.pane.set_linewidth(.5)
        a._axinfo['grid'].update(color=GRID, linewidth=.45, linestyle='-')
        a.line.set_color(GRID)
        a.line.set_linewidth(.6)
        a.set_major_locator(plt.MaxNLocator(nticks))
        a.set_tick_params(colors='#6a6a6a', pad=-2.0, labelsize=tick_fs,
                          labelcolor='#3a3a3a' if ticklabels else PANE)
    if not ticklabels:
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_ticklabels([])
    if labels:
        # pads are view dependent: at the corner view the labels sit outside the box, at
        # azim=95 both y and z land on the right and have to be pulled in.
        ax.set_xlabel('$x$ [m]', labelpad=lpad[0], fontsize=lab_fs)
        ax.set_ylabel('$y$ [m]', labelpad=lpad[1], fontsize=lab_fs)
        ax.set_zlabel('$z$ [m]', labelpad=lpad[2], fontsize=lab_fs)


def _setup(ax, lo, hi, zoom, elev=None, azim=None, boxed=False, **bkw):
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo), zoom=zoom)   # true aspect keeps the frames square
    ax.set_proj_type('ortho')
    ax.view_init(elev=ELEV if elev is None else elev,
                 azim=AZIM if azim is None else azim)
    if boxed:
        _boxed(ax, **bkw)
    else:
        ax.set_axis_off()


def _box(ax, lo, hi):
    xs, ys, zs = [lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]]
    for i in (0, 1):
        for j in (0, 1):
            ax.plot(xs, [ys[i]] * 2, [zs[j]] * 2, color=BOX, lw=.8, ls=DASH, zorder=14)
            ax.plot([xs[i]] * 2, ys, [zs[j]] * 2, color=BOX, lw=.8, ls=DASH, zorder=14)
            ax.plot([xs[i]] * 2, [ys[j]] * 2, zs, color=BOX, lw=.8, ls=DASH, zorder=14)


def _ff(fig, ax, pt):
    x2, y2, _ = proj3d.proj_transform(pt[0], pt[1], pt[2], ax.get_proj())
    px, py = ax.transData.transform((x2, y2))
    return px / fig.bbox.width, py / fig.bbox.height


def fig_teaser(png='teaser_3dinset.png', src='traj_teaser.json',
               zi=None, zoom_main=1.8, zoom_inset=1.6, probe=False,
               elev=None, azim=None, Lm=.115, Li=.07, boxed=False,
               rect_main=None, rect_inset=None, figsize=(3.5, 1.95)):
    d = json.load(open(src))
    Pd = np.asarray(d['p_dense'])
    # 'lm_bad' is the random-initialisation draw (traj_worst.py); 'lm_gnss' is the older
    # identity-initialisation dump (traj_178.py). Accept either.
    bad = 'lm_bad' if 'lm_bad' in d else 'lm_gnss'
    Po, Pf = np.asarray(d['ours']), np.asarray(d[bad])
    Ro, Rf = d['R_ours'], d['R_' + bad]
    zi = zi if zi is not None else d.get('zi', list(range(15, 21)))

    lo, hi = Pd.min(axis=0) - .18, Pd.max(axis=0) + .18
    W = np.vstack([Po[zi], Pf[zi]])
    zlo, zhi = W.min(axis=0) - .13, W.max(axis=0) + .13

    with plt.rc_context(RC):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect_main or [.0, .06, .60, .92], projection='3d')
        bx = fig.add_axes(rect_inset or [.625, .105, .355, .50], projection='3d')

        for P, col, lw, z in ((Pd, GT, 2.6, 1), (Po, OURS, 1.0, 3), (Pf, LOCAL, 1.0, 4)):
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=col, lw=lw, zorder=z)
        # body frames along the whole window as well, so the reader gets the pose sense of the
        # trajectory and not only of the magnified segment
        _frames(ax, Po, Ro, range(len(Po)), Lm, .7)
        _frames(ax, Pf, Rf, range(len(Pf)), Lm, .7)
        _setup(ax, lo, hi, zoom_main, elev, azim, boxed=boxed)    # calibrated so the track stays inside the left panel
        _box(ax, zlo, zhi)

        for P, col, lw, z in ((Pd, GT, 5.5, 1), (Po, OURS, 1.3, 3), (Pf, LOCAL, 1.3, 4)):
            bx.plot(P[:, 0], P[:, 1], P[:, 2], color=col, lw=lw, zorder=z)
        _frames(bx, Po, Ro, zi, Li, 1.5)
        _frames(bx, Pf, Rf, zi, Li, 1.5)
        _setup(bx, zlo, zhi, zoom_inset, elev, azim, boxed=boxed,
               labels=False, ticklabels=False, nticks=3)

        if probe:
            fig.canvas.draw()
            pts = np.vstack([Pd, Po, Pf, np.array(list(itertools.product(*zip(zlo, zhi))))])
            ff = np.array([_ff(fig, ax, q) for q in pts])
            print('  zoom_main=%.2f  x %.3f..%.3f  y %.3f..%.3f'
                  % (zoom_main, ff[:, 0].min(), ff[:, 0].max(), ff[:, 1].min(), ff[:, 1].max()))
            plt.close(fig)
            return ff

        bb = bx.get_position()
        fig.add_artist(Rectangle((bb.x0, bb.y0), bb.width, bb.height, transform=fig.transFigure,
                                 fill=False, ec=BOX, lw=1.0, zorder=20))

        # leader lines: of the eight projected box corners, take the two furthest right
        fig.canvas.draw()
        corners = np.array([_ff(fig, ax, c) for c in itertools.product(*zip(zlo, zhi))])
        bxr, byl, byh = corners[:, 0].max(), corners[:, 1].min(), corners[:, 1].max()
        top, bot = (bxr, byh), (bxr, byl)
        for src_, tgt in ((top, (bb.x0, bb.y1)), (bot, (bb.x0, bb.y0))):
            fig.add_artist(Line2D([src_[0], tgt[0]], [src_[1], tgt[1]],
                                  transform=fig.transFigure, color=BOX, lw=.75, ls=DASH,
                                  zorder=19))

        key = [Line2D([], [], color=GT, lw=2.6, label='ground truth'),
               Line2D([], [], color=OURS, lw=1.4, label='ours, certified'),
               Line2D([], [], color=LOCAL, lw=1.4, label='LM, local solver')]
        fig.legend(handles=key, frameon=False, loc='upper right', bbox_to_anchor=(.995, .99),
                   handlelength=1.5, handletextpad=.45, labelspacing=.2, borderpad=0,
                   fontsize=8.85)   # 1.5x the previous 5.9
        fig.savefig(png, dpi=600)
    print('wrote', png)


# The boxed layout after Dumbgen et al., RAL 2023, Fig. 1: bounded axes seen from the corner
# (two back walls plus the floor), rather than the floating-axes view of the earlier version.
# elev is 40 rather than their ~22 because this window is near planar (z spans 0.6 m against
# 3.2 x 4.5 m horizontally), and a low elevation foreshortens the loop until it is unreadable.
# The main panel is kept narrow so the z label clears the inset.
BOXED = dict(boxed=True, figsize=(3.5, 1.95), elev=40, azim=-60,
             zoom_main=1.18, rect_main=[-.01, .15, .555, .80],
             zoom_inset=1.15, rect_inset=[.685, .145, .305, .47])



# --------------------------------------------------------------------------- #
# Two-panel variant, following Dumbgen et al., RAL 2023, Fig. 1: one bounded 3D
# panel per solution, each carrying ground truth as well, titled, with a shared
# legend below. No inset -- the comparison is the two panels.
#
# Colours keep the paper's convention (grey ground truth, black ours, orange LM)
# rather than the reference's black/orange, because the Fig. 1 caption refers to
# "the orange trajectory" for the local solver and "(black)" for ours.
# Ground-truth triads are drawn thin and semi-transparent underneath each
# solution's triads, so on the left they read as coincident and on the right the
# flip is visible against them.
# --------------------------------------------------------------------------- #
TITLES = ('verified optimal solution', 'local solution')


def fig_two_panel(png='teaser_2panel.png', src='traj_teaser.json',
                  figsize=(3.5, 2.0), elev=40, azim=-60, zoom=1.16,
                  Lgt=.125, Lsol=.115, pad=.06, z_off=.068, facecolor='white',
                  lw_gt=4.8, lw_sol=2.0,
                  rect_l=(.005, .175, .47, .70), rect_r=(.475, .175, .47, .70)):
    d = json.load(open(src))
    bad = 'lm_bad' if 'lm_bad' in d else 'lm_gnss'
    Pd = np.asarray(d['p_dense'])
    Pt, Rt = np.asarray(d['p_true']), d['R_true']
    Po, Ro = np.asarray(d['ours']), d['R_ours']
    Pf, Rf = np.asarray(d[bad]), d['R_' + bad]

    lo, hi = Pd.min(axis=0) - pad, Pd.max(axis=0) + pad

    with plt.rc_context(RC):
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor(facecolor)
        axes = [fig.add_axes(rect_l, projection='3d'),
                fig.add_axes(rect_r, projection='3d')]
        for ax in axes:
            # RC sets axes.facecolor white, which would print each panel as a white block on
            # the darker canvas; only the box walls (PANE) should stay light.
            ax.patch.set_facecolor('none')

        for ax, (P, col), title, zlab in zip(
                axes, ((Po, OURS), (Pf, LOCAL)), TITLES, (False, True)):
            ax.plot(Pd[:, 0], Pd[:, 1], Pd[:, 2], color=GT, lw=lw_gt, zorder=1)
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=col, lw=lw_sol, zorder=3)
            _frames(ax, Pt, Rt, range(len(Pt)), Lgt, .5, alpha=.45, z0=6)
            _frames(ax, P, np.asarray(Ro if col is OURS else Rf),
                    range(len(P)), Lsol, .75, z0=10)
            # z ticks only on the right panel, as in the reference
            # Both panels share identical axes, so only the right one is fully labelled: the
            # left panel's y label would otherwise be overrun by the right panel's box, and a
            # duplicate adds nothing. mplot3d tick-label extents are unreliable at this view,
            # so the z label is suppressed and placed from the panel rect rather than measured.
            _setup(ax, lo, hi, zoom, elev, azim, boxed=True, lpad=(-3.0, -3.0, 0.0))
            ax.set_zlabel('')
            if not zlab:
                ax.set_ylabel('')
                ax.yaxis.set_ticklabels([])   # else the 0.0 pokes out behind the right pane
                ax.zaxis.set_ticklabels([])
            ax.set_title(title, pad=-2.0, fontsize=9.6)

        fig.text(rect_r[0] + rect_r[2] + z_off, rect_r[1] + rect_r[3] * .60,
                 '$z$ [m]', rotation=90, va='center', ha='left', fontsize=9.0)

        key = [Line2D([], [], color=GT, lw=lw_gt, label='ground truth'),
               Line2D([], [], color=OURS, lw=lw_sol * 1.4, label='Ours'),
               Line2D([], [], color=LOCAL, lw=lw_sol * 1.4, label='Levenberg\u2013Marquardt')]
        leg = fig.legend(handles=key, frameon=False, loc='lower center',
                         bbox_to_anchor=(.5, -.015), ncol=3, handlelength=1.5,
                         handletextpad=.45, columnspacing=1.6, borderpad=0, fontsize=7.4)
        for _t in leg.get_texts():
            _t.set_fontweight('bold')
        # RC sets savefig.facecolor to white, which would discard fig.patch.
        # The PDF is what the paper includes; the PNG is only for previewing.
        stem = png[:-4] if png.lower().endswith(('.png', '.pdf')) else png
        for ext in ('.pdf', '.png'):
            fig.savefig(stem + ext, dpi=600, facecolor=facecolor)
    print('wrote', stem + '.pdf', 'and', stem + '.png')


# elev=25 is the reference's front view, with z reading as the vertical axis. Panels are
# narrow and shifted right so the left x label and the right z label both stay on the canvas.
# azim=-75 is rotated further toward frontal than the reference's ~-60; the panels are
# narrower and shifted right so the higher zoom does not clip the left x ticks or the z label.
TWOPANEL = dict(elev=25, azim=-75, zoom=1.40, rect_l=(.015, .165, .45, .73),
                rect_r=(.455, .165, .45, .73))

if __name__ == '__main__':
    fig_two_panel(png='teaser_3dinset.png', **TWOPANEL)
