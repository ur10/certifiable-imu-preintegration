"""demos/out/sim_mission.json -> demos/figures/mission.mp4 (+ poster): the safety-monitor video.

Left: the simulated quadrotor at its true position with three attitudes -- ground truth (grey
ghost), the Cayley SDP estimate (blue) and the sliding-window LM estimate (orange) -- plus the
GNSS fixes it is receiving. Right: mission timelines with a moving cursor: GNSS accuracy, the
certificate per window, heading observability, and attitude error of each estimator.

At time t the estimate shown is the one from the window centred nearest t (a fixed-lag smoother
with ~8 s lag); between keyframes the attitude error is interpolated and applied to the 200 Hz
truth, so the drawing is exact at keyframes and smooth between.

    python demos/render_mission.py [--speed 4] [--fps 30]
"""
import argparse, json, os, subprocess, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from _common import OUT, FIG, BLUE, ORANGE, GREY, INK, INK2, so3_log, so3_exp, certified
import sim_quad as S

AMBER = '#c98500'
WEAK = -4.0          # log10(sigma2/sigma1) below which heading is flagged as weakly observable


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


SHORT = ['hover', 'figure-8', 'jammed', 'hover + yaw', 'dashes', 'land']


def drone_lines(R, p, s=1.3):
    """Line segments of an X-frame quadrotor (arms, rotor rings, nose, thrust mast)."""
    segs = []
    arms = [np.array([np.cos(a), np.sin(a), 0.0]) * s for a in np.radians([45, 135, 225, 315])]
    for a in arms:
        segs.append((p, p + R @ a))
        th = np.linspace(0, 2 * np.pi, 17)
        ring = [p + R @ (a + 0.32 * s * np.array([np.cos(u), np.sin(u), 0.0])) for u in th]
        segs += list(zip(ring[:-1], ring[1:]))
    segs.append((p, p + R @ np.array([0.75 * s, 0, 0])))        # nose
    segs.append((p, p + R @ np.array([0, 0, 0.6 * s])))         # thrust axis
    return segs


def main(speed=4.0, fps=30, out=None, src=None, limit=None):
    d = json.load(open(src or os.path.join(OUT, 'sim_mission.json')))
    W = d['windows']
    for w_ in W:
        w_['certified'] = certified(w_)
    t0s = np.array([w['t0'] for w in W]); Tw = d['T']
    gt = np.loadtxt(os.path.join(S.OUT, 'state_groundtruth_estimate0', 'data.csv'), delimiter=',', skiprows=1)
    tt = (gt[:, 0] - gt[0, 0]) * 1e-9; P = gt[:, 1:4]; Q = gt[:, 4:8]

    def pick(t):
        return int(np.argmin(np.abs(t0s + Tw / 2 - t)))

    def attitude(t, key):
        w = W[pick(t)]
        u = np.clip(t - w['t0'], 0, Tw - 1e-9); k = int(np.floor(u)); s = u - k
        Rt_k = [np.asarray(R) for R in w['R_true']]
        Re = [np.asarray(R) for R in (w['R_sdp'] if key == 'sdp' else w['lm']['R'])]
        E0, E1 = Re[k] @ Rt_k[k].T, Re[k + 1] @ Rt_k[k + 1].T
        E = E0 @ so3_exp(s * so3_log(E0.T @ E1))
        i = min(int(np.searchsorted(tt, t)), len(tt) - 1)
        return E @ quat_to_R(Q[i]), P[i], w

    # per-window series for the timelines (plotted at window centres)
    tc = t0s + Tw / 2
    r0 = d.get('r0_sigma')
    att = ('weak %.0f° prior on each window\'s first attitude' % np.degrees(r0 / np.sqrt(2))) if r0 else 'attitude never given'
    cert = np.array([w['certified'] for w in W]); exc = np.array([w['excitation'] for w in W])
    e_sdp = np.array([max(w['rot_err'], 1e-3) for w in W]); e_lm = np.array([max(w['lm']['rot_err'], 1e-3) for w in W])
    sig = np.array([w['sigma_p'] for w in W])

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax3 = fig.add_axes([0.0, 0.06, 0.56, 0.86], projection='3d')
    rs = [fig.add_axes([0.62, y, 0.355, h]) for y, h in ((0.81, 0.09), (0.715, 0.05), (0.565, 0.10), (0.425, 0.085), (0.08, 0.205))]
    title = fig.text(0.015, 0.955, '', fontsize=17, color=INK, weight='bold')
    sub = fig.text(0.015, 0.925, '', fontsize=12, color=INK2)
    mon = [fig.text(0.015, y, '', fontsize=12.5, color=INK, family='DejaVu Sans Mono', weight='bold')
           for y in (0.100, 0.072, 0.044, 0.016)]

    def style(a):
        for s_ in ('top', 'right'):
            a.spines[s_].set_visible(False)
        for s_ in ('left', 'bottom'):
            a.spines[s_].set_color(GREY)
        a.tick_params(colors=INK2, labelsize=9); a.set_xlim(0, S.DUR)
        for i, (s0, e0, _) in enumerate(S.PHASES):
            if i % 2:
                a.axvspan(s0, e0, color='#f1f0ec', zorder=0, lw=0)
        a.axvspan(*S.JAM, color='#f6e2c8', zorder=0, lw=0)

    a0, a1, a2, ar, a3 = rs
    style(a0); a0.step(tc, sig, where='mid', color=INK2, lw=1.5); a0.set_yscale('log'); a0.set_ylim(5e-3, 5)
    a0.set_ylabel('GNSS σ (m)', color=INK2, fontsize=10)
    for (s_, e_, _), lab in zip(S.PHASES, SHORT):
        a0.text((s_ + e_) / 2, 6.5, lab, ha='center', va='bottom', fontsize=8.5, color=INK2, clip_on=False)
    style(a1)
    for c, w_ in zip(cert, W):
        a1.add_patch(plt.Rectangle((w_['t0'] + Tw / 2 - 2, 0.1), 4, 0.8, color=BLUE if c else ORANGE, lw=0))
    a1.set_ylim(0, 1); a1.set_yticks([]); a1.set_ylabel('certified', color=INK2, fontsize=10)
    style(a2); a2.plot(tc, exc, color=INK2, lw=1.5); a2.axhline(WEAK, color=AMBER, lw=1, ls='--')
    a2.fill_between(tc, exc, WEAK, where=exc < WEAK, color=AMBER, alpha=0.25, lw=0)
    a2.set_ylabel('heading\nexcitation', color=INK2, fontsize=10)
    a2.text(2, WEAK + 0.15, 'below: heading weakly observable', color=AMBER, fontsize=8)
    style(ar)
    risk = np.array([100 * w['lm_cold']['trapped_frac'] for w in W])
    ar.bar(tc, risk, width=3.2, color=ORANGE, alpha=0.8, lw=0)
    ar.set_ylim(0, 60); ar.set_ylabel('restart\ntrapped (%)', color=INK2, fontsize=10)
    ar.text(2, 50, 'share of 20 random-attitude restarts that end in a local minimum', fontsize=8, color=INK2, va='top')
    style(a3); a3.step(tc, e_sdp, where='mid', color=BLUE, lw=2, label='Cayley SDP (no initial guess)')
    a3.step(tc, e_lm, where='mid', color=ORANGE, lw=2, label='LM, sliding window (warm-started)')
    cold = np.array([max(w['lm_cold']['rot_err'], 1e-3) for w in W]); ct = np.array([not w['lm_cold']['global_'] for w in W])
    a3.scatter(tc[ct], cold[ct], s=40, facecolors='white', edgecolors=ORANGE, linewidths=2, zorder=4,
               label='LM restarted from a random attitude: trapped')
    a3.set_yscale('log'); a3.set_ylim(0.02, 400); a3.set_ylabel('attitude error (deg)', color=INK2, fontsize=10)
    a3.set_xlabel('mission time (s)', color=INK2, fontsize=10)
    a3.axvline(d['reset'], color=INK2, lw=1, ls=':')
    a3.text(d['reset'] + 1, 300, 'scripted reset:\nsliding LM restarts\nfrom a random attitude', fontsize=8, color=INK2, va='top')
    a3.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
    cursors = [a.axvline(0, color=INK, lw=1.2) for a in rs]

    seed_seg = drone_lines(np.eye(3), P[0])
    gt_lines = Line3DCollection(seed_seg, colors=GREY, linewidths=1.6, alpha=0.6)
    sdp_lines = Line3DCollection(seed_seg, colors=BLUE, linewidths=2.4)
    lm_lines = Line3DCollection(seed_seg, colors=ORANGE, linewidths=2.4)
    for L in (gt_lines, lm_lines, sdp_lines):
        ax3.add_collection3d(L, autolim=False)
    ax3.plot(P[:, 0], P[:, 1], P[:, 2], color='#d7d6d2', lw=1)
    trail, = ax3.plot([], [], [], color=INK2, lw=1.6)
    gnss = ax3.scatter([], [], [], s=14, c=INK, depthshade=False)
    lo, hi = P.min(0) - 1.5, P.max(0) + 1.5
    ax3.set_xlim(lo[0], hi[0]); ax3.set_ylim(lo[1], hi[1]); ax3.set_zlim(0, hi[2] + 1.0)
    ax3.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] + 1.0))
    for a in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        a.set_pane_color((0.97, 0.97, 0.96, 1)); a.line.set_color(GREY)
    ax3.tick_params(colors=INK2, labelsize=8); ax3.view_init(elev=24, azim=-62)
    fig.text(0.02, 0.135, 'grey: truth    blue: Cayley SDP    orange: LM    dots: GNSS fixes (last 6 s)    '
             'drone drawn ~5x life size', fontsize=10, color=INK2)

    meas_t = np.concatenate([w['t0'] + np.arange(len(w['p_meas'])) for w in W])
    meas_p = np.concatenate([np.asarray(w['p_meas']) for w in W])
    tmp = os.path.join(OUT, '_mission_frames'); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    times = np.arange(0.0, S.DUR - 0.01, speed / fps)
    if limit:
        times = times[::max(1, len(times) // limit)]
    for fi, t in enumerate(times):
        Rs, p, w = attitude(t, 'sdp'); Rl, _, _ = attitude(t, 'lm')
        i = min(int(np.searchsorted(tt, t)), len(tt) - 1); Rt = quat_to_R(Q[i])
        gt_lines.set_segments(drone_lines(Rt, p)); sdp_lines.set_segments(drone_lines(Rs, p)); lm_lines.set_segments(drone_lines(Rl, p))
        j = max(0, i - int(10 / 0.005)); trail.set_data(P[j:i + 1:4, 0], P[j:i + 1:4, 1]); trail.set_3d_properties(P[j:i + 1:4, 2])
        m = (meas_t <= t) & (meas_t > t - 6)
        gnss._offsets3d = (meas_p[m, 0], meas_p[m, 1], meas_p[m, 2])
        for c in cursors:
            c.set_xdata([t, t])
        ph = [lab for s_, e_, lab in S.PHASES if s_ <= t < e_]
        title.set_text('Certificate as a safety monitor  ·  t = %5.1f s  ·  %s' % (t, ph[0] if ph else ''))
        sub.set_text('Simulated quadrotor, GNSS + IMU, %s. Sliding 16 s windows every 4 s, replayed at %g×.' % (att, speed))
        weak = w['excitation'] is not None and w['excitation'] < WEAK
        dis = w['lm']['vs_sdp']
        lines = [('SDP   ' + ('certified globally optimal' if w['certified'] else 'NOT certified (rank %d)' % w['rank']),
                  BLUE if w['certified'] else ORANGE),
                 (('      no heading information in the data: yaw rests on the attitude prior' if r0
                   else '      heading weakly observable: do not trust yaw') if weak else '', AMBER),
                 ('LM    ' + ('sliding window agrees with the certified optimum' if dis < 2 else
                              ('%.0f° from the certified optimum: reject' % dis if w['certified']
                               else 'cannot be checked without a certificate')),
                  INK2 if dis < 2 else ORANGE),
                 (('      a restart here from a random attitude is trapped %.0f° off: certificate rejects it'
                   % w['lm_cold']['rot_err']) if (w['certified'] and not w['lm_cold']['global_']) else '', ORANGE)]
        for tx, (msg, col) in zip(mon, lines):
            tx.set_text(msg); tx.set_color(col)
        fig.savefig(os.path.join(tmp, '%05d.png' % fi), facecolor='white')
        if fi == int(len(times) * 0.66):
            fig.savefig(os.path.join(FIG, 'mission_poster.png'), facecolor='white')
    out = out or os.path.join(FIG, 'mission.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(fps), '-i', os.path.join(tmp, '%05d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '24', '-preset', 'slow', '-movflags', '+faststart', out],
                   check=True)
    shutil.rmtree(tmp)
    print('wrote', out, '(%d frames)' % len(times))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=4.0)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--src'); ap.add_argument('--out'); ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    main(a.speed, a.fps, a.out, a.src, a.limit)
