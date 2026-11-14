"""demos/out/basin_ball.json -> a short looping clip of the basin of attraction (for thumbnails).

The same data as make_ball.py's Viser recording, rendered offline with matplotlib so it can be a
plain video: 2,000 LM runs descending through the rotation-vector ball while the camera orbits.

    python demos/web/render_ball_clip.py --out basin_clip.mp4 [--size 960x720]
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'out' / 'basin_ball.json'
BLUE, ORANGE = '#2a78d6', '#eb6834'


def main(out, size=(960, 720), fps=30, hold0=1.0, per_iter=4, max_iter=30, hold1=1.5):
    d = json.load(open(SRC))
    runs = d['runs']
    cols = np.array([BLUE if q['global_'] else ORANGE for q in runs])
    order = np.argsort([q['global_'] for q in runs])[::-1]          # draw orange on top
    paths = [np.asarray(q['path'], float)[:max_iter + 1] for q in runs]
    paths = [np.vstack([P, runs[i]['path'][-1]]) if len(runs[i]['path']) > max_iter + 1 else P
             for i, P in enumerate(paths)]
    L = max(len(P) for P in paths)

    def at(P, u):
        i = int(np.floor(u))
        if i >= len(P) - 1:
            return P[-1]
        a, b = P[i], P[i + 1]
        return (a if u - i < 0.5 else b) if np.linalg.norm(a - b) > np.pi else a + (u - i) * (b - a)

    frames_u = [0.0] * int(hold0 * fps) + list(np.arange(0, L - 1, 1 / per_iter)) + [L - 1.0] * int(hold1 * fps)
    W, H = size
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    th = np.linspace(0, 2 * np.pi, 120)
    tmp = Path(out).with_suffix('') .as_posix() + '_frames'
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    for fi, u in enumerate(frames_u):
        ax.cla(); ax.set_axis_off()
        for c in (np.c_[np.cos(th), np.sin(th), 0 * th], np.c_[0 * th, np.cos(th), np.sin(th)],
                  np.c_[np.cos(th), 0 * th, np.sin(th)]):
            ax.plot(*(np.pi * c).T, color='#cfcec9', lw=0.8)
        ax.plot([0, 0], [0, 0], [-np.pi, np.pi], color='#b9b8b2', lw=0.8)
        P = np.array([at(paths[i], u) for i in order])
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=cols[order], s=16, depthshade=False, linewidths=0)
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_zlim(-3.2, 3.2); ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-55 + 40 * fi / len(frames_u))
        fig.savefig(os.path.join(tmp, '%05d.png' % fi), facecolor='white')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(fps), '-i', os.path.join(tmp, '%05d.png'),
                    '-vf', 'crop=trunc(iw*0.6/2)*2:trunc(ih*0.6/2)*2', '-c:v', 'libx264', '-crf', '24', '-preset', 'slow',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', out], check=True)
    shutil.rmtree(tmp)
    print('wrote', out, len(frames_u), 'frames')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'build' / 'basin_clip.mp4'))
    ap.add_argument('--size', default='960x720')
    a = ap.parse_args()
    main(a.out, tuple(map(int, a.size.split('x'))))
