"""#2 "The basin of attraction": a Viser recording of demos/out/basin_ball.json.

Every point is one Levenberg-Marquardt run. Its position is the attitude error of keyframe 0 as a
rotation vector (axis x angle, inside the ball of radius pi). At the start the points fill the
ball uniformly; the recording then moves each point along its actual LM iterates (one
iterate per few frames) until it stops. Colour = where it ends: blue at the global optimum (the
point the SDP certifies, at the centre), orange at the mirror minimum ~157 deg away.

    python demos/web/make_ball.py --out <dir>        # needs viser==0.2.23, trimesh
"""
import argparse, json
from pathlib import Path
import numpy as np
import trimesh
import viser

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'out' / 'basin_ball.json'
BLUE, ORANGE, INK = np.array([42, 120, 214]), np.array([235, 104, 52]), (60, 60, 58)


def main(out, frames_per_iter=4, max_iter=30, fps=30.0, hold=2.0):
    d = json.load(open(SRC))
    runs = d['runs']
    n = len(runs)
    cols = np.array([BLUE if q['global_'] else ORANGE for q in runs], np.uint8)
    paths = []
    for q in runs:
        P = np.asarray(q['path'], float)[:max_iter + 1]
        if len(q['path']) > max_iter + 1:                # long tails: jump to the end point
            P = np.vstack([P, q['path'][-1]])
        paths.append(P)
    L = max(len(P) for P in paths)

    def at(P, u):
        """Position at fractional iterate u; no interpolation across the antipodal seam."""
        i = int(np.floor(u))
        if i >= len(P) - 1:
            return P[-1]
        a, b = P[i], P[i + 1]
        if np.linalg.norm(a - b) > np.pi:              # wrapped through |w| = pi
            return a if u - i < 0.5 else b
        return a + (u - i) * (b - a)

    server = viser.ViserServer(port=8157)
    rec = server.get_scene_serializer()
    server.scene.set_up_direction('+z')
    shell = trimesh.creation.icosphere(subdivisions=4, radius=np.pi)
    server.scene.add_mesh_simple('/shell', shell.vertices, shell.faces, color=(200, 200, 196),
                                 opacity=0.07, side='double')
    th = np.linspace(0, 2 * np.pi, 97)
    for nm, pts in (('eq', np.c_[np.cos(th), np.sin(th), 0 * th]), ('mx', np.c_[0 * th, np.cos(th), np.sin(th)]),
                    ('my', np.c_[np.cos(th), 0 * th, np.sin(th)])):
        server.scene.add_spline_catmull_rom('/rings/' + nm, np.pi * pts, line_width=1.0, color=(170, 170, 166))
    for ax, v, lab in ((0, (1, 0, 0), 'x'), (1, (0, 1, 0), 'y'), (2, (0, 0, 1), 'z (heading)')):
        v = np.array(v, float)
        server.scene.add_spline_catmull_rom('/axes/%d' % ax, np.array([-np.pi * v, np.pi * v]), line_width=1.5,
                                            color=(140, 140, 136))
        server.scene.add_label('/axes/l%d' % ax, lab, position=1.15 * np.pi * v)
    server.scene.add_icosphere('/sdp', radius=0.045, color=(20, 20, 20))
    server.scene.add_label('/sdp_label', 'global optimum (Cayley SDP)',
                           position=(0.0, 0.0, -0.5))
    bad = [q for q in runs if not q['global_']]
    if bad:
        m = np.median(np.asarray([q['path'][-1] for q in bad]), axis=0)
        server.scene.add_label('/mirror_label', 'local minima, ~%.0f° off, cost x%.0f'
                               % (np.degrees(np.linalg.norm(m)),
                                  np.median([q['cost'] for q in bad]) / d['ref_cost']),
                               position=m * 1.12 + (0, 0, 0.35))

    cloud = server.scene.add_point_cloud('/runs', np.array([P[0] for P in paths]), colors=cols,
                                         point_size=0.06, point_shape='circle')
    total = (L - 1) * frames_per_iter
    for f in range(total + 1):
        u = f / frames_per_iter
        pts = np.array([at(P, u) for P in paths], np.float32)
        if f == 0:                                       # force frame 0 (viser skips unchanged setters)
            cloud.points = pts + 1e-4
        cloud.points = pts
        rec.insert_sleep(1.6 if f == 0 else 1.0 / fps)      # hold the starting grid briefly
    rec.insert_sleep(hold)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fn = out / 'basin_ball.viser'
    fn.write_bytes(rec.serialize())
    server.stop()
    print('wrote %s (%.1f MB): %d runs, %d frames, %d end in the mirror minimum'
          % (fn, fn.stat().st_size / 1e6, n, total + 1, len(bad)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'build'))
    a = ap.parse_args()
    main(a.out)
