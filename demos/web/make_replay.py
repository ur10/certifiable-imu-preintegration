"""#1 "Same path, wrong drone": a Viser recording of the MH_04 window from demos/basin_ball.py.

Two copies of the scene side by side. Left: the attitude a trapped Levenberg-Marquardt run
converged to. Right: the certified Cayley SDP estimate. A translucent grey drone flies the ground
truth in both. Positions are the ground-truth track (GNSS pins them to ~1 cm in both estimates,
so they are indistinguishable); only the attitude differs.

Between keyframes the attitude ERROR is interpolated (slerp of R_est R_true^T) and applied to the
200 Hz ground-truth attitude, so each estimate is exact at its 17 keyframes and smooth between.

Needs viser==0.2.23 (the version of the deployed web viewer) and trimesh, not MOSEK; run it in any env that has them:
    python demos/web/make_replay.py --out <dir>        # reads demos/out/replay_window.json
"""
import argparse, json, os, shutil
from pathlib import Path
import numpy as np
import trimesh
import viser

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE.parent / 'out' / 'replay_window.json'
BLUE, ORANGE, GREY = (42, 120, 214), (235, 104, 52), (150, 150, 146)


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_wxyz(R):
    q = trimesh.transformations.quaternion_from_matrix(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]))
    return q  # trimesh returns w, x, y, z


def log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-6:
        A = (R + np.eye(3)) / 2; ax = A[:, np.argmax(np.diag(A))]; return ax / np.linalg.norm(ax) * th
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th)) * th


def exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def slerp(Ra, Rb, s):
    return Ra @ exp(s * log(Ra.T @ Rb))


def drone_parts(span=0.5):
    """(name, mesh, role) for a simple X-frame quadrotor; +x is the nose."""
    parts = []
    body = trimesh.creation.box((0.16, 0.10, 0.05)); parts.append(('body', body, 'main'))
    nose = trimesh.creation.cone(0.035, 0.09); nose.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])); nose.apply_translation((0.08, 0, 0))
    parts.append(('nose', nose, 'nose'))
    for i, a in enumerate((45, 135, 225, 315)):
        th = np.radians(a); L = span / 2
        arm = trimesh.creation.box((L, 0.025, 0.02))
        arm.apply_translation((L / 2, 0, 0)); arm.apply_transform(trimesh.transformations.rotation_matrix(th, [0, 0, 1]))
        parts.append(('arm%d' % i, arm, 'main'))
        rot = trimesh.creation.cylinder(0.085, 0.012, sections=32)
        rot.apply_translation((L * np.cos(th), L * np.sin(th), 0.025))
        parts.append(('rotor%d' % i, rot, 'front' if a in (45, 315) else 'main'))
    # thrust axis: a thin mast on +z, so "upside down" reads at a glance
    mast = trimesh.creation.cylinder(0.01, 0.16, sections=12); mast.apply_translation((0, 0, 0.1))
    parts.append(('mast', mast, 'nose'))
    return parts


def tint(c, f):
    return tuple(int(min(255, v + (255 - v) * f)) for v in c)


DRONE_SCALE = 3.0      # drawn 3x life size so the attitude reads at trajectory scale


def add_drone(server, name, color, opacity=None):
    root = server.scene.add_frame(name, show_axes=False)
    for pn, m, role in drone_parts():
        m = m.copy(); m.apply_scale(DRONE_SCALE)
        col = {'main': color, 'front': tint(color, 0.55), 'nose': (30, 30, 30)}[role]
        server.scene.add_mesh_simple('%s/%s' % (name, pn), m.vertices, m.faces, color=col,
                                     opacity=opacity, flat_shading=True)
    return root


def main(out, fps=30.0, speed=1.0):
    d = json.load(open(OUT_JSON))
    if d['R_lm_bad'] is None:
        raise SystemExit('no trapped LM run in replay_window.json')
    t = np.asarray(d['dense']['t']); p = np.asarray(d['dense']['p'])
    q = np.asarray(d['dense']['q_wxyz']); Rt = [quat_to_R(x) for x in q]
    kf = np.asarray(d['kf_t'])
    R_true_k = [np.asarray(R) for R in d['R_true']]
    E = {k: [np.asarray(Re) @ Rtk.T for Re, Rtk in zip(d[k], R_true_k)] for k in ('R_sdp', 'R_lm_bad')}
    p0 = p[0].copy(); P = p - p0                         # local coordinates for the viewer
    span_x = np.ptp(P[:, 0]) + 2.5
    lanes = {'lm': np.array([-span_x / 2 - P[:, 0].mean(), -P[:, 1].mean(), 0]),
             'sdp': np.array([span_x / 2 - P[:, 0].mean(), -P[:, 1].mean(), 0])}

    server = viser.ViserServer(port=8156)
    rec = server.get_scene_serializer()
    server.scene.set_up_direction('+z')
    zf = P[:, 2].min() - 0.6
    for lane, off in lanes.items():
        server.scene.add_grid('/%s/floor' % lane, width=np.ptp(P[:, 0]) + 2.5, height=np.ptp(P[:, 1]) + 2.5,
                              cell_size=0.5, position=(off[0] + P[:, 0].mean(), off[1] + P[:, 1].mean(), zf))
        server.scene.add_spline_catmull_rom('/%s/track' % lane, P[::10] + off, line_width=2.0,
                                            color=(120, 120, 116))
        pm = np.asarray(d['p_meas']) - p0
        server.scene.add_point_cloud('/%s/gnss' % lane, pm + off, colors=(40, 40, 40), point_size=0.06,
                                     point_shape='circle')
    s = d['sdp']; b = d['lm_bad']
    server.scene.add_label('/lm/title', 'LM: converged, %.0f° off' % b['rot_err'],
                           position=lanes['lm'] + [P[:, 0].mean(), P[:, 1].mean(), P[:, 2].max() + 2.6])
    server.scene.add_label('/sdp/title', 'Cayley SDP: certified, %.1f° off' % s['rot_err'],
                           position=lanes['sdp'] + [P[:, 0].mean(), P[:, 1].mean(), P[:, 2].max() + 2.6])
    drones = {('lm', 'est'): add_drone(server, '/lm/est', ORANGE),
              ('lm', 'gt'): add_drone(server, '/lm/gt', GREY, opacity=0.35),
              ('sdp', 'est'): add_drone(server, '/sdp/est', BLUE),
              ('sdp', 'gt'): add_drone(server, '/sdp/gt', GREY, opacity=0.35)}

    dt_frame = speed / fps
    times = np.arange(0.0, t[-1], dt_frame)
    for fi, tf in enumerate(times):
        i = min(int(np.searchsorted(t, tf)), len(t) - 1)
        k = min(int(np.floor(tf)), len(kf) - 2); sfrac = tf - kf[k]
        for lane, key in (('lm', 'R_lm_bad'), ('sdp', 'R_sdp')):
            Ek = slerp(E[key][k], E[key][k + 1], sfrac)
            for which, R in (('est', Ek @ Rt[i]), ('gt', Rt[i])):
                h = drones[(lane, which)]
                wxyz = R_to_wxyz(R); pos = P[i] + lanes[lane]
                if fi == 0:      # viser drops unchanged setters; force frame 0 so the loop restarts
                    h.wxyz, h.position = -wxyz, pos + 1e-4
                h.wxyz, h.position = wxyz, pos
        rec.insert_sleep(dt_frame / speed)
    rec.insert_sleep(1.0)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fn = out / 'drone_replay.viser'
    fn.write_bytes(rec.serialize())
    server.stop()
    cam = dict(position=(0.0, -(np.ptp(P[:, 1]) + 6.5), P[:, 2].mean() + 8.0),
               look_at=(0.0, 0.0, P[:, 2].mean()))
    json.dump(cam, open(out / 'drone_replay_camera.json', 'w'))
    print('wrote %s (%.1f MB), %d frames; camera %s' % (fn, fn.stat().st_size / 1e6, len(times), cam))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'build'))
    a = ap.parse_args()
    main(a.out)
