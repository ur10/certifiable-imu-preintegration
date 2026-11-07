"""Conditioning of the Cayley SDP: exact change of variables instead of a scalar obj_scale.

The cost is Q = B^T B, where B stacks every whitened residual row (IMU edges, GNSS priors, bias
priors, bias random walk). Forming Q squares B, and the objective at the optimum is a small
difference of huge entries: on synthetic sick at sigma = 1e-3, f* = 1.35 while max|Q| = 4.7e9.
A scalar obj_scale leaves the ratio max|Q| / f* unchanged, so it can only trade a convergence
failure (scale too large) against an absolute-tolerance floor (scale too small).

Here the homogenised vector is substituted as x = T x_t with

    x_i = d_i x_t_i + xbar_i s     (i != s),       x_s = x_t_s,

i.e. centre on a reference xbar (built a priori from the anchor, the chained Delta R, the GNSS
priors and the bias prior, never from the SDP solution) and then whiten by D = diag(d). This is
a congruence Z = T Z_t T^T: the feasible sets are in bijection, ranks and the optimum are
identical, so it is the same relaxation. Only the numbers MOSEK sees change:

    Q_t = (B T)^T (B T),  whose s-column is B xbar -- the residual at the reference, small,
    A_t = T^T A T,        with the Z_t[s,s] coefficient folded into the right-hand side.

Each variant's Z_t is mapped back to Z = T Z_t T^T and runs through the runner's own extraction
and rank test, so rank and accuracy are comparable across variants. eta is recomputed
accurately: f_hat = ||B x_hat||^2 (no cancellation) against p* = <Q_t, Z_t> and against the dual
bound b^T y + min(0, lambda_min(S)) tr(Z_t), with S = Q_t - sum y_m A_m.

    .venv/bin/python cayley_conditioning.py --nkf 8
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
import argparse, json, time
import numpy as np
from mosek.fusion import Model, Domain, Expr, ObjectiveSense, AccSolutionStatus
import sdp_preintegration_cayley as C
import cayley_iso_propagation as P

I3 = np.eye(3)
VARIANTS = {            # (centre, whiten)
    'base':     (False, None),       # T = I: must reproduce the library's obj_scale='auto'
    'whiten':   (False, 'all'),
    'centre':   (True, None),
    'c+w_lin':  (True, 'nonrot'),     # whiten v, p, b, Z^v, Z^p; leave R, Y, Y' at unit scale
    'c+w_all':  (True, 'all'),
}
BASE = {k: v for k, v in dict(P.NOMINAL, iso_reduce='mean', iso_propagate=False).items()
        if k != 'obj_scale'}


# --------------------------------------------------------------------------- #
def residual_matrix(lay, edges, scalar_priors=None, rw_pairs=None):
    """B with Q = B^T B equal to build_cost_matrix (same rows, unsquared)."""
    n = lay.n; rows = []
    for e, ed in enumerate(edges):
        i, j, dt = ed.i, ed.j, ed.dt_ij
        tR, tv, tp = np.sqrt(ed.tauR), np.sqrt(ed.tauv), np.sqrt(ed.taup)
        B = np.zeros((15, n))
        B[0:9, 9 * j:9 * (j + 1)] = tR * np.kron((I3 + 0.5 * ed.Gam).T, I3)
        B[0:9, 9 * i:9 * (i + 1)] = -tR * np.kron(ed.Rbr.T, I3)
        B[0:9, lay.Y0 + 9 * e:lay.Y0 + 9 * (e + 1)] = -0.5 * tR * np.eye(9)
        B[0:9, lay.Yp0 + 9 * e:lay.Yp0 + 9 * (e + 1)] = -0.5 * tR * np.eye(9)
        B[9:12, lay.v(j, 0):lay.v(j, 0) + 3] = tv * I3
        B[9:12, lay.v(i, 0):lay.v(i, 0) + 3] = -tv * I3
        B[9:12, lay.si] = -tv * C.GRAVITY * dt
        B[9:12, 9 * i:9 * (i + 1)] = -tv * np.kron(ed.vbr.reshape(1, 3), I3)
        B[9:12, lay.Zv0 + 3 * e:lay.Zv0 + 3 * e + 3] = -tv * I3
        B[12:15, lay.p(j, 0):lay.p(j, 0) + 3] = tp * I3
        B[12:15, lay.p(i, 0):lay.p(i, 0) + 3] = -tp * I3
        B[12:15, lay.v(i, 0):lay.v(i, 0) + 3] = -tp * dt * I3
        B[12:15, lay.si] = -tp * 0.5 * C.GRAVITY * dt * dt
        B[12:15, 9 * i:9 * (i + 1)] = -tp * np.kron(ed.pbr.reshape(1, 3), I3)
        B[12:15, lay.Zp0 + 3 * e:lay.Zp0 + 3 * e + 3] = -tp * I3
        rows.append(B)
    for m0, m1, tau_g, tau_a in (rw_pairs or []):
        for idx_f, tau in ((lay.bg, tau_g), (lay.ba, tau_a)):
            if tau <= 0: continue
            for c in range(3):
                r = np.zeros((1, n)); r[0, idx_f(m1, c)] = np.sqrt(tau); r[0, idx_f(m0, c)] = -np.sqrt(tau)
                rows.append(r)
    for idx, ref, tau in (scalar_priors or []):
        if tau <= 0: continue
        r = np.zeros((1, n)); r[0, idx] = np.sqrt(tau); r[0, lay.si] = -np.sqrt(tau) * float(ref)
        rows.append(r)
    return np.vstack(rows)


def lift(lay, edges, R, v, p, bg, ba):
    """Homogenised x from (R, v, p, b), lifting variables included (as in rank_and_gap)."""
    x = np.zeros(lay.n); bg = np.atleast_2d(bg); ba = np.atleast_2d(ba)
    for k in range(lay.N):
        x[9 * k:9 * (k + 1)] = R[k].flatten('F')
        x[lay.v(k, 0):lay.v(k, 0) + 3] = v[k]; x[lay.p(k, 0):lay.p(k, 0) + 3] = p[k]
    for m in range(lay.M):
        x[lay.bg(m, 0):lay.bg(m, 0) + 3] = bg[m]; x[lay.ba(m, 0):lay.ba(m, 0) + 3] = ba[m]
    for e, ed in enumerate(edges):
        vv = ed.JR @ bg[ed.blk]
        x[lay.Y0 + 9 * e:lay.Y0 + 9 * (e + 1)] = (R[ed.i] @ ed.dRt @ C.skew(vv)).flatten('F')
        x[lay.Yp0 + 9 * e:lay.Yp0 + 9 * (e + 1)] = (R[ed.j] @ C.skew(vv)).flatten('F')
        x[lay.Zv(e, 0):lay.Zv(e, 0) + 3] = R[ed.i] @ (ed.Avg @ bg[ed.blk] + ed.Ava @ ba[ed.blk])
        x[lay.Zp(e, 0):lay.Zp(e, 0) + 3] = R[ed.i] @ (ed.Apg @ bg[ed.blk] + ed.Apa @ ba[ed.blk])
    x[lay.si] = 1.0
    return x


def reference(lay, edges, anchors, scalar_priors):
    """A priori xbar: R_0 from the anchor (or its soft prior), R_k by chaining Delta R; v, p from
    the GNSS priors (or anchors); every bias block at the phi^B reference. No SDP output used."""
    ref = {idx: r for idx, r, tau in (scalar_priors or [])}
    if anchors and 'R0' in anchors:
        R0 = np.asarray(anchors['R0'])
    elif all(lay.R(0, r, c) in ref for r in range(3) for c in range(3)):
        R0 = C.project_to_so3(np.array([[ref[lay.R(0, r, c)] for c in range(3)] for r in range(3)]))
    else:
        raise ValueError('centring needs a rotation anchor or an R_0 prior (gauge R / R_prior)')
    R = [R0]
    for ed in edges: R.append(R[-1] @ ed.dRt)

    def vec(f, k, key):
        if k == 0 and anchors and key in anchors: return np.asarray(anchors[key], float)
        return np.array([ref[f(k, c)] for c in range(3)])
    v = [vec(lay.v, k, 'v0') for k in range(lay.N)]
    p = [vec(lay.p, k, 'p0') for k in range(lay.N)]
    bg = np.array([[ref[lay.bg(0, c)] for c in range(3)]] * lay.M)
    ba = np.array([[ref[lay.ba(0, c)] for c in range(3)]] * lay.M)
    return lift(lay, edges, R, v, p, bg, ba)


# --------------------------------------------------------------------------- #
LAST = {}   # per-solve diagnostics, read back by main()
SCALE = 'auto'   # 'auto' (min(1, 1e4/max|Q_t|)) or a fixed float; set by --scale


def make_solver(centre, whiten):
    def solve(layout, edges, anchors, scalar_priors=None, add_redundant=True, obj_scale=1.0,
              solver_params=None, accept_any=False, diag=None, rw_pairs=None, **_ignored):
        lay = layout; n = lay.n; s = lay.si
        B = residual_matrix(lay, edges, scalar_priors, rw_pairs)
        Qlib = C.build_cost_matrix(lay, edges, scalar_priors=scalar_priors, rw_pairs=rw_pairs)
        qerr = float(np.abs(B.T @ B - Qlib).max() / np.abs(Qlib).max())
        xbar = reference(lay, edges, anchors, scalar_priors) if centre else np.eye(n)[s]
        xbar = xbar.copy(); xbar[s] = 0.0                    # T[:, s] = xbar + e_s
        Bc = B.copy(); Bc[:, s] = B @ (xbar + np.eye(n)[s])   # centred residual column
        d = np.ones(n)
        if whiten:
            dq = np.sqrt(np.einsum('ij,ij->j', Bc, Bc))       # sqrt(diag of centred Q)
            mask = np.ones(n, bool); mask[s] = False
            if whiten == 'nonrot':
                mask[:lay.v0] = False; mask[lay.Y0:lay.Zv0] = False
            d[mask & (dq > 0)] = 1.0 / dq[mask & (dq > 0)]
        Bt = Bc * d[None, :]; Bt[:, s] = Bc[:, s]
        Qt = Bt.T @ Bt
        qmax = float(np.abs(Qt).max())
        sc = min(1.0, 1e4 / qmax) if SCALE == 'auto' else float(SCALE)
        A_list, b_list, cons = [], [], []
        # Row a of T: (a, d_a) and (s, xbar_a); row s: (s, 1).
        Trow = [((a, d[a]), (s, xbar[a])) if a != s else ((s, 1.0),) for a in range(n)]

        with Model('cond') as M:
            Z = M.variable('Z', Domain.inPSDCone(n))
            M.objective(ObjectiveSense.Minimize, Expr.dot((Qt * sc).tolist(), Z))

            def put(dct, rhs):
                mx = max([abs(v) for v in dct.values()] + [0.0])
                dct = {k: v / mx for k, v in dct.items() if abs(v) > 1e-13 * mx} if mx > 0 else {}
                if not dct: return
                rhs = rhs / mx
                ex = [Expr.mul(v, Z.index(a, b)) for (a, b), v in dct.items()]
                cons.append(M.constraint(Expr.add(ex) if len(ex) > 1 else ex[0], Domain.equalsTo(rhs)))
                A = np.zeros((n, n))
                for (a, b), v in dct.items():
                    if a == b: A[a, a] += v
                    else: A[a, b] += v / 2; A[b, a] += v / 2
                A_list.append(A); b_list.append(rhs)

            put({(s, s): 1.0}, 1.0)
            first = [True]

            def add(terms, rhs=0.0):
                if first[0]: first[0] = False; return   # the s^2 = 1 row, added above
                dct = {}
                for c, a, b in terms:
                    for k, ta in Trow[a]:
                        for l, tb in Trow[b]:
                            co = float(c) * ta * tb
                            if co == 0.0: continue
                            key = (k, l) if k <= l else (l, k)
                            dct[key] = dct.get(key, 0.0) + co
                rhs = float(rhs) - dct.pop((s, s), 0.0)      # fold: Z_t[s,s] = 1
                put(dct, rhs)

            C._constraints(lay, edges, anchors, add, redundant=add_redundant)
            for k, v in (solver_params or {}).items(): M.setSolverParam(k, v)
            if accept_any: M.acceptedSolutionStatus(AccSolutionStatus.Anything)
            t0 = time.time(); M.solve(); tsolve = time.time() - t0
            solsta = str(M.getPrimalSolutionStatus()); prosta = str(M.getProblemStatus())
            Zt = np.array(Z.level()).reshape(n, n)
            y = np.array([float(np.asarray(c.dual()).ravel()[0]) for c in cons])
        # dual certificate in the solver's units: S = sc Q_t - sum y A, bound = b^T y
        best = None
        for sgn in (1.0, -1.0):
            S = sc * Qt - sgn * np.tensordot(y, np.array(A_list), 1)
            lmin = float(np.linalg.eigvalsh((S + S.T) / 2)[0])
            lb = (sgn * float(np.dot(y, b_list)) + min(0.0, lmin) * float(np.trace(Zt))) / sc
            if best is None or lmin > best[0]: best = (lmin / sc, lb)
        T = np.diag(d); T[:, s] += xbar; T[s, s] = 1.0
        Zo = T @ Zt @ T.T
        LAST.clear(); LAST.update(B=B, layout=lay, edges=edges, solsta=solsta, prosta=prosta,
                                  qmax=qmax, scale=sc, pstar=float(np.sum(Qt * Zt)),
                                  lmin_S=best[0], dual_lb=best[1], tsolve=tsolve,
                                  rank_t=int(np.sum((w := np.linalg.eigvalsh(Zt)) > 1e-6 * w.max())),
                                  fbar=float(Qt[s, s]), qerr=qerr)
        if diag is not None: diag.update(solsta=solsta, prosta=prosta, obj_scale=sc, qmax=qmax)
        Q = B.T @ B
        return Zo, Q, float(np.sum(Q * Zo))
    return solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nkf', type=int, default=8)
    ap.add_argument('--kpb', type=int, default=None)
    ap.add_argument('--ds', nargs='*', default=['euroc', 'medium', 'deep', 'sick', 'loop'])
    ap.add_argument('--offsets', nargs='*', type=float, default=P.OFFSETS)
    ap.add_argument('--sigmas', nargs='*', type=float, default=[1e-3, 1e-2, 1e-1])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    ap.add_argument('--scale', default='auto', help="'auto' or a fixed obj scale, e.g. 1.0")
    ap.add_argument('--out', default='conditioning.json')
    a = ap.parse_args()
    global SCALE
    SCALE = a.scale if a.scale == 'auto' else float(a.scale)
    jobs = []
    for ds in a.ds:
        jobs += ([(ds, None, s, sd) for s in a.sigmas for sd in range(a.seeds)]
                 if ds in ('sick', 'loop') else [(ds, off, None, 0) for off in a.offsets])
    orig = C.solve_cayley_sdp; rows = []
    for ds, off, sig, seed in jobs:
        for name in a.variants:
            C.solve_cayley_sdp = make_solver(*VARIANTS[name]); t0 = time.time()
            try:
                r = P._solve(ds, off, a.nkf, a.kpb, seed, sig, obj_scale='auto', **BASE)
                d = P._metrics(r)
                lay, edges, B = LAST['layout'], LAST['edges'], LAST['B']
                v_est = r.get('v_est', r.get('vel_est')); p_est = r.get('p_est', r.get('pos_est'))
                xh = lift(lay, edges, r['R_est'], v_est, p_est, r['bg'], r['ba'])
                f = float(np.sum((B @ xh) ** 2)); ps = LAST['pstar']; lb = LAST['dual_lb']
                d.update({k: LAST[k] for k in ('solsta', 'prosta', 'qmax', 'scale', 'lmin_S',
                                               'tsolve', 'rank_t', 'fbar', 'qerr')},
                         f_acc=f, pstar_acc=ps, dual_lb=lb,
                         eta_acc=abs(f - ps) / (1 + abs(f) + abs(ps)),
                         eta_dual=(f - lb) / (1 + abs(f) + abs(lb)))
            except Exception as exc:
                d = dict(rank=None, status='EXC %r' % exc)
            finally:
                C.solve_cayley_sdp = orig
            d.update(variant=name, scale_mode=str(a.scale), ds=ds, off=off, sigma=sig, seed=seed, nkf=a.nkf, kpb=a.kpb,
                     secs=time.time() - t0)
            rows.append(d)
            print('%-8s %-6s off=%-5s sig=%-6s seed=%d qmax=%-8.1e fbar=%-8.1e %-8s rank=%-3s '
                  'eta=%-8.1e eta_acc=%-8.1e eta_dual=%-8.1e rot=%.4f pos=%.4f (%.1fs)' % (
                      name, ds, off, sig, seed, d.get('qmax', np.nan), d.get('fbar', np.nan),
                      str(d.get('solsta', d.get('status'))).split('.')[-1][:8], d.get('rank'),
                      d.get('eta', np.nan), d.get('eta_acc', np.nan), d.get('eta_dual', np.nan),
                      d.get('rot_mean', np.nan), d.get('pos_rmse', np.nan), d['secs']), flush=True)
            json.dump(rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
