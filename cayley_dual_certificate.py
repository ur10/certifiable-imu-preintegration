"""Dual certificate for the Cayley IMU relaxation.

MOSEK's primal objective <Q,Z> is not a usable lower bound here: it comes out slightly ABOVE
the rounded feasible cost and no tolerance setting closes the sign (the optimum is exactly
rank one, so Z sits on the deepest face of the PSD cone and interior-point conditioning
degrades). This module instead extracts the dual multipliers y and forms

    S = Q - sum_m y_m A_m ,        bound = sum_m y_m b_m ,

so that S >= 0 certifies bound <= p*, independently of the primal value. With a candidate
x_hat from rounding, S x_hat = 0 and S >= 0 together certify x_hat as globally optimal.
"""
import numpy as np
from mosek.fusion import Model, Domain, Expr, ObjectiveSense, AccSolutionStatus
import sdp_preintegration_cayley as C


def solve_with_duals(layout, edges, anchors, scalar_priors=None, pair_priors=None,
                     add_redundant=True, obj_scale='auto', solver_params=None, rw_pairs=None):
    """Same SDP as solve_cayley_sdp, but retains constraint handles so the duals y_m and the
    constraint matrices A_m can be recovered afterwards."""
    kw = {}
    if pair_priors is not None: kw['pair_priors'] = pair_priors
    if rw_pairs is not None: kw['rw_pairs'] = rw_pairs
    try:
        Q = C.build_cost_matrix(layout, edges, scalar_priors=scalar_priors, **kw)
    except TypeError:
        Q = C.build_cost_matrix(layout, edges, scalar_priors=scalar_priors)
    n = layout.n
    if isinstance(obj_scale, str):
        qmax = float(np.abs(Q).max()); obj_scale = min(1.0, 1e4 / qmax) if qmax > 0 else 1.0
    Qs = Q * float(obj_scale)

    A_list, b_list = [], []
    with Model('dualcert') as M:
        Z = M.variable('Z', Domain.inPSDCone(n))
        M.objective(ObjectiveSense.Minimize, Expr.dot(Qs.tolist(), Z))
        cons = []

        def add(terms, rhs=0.0):
            d = {}
            for c, a, b in terms:
                a, b = (a, b) if a <= b else (b, a)
                d[(a, b)] = d.get((a, b), 0.0) + float(c)
            d = {k: v for k, v in d.items() if abs(v) > 1e-14}
            if not d:
                return
            ex = [Expr.mul(v, Z.index(a, b)) for (a, b), v in d.items()]
            cons.append(M.constraint(Expr.add(ex) if len(ex) > 1 else ex[0],
                                     Domain.equalsTo(float(rhs))))
            # symmetric A with <A,Z> = sum_{(a,b)} v_ab Z_ab
            A = {}
            for (a, b), v in d.items():
                if a == b: A[(a, a)] = A.get((a, a), 0.0) + v
                else:
                    A[(a, b)] = A.get((a, b), 0.0) + 0.5 * v
                    A[(b, a)] = A.get((b, a), 0.0) + 0.5 * v
            A_list.append(A); b_list.append(float(rhs))

        C._constraints(layout, edges, anchors, add, redundant=add_redundant)
        for k, v in (solver_params or {}).items():
            M.setSolverParam(k, v)
        M.acceptedSolutionStatus(AccSolutionStatus.Anything)
        M.solve()
        Zo = np.array(Z.level()).reshape(n, n)
        y = np.array([c.dual()[0] for c in cons])
    return Q, Qs, float(obj_scale), Zo, A_list, np.array(b_list), y


def certify(Q, Qs, obj_scale, A_list, b, y, x_hat, tol=1e-9):
    """Form S = Qs - sum y_m A_m for both dual sign conventions; report the PSD one."""
    out = {}
    for sgn in (+1.0, -1.0):
        S = Qs.copy()
        for A, ym in zip(A_list, sgn * y):
            for (i, j), v in A.items(): S[i, j] -= ym * v
        S = 0.5 * (S + S.T)
        ev = np.linalg.eigvalsh(S)
        bound = float((sgn * y) @ b) / obj_scale       # de-scale to true Q units
        out[sgn] = dict(min_eig=float(ev.min()), bound=bound,
                        Sx=float(np.linalg.norm(S @ x_hat)), S=S, ev=ev)
    best = min(out, key=lambda s: -out[s]['min_eig'])   # the one closest to PSD
    return out, best
