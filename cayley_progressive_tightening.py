"""Progressive tightening on synthetic trajectories (Sec. VII-A of the paper).

Adds the redundant families of Sec. VI cumulatively, in the order they are derived, and sweeps
injected sensor noise at each stage. The point is to identify which families are actually
responsible for certifiability, rather than only contrasting "bare" against "everything".

Stage 0  none                      the base QCQP constraints only
Stage 1  + A1, A2                  the remaining SO(3) descriptions
Stage 2  + B2                      wedge kernel identity
Stage 3  + B3                      wedge Gram identity
Stage 4  + B1c                     the shared-axis tie between Y and Y'
Stage 5  + C2                      vector-lifting Gram identity  (= full catalog)
"""
import argparse, json, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sdp_preintegration_cayley as C
from sdp_preintegration_cayley import skew
import cayley_dual_certificate as D

_CAP = {}
def _wrapped(layout, edges, anchors, scalar_priors=None, add_redundant=True, obj_scale=1.0,
             solver_params=None, accept_any=False, diag=None, rw_pairs=None):
    Q, Qs, osc, Z, A, b, y = D.solve_with_duals(
        layout, edges, anchors, scalar_priors=scalar_priors, add_redundant=add_redundant,
        obj_scale=obj_scale, solver_params=solver_params, rw_pairs=rw_pairs)
    _CAP.update(Q=Q, Qs=Qs, osc=osc, Z=Z, A=A, b=b, y=y, lay=layout, ed=edges)
    return Z, Q, float(np.sum(Q * Z))
C.solve_cayley_sdp = _wrapped

def _xhat(lay, edges, R, v, p, bg, ba):
    x = np.zeros(lay.n)
    for k in range(lay.N):
        x[lay.R0+9*k:lay.R0+9*(k+1)] = R[k].flatten('F')
        x[lay.v(k,0):lay.v(k,0)+3] = v[k]; x[lay.p(k,0):lay.p(k,0)+3] = p[k]
    bg = np.atleast_2d(bg); ba = np.atleast_2d(ba)
    for m in range(lay.M):
        x[lay.bg(m,0):lay.bg(m,0)+3] = bg[m]; x[lay.ba(m,0):lay.ba(m,0)+3] = ba[m]
    for e, ed in enumerate(edges):
        vv = ed.JR @ bg[ed.blk]
        x[lay.Y0+9*e:lay.Y0+9*(e+1)] = (R[ed.i]@ed.dRt@skew(vv)).flatten('F')
        x[lay.Yp0+9*e:lay.Yp0+9*(e+1)] = (R[ed.j]@skew(vv)).flatten('F')
        x[lay.Zv(e,0):lay.Zv(e,0)+3] = R[ed.i]@(ed.Avg@bg[ed.blk]+ed.Ava@ba[ed.blk])
        x[lay.Zp(e,0):lay.Zp(e,0)+3] = R[ed.i]@(ed.Apg@bg[ed.blk]+ed.Apa@ba[ed.blk])
    x[lay.si] = 1.0
    return x

STAGES = [("none",            ()),
          ("+A1,A2",          ('A1','A2')),
          ("+B2",             ('A1','A2','B2')),
          ("+B3",             ('A1','A2','B2','B3')),
          ("+B1c",            ('A1','A2','B2','B3','B1c')),
          ("+C2 (full)",      ('A1','A2','B2','B3','B1c','C2'))]
SIGMAS = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
NOMINAL = 1e-3      # modelled noise density -> fixes the weights; only the injection varies
# Same solver/weighting configuration as the EuRoC and UrbanNav runs of
# cayley_multibias_experiment.py, so the three experiments are directly comparable.
BASE = dict(weight_mode='dataset_cov', iso_reduce='mean', obj_scale='auto',
            prior_sigmas=(1e-2, 1e-2, 1e-3, 1e-3), accept_any=True)
SURF='#fcfcfb'; GRID='#e1e0d9'; AXIS='#c3c2b7'; MUTED='#898781'; INK='#0b0b0b'; INK2='#52514e'
COLS = ['#e34948','#c77b1e','#b0a02a','#2a9d5c','#2a78d6','#008300']


def run(scenarios=('sick','loop'), T=7.0, dt=0.01):
    res = {}
    for sc in scenarios:
        res[sc] = {}
        print("=== %s ===" % sc, flush=True)
        for lab, fams in STAGES:
            rk, gp = [], []
            for s in SIGMAS:
                red = True if len(fams) == len(C.ALL_FAMILIES) else (fams if fams else False)
                try:
                    r = C.run_synthetic(sc, T=T, dt=dt, gyro_sigma=NOMINAL,
                                        accel_sigma=NOMINAL, noise_sigma=s,
                                        add_redundant=red, **BASE)
                    if r['rank'] is None:
                        rk.append(np.nan); gp.append(np.nan)
                    else:
                        Q,Qs,osc,Z,A,b_,y,lay,ed = (_CAP[k] for k in
                            ('Q','Qs','osc','Z','A','b','y','lay','ed'))
                        R,v,pp,bg,ba = C.extract_solution(Z, lay)
                        xh = _xhat(lay, ed, R, v, pp, bg, ba)
                        rounded = float(xh @ Q @ xh)
                        out, best = D.certify(Q, Qs, osc, A, b_, y, xh)
                        rk.append(r['rank']); gp.append(max(rounded - out[best]['bound'], 1e-14))
                except Exception as e:
                    rk.append(np.nan); gp.append(np.nan)
            res[sc][lab] = dict(rank=rk, gap=gp)
            print("  %-12s rank %-28s gap %s" % (lab, rk,
                  ' '.join('%.1e'%g for g in gp)), flush=True)
    return res


def plot(res, png):
    """Top: certified duality gap, rounded feasible cost minus the verified dual lower bound
    b'y with S = Q - sum y_m A_m >= 0. Bottom: rank. The ground truth is chained from the
    noise-free increments while the estimator sees the noisy ones, so the optimum scales with
    the injected noise and both panels are meaningful."""
    scs = list(res)
    fig, axes = plt.subplots(2, len(scs), figsize=(6.2*len(scs), 7.0), squeeze=False)
    fig.patch.set_facecolor(SURF)
    xs = np.where(np.asarray(SIGMAS) == 0.0, SIGMAS[1]/3, SIGMAS)
    for c, sc in enumerate(scs):
        for row, key, ylab, logy in [(0,'gap','certified duality gap',True),
                                     (1,'rank',r'rank $\hat Z$',False)]:
            ax = axes[row][c]; ax.set_facecolor(SURF)
            for k,(lab,_) in enumerate(STAGES):
                y = np.asarray(res[sc][lab][key], float)
                (ax.loglog if logy else ax.semilogx)(xs, y, marker='o', ms=4.5, lw=1.8,
                                                    color=COLS[k], label=lab)
            ax.grid(True, which='both', color=GRID, lw=0.6); ax.set_axisbelow(True)
            for sp in ax.spines.values(): sp.set_color(AXIS)
            ax.tick_params(colors=MUTED)
            if row == 0:
                ax.set_title(sc, color=INK, fontsize=11)
                if c == 0:
                    lg=ax.legend(fontsize=8, frameon=False); [t.set_color(INK2) for t in lg.get_texts()]
            else:
                ax.axhline(1, color=MUTED, ls=':', lw=1)
                ax.set_xlabel('injected noise stddev', color=INK2)
            if c == 0: ax.set_ylabel(ylab, color=INK2)
    fig.suptitle('Progressive tightening: redundant families added cumulatively (synthetic)',
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(png, dpi=165, facecolor=SURF)
    print("saved", png)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--T", type=float, default=7.0)
    a = p.parse_args()
    t0=time.time()
    res = run(T=a.T)
    json.dump(dict(sigmas=SIGMAS, stages=[s[0] for s in STAGES], res=res),
              open('progressive_tightening.json','w'), indent=1)
    plot(res, 'progressive_tightening.png')
    print("total %.0f s" % (time.time()-t0))
