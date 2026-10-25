"""Uniform experiment suite: one configuration everywhere, weight_mode the only variation.

Fixed across every run in this file:
  * obj_scale = 'auto'          (Q rescaled so max|Q| ~ 1e4, keeps MOSEK in its tolerance band)
  * gauge_fix = False           (no hard anchor; GNSS v/p priors on *every* node, first_node=0)
  * prior_sigmas = (1e-2, 1e-2, 1e-3, 1e-3)
  * kf_interval = 1.0 s, accept_any = True, full redundant catalog unless a stage says otherwise

The only thing that varies is weight_mode in {'mle', 'dataset_cov'}.

Cost tightness is the standard relative suboptimality

    eta = |f_hat - p*| / (1 + |f_hat| + |p*|)

computed in sdp_preintegration_cayley.rank_and_gap. We also record f_hat and the plain
relative discrepancy |f_hat - p*| / |f_hat|, because eta's `+1` dominates whenever the optimum
is itself small -- in that regime eta reports an absolute difference and understates how far
apart the primal and rounded costs really are.
"""
import argparse, json, time
import numpy as np
import sdp_preintegration_cayley as C

import datasets
EUROC = datasets.EUROC          # override with $EUROC_MH01, see datasets.py
URBANNAV = datasets.URBANNAV    # override with $URBANNAV_ROOT

# --------------------------------------------------------------------------- #
# The one configuration
# --------------------------------------------------------------------------- #
UNIFORM = dict(kf_interval=1.0, prior_sigmas=(1e-2, 1e-2, 1e-3, 1e-3),
               obj_scale='auto', accept_any=True,
               gauge_fix=False, R0_prior_sigma=None)

MODES = {'mle': {}, 'dataset_cov': dict(iso_reduce='mean')}

STAGES = [("none",       ()),
          ("+A1,A2",     ('A1', 'A2')),
          ("+B2",        ('A1', 'A2', 'B2')),
          ("+B3",        ('A1', 'A2', 'B2', 'B3')),
          ("+B1c",       ('A1', 'A2', 'B2', 'B3', 'B1c')),
          ("+C2 (full)", ('A1', 'A2', 'B2', 'B3', 'B1c', 'C2'))]

SIGMAS = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
NOMINAL = 1e-3          # modelled density; only the injected noise varies across the sweep


def _kw(mode, **over):
    kw = dict(UNIFORM); kw.update(MODES[mode]); kw['weight_mode'] = mode; kw.update(over)
    return kw


def _rec(r, **meta):
    """Flatten one run_* result, adding the metrics eta cannot express on its own."""
    d = dict(meta)
    if r is None or r.get('rank') is None:
        d.update(rank=None, status=str(None if r is None else r.get('status')))
        return d
    f, p = float(r['rounded']), float(r['sdp_opt'])
    d.update(rank=int(r['rank']), eta=float(r['eta']), f_hat=f, p_star=p,
             rel=abs(f - p) / abs(f) if abs(f) > 0 else float('nan'),
             rot_mean=float(np.mean(r['rot_err'])), rot_max=float(np.max(r['rot_err'])),
             n_var=int(r.get('n_var', 0)), num_bias=int(r.get('num_bias', 1)),
             status=str(r.get('status', 'ok')))
    return d


def _solve(ds, mode, **over):
    kw = _kw(mode, **over)
    if ds == 'sick' or ds == 'loop':
        return C.run_synthetic(ds, **kw)
    if ds == 'euroc':
        return C.run_euroc(EUROC, **kw)
    return C.run_urbannav(URBANNAV, ds, **kw)


# --------------------------------------------------------------------------- #
# 1. Progressive tightening, synthetic
# --------------------------------------------------------------------------- #
def exp_stages(out='uniform_stages.json', T=7.0, dt=0.01):
    rows = []
    for mode in MODES:
        for sc in ('sick', 'loop'):
            for lab, fams in STAGES:
                red = True if len(fams) == len(C.ALL_FAMILIES) else (fams if fams else False)
                for s in SIGMAS:
                    t0 = time.time()
                    try:
                        r = _solve(sc, mode, T=T, dt=dt, gyro_sigma=NOMINAL,
                                   accel_sigma=NOMINAL, noise_sigma=s, add_redundant=red)
                    except Exception as exc:
                        r = None
                        print('  EXC %s %s %s %g: %s' % (mode, sc, lab, s, exc), flush=True)
                    rows.append(_rec(r, exp='stages', mode=mode, scen=sc, stage=lab,
                                     sigma=s, secs=time.time() - t0))
                    q = rows[-1]
                    print('%-11s %-5s %-11s sig=%-6.0e rank=%-4s eta=%-10.3e rel=%-10.3e f=%.3e'
                          % (mode, sc, lab, s, q.get('rank'), q.get('eta', np.nan),
                             q.get('rel', np.nan), q.get('f_hat', np.nan)), flush=True)
    json.dump(rows, open(out, 'w'), indent=1)
    return rows


# --------------------------------------------------------------------------- #
# 2. Real data
# --------------------------------------------------------------------------- #
OFFSETS = [6.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0]


def exp_real(out='uniform_real.json', T=8.0):
    rows = []
    for mode in MODES:
        for ds in ('euroc', 'medium', 'deep'):
            for off in OFFSETS:
                t0 = time.time()
                try:
                    r = _solve(ds, mode, T=T, t_start=off, add_redundant=True)
                except Exception as exc:
                    r = None
                    print('  EXC %s %s %g: %s' % (mode, ds, off, exc), flush=True)
                rows.append(_rec(r, exp='real', mode=mode, ds=ds, off=off,
                                 secs=time.time() - t0))
                q = rows[-1]
                print('%-11s %-7s off=%-5.0f rank=%-4s eta=%-10.3e rel=%-10.3e rot=%.3f'
                      % (mode, ds, off, q.get('rank'), q.get('eta', np.nan),
                         q.get('rel', np.nan), q.get('rot_mean', np.nan)), flush=True)
    json.dump(rows, open(out, 'w'), indent=1)
    return rows


# --------------------------------------------------------------------------- #
# 3. Multiple bias blocks
# --------------------------------------------------------------------------- #
MB_ROWS = [('euroc', 8, None), ('euroc', 16, None), ('euroc', 16, 8),
           ('euroc', 24, None), ('euroc', 24, 8),
           ('medium', 16, 8), ('medium', 24, 8), ('deep', 16, 8), ('deep', 24, 8)]
MB_OFF = [6.0, 40.0, 90.0, 132.0]


def exp_multibias(out='uniform_multibias.json'):
    rows = []
    for mode in MODES:
        for ds, nkf, kpb in MB_ROWS:
            for off in MB_OFF:
                t0 = time.time()
                try:
                    r = _solve(ds, mode, T=float(nkf) + 0.5, t_start=off,
                               kf_per_bias=kpb, add_redundant=True)
                except Exception as exc:
                    r = None
                    print('  EXC %s %s %d %s %g: %s' % (mode, ds, nkf, kpb, off, exc), flush=True)
                rows.append(_rec(r, exp='multibias', mode=mode, ds=ds, nkf=nkf,
                                 kpb=kpb, off=off, secs=time.time() - t0))
                q = rows[-1]
                print('%-11s %-7s kf=%-3d blk=%-4s off=%-5.0f rank=%-4s M=%s eta=%-10.3e rot=%.3f'
                      % (mode, ds, nkf, str(kpb), off, q.get('rank'), q.get('num_bias'),
                         q.get('eta', np.nan), q.get('rot_mean', np.nan)), flush=True)
    json.dump(rows, open(out, 'w'), indent=1)
    return rows


# --------------------------------------------------------------------------- #
def summarize(rows, keys, title):
    print('\n=== %s ===' % title)
    seen = []
    for r in rows:
        k = tuple(r.get(x) for x in keys)
        if k not in seen:
            seen.append(k)
    for k in seen:
        g = [r for r in rows if tuple(r.get(x) for x in keys) == k]
        ok = [r for r in g if r.get('rank') is not None]
        if not ok:
            print('  %-42s ALL FAILED' % (' '.join(map(str, k)))); continue
        e = np.array([r['eta'] for r in ok]); rl = np.array([r['rel'] for r in ok])
        print('  %-42s n=%d rank={%s} eta med %.2e max %.2e | rel med %.2e max %.2e | rot %.3f'
              % (' '.join(map(str, k)), len(ok), ','.join(map(str, sorted({r['rank'] for r in ok}))),
                 np.median(e), e.max(), np.median(rl), rl.max(),
                 np.mean([r['rot_mean'] for r in ok])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('which', nargs='*', default=['stages', 'real', 'multibias'])
    a = ap.parse_args()
    if 'stages' in a.which:
        summarize(exp_stages(), ['mode', 'scen', 'stage'], 'progressive tightening')
    if 'real' in a.which:
        summarize(exp_real(), ['mode', 'ds'], 'real data')
    if 'multibias' in a.which:
        summarize(exp_multibias(), ['mode', 'ds', 'nkf', 'kpb'], 'multiple bias blocks')


# --------------------------------------------------------------------------- #
def plot(png='uniform_stages.png', src='uniform_stages.json'):
    """Progressive tightening, both weight modes, one panel per trajectory."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = json.load(open(src))
    SURF = '#fcfcfb'; GRID = '#e1e0d9'; AXIS = '#c3c2b7'; MUTED = '#898781'; INK = '#0b0b0b'
    COL = {'mle': '#e34948', 'dataset_cov': '#2a78d6'}
    labs = [s[0] for s in STAGES]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.0), facecolor=SURF)
    for c, sc in enumerate(('sick', 'loop')):
        for mode in MODES:
            rk, et, rl = [], [], []
            for lab in labs:
                g = [r for r in rows if r['mode'] == mode and r['scen'] == sc
                     and r['stage'] == lab and r.get('rank') is not None]
                rk.append(np.median([r['rank'] for r in g]) if g else np.nan)
                et.append(np.median([r['eta'] for r in g]) if g else np.nan)
                rl.append(np.median([r['rel'] for r in g]) if g else np.nan)
            axes[0, c].plot(labs, rk, 'o-', color=COL[mode], label=mode, lw=1.8, ms=4)
            axes[1, c].semilogy(labs, et, 'o-', color=COL[mode], label=mode, lw=1.8, ms=4)
            axes[2, c].semilogy(labs, rl, 'o-', color=COL[mode], label=mode, lw=1.8, ms=4)
        axes[0, c].set_title(sc, color=INK)
        axes[0, c].set_yscale('log'); axes[0, c].axhline(1, color=MUTED, ls=':', lw=1)
        axes[0, c].set_ylabel(r'median $\mathrm{rank}\,\hat Z$', color=INK)
        axes[1, c].set_ylabel(r'median $\eta$  (not scale-invariant)', color=INK)
        axes[2, c].set_ylabel(r'median $|\hat f-p^\star|/|\hat f|$', color=INK)
        for ax in (axes[0, c], axes[1, c], axes[2, c]):
            ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=.6)
            for sp in ax.spines.values(): sp.set_color(AXIS)
            ax.tick_params(colors=MUTED, labelsize=8)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        axes[0, c].legend(frameon=False, fontsize=8)
    fig.suptitle('Progressive tightening, uniform config (no anchor, auto scaling).\n'
                 r"$\eta$ favours mle only because its objective is ~7 orders smaller; "
                 'the bottom row is the scale-invariant comparison.', color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(png, dpi=160, facecolor=SURF)
    print('wrote', png)
