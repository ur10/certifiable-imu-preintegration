"""Demo 1 -- where a local solver ends up, depending on where it starts.

One 16 s window of EuRoC MH_04_difficult. We start Levenberg-Marquardt from N attitudes drawn
uniformly on SO(3) (GNSS supplies the initial positions and velocities, as it would in practice)
and record where each run converges. The Cayley SDP is solved once, with no initial guess, and
its certificate says whether its answer is the global optimum.

    python demos/basins.py            # ~2 min; writes demos/out/basins.json
    python demos/plot_basins.py       # -> demos/figures/basins.png
"""
import argparse, json, os, time
import numpy as np
from _common import (OUT, solve_window, run_lm, initial_attitude, rot_angle, certified,
                     heading_tilt)


def main(seq='MH_04_difficult', t_start=50.0, n=100, gnss_seed=0):
    t0 = time.time()
    r, W = solve_window(seq, t_start, gnss_seed)
    sdp_secs = time.time() - t0
    R_t = r['R_true']
    ref = run_lm(W, R_t, 'gt')                 # LM started at the truth: the reference optimum
    rows = []
    for init, seeds in (('gravity', [0]), ('gnss', [0]), ('haar_shared', range(n))):
        for sd in seeds:
            g = run_lm(W, R_t, init, seed=sd)
            rows.append(dict(init=init, seed=int(sd),
                             init_err=rot_angle(initial_attitude(W, R_t, init, sd), R_t[0]),
                             rot_err=float(np.mean([rot_angle(a, b) for a, b in zip(g['R_est'], R_t)])),
                             vs_sdp=float(np.mean([rot_angle(a, b) for a, b in zip(g['R_est'], r['R_est'])])),
                             cost=g['cost'], iters=g['iters'], secs=g['secs'],
                             global_=bool(g['cost'] <= ref['cost'] * 1.01 + 1e-9)))
    hd, tl = heading_tilt(r['R_est'], R_t)
    out = dict(seq=seq, t_start=t_start, T=16.0, gnss_seed=gnss_seed,
               sdp=dict(rank=r['rank'], eta=r['eta'], eta_dual=r['eta_dual'], log_svr=r['log_svr'],
                        certified=certified(r), secs=sdp_secs, solsta=r.get('solsta'),
                        rot_err=float(np.mean(r['rot_err'])), heading_err=hd, tilt_err=tl,
                        n_var=r['n_var']),
               lm_ref=dict(cost=ref['cost'], rot_err=float(np.mean([rot_angle(a, b) for a, b in zip(ref['R_est'], R_t)])),
                           vs_sdp=float(np.mean([rot_angle(a, b) for a, b in zip(ref['R_est'], r['R_est'])]))),
               lm=rows)
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, 'basins.json')
    json.dump(out, open(fn, 'w'), indent=1)
    rnd = [q for q in rows if q['init'] == 'haar_shared']
    bad = [q for q in rnd if not q['global_']]
    print('SDP: rank %d, eta %.1e, dual eta %.1e, certified %s, attitude error %.2f deg, %.0f s'
          % (r['rank'], r['eta'], r['eta_dual'], certified(r), np.mean(r['rot_err']), sdp_secs))
    print('LM from the truth: %.2f deg (%.2f deg from the SDP estimate)'
          % (out['lm_ref']['rot_err'], out['lm_ref']['vs_sdp']))
    for q in rows[:2]:
        print('LM from %-8s init error %6.1f deg -> %7.2f deg, %s'
              % (q['init'], q['init_err'], q['rot_err'], 'global' if q['global_'] else 'LOCAL MINIMUM'))
    print('LM from %d uniformly random attitudes: %d reach the global optimum, %d stop in a local '
          'minimum (median attitude error there %.1f deg)'
          % (len(rnd), len(rnd) - len(bad), len(bad),
             np.median([q['rot_err'] for q in bad]) if bad else float('nan')))
    print('wrote', fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='MH_04_difficult')
    ap.add_argument('--t', type=float, default=50.0)
    ap.add_argument('--n', type=int, default=100)
    a = ap.parse_args()
    main(a.seq, a.t, a.n)
