"""Attribute the Fig. 3 stage-3 change (A1,A2,B2,B3) to the gauge or to the conditioning.
Same call as cayley_ablation_plots.run (T=8, kf_per_bias=4, NOMINAL=1e-3), 4 configurations."""
import sys, json, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np
import cayley_uniform_experiments as U
CONFIGS = {'gauge free, legacy auto': dict(gauge_fix=False, conditioning=None, obj_scale='auto'),
           'gauge free, centre_whiten (falls back: no R ref)': dict(gauge_fix=False, conditioning='centre_whiten', obj_scale=1.0),
           'shipped: gauge True, legacy auto': dict(gauge_fix=True, conditioning=None, obj_scale='auto'),
           'gauge True, centre_whiten':        dict(gauge_fix=True, conditioning='centre_whiten', obj_scale=1.0),
           'gauge R, legacy auto':             dict(gauge_fix='R', conditioning=None, obj_scale='auto'),
           'new: gauge R, centre_whiten':      dict(gauge_fix='R', conditioning='centre_whiten', obj_scale=1.0)}
rows = []
for name, kw in CONFIGS.items():
    for s in (1e-2, 0.5):
        for sc in ('sick', 'loop'):
            for sd in range(12):
                r = U._solve(sc, 'dataset_cov', T=8.0, dt=0.01, kf_per_bias=4, gyro_sigma=1e-3, accel_sigma=1e-3,
                             noise_sigma=s, add_redundant=('A1', 'A2', 'B2', 'B3'), seed=sd, **kw)
                rows.append(dict(config=name, sigma=s, scen=sc, seed=sd, rank=r['rank'], log_svr=r['log_svr'],
                                 eta=r['eta'], eta_dual=r['eta_dual'], solsta=r['solsta']))
        for s in (1e-2, 0.5):
            g = [q for q in rows if q['config'] == name and q['sigma'] == s]
            print('%-34s sigma=%-5g certified %2d/%d  ranks %s  eta_dual med %.1e' % (
                name, s, sum(q['log_svr'] > 6 for q in g), len(g), sorted(q['rank'] for q in g),
                np.median([abs(q['eta_dual']) for q in g])), flush=True)
json.dump(rows, open('attribute_stage3.json', 'w'), indent=1)
