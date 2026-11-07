"""Summarise runs/centre_whiten/*.json against the shipped results/*.json.

Shipped = paper runs (gauge_fix=False, obj_scale='auto', unconditioned; eta from x^T Q x and <Q,Z>).
New     = gauge_fix='R', conditioning='centre_whiten', obj_scale=1.0; eta from ||B x||^2 and
          <Q_t, Z_t>, plus the dual-bound eta_dual.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OLD = os.path.join(HERE, '..', '..', 'results')


def load(name, where):
    p = os.path.join(where, name)
    return json.load(open(p)) if os.path.exists(p) else None


def med(x): x = [v for v in x if v is not None]; return np.median(x) if x else np.nan
def mx(x): x = [abs(v) for v in x if v is not None]; return max(x) if x else np.nan


def ablation():
    new, old = load('ablation_svr.json', HERE), load('ablation_svr.json', OLD)
    if new is None: return print('ablation: not run yet')
    print('\n=== Fig. 3 ablation: certified (log SVR > 6) out of 24 per cell, shipped -> new ===')
    sig = sorted({r['sigma'] for r in new}); st = sorted({r['stage'] for r in new})
    lab = {r['stage']: r['stage_lab'] for r in new}
    print('%-16s' % 'stage' + ''.join('%14g' % s for s in sig))
    for k in st:
        cells = []
        for s in sig:
            c = lambda rows: sum(1 for r in rows if r['stage'] == k and r['sigma'] == s and (r['log_svr'] or -1) > 6)
            cells.append('%5d -> %-5d' % (c(old), c(new)))
        print('%-16s' % lab[k][:16] + ''.join('%14s' % c for c in cells))
    full = [r for r in new if r['stage'] == max(st) and r['rank'] is not None]
    print('full catalog, new: eta med %.1e max %.1e | eta_dual max %.1e | Optimal %d/%d' % (
        med([r['eta'] for r in full]), mx([r['eta'] for r in full]), mx([r.get('eta_dual') for r in full]),
        sum('Optimal' in str(r.get('solsta')) for r in full), len(full)))
    for s in sig:
        g = [r for r in full if r['sigma'] == s]; go = [r for r in old if r['stage'] == max(st) and r['sigma'] == s and r['eta'] is not None]
        print('   sigma=%-7g eta med shipped %.1e -> new %.1e | max %.1e -> %.1e | eta_dual max %.1e' % (
            s, med([r['eta'] for r in go]), med([r['eta'] for r in g]), mx([r['eta'] for r in go]),
            mx([r['eta'] for r in g]), mx([r.get('eta_dual') for r in g])))


def table2():
    new, old = load('uniform_multibias.json', HERE), load('uniform_multibias.json', OLD)
    if new is None: return print('multibias: not run yet')
    print('\n=== Table II (uniform_multibias), dataset_cov | shipped -> new ===')
    print('%-7s %3s %4s  %-11s %-21s %-21s %-9s %-15s' % ('ds', 'K', 'kpb', 'rank1', 'eta median', 'eta max', 'eta_dual', 'rot mean (deg)'))
    keys = []
    for r in new:
        k = (r['ds'], r['nkf'], r['kpb'])
        if r['mode'] == 'dataset_cov' and k not in keys: keys.append(k)
    for k in keys:
        f = lambda rows: [r for r in rows if r['mode'] == 'dataset_cov' and (r['ds'], r['nkf'], r['kpb']) == k and r.get('rank') is not None]
        n, o = f(new), f(old)
        print('%-7s %3d %4s  %d/%d -> %d/%d  %.1e -> %-9.1e  %.1e -> %-9.1e  %-9.1e %.3f -> %.3f' % (
            k[0], k[1], k[2], sum(r['rank'] == 1 for r in o), len(o), sum(r['rank'] == 1 for r in n), len(n),
            med([r['eta'] for r in o]), med([r['eta'] for r in n]), mx([r['eta'] for r in o]), mx([r['eta'] for r in n]),
            mx([r.get('eta_dual') for r in n]), med([r['rot_mean'] for r in o]), med([r['rot_mean'] for r in n])))
    m = [r for r in new if r['mode'] == 'mle' and r.get('rank') is not None]
    mo = [r for r in old if r['mode'] == 'mle' and r.get('rank') is not None]
    print("mode 'mle' (all rows): rank1 %d/%d -> %d/%d" % (sum(r['rank'] == 1 for r in mo), len(mo), sum(r['rank'] == 1 for r in m), len(m)))


def real_and_stages():
    for name, keys in (('uniform_real.json', ('mode', 'ds')), ('uniform_stages.json', ('mode', 'scen', 'stage'))):
        new, old = load(name, HERE), load(name, OLD)
        if new is None: print('%s: not run yet' % name); continue
        print('\n=== %s%s ===' % (name, '' if old else '  (no shipped copy)'))
        seen = []
        for r in new:
            k = tuple(r[x] for x in keys)
            if k not in seen: seen.append(k)
        for k in seen:
            f = lambda rows: [r for r in (rows or []) if tuple(r[x] for x in keys) == k and r.get('rank') is not None]
            n, o = f(new), f(old)
            print('  %-34s rank1 %s -> %d/%d  eta med %s -> %.1e max %.1e  eta_dual max %.1e' % (
                ' '.join(map(str, k)), '%d/%d' % (sum(r['rank'] == 1 for r in o), len(o)) if old else '-',
                sum(r['rank'] == 1 for r in n), len(n), '%.1e' % med([r['eta'] for r in o]) if old else '-',
                med([r['eta'] for r in n]), mx([r['eta'] for r in n]), mx([r.get('eta_dual') for r in n])))


def table3():
    new, old = load('matched.json', HERE), load('matched.json', OLD)
    if new is None: return print('matched: not run yet')
    print('\n=== Table III (matched): mean rotation error deg, shipped -> new ===')
    print('%-7s %5s  %-19s %-19s %-19s %-6s %-9s %-9s %s' % ('ds', 'off', 'cayley-sdp', 'gtsam-gt', 'gtsam-gnss', 'rank', 'eta', 'eta_dual', 'sdp secs'))
    for r in new:
        if r['method'] != 'cayley-sdp': continue
        g = lambda rows, m: next((q for q in rows if q['ds'] == r['ds'] and q['off'] == r['off'] and q['method'] == m), {})
        cells = ['%7.3f -> %-7.3f' % (g(old, m).get('rot_mean', np.nan), g(new, m).get('rot_mean', np.nan))
                 for m in ('cayley-sdp', 'gtsam-gt', 'gtsam-gnss')]
        print('%-7s %5g  %s %-6s %-9.1e %-9.1e %.1f' % (r['ds'], r['off'], ' '.join(cells), r.get('rank'),
                                                    r.get('eta', np.nan), r.get('eta_dual') or np.nan, r.get('secs', np.nan)))
    new, old = load('divergence.json', HERE), load('divergence.json', OLD)
    if new:
        print('\n=== divergence (EuRoC 6 s, GNSS sigma sweep) ===')
        for s in sorted({r['sigma'] for r in new}):
            o = next((q for q in new if q['sigma'] == s and q['method'] == 'cayley-sdp'), {})
            rnd = [q for q in new if q['sigma'] == s and q['method'] == 'gtsam-random_rot']
            print('  sigma=%-6g ours rot %7.3f rank %s eta %.1e eta_dual %.1e | LM random-init missed %d/%d' % (
                s, o.get('rot_mean', np.nan), o.get('rank'), o.get('eta', np.nan), o.get('eta_dual') or np.nan,
                sum(q['missed'] for q in rnd), len(rnd)))


def teaser():
    new, old = load('traj_teaser.json', HERE), load('traj_teaser.json', OLD)
    if new is None: return print('teaser: not run yet')
    print('\n=== Fig. 1 teaser (EuRoC 90 s) shipped -> new ===')
    for k in ('ours_rot', 'ours_rank', 'lm_gt_rot', 'lm_gnss_rot'):
        print('  %-12s %10.3f -> %.3f' % (k, old[k], new[k]))


def progressive():
    new, old = load('progressive_tightening.json', HERE), load('progressive_tightening.json', OLD)
    if new is None: return print('progressive: not run yet')
    print('\n=== progressive tightening: rank / certified gap per sigma, shipped -> new ===')
    for sc in new['res']:
        for lab in new['stages']:
            n = new['res'][sc][lab]; o = old['res'][sc].get(lab) if old else None
            print('  %-5s %-11s rank %s -> %s | gap %s -> %s' % (sc, lab, o and o['rank'], n['rank'],
                  o and ' '.join('%.0e' % g for g in o['gap']), ' '.join('%.0e' % g for g in n['gap'])))


if __name__ == '__main__':
    for f in (ablation, real_and_stages, table2, table3, teaser, progressive):
        f()
