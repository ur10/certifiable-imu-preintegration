"""demos/out/urbannav_map.json -> compact JSON for the Leaflet map on the project page.

    python demos/web/make_map_data.py --out <project>/media/urbannav.json
"""
import argparse, json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import certified  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'out' / 'urbannav_map.json'
WEAK = -4.0          # same heading-excitation threshold as the mission video
LABEL = {'medium': 'Medium urban (Tsim Sha Tsui), 786 s', 'deep': 'Deep urban (Whampoa), 1,538 s'}


def main(out):
    rows = [r for r in json.load(open(SRC)) if 'failed' not in r]
    for r in rows:
        r['certified'] = certified(r)
    res = {}
    for seq in ('medium', 'deep'):
        g = sorted([r for r in rows if r['seq'] == seq], key=lambda r: r['t0'])
        W = []
        for r in g:
            W.append(dict(t0=int(r['t0']), c=int(r['certified']), rank=r['rank'], gap=float(r['eta_dual']),
                          exc=float(r['excitation']), weak=int(r['excitation'] < WEAK),
                          rot=float(r['rot_err']), hd=float(r['heading_err']), v=float(r['mean_speed']),
                          lmI=int(r['lm']['gnss']['global_']), lmIerr=float(r['lm']['gnss']['rot_err']),
                          lmG=int(r['lm']['gravity']['global_']),
                          pts=[[round(a, 7), round(b, 7), round(h1, 1), round(h2, 1)]
                               for a, b, h1, h2 in zip(r['lat'], r['lon'], r['hdg_true'], r['hdg_sdp'])]))
        n = len(W); c = sum(w['c'] for w in W)
        weak = [w for w in W if w['c'] and w['weak']]
        strong = [w for w in W if w['c'] and not w['weak']]
        lm_bad = sum(not w['lmI'] for w in W)
        med = np.median([w['hd'] for w in strong]) if strong else float('nan')
        medw = np.median([w['hd'] for w in weak]) if weak else float('nan')
        unc = [w for w in W if not w['c']]
        stops = sum(w['v'] < 0.5 for w in unc)
        hdc = np.array([w['hd'] for w in W if w['c']])
        summary = ('<b>%d / %d</b> windows certified, heading error median <b>%.2f&deg;</b> (worst %.1f&deg;). '
                   '%s of the %d uncertified windows the car is stopped. LM from the identity attitude is trapped on '
                   '%d window%s.' % (c, n, np.median(hdc), hdc.max(),
                                     'In all' if stops == len(unc) else 'In %d' % stops, len(unc), lm_bad,
                                     '' if lm_bad == 1 else 's'))
        res[seq] = dict(label=LABEL[seq], windows=W, summary=summary)
        print(seq, '%d windows, %d certified, %d weak, LM-identity trapped %d, LM-gravity trapped %d, '
              'heading err median strong %.2f weak %.2f, max weak %.1f'
              % (n, c, len(weak), lm_bad, sum(not w['lmG'] for w in W), med, medw,
                 max([w['hd'] for w in weak] or [np.nan])))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, 'w'), separators=(',', ':'))
    print('wrote', out, '%.0f kB' % (Path(out).stat().st_size / 1e3))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE / 'build' / 'urbannav.json'))
    main(ap.parse_args().out)
