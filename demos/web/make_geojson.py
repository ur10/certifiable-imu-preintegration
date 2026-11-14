"""demos/out/urbannav_map.json -> one GeoJSON per UrbanNav drive, which GitHub renders as an interactive map.

Each certified 16 s window is a line segment, coloured by its heading error (light to dark blue,
0 to 2 deg and above); a thinner, translucent segment marks a weakly observable heading. Windows
that are not certified (all of them the car at a stop) are grey points. Click any feature for
its numbers.

    python demos/web/make_geojson.py      # -> demos/figures/urbannav_medium.geojson, urbannav_deep.geojson
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'out', 'urbannav_map.json')
FIG = os.path.join(HERE, '..', 'figures')
WEAK = -4.0
RAMP = ['#bcd6f5', '#7fb0eb', '#2a78d6', '#1d5aa8', '#123f7a']      # 0, 0.5, 1, 1.5, >=2 deg
NAME = dict(medium='Medium urban, Tsim Sha Tsui', deep='Deep urban, Whampoa')


def certified(w):
    return w.get('rank') == 1 and w.get('eta_dual') is not None and abs(w['eta_dual']) <= 1e-6


def colour(err):
    return RAMP[min(int(err / 0.5), len(RAMP) - 1)]


def main():
    D = json.load(open(SRC))
    for seq in ('medium', 'deep'):
        feats = []
        for w in (x for x in D if x['seq'] == seq):
            t = '%.0f–%.0f s' % (w['t0'], w['t0'] + 16)
            common = {'window': t, 'speed (m/s)': round(w['mean_speed'], 1),
                      'heading excitation (log10 σ2/σ1)': round(w['excitation'], 1)}
            if certified(w):
                weak = w['excitation'] < WEAK
                props = dict(title='%s · certified' % t,
                             description='Certified globally optimal (dual gap %.0e). Attitude error %.2f°, heading %.2f°.%s'
                             % (abs(w['eta_dual']), w['rot_err'], w['heading_err'],
                                ' Heading weakly observable.' if weak else ''),
                             **common, **{'attitude error (deg)': round(w['rot_err'], 2),
                                          'heading error (deg)': round(w['heading_err'], 2),
                                          'dual gap': float('%.1e' % abs(w['eta_dual'])),
                                          'heading weakly observable': weak},
                             stroke=colour(w['heading_err']), **{'stroke-width': 3 if weak else 6,
                                                                 'stroke-opacity': 0.6 if weak else 0.95})
                geom = dict(type='LineString', coordinates=[[round(lo, 7), round(la, 7)]
                                                            for la, lo in zip(w['lat'], w['lon'])])
            else:
                props = dict(title='%s · not certified (car stopped)' % t,
                             description='Not certified (rank %d). The car is at a stop, so heading is undefined.' % w['rank'],
                             **common, **{'marker-color': '#9a9994', 'marker-size': 'small', 'marker-symbol': 'car'})
                geom = dict(type='Point', coordinates=[round(w['lon'][0], 7), round(w['lat'][0], 7)])
            feats.append(dict(type='Feature', properties=props, geometry=geom))
        out = os.path.join(FIG, 'urbannav_%s.geojson' % seq)
        json.dump(dict(type='FeatureCollection', name=NAME[seq], features=feats), open(out, 'w'),
                  ensure_ascii=False, separators=(',', ':'))
        print('wrote %s: %d features (%.0f kB)' % (out, len(feats), os.path.getsize(out) / 1e3))


if __name__ == '__main__':
    main()
