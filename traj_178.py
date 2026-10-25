"""Dump the EuRoC t=90s window trajectories for the 178-degree divergence figure.

This is the one window in `matched.json` where the local solver, started from the GNSS
measurements rather than from ground truth, converges to a gravity-flipped minimum. We
re-solve it and save the ground-truth and estimated position tracks so the failure can be
plotted against the trajectories that succeed.
"""
import json
import numpy as np
import compare_gtsam_cayley as X

DS, T, OFF = 'euroc', 16.0, 90.0

# The paper's Table III row uses a 1 s keyframe interval (K=17, M=2). For the teaser we sample
# the same window at 0.5 s so the triads are dense enough to read; the bias partition is still
# M=2 and the failure is unchanged (LM/GNSS 178.11 deg, ours 0.67 deg at rank one).
KFI, KFPB = 0.5, 16


def main(out='traj_teaser.json'):
    W = X.load_window(DS, T, KFI, OFF, KFPB)
    rec = {'p_true': np.asarray(W['p_t']).tolist()}

    # ground truth at its native 200 Hz over the same window, recentred the same way, so the
    # teaser can draw a smooth path instead of a 17-point polyline
    gt = np.loadtxt('%s/state_groundtruth_estimate0/data.csv' % X.EUROC, delimiter=',',
                    skiprows=1)
    gt_t = gt[:, 0].astype(np.int64)
    t0 = gt_t[0] + int(OFF * 1e9)
    m = (gt_t >= t0) & (gt_t <= t0 + int(T * 1e9))
    rec['p_dense'] = (gt[m, 1:4] - np.asarray(W['p_t'][0]) - (gt[np.argmin(np.abs(gt_t - t0)), 1:4]
                      - np.asarray(W['p_t'][0]))).tolist()

    rec['R_true'] = np.asarray(W['R_t']).tolist()

    r = X.solve_ours(DS, T, KFI, OFF, KFPB, X.PRIOR_SIGMAS)
    rec['ours'] = np.asarray(r['p_est']).tolist()
    rec['R_ours'] = np.asarray(r['R_est']).tolist()
    rec['ours_err'] = X.rot_errors(r['R_est'], W['R_t']).tolist()
    rec['ours_rot'] = float(np.mean(rec['ours_err']))
    rec['ours_rank'] = int(r['rank'])

    for init in ('gt', 'gnss'):
        g = X.run_gtsam(W, X.PRIOR_SIGMAS, init)
        rec['lm_' + init] = np.asarray(g['p_est']).tolist()
        rec['R_lm_' + init] = np.asarray(g['R_est']).tolist()
        rec['lm_%s_err' % init] = X.rot_errors(g['R_est'], W['R_t']).tolist()
        rec['lm_%s_rot' % init] = float(np.mean(rec['lm_%s_err' % init]))

    json.dump(rec, open(out, 'w'))
    print('ours      rot %.3f deg (rank %d)' % (rec['ours_rot'], rec['ours_rank']))
    print('LM truth  rot %.3f deg' % rec['lm_gt_rot'])
    print('LM GNSS   rot %.3f deg' % rec['lm_gnss_rot'])
    print('wrote', out)


if __name__ == '__main__':
    main()
