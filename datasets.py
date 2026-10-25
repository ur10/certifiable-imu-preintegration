"""Where the datasets live.

EuRoC and UrbanNav are large and licensed separately, so they are not vendored here. Point the
two environment variables below at your own copies, or edit the fallbacks:

    export EUROC_MH01=/path/to/MH_01_easy/mav0
    export URBANNAV_ROOT=/path/to/urbannav

`EUROC_MH01` must be the `mav0` directory itself -- the one containing `imu0/data.csv` and
`state_groundtruth_estimate0/data.csv`.

`URBANNAV_ROOT` must be the directory holding the UrbanNav sequences; see `urbannav_data.py`
for the layout it expects. Only the medium-urban and deep-urban sequences are used.

Nothing in this module touches the disk at import time, so the synthetic experiments
(`cayley_progressive_tightening.py`) run without either dataset present.
"""
import os

_DEFAULT_EUROC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'data', 'MH_01_easy', 'mav0')
_DEFAULT_URBANNAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'urbannav')

EUROC = os.environ.get('EUROC_MH01', _DEFAULT_EUROC)
URBANNAV = os.environ.get('URBANNAV_ROOT', _DEFAULT_URBANNAV)


def require(path, what, env_var, example):
    """Fail with something actionable rather than a stack trace from numpy.loadtxt."""
    if not os.path.exists(path):
        raise SystemExit(
            '%s not found at:\n    %s\n\n'
            'Set %s to your copy:\n    export %s=%s\n\n'
            'See the README section "Datasets".' % (what, path, env_var, env_var, example))
    return path


def euroc():
    return require(EUROC, 'EuRoC MH_01 (mav0)', 'EUROC_MH01',
                   '/path/to/MH_01_easy/mav0')


def urbannav():
    return require(URBANNAV, 'UrbanNav root', 'URBANNAV_ROOT',
                   '/path/to/urbannav')
