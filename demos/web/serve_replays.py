"""Watch the two 3D replays interactively in your browser, from the shipped JSON (no MOSEK, no datasets).

Builds the Viser recordings if they are missing (make_replay.py, make_ball.py), copies Viser's
web viewer next to them, serves the folder locally and prints one link per replay:

    "Same path, wrong drone"   the MH_04 window: a trapped LM attitude next to the certified one
    "Basin of attraction"      2,000 LM runs descending through the ball of attitude errors

    uv venv .venv-web && uv pip install -p .venv-web viser==0.2.23 trimesh
    .venv-web/bin/python demos/web/serve_replays.py [--port 8000]
"""
import argparse, functools, http.server, shutil
from pathlib import Path
import viser

HERE = Path(__file__).resolve().parent
BUILD = HERE / 'build'
CAMERA = {
    'drone_replay.viser': 'initialCameraPosition=0,-9.6,5.6&initialCameraLookAt=0,0,0.6&initialCameraUp=0,0,1',
    'basin_ball.viser': 'initialCameraPosition=4.3,-5.2,2.6&initialCameraLookAt=0,0,0&initialCameraUp=0,0,1',
}
TITLE = {'drone_replay.viser': 'Same path, wrong drone', 'basin_ball.viser': 'Basin of attraction'}


def main(port):
    BUILD.mkdir(exist_ok=True)
    if not (BUILD / 'drone_replay.viser').exists():
        import make_replay; make_replay.main(BUILD)
    if not (BUILD / 'basin_ball.viser').exists():
        import make_ball; make_ball.main(BUILD)
    viewer = BUILD / 'viewer'
    if not (viewer / 'index.html').exists():
        shutil.copytree(Path(viser.__file__).parent / 'client' / 'build', viewer, dirs_exist_ok=True)
    for f, cam in CAMERA.items():
        print('%-24s http://localhost:%d/viewer/index.html?playbackPath=../%s&%s' % (TITLE[f], port, f, cam), flush=True)
    print('serving %s (Ctrl-C to stop)' % BUILD, flush=True)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(BUILD))
    http.server.ThreadingHTTPServer(('localhost', port), handler).serve_forever()


if __name__ == '__main__':
    import sys; sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    main(ap.parse_args().port)
