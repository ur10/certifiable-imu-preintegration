"""Screenshot a page in headless Chrome after a real-time wait (for poster stills of Viser replays).

    python demos/web/snap.py URL out.png --wait 8 --size 1400x800

Uses the DevTools protocol directly (only needs `websockets`, which viser already depends on), so
animations actually advance, unlike --virtual-time-budget.
"""
import argparse, asyncio, base64, json, subprocess, tempfile, time, urllib.request

CHROME = '/opt/google/chrome/chrome'


async def shoot(url, out, wait, w, h, port=9333):
    import websockets
    prof = tempfile.mkdtemp()
    proc = subprocess.Popen([CHROME, '--headless=new', '--no-sandbox', '--use-angle=swiftshader',
                             '--enable-unsafe-swiftshader', '--hide-scrollbars', '--disable-background-timer-throttling',
                             '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows', '--remote-debugging-port=%d' % port,
                             '--user-data-dir=' + prof, '--window-size=%d,%d' % (w, h), 'about:blank'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen('http://127.0.0.1:%d/json' % port))
                break
            except Exception:
                time.sleep(0.2)
        ws_url = [t for t in tabs if t['type'] == 'page'][0]['webSocketDebuggerUrl']
        async with websockets.connect(ws_url, max_size=2 ** 28) as ws:
            n = [0]

            async def call(method, **params):
                n[0] += 1
                await ws.send(json.dumps(dict(id=n[0], method=method, params=params)))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get('id') == n[0]:
                        return msg.get('result', {})
            await call('Emulation.setDeviceMetricsOverride', width=w, height=h, deviceScaleFactor=1,
                       mobile=False)
            await call('Page.navigate', url=url)
            await asyncio.sleep(wait)
            r = await call('Page.captureScreenshot', format='png')
            open(out, 'wb').write(base64.b64decode(r['data']))
    finally:
        proc.kill()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('url'); ap.add_argument('out')
    ap.add_argument('--wait', type=float, default=6.0)
    ap.add_argument('--size', default='1400x800')
    a = ap.parse_args()
    w, h = map(int, a.size.split('x'))
    asyncio.run(shoot(a.url, a.out, a.wait, w, h))
    print('wrote', a.out)
