#!/usr/bin/env python3
"""Drive the real console in headless Chrome (DevTools protocol): new call, chat turn, guided scenario, run-all, screenshot.

Usage: python scripts/ui_smoke.py [--shot path.png]     Needs Google Chrome installed.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import websockets  # noqa: E402

from app.config import ROOT  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT, DBG = 7861, 9333


class Cdp:
    def __init__(self, ws):
        self.ws, self.n, self.errors = ws, 0, []
        self.pending: dict[int, asyncio.Future] = {}

    async def pump(self):
        async for m in self.ws:
            d = json.loads(m)
            if "id" in d and d["id"] in self.pending:
                self.pending[d["id"]].set_result(d)
            elif d.get("method") == "Runtime.exceptionThrown":
                self.errors.append(d["params"]["exceptionDetails"].get("text", "") + str(d["params"]["exceptionDetails"].get("exception", {}).get("description", ""))[:200])
            elif d.get("method") == "Log.entryAdded" and d["params"]["entry"]["level"] == "error":
                self.errors.append(d["params"]["entry"]["text"][:200])

    async def call(self, method, **params):
        self.n += 1
        fut = asyncio.get_event_loop().create_future()
        self.pending[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        return (await asyncio.wait_for(fut, 30)).get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")


async def wait_for(cdp, expr, secs=20):
    for _ in range(int(secs * 4)):
        v = await cdp.js(expr)
        if v:
            return v
        await asyncio.sleep(0.25)
    return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default=str(ROOT / "logs" / "ui_smoke.png"))
    a = ap.parse_args()
    env = dict(os.environ, UI_PORT=str(PORT), LOG_DIR=str(ROOT / "logs" / "ui"), PYTHONWARNINGS="ignore")
    srv = subprocess.Popen([sys.executable, "-m", "app.server"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    prof = tempfile.mkdtemp()
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={DBG}", f"--user-data-dir={prof}",
                               "--window-size=1500,950", "--no-first-run", "--disable-gpu", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results: list[tuple[str, bool]] = []
    try:
        for _ in range(120):
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/api/config", timeout=1).status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        for _ in range(60):
            try:
                tabs = httpx.get(f"http://127.0.0.1:{DBG}/json", timeout=1).json()
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            cdp = Cdp(ws)
            pump = asyncio.ensure_future(cdp.pump())
            await cdp.call("Runtime.enable")
            await cdp.call("Log.enable")
            await cdp.call("Page.enable")
            await cdp.call("Page.navigate", url=f"http://127.0.0.1:{PORT}/")
            ok = bool(await wait_for(cdp, "document.querySelectorAll('#scList li').length >= 40"))
            results.append((f"scenario list rendered ({await cdp.js(chr(100)+'ocument.querySelectorAll(\"#scList li\").length')} items)", ok))
            cfg = await cdp.js("document.getElementById('cfg').textContent")
            results.append((f"config line shown: {cfg[:90]!r}", bool(cfg and "router" in cfg)))

            await cdp.js("document.getElementById('newCall').click()")
            ok = bool(await wait_for(cdp, "document.querySelectorAll('#log .msg.bot').length >= 1"))
            results.append(("New call shows the greeting", ok))

            await cdp.js("""(()=>{const m=document.getElementById('msg');m.value='cbc ka test ghar par karwana hai';
                document.getElementById('form').dispatchEvent(new Event('submit',{cancelable:true,bubbles:true}));})()""")
            ok = bool(await wait_for(cdp, "document.querySelector('#log').textContent.includes('Mareez ka naam')"))
            results.append(("typed message gets a bot reply", ok))
            flow = await cdp.js("document.getElementById('flow').textContent")
            slots = await cdp.js("document.getElementById('slots').textContent")
            results.append((f"flow/slots panels update ({flow[:40]!r} / {slots[:40]!r})", "home_sampling" in flow and "Complete Blood Count" in slots))
            router = await cdp.js("document.getElementById('router').textContent")
            results.append((f"router panel shows the decision ({router[:50]!r})", "switch" in router))

            await cdp.js("document.querySelector('button[data-g=\"hs_one_sentence\"]').click()")
            await wait_for(cdp, "!document.getElementById('guided').classList.contains('hidden')")
            await cdp.js("document.getElementById('guidedSend').click()")
            ok = bool(await wait_for(cdp, "document.getElementById('guidedChecks').textContent.includes('matched')"))
            results.append(("guided scenario: scripted line checked against expectations", ok))
            ok = bool(await wait_for(cdp, "document.querySelector('#log').textContent.includes('Scenario passed')"))
            results.append(("guided scenario completes with a pass", ok))

            await cdp.js("document.getElementById('tagSel').value='smoke'; document.getElementById('tagSel').dispatchEvent(new Event('change'))")
            await cdp.js("document.getElementById('runTag').click()")
            ok = bool(await wait_for(cdp, "/passed/.test(document.getElementById('runSummary').textContent)", 60))
            summ = await cdp.js("document.getElementById('runSummary').textContent")
            results.append((f"Run all (smoke): {summ}", bool(ok and summ.split('/')[0].strip().isdigit() and summ.split()[0].split('/')[0] == summ.split()[0].split('/')[1])))

            await cdp.js("document.getElementById('modeVoice').click()")
            vb = await cdp.js("document.getElementById('modeVoice').classList.contains('on')")
            results.append(("Voice mode toggle works", bool(vb)))
            await cdp.js("document.getElementById('modeChat').click()")

            shot = await cdp.call("Page.captureScreenshot", format="png")
            Path(a.shot).parent.mkdir(parents=True, exist_ok=True)
            Path(a.shot).write_bytes(base64.b64decode(shot["data"]))
            results.append((f"no JavaScript errors in the console ({len(cdp.errors)})", not cdp.errors))
            for e in cdp.errors[:5]:
                print("   JS error:", e)
            pump.cancel()
    finally:
        chrome.terminate()
        srv.terminate()
    bad = 0
    for name, ok in results:
        print(("PASS " if ok else "FAIL ") + name)
        bad += not ok
    print("screenshot:", a.shot)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
