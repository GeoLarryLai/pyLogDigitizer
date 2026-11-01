"""Record the README demo clip of assisted tracing in the real app (recording tool only, not a package dependency).

Needs Playwright (+ Chromium) and ffmpeg in the environment that runs this script:
    pip install playwright && playwright install chromium
    python scripts/record_readme_demo.py --example 0 --out image/assisted_tracing_demo

Starts ``pylogdigitizer-app`` on a free port, opens one shipped example scan, accepts the automatic
steps up to "Fix the trace", then plays a fixed script: auto trace -> pin on the real curve -> drag
the pin -> cursor-circle re-trace -> before/after overlay. Writes <out>.webp and <out>.mp4.
"""
from __future__ import annotations

import argparse, shutil, socket, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

W, H = 1000, 640


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--example", type=int, default=0, help="index in the app's example catalog")
    ap.add_argument("--project", default=".", help="project root with example_data/")
    ap.add_argument("--out", default="image/assisted_tracing_demo")
    ap.add_argument("--window-ft", type=float, default=250, help="depth window shown")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    port = free_port(); url = f"http://127.0.0.1:{port}"
    srv = subprocess.Popen([sys.executable, "-c", "import sys; from pylogdigitizer.app import main; sys.exit(main(sys.argv[1:]))", a.project, "--port", str(port), "--no-browser"])
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(url + "/app/api/info", timeout=1); break
            except Exception:
                time.sleep(0.5)
        vid = Path(tempfile.mkdtemp())
        with sync_playwright() as pw:
            br = pw.chromium.launch()
            ctx = br.new_context(viewport=dict(width=W, height=H), record_video_dir=str(vid), record_video_size=dict(width=W, height=H))
            pg = ctx.new_page(); pg.goto(url + "/")
            pg.add_style_tag(content="#demoCap{position:fixed;left:50%;bottom:14px;transform:translateX(-50%);z-index:99;background:rgba(17,24,39,.85);"
                                     "color:#fff;font:600 17px -apple-system,sans-serif;padding:6px 14px;border-radius:8px}"
                                     "#demoCur{position:fixed;z-index:98;width:14px;height:14px;margin:-7px;border:2px solid #111;border-radius:50%;pointer-events:none}")
            pg.evaluate("document.body.insertAdjacentHTML('beforeend','<div id=demoCap></div><div id=demoCur></div>')")
            cap = lambda t, s=1.6: (pg.evaluate("t => document.getElementById('demoCap').textContent = t", t), pg.wait_for_timeout(int(s * 1000)))

            def move(x, y, steps=15):
                pg.mouse.move(x, y, steps=steps)
                pg.evaluate("([x, y]) => { const c = document.getElementById('demoCur'); c.style.left = x + 'px'; c.style.top = y + 'px'; }", [x, y])

            def el(row, x):                    # image (sheet) coords -> page coords
                return pg.evaluate("""([r, x]) => { const p = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(x, r));
                    const b = document.getElementById('viewer').getBoundingClientRect(); return [b.left + p.x, b.top + p.y]; }""", [row, x])

            pg.wait_for_function("INFO !== null")
            pg.select_option("#cat", str(a.example)); pg.click("#mainBtn")
            pg.wait_for_function("SID !== null && STEP !== 'load'", timeout=600_000)
            for _ in range(12):                # accept the automatic steps
                if pg.evaluate("STEP") == "trace":
                    break
                pg.wait_for_function("!document.getElementById('mainBtn').disabled", timeout=600_000)
                pg.click("#mainBtn"); pg.wait_for_timeout(800)
            spots = pg.evaluate("api('GET', A('/checks'))")["spots"]
            sec = pg.evaluate("OV.section ? OV.section.rows : [0, ST.height]"); r0, r1 = sec[0] + 0.15 * (sec[1] - sec[0]), sec[1] - 0.1 * (sec[1] - sec[0])
            cand = [s for s in spots if s["kind"] == "crossover" and r0 < s["row"] < r1] or [s for s in spots if r0 < s["row"] < r1] or spots
            sp = max(cand, key=lambda s: s["score"]) if cand else None
            if sp is None:
                sys.exit("no tricky spot on this example; try another --example")
            alt = next(c for c in sp["choices"] if not c.get("current"))
            ppf = pg.evaluate("ST.px_per_ft || 10") or 10
            pg.evaluate("""([r, h]) => { const xs = OV.trace.filter(t => Math.abs(t[0] - r) < h / 2).map(t => t[1]);
                const x0 = Math.min(...xs) - 150, x1 = Math.max(...xs) + 150, w = Math.max(x1 - x0, 0.9 * h);
                viewer.viewport.fitBounds(viewer.viewport.imageToViewportRectangle((x0 + x1 - w) / 2, r - h / 2, w, h), true); }""",
                        [sp["row"], a.window_ft * ppf])
            pg.evaluate("""() => { window.DEMO_BEFORE = OV.trace.map(t => [t[0], t[1]]); window.DEMO_SHOW = false; const d0 = draw;
                draw = function () { d0(); if (!DEMO_SHOW) return; const c = document.getElementById('ov').getContext('2d'); c.strokeStyle = 'rgba(37,99,235,.9)';
                  c.lineWidth = 2; c.setLineDash([6, 4]); c.beginPath(); DEMO_BEFORE.forEach(([r, x], i) => { const p = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(x, r));
                  i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y); }); c.stroke(); c.setLineDash([]); }; }""")
            pg.wait_for_timeout(1500); cap("1. Automatic trace — it jumps to the wrong curve here", 2.5)
            pg.evaluate("setMode('fix')"); x, y = el(sp["row"], alt["x"]); move(x, y)
            cap("2. Click the real curve: a pin, the trace re-routes through it", 0.6); pg.mouse.click(x, y); pg.wait_for_timeout(2200)
            r2 = sp["row"] + 0.25 * (sp["row1"] - sp["row0"] + 40); x2, y2 = el(r2, alt["x"])
            cap("3. Drag the pin: re-traces around it", 0.6); move(x, y); pg.mouse.down(); move(x2, y2, 25); pg.mouse.up(); pg.wait_for_timeout(2000)
            pg.evaluate("setMode('circle')"); rc = sp["row"] - 0.3 * (sp["row1"] - sp["row0"] + 40); xc, yc = el(rc, alt["x"])
            cap("4. Cursor circle: re-trace only inside it", 0.6); move(xc, yc, 25); pg.wait_for_timeout(700)
            pg.keyboard.down("Shift"); pg.mouse.click(xc, yc); pg.keyboard.up("Shift"); pg.wait_for_timeout(2200)
            pg.evaluate("() => { DEMO_SHOW = true; draw(); }"); cap("5. Before (blue dashed) vs after (red)", 3.0)
            ctx.close(); br.close()
        raw = next(vid.glob("*.webm")); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        ff = shutil.which("ffmpeg") or sys.exit("ffmpeg not found")
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(raw), "-vf", "fps=12,scale=900:-1", "-loop", "0", "-q:v", "60", str(out.with_suffix(".webp"))], check=True)
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(raw), "-vf", "scale=900:-2", "-c:v", "libx264", "-crf", "30", "-pix_fmt", "yuv420p", "-an",
                        str(out.with_suffix(".mp4"))], check=True)
        for f in (out.with_suffix(".webp"), out.with_suffix(".mp4")):
            print(f, f"{f.stat().st_size / 1e6:.2f} MB")
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
