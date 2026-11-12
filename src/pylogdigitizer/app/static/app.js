// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
// the scan is always world item 0; the highlight overlay is item 1 with the same size
const BASE = () => viewer.world.getItemAt(0);
// pyLogDigitizer app: five tabs (Open scan · Tracks and depth · Curves · Trace and fix · Check and save).
// Viewer coordinates = sheet pixels. Every curve has its own working image and session on the server;
// the selected curve's session answers the editor API (/core/api/sessions/..). Long tasks are app jobs
// (/app/api/..) with n / N progress. Files are written only in Check and save.
"use strict";
const {$, api, runJob, toast, help} = Shell;
const TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];
const STEPS = [{id: "open", title: "Open scan"}, {id: "tracks", title: "Tracks and depth"}, {id: "curves", title: "Curves"},
  {id: "trace", title: "Trace and fix"}, {id: "save", title: "Check and save"}];
let CHK = null, CUR = null, CPOS = 0, STROKE = [], DRAG = null, PROG = null, NODE = null, HOVER = null, HW = [];
let INFO = null, SID = null, ST = null, LAY = null, OV = null, HDR = null, viewer = null, mode = "pan", pts = [], STEP = "open";
let LBL = [], LBLsel = new Set(), CURV = null, TR = null, GRID = null, HLITEM = null;
let CVIEW = "curves", HLN = null, LOGS = null, LOGSEL = null, SHOWIMG = true, LASSTEP = "0.5";   // Curves view, header lines, Check and save plot, typed LAS step (kept across re-renders)
const GREY = () => ["tracks", "curves", "trace"].includes(STEP);              // working tabs show the scan in greyscale
const LAYERS = {scan: true, hl: false, grid: true, traces: true};
const fmt = (v, d = 1) => v == null || !isFinite(v) ? "–" : (+v).toFixed(d);
const E = p => `/core/api/sessions/${SID}${p}`, A = p => `/app/api/${SID}${p}`;
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const off = f => f && !f.ok ? `disabled title="${esc(f.hint)}"` : "";
const hint = f => f && !f.ok ? `<div class="hint">${esc(f.hint)}</div>` : "";
const KEYS = (...r) => `<div class="controls">${r.map(x => `<div>${x}</div>`).join("")}</div>`;
const NAV = "<kbd>drag</kbd> pan · <kbd>wheel</kbd> zoom · <kbd>double-click</kbd> zoom in · <kbd>Esc</kbd> back to pan";
const active = () => (CURV?.curves || []).find(c => c.id === CURV.active);
const ocrLine = () => INFO.features.ocr.ok ? `<div class="hint">OCR: RapidOCR</div>` : `<div class="hint">OCR is not installed - ${esc(INFO.features.ocr.hint || "")}. Type the values by hand.</div>`;

// ------------------------------------------------------------------ state + viewer + layers
async function refresh(over = true) {
  ST = await api("GET", A("/state")); $("undoBtn").disabled = !ST.undo.length;
  $("undoBtn").title = ST.undo.length ? `Undo: ${ST.undo[ST.undo.length - 1]} (Z)` : "nothing to undo";
  $("doc").textContent = `${ST.well} · ${ST.scan_name || ST.scan.split("/").pop()} · ${ST.n_curves} curve(s)` + (ST.depth_top_base ? ` · ${fmt(ST.depth_top_base[0], 0)}–${fmt(ST.depth_top_base[1], 0)} ft` : "");
  if (over) { OV = await api("GET", E("/overlay")); TR = await api("GET", A("/traces")); draw(); }
}
async function loadCurves() { CURV = await api("GET", A("/curves")); renderCurvePanel(); }
function openViewer() {
  if (viewer) viewer.destroy(); $("empty").hidden = true; HLITEM = null;
  viewer = OpenSeadragon({id: "viewer", tileSources: E("/image.dzi"), showNavigationControl: false, prefixUrl: "",
    gestureSettingsMouse: {clickToZoom: false, dblClickToZoom: true}, maxZoomPixelRatio: 4, minZoomImageRatio: 0.3, visibilityRatio: 0.2});
  viewer.addHandler("canvas-click", onClick); viewer.addHandler("canvas-press", onPress); viewer.addHandler("canvas-drag", onDrag); viewer.addHandler("canvas-release", onRelease);
  new OpenSeadragon.MouseTracker({element: viewer.canvas, moveHandler: e => { if (mode !== "circle" && mode !== "node") return; const ip = BASE().viewerElementToImageCoordinates(e.position); HOVER = {row: ip.y, x: ip.x}; drawSoon(); },
    scrollHandler: e => { if ((mode !== "circle" && mode !== "node") || !e.originalEvent.shiftKey) return; const el = $("circR"); if (!el) return; el.value = Math.max(10, Math.min(600, +el.value * (e.scroll > 0 ? 1.15 : 1 / 1.15))) | 0; e.preventDefault = true; drawSoon(); }});
  for (const ev of ["update-viewport", "animation", "animation-finish", "resize", "tile-drawn"]) viewer.addHandler(ev, drawSoon);
  $("layers").hidden = false;
  $("logPane").onwheel = ev => { ev.preventDefault(); const b = viewer.viewport.getBounds(); viewer.viewport.panBy(new OpenSeadragon.Point(0, ev.deltaY / $("logPane").clientHeight * b.height)); };
  return new Promise(r => viewer.addOnceHandler("open", () => { applyLayers(); loadHighlight(); focusStep(); r(); }));
}
function loadHighlight() {                 // highlight layer: the selected curve's own ink, as a tiled overlay
  if (!viewer || !CURV) return;
  if (HLITEM) { viewer.world.removeItem(HLITEM); HLITEM = null; }
  viewer.addTiledImage({tileSource: A(`/highlight/${CURV.active}.dzi`), opacity: LAYERS.hl ? 1 : 0, success: ev => { HLITEM = ev.item; applyLayers(); }});
}
function applyLayers() {   // every layer state looks different: scan full, washed under the grid, dimmed under the highlight
  for (const [k, id] of [["scan", "lyScan"], ["hl", "lyHl"], ["grid", "lyGrid"], ["traces", "lyTr"]]) if ($(id)) $(id).checked = LAYERS[k];
  const base = viewer?.world.getItemAt(0), save = STEP === "save";
  if (base) base.setOpacity(save ? (SHOWIMG ? 1 : 0) : !LAYERS.scan ? 0 : LAYERS.hl ? 0.3 : LAYERS.grid ? 0.8 : 1);
  if (HLITEM) HLITEM.setOpacity(LAYERS.hl && !save ? 1 : 0);
  const m = $("main"); for (const k of Object.keys(LAYERS)) m.classList.toggle("ly-" + k, !!LAYERS[k]);
  const parts = [LAYERS.scan && (LAYERS.hl ? "scan dimmed" : GREY() ? "scan in greyscale" : "scan in its own colours"),
    LAYERS.hl && "selected curve ink (dark)", LAYERS.grid && "tracks and grid (teal)", LAYERS.traces && "traced curves (colour)"].filter(Boolean);
  $("lyState").textContent = parts.length ? parts.join(" + ") : "nothing shown";
  draw();
}
for (const [k, id] of [["scan", "lyScan"], ["hl", "lyHl"], ["grid", "lyGrid"], ["traces", "lyTr"]]) $(id).onchange = () => { LAYERS[k] = $(id).checked; applyLayers(); };
function fitRows(r0, r1, x0 = 0, x1 = ST.width) {
  if (viewer && BASE()) viewer.viewport.fitBounds(BASE().imageToViewportRectangle(x0, r0, x1 - x0, Math.max(50, r1 - r0)), true);
}
function focusStep() {
  if (!viewer || !ST) return;
  const sec = ST.section, w = ST.width;
  if (STEP === "open") viewer.viewport.goHome(true);
  else if (STEP === "tracks" && LAY?.labels?.length) { const r = LAY.labels[0][0]; fitRows(Math.max(0, r - 0.9 * w), r + 0.9 * w); }
  else if (STEP === "curves" && CVIEW === "header" && HLN?.strip?.[1]) fitRows(0, HLN.strip[1] * 2.7);
  else if (STEP === "curves" && sec) fitRows(Math.max(0, sec.rows[0] - 0.25 * w), sec.rows[0] + 0.9 * w);
  else if (STEP === "save" && sec) fitRows(sec.rows[0] - 0.12 * w, sec.rows[0] + 1.6 * w);
  else if (sec) fitRows(sec.rows[0], sec.rows[0] + 1.3 * w);
  else viewer.viewport.goHome(true);
}
let _drawReq = 0;
function drawSoon() { if (!_drawReq) _drawReq = requestAnimationFrame(() => { _drawReq = 0; draw(); }); }
function poly(c, P, pts, vis) { c.beginPath(); let f = true; for (const [r, x] of pts) { if (!vis(r)) { f = true; continue; } const p = P(x, r); f ? c.moveTo(p.x, p.y) : c.lineTo(p.x, p.y); f = false; } c.stroke(); }
function draw() {
  const cv = $("ov"), c = cv.getContext("2d"); cv.width = cv.clientWidth; cv.height = cv.clientHeight; c.clearRect(0, 0, cv.width, cv.height);
  if (!viewer || !viewer.world.getItemAt(0) || !ST) return;
  const P = (x, y) => BASE().imageToViewerElementCoordinates(new OpenSeadragon.Point(x, y));
  const top = BASE().viewerElementToImageCoordinates(new OpenSeadragon.Point(0, 0)).y;
  const bot = BASE().viewerElementToImageCoordinates(new OpenSeadragon.Point(0, cv.height)).y;
  const vis = r => r > top - 200 && r < bot + 200;
  if (STEP === "open" && (ST.depth_sections || []).length > 1) {        // log sections: numbered bands to click
    ST.depth_sections.forEach((d, i) => { const a = P(0, d.row0), b = P(ST.width, d.row1), cur = ST.depth_section === d.index;
      c.fillStyle = cur ? "rgba(37,99,235,.22)" : `${TAB10[i % 10]}33`; c.fillRect(a.x, a.y, b.x - a.x, b.y - a.y);
      c.strokeStyle = cur ? "#2563eb" : TAB10[i % 10]; c.lineWidth = cur ? 3 : 2; c.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
      c.font = "bold 16px -apple-system,sans-serif"; c.fillStyle = c.strokeStyle; c.fillText(`Section ${i + 1}: ${fmt(d.top, 0)}–${fmt(d.base, 0)} ft`, a.x + 8, Math.max(a.y, 0) + 20); });
  }
  if (STEP === "save") { drawLogs(); return; }                           // Check and save: the scan as printed, logs on the right
  const hdrView = STEP === "curves" && CVIEW === "header";
  if (hdrView) { drawHeaderView(c, P, cv); return; }
  if (LAYERS.grid) {                                                     // grid / tracks layer (teal)
    if (GRID?.available) {
      c.strokeStyle = "rgba(13,148,136,.5)"; c.lineWidth = 1;
      for (const xs of Object.values(GRID.v)) for (const x of xs) { const a = P(x, Math.max(top, 0)), b = P(x, bot); c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke(); }
      const step = Math.max(1, Math.ceil(GRID.h.length / 400 * (bot - top) / ST.height));
      GRID.h.forEach((r, i) => { if (!vis(r) || i % step) return; const a = P(GRID.tracks[0]?.left ?? 0, r), b = P(GRID.tracks[GRID.tracks.length - 1]?.right ?? ST.width, r);
        c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke(); });
    }
    if (LAY?.available) { c.lineWidth = 3; Object.values(LAY.edges).forEach(pl => { c.strokeStyle = "rgba(15,118,110,.95)"; poly(c, P, pl, vis); }); }
    (GRID?.tracks || []).forEach(t => { const a = P((t.left + t.right) / 2, 0), w = 64;   // track names on a teal tab at the top
      c.fillStyle = "rgba(15,118,110,.92)"; c.fillRect(a.x - w / 2, 40, w, 18); c.fillStyle = "#fff"; c.font = "bold 11px -apple-system,sans-serif";
      c.textAlign = "center"; c.fillText(t.name, a.x, 53); c.textAlign = "left"; });
    if (LAY?.available) { c.font = "12px -apple-system,sans-serif";
      LBL.forEach((l, i) => { if (!vis(l.row)) return; const dx = (GRID?.tracks || []).length > 1 ? (GRID.tracks[0].right + GRID.tracks[1].left) / 2 : ST.width / 2;
        const a = P(dx - 60, l.row), b = P(dx + 60, l.row);
        c.strokeStyle = LBLsel.has(i) ? "#2563eb" : "#d62728"; c.lineWidth = 3; c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
        if (STEP === "tracks") { c.fillStyle = l.value == null ? "#d97706" : "#b91c1c"; c.font = "bold 13px -apple-system,sans-serif"; c.fillText(l.value == null ? "?" : l.value + " ft", b.x + 6, b.y + 4); } });
    }
  }
  if (STEP === "curves" && HDR && LAYERS.grid) drawWords(c, P);
  if (LAYERS.traces && TR) {                                              // traced curves layer: every curve in its colour
    const hl = LAYERS.hl;                                                   // with the dark ink highlight on, other traces step back
    for (const t of TR.traces) { const act = t.id === TR.active, lw = act ? (hl ? 1.8 : 3) : 2.2;
      if (!hl) { c.save(); c.strokeStyle = "rgba(255,255,255,.9)"; c.lineWidth = lw + 3; poly(c, P, t.trace, vis); c.restore(); }   // halo on the grey scan
      c.strokeStyle = t.color; c.globalAlpha = act ? 1 : (hl ? 0.18 : 0.95); c.lineWidth = lw; poly(c, P, t.trace, vis); }
    c.globalAlpha = 1;
  }
  if (STEP === "trace" && OV) drawFix(c, P, vis, cv);
  drawCircle(c, P);
  c.strokeStyle = "#2563eb"; c.lineWidth = 2; for (const q of pts) { const p = P(q[1], q[0]); c.strokeRect(p.x - 5, p.y - 5, 10, 10); }
}
function drawHeaderView(c, P, cv) {      // Curves > Header & scales: body covered, header lines boxed, confidence bars
  if (!HLN?.strip?.[1]) return;
  const yb = Math.max(0, P(0, HLN.strip[1] + 6).y);
  c.fillStyle = "rgba(30,41,59,.66)"; c.fillRect(0, yb, cv.width, cv.height - yb);
  for (const L of HLN.lines) { const a = P(L.x0 - 4, L.y0 - 3), b = P(L.x1 + 4, L.y1 + 3), cu = (CURV?.curves || []).find(x => x.id === L.curve);
    if (cu) { c.globalAlpha = 0.14; c.fillStyle = cu.color; c.fillRect(a.x, a.y, b.x - a.x, b.y - a.y); c.globalAlpha = 1; c.strokeStyle = cu.color; c.lineWidth = 2.5; }
    else { c.strokeStyle = "rgba(100,116,139,.8)"; c.lineWidth = 1; c.setLineDash([4, 3]); }
    c.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y); c.setLineDash([]);
    c.fillStyle = L.conf >= 90 ? "#16a34a" : L.conf >= 70 ? "#d97706" : "#dc2626"; c.fillRect(a.x, b.y + 1, (b.x - a.x) * Math.max(0.05, L.conf / 100), 3); }
  drawWords(c, P); placePops(P, yb);
  const m = $("main").getBoundingClientRect();
  document.querySelectorAll("#scalePops .pop").forEach(el => { const L = HLN.lines.find(q => q.curve === el.dataset.id); if (!L) return;
    const a = P((L.x0 + L.x1) / 2, L.y1 + 4), r = el.getBoundingClientRect();
    c.strokeStyle = el.dataset.color; c.lineWidth = 2; c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(r.left - m.left + r.width / 2, r.top - m.top); c.stroke(); });
}
function placePops(P, yb) {               // scale pop-ups under their header lines, spread so they never overlap
  const els = [...document.querySelectorAll("#scalePops .pop")], W = $("main").clientWidth; let right = 8;
  els.map(el => { const L = HLN.lines.find(q => q.curve === el.dataset.id); return [L ? P((L.x0 + L.x1) / 2, 0).x : 0, el]; })
    .sort((a, b) => a[0] - b[0]).forEach(([x, el]) => { const w = el.offsetWidth || 230, l = Math.min(Math.max(right, x - w / 2), W - w - 8);
      el.style.left = l + "px"; el.style.top = Math.min(yb + 26, $("main").clientHeight - el.offsetHeight - 8) + "px"; right = l + w + 12; });
}
function drawLogs() {                     // Check and save: every chosen log in its track, row for row with the scan on the left
  const pane = $("logPane"); if (pane.hidden || !LOGS || !viewer || !BASE()) return;
  const cv = $("logCv"), c = cv.getContext("2d"); cv.width = cv.clientWidth; cv.height = cv.clientHeight; c.clearRect(0, 0, cv.width, cv.height);
  // viewer-element y, shifted by the gap between the two boxes' tops (borders / padding), so rows line up exactly
  const dy = viewer.element.getBoundingClientRect().top - cv.getBoundingClientRect().top;
  const Y = r => BASE().imageToViewerElementCoordinates(new OpenSeadragon.Point(0, r)).y + dy;
  const tr = LOGS.tracks.length ? LOGS.tracks : [{name: null, left: 0, right: LOGS.width}], sel = LOGS.curves.filter(q => LOGSEL.has(q.id));
  const x0 = Math.min(...tr.map(t => t.left)), x1 = Math.max(...tr.map(t => t.right)), ml = 50, mr = 10, X = v => ml + (v - x0) / (x1 - x0) * (cv.width - ml - mr);
  const nmax = Math.max(1, ...tr.map(t => sel.filter(q => q.track === t.name).length)), HD = 8 + 15 * nmax;
  const ln = (a, b, p, q) => { c.beginPath(); c.moveTo(a, b); c.lineTo(p, q); c.stroke(); };
  c.font = "11px -apple-system,sans-serif"; c.textAlign = "right";
  for (const [d, r, major] of LOGS.ticks) { const y = Y(r); if (y < HD || y > cv.height) continue;
    c.strokeStyle = major ? "#94a3b8" : "#e2e8f0"; c.lineWidth = major ? 1 : 0.6; ln(ml, y, cv.width - mr, y);
    if (major) { c.fillStyle = "#334155"; c.fillText(String(d), ml - 6, y + 4); } }
  c.strokeStyle = "#475569"; c.lineWidth = 1; tr.forEach(t => { ln(X(t.left), HD, X(t.left), cv.height); ln(X(t.right), HD, X(t.right), cv.height); });
  for (const q of sel) { const t = tr.find(t => t.name === q.track) || tr[0]; let pen = false;
    c.strokeStyle = q.color; c.lineWidth = 1.4; c.beginPath();
    for (let i = 0; i < q.row.length; i++) { const f = q.frac[i], y = Y(q.row[i]);
      if (f == null || f < -0.01 || f > 1.01 || y < HD || y > cv.height) { pen = false; continue; }
      const x = X(t.left + f * (t.right - t.left)); pen ? c.lineTo(x, y) : c.moveTo(x, y); pen = true; }
    c.stroke(); }
  c.fillStyle = "#fff"; c.fillRect(0, 0, cv.width, HD); c.strokeStyle = "#cbd5e1"; ln(0, HD, cv.width, HD);
  tr.forEach(t => sel.filter(q => q.track === t.name).forEach((q, i) => { const y = 15 + 15 * i, a = X(t.left), b = X(t.right);
    c.fillStyle = q.color; c.font = "bold 11px -apple-system,sans-serif"; c.textAlign = "center"; c.fillText(q.name, (a + b) / 2, y);
    c.font = "10px -apple-system,sans-serif"; c.textAlign = "left"; c.fillText(String(q.left), a + 3, y); c.textAlign = "right"; c.fillText(String(q.right), b - 3, y); }));
  c.textAlign = "left";
}
function drawWords(c, P) {
  if (mode !== "word") return; c.lineWidth = 1.5;
  HW.forEach(w => { const a = P(w.x0, w.y0), b = P(w.x1, w.y1);
    c.strokeStyle = w.corrected ? "#16a34a" : (w.conf < 60 ? "#d97706" : "rgba(37,99,235,.7)"); c.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y); });
}
function drawCircle(c, P) {
  if ((mode !== "circle" && mode !== "node") || !HOVER) return; const r = +($("circR")?.value || 60), a = P(HOVER.x, HOVER.row), b = P(HOVER.x + r, HOVER.row);
  c.strokeStyle = "rgba(217,119,6,.9)"; c.lineWidth = 2; c.setLineDash([5, 4]); c.beginPath(); c.arc(a.x, a.y, Math.abs(b.x - a.x), 0, 2 * Math.PI); c.stroke(); c.setLineDash([]);
}
const CAND_COL = ["#d62728", "#2563eb", "#16a34a", "#9333ea"];
function drawFix(c, P, vis, cv) {         // the selected curve while fixing: off-scale orange, uncertain yellow, pins, guides, spots
  const dis = CHK?.disagree || [], inDis = r => dis.some(([a, b]) => r >= a && r <= b);
  for (const cd of (OV.candidates || [])) { if (cd.picked) continue;
    c.strokeStyle = CAND_COL[cd.index % CAND_COL.length]; c.globalAlpha = 0.7; c.lineWidth = 2; c.setLineDash([7, 5]); poly(c, P, cd.trace, vis); c.setLineDash([]); c.globalAlpha = 1; }
  let prev = null; const col = active()?.color || "#d62728";
  for (const [r, x, o] of OV.trace) { if (!vis(r)) { prev = null; continue; } const p = P(x, r);
    if (prev) { if (inDis(r)) { c.strokeStyle = "rgba(250,204,21,.55)"; c.lineWidth = 9; c.beginPath(); c.moveTo(prev.x, prev.y); c.lineTo(p.x, p.y); c.stroke(); }
      c.lineWidth = 3; c.strokeStyle = o ? "rgba(255,140,0,.95)" : col; c.beginPath(); c.moveTo(prev.x, prev.y); c.lineTo(p.x, p.y); c.stroke(); }
    prev = p; }
  c.strokeStyle = "rgba(37,99,235,.85)"; c.lineWidth = 5; c.lineCap = "round";
  for (const st of (OV.strokes || []).concat(STROKE.length ? [STROKE] : [])) { c.beginPath(); st.forEach(([r, x], i) => { const p = P(x, r); i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y); }); c.stroke(); }
  c.lineCap = "butt";
  if (CUR) { const sp = CUR, cols = ["#2563eb", "#f97316", "#16a34a"];
    const a = P(0, sp.row0), b = P(0, sp.row1); c.fillStyle = "rgba(250,204,21,.12)"; c.fillRect(0, a.y, cv.width, b.y - a.y);
    sp.choices.forEach((ch, i) => { const p = P(ch.x, sp.row); c.strokeStyle = cols[i % 3]; c.lineWidth = 4; c.beginPath(); c.arc(p.x, p.y, 16, 0, 7); c.stroke();
      c.font = "bold 15px -apple-system,sans-serif"; c.fillStyle = cols[i % 3]; c.fillText(String(i + 1), p.x - 4, p.y - 22); }); }
  c.fillStyle = "rgba(255,140,0,.15)"; for (const [a, b] of OV.backup_intervals || []) { const p0 = P(0, a), p1 = P(0, b); c.fillRect(0, p0.y, cv.width, p1.y - p0.y); }
  c.fillStyle = "rgba(100,116,139,.16)"; for (const [a, b] of OV.hidden_intervals || []) { const p0 = P(0, a), p1 = P(0, b); c.fillRect(0, p0.y, cv.width, p1.y - p0.y); }
  OV.anchors.forEach((a, i) => { const p = DRAG && DRAG.i === i ? P(DRAG.x, DRAG.row) : P(a[1], a[0]); c.fillStyle = "#16a34a"; c.beginPath(); c.arc(p.x, p.y, 7, 0, 7); c.fill();
    c.strokeStyle = "#fff"; c.lineWidth = 2; c.stroke(); });
  c.strokeStyle = "#e377c2"; c.lineWidth = 2.5; for (const a of OV.excluded || []) { const p = P(a[1], a[0]);
    c.beginPath(); c.moveTo(p.x - 6, p.y - 6); c.lineTo(p.x + 6, p.y + 6); c.moveTo(p.x + 6, p.y - 6); c.lineTo(p.x - 6, p.y + 6); c.stroke(); }
}
window.addEventListener("resize", draw);

// ------------------------------------------------------------------ clicks + keys
async function onClick(ev) {
  if (!ev.quick || mode === "pan") return; ev.preventDefaultAction = true;
  const ip = BASE().viewerElementToImageCoordinates(ev.position), row = ip.y, x = ip.x;
  try {
    if (mode === "word") { const w = HW.find(w => row >= w.y0 - 3 && row <= w.y1 + 3 && x >= w.x0 - 3 && x <= w.x1 + 3); if (!w) return;
      const t = prompt(`OCR read "${w.ocr_text || w.text}" (confidence ${Math.round(w.conf)}). Correct text (empty = delete):`, w.text); if (t === null || t === w.text) return;
      HDR = Object.assign(HDR || {}, await api("POST", A("/header/words"), {track: w.track, edits: [{i: w.i, text: t}]}));
      HW = (await api("GET", A("/header/words"))).words; draw(); return; }
    if (mode === "circle") { await fixCall("/fix/circle", {row, x, radius: +($("circR")?.value || 60), through: ev.originalEvent?.shiftKey || false}); return; }
    if (mode === "fix") { if (Date.now() - (DRAG?.t || 0) < 300) return; await fixCall("/fix/pin", {row, x}); return; }
    if (mode === "exclude") { await api("POST", E("/exclude"), {row, x}); }
    else if (mode === "seed") { await api("POST", E("/trace"), {seed: [row, x]}); setMode("fix"); }
    else if (mode === "wrap") { pts.push([row, x]); draw(); if (pts.length < 2) return;
      const [a, b] = pts; pts = []; await api("POST", E("/backup_interval"), {rows: [a[0], b[0]]}); }
    else if (mode === "box") { pts.push([row, x]); draw(); if (pts.length < 2) return; const [a, b] = pts; pts = []; setMode("pan");
      await runJob(A("/section"), {box: [a[1], b[1]], rows: [Math.min(a[0], b[0]), Math.max(a[0], b[0])]}, "section from box"); await refresh(); render(false); return; }
    else if (mode === "dsec") { const d = (ST.depth_sections || []).find(d => row >= d.row0 && row <= d.row1);
      if (d) await chooseSection(d.index); return; }
    else if (mode === "sample") { const r = await api("POST", A("/color/sample"), {row, x});
      const name = prompt(`RGB ${r.rgb.join(", ")}. Name for this colour:`, "custom"); if (!name) return;
      PAL.palette[name.trim()] = r.rgb; await api("POST", A("/color"), {palette: PAL.palette}); renderPalette(); return; }
    else if (mode === "label") { const v = prompt(`Depth (ft) of the label at row ${row.toFixed(0)}?`);
      if (v === null || v === "" || !isFinite(+v)) return; LBL.push({row, ocr: null, value: +v, added: true}); LBL.sort((a, b) => a.row - b.row); dirty(true); renderLabels(); draw(); return; }
    await refresh(); if (STEP === "trace") renderAnchors();
  } catch (e) { console.warn(e); }
}
const imgPt = ev => BASE().viewerElementToImageCoordinates(ev.position);
function pinNear(ev) { const ip = imgPt(ev), z = BASE().viewportToImageZoom(viewer.viewport.getZoom(true)), tol = 12 / z;
  return (OV?.anchors || []).findIndex(a => Math.hypot(a[1] - ip.x, a[0] - ip.y) < tol); }
function onPress(ev) {
  if (mode === "guide") { const ip = imgPt(ev); STROKE = [[ip.y, ip.x]]; ev.preventDefaultAction = true; return; }
  if (mode === "node") { const ip = imgPt(ev); NODE = {row: ip.y, x: ip.x}; viewer.setMouseNavEnabled(false); ev.preventDefaultAction = true; return; }
  if (mode === "fix") { const i = pinNear(ev); if (i >= 0) { const ip = imgPt(ev); DRAG = {i, row: ip.y, x: ip.x, moved: false}; viewer.setMouseNavEnabled(false); } }
}
function onDrag(ev) {
  if (mode === "guide" && STROKE.length) { const ip = imgPt(ev); STROKE.push([ip.y, ip.x]); ev.preventDefaultAction = true; drawSoon(); return; }
  if (mode === "node" && NODE) { const ip = imgPt(ev); NODE = {row: ip.y, x: ip.x}; HOVER = NODE; ev.preventDefaultAction = true; drawSoon(); return; }
  if (DRAG) { const ip = imgPt(ev); DRAG.row = ip.y; DRAG.x = ip.x; DRAG.moved = true; ev.preventDefaultAction = true; drawSoon(); }
}
async function onRelease() {
  if (mode === "node" && NODE) { const n = NODE; NODE = null; viewer.setMouseNavEnabled(true);
    try { await fixCall("/fix/node", {row: n.row, x: n.x, radius: +($("circR")?.value || 60)}); } catch (e) { draw(); } return; }
  if (mode === "guide" && STROKE.length) { const st = STROKE; STROKE = [];
    if (st.length > 2) { try { await fixCall("/fix/stroke", {points: st}); } catch (e) { draw(); } } else draw(); return; }
  if (DRAG) { const d = DRAG; viewer.setMouseNavEnabled(true); DRAG = {t: Date.now()}; if (d.moved) await fixCall("/fix/move", {index: d.i, row: d.row, x: d.x}); else { DRAG = null; } }
}
async function fixCall(path, body) {
  const r = await api("POST", A(path), body); PROG = r.progress; await refresh(); CHK = null; await loadChecks();
  if (STEP === "trace") { renderAnchors(); renderProgress(r); renderCands(); } return r;
}
async function loadChecks() { if (!SID || !ST?.traced) return; CHK = await api("GET", A("/checks")); if (STEP === "trace") { renderAnchors(); showSpot(false); } draw(); }
function renderProgress(r) {
  const el = $("prog7"); if (!el || !PROG) return; const rec = PROG.recent || [], mx = Math.max(PROG.small_ft * 4 || 20, ...rec);
  el.innerHTML = `<div class="meter">${rec.map(v => `<i style="height:${Math.max(2, 34 * v / mx)}px;background:${v <= PROG.small_ft ? "#16a34a" : "#2563eb"}" title="${v} ft changed"></i>`).join("")}</div>
    <div class="${PROG.settled ? "okc" : ""}">${esc(PROG.message)}</div>${r?.ms != null ? `<div class="note">re-traced in ${r.ms} ms</div>` : ""}`;
}
function setMode(m) { mode = m; pts = []; document.querySelectorAll("button.tool").forEach(b => b.classList.toggle("active", b.dataset.mode === m));
  if (viewer) viewer.setMouseNavEnabled(true); draw(); }
document.addEventListener("keydown", e => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  const k = e.key.toLowerCase(), b = document.querySelector(`button.tool[data-key="${k}"]`);
  if (k === "n" && STEP === "trace") { nextUncertain(); return; }
  if (STEP === "trace" && CUR && /^[1-3]$/.test(k)) { const ch = CUR.choices[+k - 1]; if (ch) answer(ch.id); return; }
  if (STEP === "trace" && CUR && k === "k") { answer("skip"); return; }
  if (k === "escape") setMode("pan"); else if (k === "z" && !e.metaKey && !e.ctrlKey) undo(); else if (b && !b.disabled) b.click();
  else if ((k === "delete" || k === "backspace") && STEP === "tracks") delLabels();
});
async function undo() { if (!SID || !ST?.undo.length) return; const r = await api("POST", A("/undo")); toast("undone: " + r.undone);
  await loadCurves(); if (STEP === "tracks") await loadLabels(); await refresh(); loadHighlight(); render(false); }
$("undoBtn").onclick = undo;
const dirty = on => $("dirty").innerHTML = on ? '<span class="warnc">● unapplied changes - press the blue button to apply</span>' : "";
const tool = (m, key, label, dis) => `<button class="tool" data-mode="${m}" data-key="${key}" ${dis || ""}>${label} <span class="kbd">${key.toUpperCase()}</span></button>`;
function bindTools() { document.querySelectorAll("button.tool").forEach(b => b.onclick = () => setMode(mode === b.dataset.mode ? "pan" : b.dataset.mode)); }

// ------------------------------------------------------------------ SVG figures
function chart(o) {   // o: {w,h,title,xlabel,ylabel,invertY,series:[{pts:[[x,y]],color,dots,dash}],xr,yr}
  const W = o.w || 360, H = o.h || 220, m = {l: 46, r: 10, t: 8, b: 30}; const all = o.series.flatMap(s => s.pts).filter(p => isFinite(p[0]) && isFinite(p[1]));
  if (!all.length) return `<p class="note">no data</p>`;
  const ext = i => { let lo = Math.min(...all.map(p => p[i])), hi = Math.max(...all.map(p => p[i])); if (hi === lo) { lo -= 1; hi += 1; } return [lo, hi]; };
  const [x0, x1] = o.xr || ext(0), [y0, y1] = o.yr || ext(1);
  const X = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r), Y = v => o.invertY ? m.t + (v - y0) / (y1 - y0) * (H - m.t - m.b) : H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  const tick = (lo, hi) => { const s = Math.pow(10, Math.floor(Math.log10((hi - lo) / 4))), st = [1, 2, 5, 10].map(k => k * s).find(k => (hi - lo) / k <= 6); const r = []; for (let v = Math.ceil(lo / st) * st; v <= hi; v += st) r.push(+v.toPrecision(10)); return r; };
  let g = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="none" stroke="#bbb"/>`;
  tick(x0, x1).forEach(v => g += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - m.b}" y2="${H - m.b + 4}" stroke="#888"/><text x="${X(v)}" y="${H - m.b + 15}" text-anchor="middle">${v}</text>`);
  tick(y0, y1).forEach(v => g += `<line x1="${m.l - 4}" x2="${m.l}" y1="${Y(v)}" y2="${Y(v)}" stroke="#888"/><text x="${m.l - 6}" y="${Y(v) + 3}" text-anchor="end">${v}</text>`);
  for (const s of o.series) {
    const p = s.pts.filter(q => isFinite(q[0]) && isFinite(q[1]));
    if (s.dots) g += p.map(q => `<circle cx="${X(q[0]).toFixed(1)}" cy="${Y(q[1]).toFixed(1)}" r="${s.r || 1.6}" fill="${s.color}" fill-opacity="${s.alpha || 0.6}"/>`).join("");
    else { let d = "", pen = false; s.pts.forEach(q => { if (!isFinite(q[0]) || !isFinite(q[1])) { pen = false; return; } d += (pen ? "L" : "M") + X(q[0]).toFixed(1) + " " + Y(q[1]).toFixed(1); pen = true; });
      g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.lw || 0.9}" ${s.dash ? 'stroke-dasharray="5 3"' : ""} opacity="${s.alpha || 1}"/>`; }
  }
  g += `<text x="${(W + m.l) / 2}" y="${H - 2}" text-anchor="middle">${esc(o.xlabel || "")}</text><text transform="translate(11 ${(H - m.b) / 2}) rotate(-90)" text-anchor="middle">${esc(o.ylabel || "")}</text></svg>`;
  return (o.title ? `<div class="ttl">${esc(o.title)}</div>` : "") + g;
}
function fig(html, small) { $("fig").innerHTML = html; $("fig").hidden = !html; $("fig").classList.toggle("small", !!small); }

// ------------------------------------------------------------------ tabs
const R = {};
async function render(refit = true) { const st = STEPS.findIndex(s => s.id === STEP); setMode("pan"); dirty(false); fig("");
  $("stepTitle").textContent = `${st + 1} · ${STEPS[st].title}`; $("mainBtn").disabled = false;
  document.body.dataset.step = $("main").dataset.step = STEP; $("tabBadge").textContent = `${st + 1} · ${STEPS[st].title}`;
  const save = STEP === "save" && !!SID;                   // Check and save: scan left, logs right
  $("main").classList.toggle("split", save); $("logPane").hidden = !save; $("layers").hidden = !viewer || save; $("lyState").hidden = !viewer || save;
  if (viewer) viewer.forceResize?.();
  $("scalePops").innerHTML = ""; $("curvePanel").hidden = !(SID && STEP === "curves" && CVIEW === "curves");
  try { await R[STEP](); } catch (e) { console.warn(e); } bindTools(); applyLayers(); if (refit) focusStep(); draw(); }
Shell.init({steps: STEPS, onStep: (i, s) => { STEP = s.id; render(); }});
const main = (label, fn) => { $("mainBtn").textContent = label; $("mainBtn").onclick = async () => { $("mainBtn").disabled = true; try { await fn(); } catch (e) { console.warn(e); } finally { $("mainBtn").disabled = false; } }; };
const VEC = () => ST?.input?.used === "vector";
const VNOTE = "vector file: tracing not needed";
function applyInputMode() {           // vector files: the tracing steps do not apply, so they are hidden (not greyed out)
  for (const s of STEPS) if (["tracks", "curves", "trace"].includes(s.id)) { s.hidden = VEC(); if (s.hidden) s.note = VNOTE; }
  if (VEC()) { for (const i of [0, 1, 2, 3]) Shell.complete(i, false); } else Shell.reset(1);
}
const next = () => { if (VEC() && STEP === "open") { applyInputMode(); Shell.go(4); return; } Shell.complete(Shell.cur); };
const on = (id, fn) => { const b = $(id); if (b) b.onclick = async () => { try { await fn(); } catch (e) { console.warn(e); } }; };
async function upload(file) {
  const r = await fetch("/app/api/upload", {method: "POST", headers: {"X-Filename": encodeURIComponent(file.name)}, body: file});
  const j = await r.json(); if (!r.ok) { toast(j.detail || "upload failed", true); throw new Error(j.detail); } return j.path;
}
async function afterLoad(sid) {
  SID = sid; ST = null; Shell.reset(0); await refresh(false); await loadCurves();
  LAY = await api("GET", A("/layout")); GRID = await api("GET", A("/grid_lines")); await loadLabels(); await openViewer(); await refresh();
  const errs = Object.entries(ST.errors || {}).filter(([k]) => !["depth_section", "color"].includes(k)); if (errs.length) toast(errs.map(([k, v]) => `${k}: ${v}`).join("\n"), true);
}

R.open = () => {
  $("stepLead").textContent = "Pick a scanned log. Everything is then found automatically once: tracks, depth numbers, header scales and every curve.";
  const f = INFO.features, multi = ST && (ST.depth_sections || []).length > 1, sk = ST?.skew || [];
  $("stepBody").innerHTML = `<h3>Scan</h3><div class="row"><select id="cat" style="width:100%">${INFO.catalog.map((c, i) => `<option value="${i}">${esc(c.well)} · ${esc(c.scan.split("/").pop())}${c.las ? " · with LAS" : ""}</option>`).join("")}</select></div>
    <div class="row"><label class="filebtn">or choose a file… <input type="file" id="filePick" accept=".tif,.tiff,.png,.jpg,.jpeg,.pdf"></label><span class="note" id="fileName"></span></div>
    <div class="row"><label class="filebtn">reference LAS (optional) <input type="file" id="lasPick" accept=".las"></label><span class="note" id="lasName"></span></div>
    <div id="kindBox" hidden><h3>Input type</h3><div class="card" id="kindCard"><span class="note">checking the file…</span></div>
    <div class="row"><label>method</label><select id="method" style="flex:1"><option value="auto">as detected (recommended)</option>
      <option value="trace">image: clean up and trace the curves</option><option value="vector">vector PDF: read the curve paths</option>
      <option value="mixed">mixed PDF: header from the PDF text, curves traced</option></select></div>
    <div class="row" id="pageRow" hidden><label>PDF page</label><input type="number" id="pdfPage" min="1" value="1" style="width:5em"></div></div>
    <div class="row"><label><input type="checkbox" id="autoLayout" checked> automatic layout</label></div>
    ${ST && VEC() ? `<div class="card">Loaded: <b>${esc(ST.well)}</b> · PDF page ${(ST.vector?.page ?? 0) + 1} · ${ST.n_curves} curve(s) read from their paths</div>
      <div class="card vnote"><b>${VNOTE}.</b> The curves, depth numbers and scales were read from the PDF itself, so the
      <i>Tracks and depth</i>, <i>Curves</i> and <i>Trace and fix</i> steps are hidden. Continue to <b>Check and save</b>.</div>
      ${(ST.vector?.notes || []).length ? `<div class="note">${ST.vector.notes.map(esc).join("<br>")}</div>` : ""}` : ""}
    ${ST && !VEC() ? `<div class="card">Loaded: <b>${esc(ST.well)}</b> · ${ST.width}×${ST.height} px · ${ST.n_curves} curve(s) found (${ST.kind === "color" ? "by colour" : ST.kind === "style" ? "by line style" : "one curve"})</div>
      ${ST.mixed ? `<div class="note">Header from the PDF text: ${(ST.mixed.applied || []).map(esc).join("<br>") || "no matching track"}</div>` : ""}
      ${multi ? `<h3>Depth section</h3><div class="note">This sheet holds more than one log section. Pick one (or click its band on the scan).</div>
        ${ST.depth_sections.map((d, i) => `<div class="row"><button id="dsec${d.index}" style="width:100%" ${ST.depth_section === d.index ? 'class="primary"' : ""}>Section ${i + 1}: ${fmt(d.top, 0)}–${fmt(d.base, 0)} ft · ${fmt(d.px_per_ft, 2)} px/ft</button></div>`).join("")}` : ""}
      <h3>Straight?</h3><div class="note">Deskewed automatically (${sk.length ? `${fmt(Math.min(...sk), 2)}° to ${fmt(Math.max(...sk), 2)}°` : "no blocks"}). If the grid still leans, turn it a little:</div>
      <div class="row"><label>rotate</label><input type="number" id="rot" step="0.1" min="-10" max="10" value="${ST.rotate || 0}" style="width:6em"> °<button id="rotApply">Apply</button></div>` : ""}
    <details class="more"><summary>Installed features</summary><div class="card">Tracer backend: <b>${esc(f.backend)}</b>
    <div class="${f.ocr.ok ? "" : "disabled-feature"}">OCR (depth labels, header): ${f.ocr.ok ? `<span class="okc">RapidOCR</span>` : "no"}</div>${hint(f.ocr)}
    <div class="${f.ml.ok ? "" : "disabled-feature"}">ML U-Net cost map: ${f.ml.ok ? '<span class="okc">yes</span>' : "no"}</div>${hint(f.ml)}
    <div class="${f.plot.ok ? "" : "disabled-feature"}">QC plot PNG export: ${f.plot.ok ? '<span class="okc">yes</span>' : "no"}</div>${hint(f.plot)}</div>
    <div class="note">Project: ${esc(INFO.project)} · presets from <code>dataset_profiles/*_wells.yaml</code></div></details>`;
  let UP = null;                     // uploaded file path (uploaded once, when picked, so its type can be shown)
  const KIND = {scan: "image scan", vector: "vector PDF", mixed: "mixed PDF (scan with PDF text)"};
  const showKind = async (path) => {
    const el = $("kindCard"); if (!el) return;
    try { renderKind(await api("GET", "/app/api/detect?scan=" + encodeURIComponent(path)), /\.pdf$/i.test(path)); }
    catch (e) { el.innerHTML = `<span class="note">could not inspect the file</span>`; }
  };
  const renderKind = (d, pdf) => {
    const el = $("kindCard"); if (!el || !d) return; const ev = d.evidence || {};
    { $("pageRow").hidden = !pdf; $("kindBox").hidden = !pdf; if (!pdf) $("method").value = "auto";   // image scans: the Open tab stays as before (always the image workflow)
      if (pdf && ev.curve_pages?.length && d.kind === "vector") $("pdfPage").value = ev.curve_pages[0];
      el.dataset.kind = d.kind;
      el.innerHTML = `Detected: <b>${esc(KIND[d.kind] || d.kind)}</b>` + (pdf ? `<div class="note">page ${d.page + 1} of ${d.n_pages}: ${ev.n_curve_paths} curve path(s), ${ev.text_chars} text characters, pictures cover ${Math.round(100 * (ev.image_coverage || 0))}% of the page${ev.curve_pages?.length ? ` · curves on page ${ev.curve_pages.join(", ")}` : ""}</div>`
        : `<div class="note">${ev.size_px ? `${ev.size_px[0]}×${ev.size_px[1]} px ${esc(ev.format || "")}` : ""}</div>`)
        + `<div>Recommended: ${esc(String(d.method_label || "").replace(/\s*\([a-z_]+\)$/, ""))}</div>` + (d.kind === "vector" ? `<div class="note">${VNOTE}</div>` : "");
      $("method").querySelector('option[value="auto"]').textContent = `as detected: ${KIND[d.kind] || d.kind} (recommended)`;
      for (const v of ["vector", "mixed"]) $("method").querySelector(`option[value="${v}"]`).disabled = !pdf;
    }
  };
  if (ST?.scan) { const i = INFO.catalog.findIndex(c => String(ST.scan).replace(/\\/g, "/").endsWith(c.scan)); if (i >= 0) $("cat").value = String(i); }
  const catPath = () => INFO.catalog[+$("cat").value]?.scan;
  if (ST?.input && !$("filePick").files.length) { $("method").value = ST.input.method || "auto"; }
  if (ST?.input?.kind) { renderKind(ST.input, /\.pdf$/i.test(ST.scan_name || "")); if (VEC()) $("pdfPage").value = (ST.vector?.page ?? 0) + 1; }
  else if (catPath()) showKind(catPath());
  $("filePick").onchange = async () => { const fp = $("filePick").files[0]; $("fileName").textContent = fp?.name || ""; UP = null;
    if (fp) { UP = await upload(fp); await showKind(UP); if (ST) $("mainBtn").textContent = "Open & analyse (replaces current)"; } };
  $("lasPick").onchange = () => { $("lasName").textContent = $("lasPick").files[0]?.name || ""; };
  if (multi) { ST.depth_sections.forEach(d => on("dsec" + d.index, () => chooseSection(d.index))); setTimeout(() => { mode = "dsec"; }, 0); }
  on("rotApply", async () => { const r = await runJob(A("/rotate"), {deg: +$("rot").value || 0}, "rotate and analyse again"); await afterLoad(r.sid); render(false); });
  main(ST ? "Continue" : "Open & analyse", async () => {
    const fp = $("filePick").files[0], lp = $("lasPick").files[0];
    if (ST && !fp && $("cat").dataset.touched !== "1" && $("method").dataset.touched !== "1") { if (multi && ST.errors?.depth_section) { toast("pick a depth section first", true); return; } next(); return; }
    let body;
    if (fp) body = {scan: UP || await upload(fp), curve: "GR", well: fp.name.replace(/\.[^.]+$/, ""), las: lp ? await upload(lp) : null, settings: {}, auto_layout: $("autoLayout").checked};
    else { const c = INFO.catalog[+$("cat").value]; body = {scan: c.scan, curve: c.curve, well: c.well, las: lp ? await upload(lp) : c.las, settings: c.settings, auto_layout: $("autoLayout").checked}; }
    body.method = $("method").value; if (!$("pageRow").hidden) body.page = Math.max(0, (+$("pdfPage").value || 1) - 1);
    const r = await runJob("/app/api/open", body, "loading");
    await afterLoad(r.sid); applyInputMode(); render(false);
    if (VEC()) { Shell.go(4); return; }
    if (!((ST.depth_sections || []).length > 1 && ST.errors?.depth_section)) next();
  });
  $("cat").onchange = () => { $("cat").dataset.touched = "1"; showKind(catPath()); if (ST) $("mainBtn").textContent = "Open & analyse (replaces current)"; };
  $("method").onchange = () => { $("method").dataset.touched = "1"; if (ST) $("mainBtn").textContent = "Open & analyse (replaces current)"; };
  help(`Choose a scan from the list or a file, then <b>Open & analyse</b>. Progress shows as n / N below. ${NAV}`);
};
async function chooseSection(i) { await runJob(A("/depth_section"), {index: i}, "depth section"); LAY = await api("GET", A("/layout")); await loadLabels(); await refresh(); render(false); }

// Tracks and depth ---------------------------------------------------
async function loadLabels() { LAY = await api("GET", A("/layout")); LBL = (LAY.labels || []).map(([row, v]) => ({row, ocr: v, value: v})); LBLsel.clear(); }
function labelDoubt(l) {
  if (l.added || l.value !== l.ocr) return "";
  if (l.ocr == null) return "could not be read";
  const a = (ST.calibration?.depth_anchors || []).slice().sort((p, q) => p[0] - q[0]);
  if (a.length < 2) return "";
  let j = a.findIndex(p => p[0] >= l.row); if (j <= 0) j = j === 0 ? 1 : a.length - 1;
  const [r0, d0] = a[j - 1], [r1, d1] = a[j], d = d0 + (d1 - d0) * (l.row - r0) / ((r1 - r0) || 1);
  return Math.abs(d - l.ocr) > 1 ? `read ${l.ocr}, but the others suggest ${d.toFixed(0)}` : "";
}
function renderLabels() {
  if (!$("lblTbl")) return;
  const nd = LBL.filter(l => labelDoubt(l)).length;
  if ($("lblDoubt")) $("lblDoubt").innerHTML = nd ? `<span class="warnc">${nd} number(s) need a look (orange)</span>` : "";
  $("lblTbl").innerHTML = `<tr><th>#</th><th>row (px)</th><th>read</th><th>depth (ft)</th></tr>` + LBL.map((l, i) =>
    `<tr data-i="${i}" class="${LBLsel.has(i) ? "sel" : ""} ${l.added ? "added" : (l.value !== l.ocr ? "edited" : "")}" ${labelDoubt(l) ? `style="background:#fff4e0" title="${esc(labelDoubt(l))}"` : ""}><td>${i + 1}</td><td>${l.row.toFixed(0)}</td><td>${l.ocr ?? "?"}</td>
     <td><input data-i="${i}" value="${l.value ?? ""}" style="width:80px"></td></tr>`).join("");
  $("lblTbl").querySelectorAll("tr[data-i]").forEach(tr => { const i = +tr.dataset.i;
    tr.onclick = ev => { if (ev.target.tagName !== "INPUT") { LBLsel.has(i) ? LBLsel.delete(i) : LBLsel.add(i); renderLabels(); draw(); fitRows(LBL[i].row - 300, LBL[i].row + 300); } }; });
  $("lblTbl").querySelectorAll("input").forEach(inp => inp.onchange = () => { const i = +inp.dataset.i; LBL[i].value = inp.value === "" ? null : +inp.value; dirty(true); renderLabels(); draw(); });
}
function delLabels() { if (!LBLsel.size) return; LBL = LBL.filter((_, i) => !LBLsel.has(i)); LBLsel.clear(); dirty(true); renderLabels(); draw(); }
R.tracks = async () => {
  const c = ST.calibration || {}, f = INFO.features.ocr, tr = (GRID?.tracks || []);
  $("stepLead").textContent = "The track borders (blue), the grid and the depth numbers (red) found on the scan, in one view. Fix any depth number or border that is off.";
  $("stepBody").innerHTML = `<div class="card">${c.px_per_ft ? `<b>${esc(c.depth_source)}</b> · ${c.depth_anchors.length} depth anchors · ${fmt(c.px_per_ft, 3)} px/ft` : '<span class="err">no depth calibration</span>'}
     · ${tr.length} track(s)${ST.errors?.depth ? `<div class="err">${esc(ST.errors.depth)}</div>` : ""}</div>${f.ok ? "" : hint(f)}
    <h3>Depth numbers</h3><div class="note" id="lblDoubt"></div><div class="list" style="max-height:210px"><table class="t" id="lblTbl"></table></div>
    <div class="row">${tool("label", "l", "+ Add label")}<button id="lDel" class="danger">Delete selected</button><button id="lReset">Reset to automatic</button></div>
    <details class="more"><summary>Or type one known depth</summary><div class="note">Depth of the first number below the header, and the step between numbers (often 100).</div><div class="row"><label>first label</label><input type="number" id="fd" placeholder="ft"><label style="min-width:0">step</label><input type="number" id="li" placeholder="100"></div>
    <div class="row"><button id="lAuto">Re-run automatic</button></div>${ocrLine()}</details>
    <h3>Track borders</h3><div class="row"><label>track</label><select id="trSel">${tr.map(t => `<option value="${t.name}" ${active()?.track === t.name ? "selected" : ""}>${t.name} · x ${fmt(t.left, 0)}–${fmt(t.right, 0)} px</option>`).join("")}</select></div>
    <div class="row"><label>move left</label><input type="number" id="eL" value="0" step="1" style="width:5em"><label style="min-width:0">right</label><input type="number" id="eR" value="0" step="1" style="width:5em"><button id="eApply">Move</button></div>
    <div class="row">${tool("box", "b", "Draw box")}<span class="note">no track found? click two corners of the curve's track</span></div>
    ${KEYS("<b>L</b> add a depth label · <kbd>Delete</kbd> removes selected rows", "Moving a border re-traces every curve in that track", "Applying depths re-traces every curve; <kbd>Z</kbd> undoes", NAV)}`;
  renderLabels();
  if (ST.residuals?.length) fig(chart({title: "Depth anchors: leave-one-out residuals", series: [{pts: ST.residuals, color: TAB10[0], dots: true, r: 2.5, alpha: 1}], xlabel: "depth (ft)", ylabel: "LOO residual (ft)", w: 350, h: 170}), true);
  on("lDel", delLabels);
  on("lReset", async () => { await runJob(A("/depth"), {mode: "reset"}, "depth"); await loadLabels(); await refresh(); render(false); });
  on("lAuto", async () => { await runJob(A("/depth"), {mode: "auto", first_depth: $("fd").value, label_interval: $("li").value}, "depth"); await loadLabels(); await refresh(); render(false); });
  on("eApply", async () => { const t = $("trSel").value, c0 = CURV.curves.find(c => c.track === t);
    if (c0 && c0.id !== CURV.active) await selectCurve(c0.id, false);
    await runJob(A("/calib/edges"), {left_dx: +$("eL").value || 0, right_dx: +$("eR").value || 0}, "track borders"); GRID = await api("GET", A("/grid_lines")); LAY = await api("GET", A("/layout")); await refresh(); render(false); });
  main("Apply depths — continue", async () => {
    const edited = LBL.some(l => l.added || l.value !== l.ocr) || LBL.length !== (LAY.labels || []).length;
    if (edited) { await runJob(A("/depth"), {mode: "labels", labels: LBL.filter(l => l.value != null).map(l => [l.row, l.value])}, "depth from labels"); await refresh(); }
    else if (!c.px_per_ft) { toast("no depth calibration yet: edit/add labels (≥ 2) or type the first label + step", true); return; }
    next();
  });
  help(`Blue = track borders, thin lines = grid, red ticks = depth numbers (orange ? = could not be read). <b>L</b> add label. ${NAV}`);
};

// Curves -------------------------------------------------------------
function identLabel(c) {
  if (c.kind === "single") return "one curve (no tracks found)";
  if (c.kind === "style") return `${c.cls} line`;
  if (c.kind === "color+style") return `${c.cls.split(" ")[0]} ink, ${c.cls.split(" ").slice(1).join(" ")} line`;
  return `${c.cls} ink`;
}
function scaleForm(c, k) {               // name / unit / scale / type / backup of one curve: beside the scan and in the scale pop-ups
  const bk = c.backup || [null, null];
  return `<div class="cp-grid"><label>name</label><input id="${k}Name" value="${esc(c.name)}"><label>unit</label><input id="${k}Unit" value="${esc(c.unit || "")}">
      <label>scale</label><span class="pair"><input id="${k}L" type="number" step="any" value="${c.left}"><input id="${k}R" type="number" step="any" value="${c.right}"></span>
      <label>type</label><select id="${k}T"><option ${c.type === "linear" ? "selected" : ""}>linear</option><option ${c.type === "log" ? "selected" : ""}>log</option></select>
      <label>backup</label><span class="pair"><input id="${k}BL" type="number" step="any" value="${bk[0] ?? ""}" placeholder="–"><input id="${k}BR" type="number" step="any" value="${bk[1] ?? ""}" placeholder="–"></span></div>`;
}
function bindScaleForm(c, k) {
  $(k + "Apply").onclick = async () => {
    const bl = $(k + "BL").value, br = $(k + "BR").value;
    await api("POST", A(`/curves/${c.id}`), {name: $(k + "Name").value, unit: $(k + "Unit").value, left: +$(k + "L").value, right: +$(k + "R").value, type: $(k + "T").value,
      backup: bl !== "" && br !== "" ? [+bl, +br] : null});
    toast(`${$(k + "Name").value}: scale applied`); await loadCurves(); await refresh(); if (STEP === "curves") renderCurvesView(false);
  };
}
function renderCurvePanel() {
  const el = $("curvePanel"); if (!el || !CURV) return; const c = active(); if (!c) return;
  el.innerHTML = `<div class="cp-head"><label for="curveSel">Curve</label><select id="curveSel">${CURV.curves.map(x => `<option value="${x.id}" ${x.id === CURV.active ? "selected" : ""}>${esc(x.name)} · ${esc(x.track)}</option>`).join("")}</select></div>
    <div class="cp-id"><span class="chip" style="background:${c.color}"></span>${esc(identLabel(c))} · ${esc(c.track)}${c.coverage != null ? ` · ${Math.round(100 * c.coverage)} % of depth` : ""}</div>
    ${(c.review || []).length ? `<div class="cp-rev">check: ${esc(c.review.join(", "))}</div>` : ""}
    ${scaleForm(c, "cp")}
    <div class="cp-foot">${c.header ? `<span class="note" title="${esc(c.header)}">header: ${esc(c.header.slice(0, 34))}</span>` : `<span class="note">no header row</span>`}<button id="cpApply">Apply</button></div>`;
  $("curveSel").onchange = () => selectCurve($("curveSel").value);
  bindScaleForm(c, "cp");
}
async function selectCurve(id, highlight = true) {
  await api("POST", A("/curves/select"), {id}); await loadCurves(); await refresh();
  if (highlight) LAYERS.hl = true; loadHighlight(); applyLayers();
  if (STEP === "curves" && CVIEW === "curves") renderCurveList(); if (STEP === "trace") render(false);
}
function renderCurveList() {
  const el = $("curveList"); if (!el || !CURV) return;
  el.innerHTML = CURV.curves.map(c => `<div class="citem ${c.id === CURV.active ? "on" : ""}" data-id="${c.id}"><span class="chip" style="background:${c.color}"></span><b>${esc(c.name)}</b>
    <span class="muted">${esc(c.track)} · ${esc(identLabel(c))} · ${fmt(c.left, c.type === "log" ? 1 : 0)}–${fmt(c.right, c.type === "log" ? 0 : 0)}${c.type === "log" ? " log" : ""}${c.unit ? " " + esc(c.unit) : ""}</span>${(c.review || []).length ? ' <span class="warnc">●</span>' : ""}</div>`).join("");
  el.querySelectorAll(".citem").forEach(d => d.onclick = () => selectCurve(d.dataset.id));
}
let PAL = {palette: {}, curve_color: null};
function renderPalette() {
  if (!$("palTbl")) return;
  const names = Object.keys(PAL.palette || {}), css = rgb => `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  $("palTbl").innerHTML = names.map(n => { const rgb = PAL.palette[n], lk = (n === "white" || n === "black");
    return `<tr data-n="${esc(n)}"><td><span class="pal-swatch" style="background:${css(rgb)}"></span></td><td>${lk ? `<b>${esc(n)}</b>` : `<input data-k="name" value="${esc(n)}">`}</td>
      <td><input data-k="rgb" value="${rgb.join(", ")}" ${lk ? "readonly" : ""} style="width:110px"></td><td>${lk ? "" : `<button data-del="${esc(n)}" class="danger">✕</button>`}</td></tr>`; }).join("");
  $("palTbl").querySelectorAll("button[data-del]").forEach(b => b.onclick = async () => { delete PAL.palette[b.dataset.del]; await api("POST", A("/color"), {palette: PAL.palette}); renderPalette(); });
}
function readPalette() {
  const pal = {};
  document.querySelectorAll("#palTbl tr").forEach(tr => { const old = tr.dataset.n, ni = tr.querySelector("input[data-k=name]"), ri = tr.querySelector("input[data-k=rgb]");
    const name = ((ni ? ni.value : old) || "").trim(), rgb = (ri ? ri.value : "").split(/[,\s]+/).filter(Boolean).map(Number);
    if (name && rgb.length === 3 && rgb.every(n => Number.isFinite(n) && n >= 0 && n <= 255)) pal[name] = rgb.map(n => n | 0); });
  pal.white = pal.white || PAL.palette.white || [255, 255, 255]; pal.black = pal.black || PAL.palette.black || [0, 0, 0]; return pal;
}
function popCurves() {                   // up to three scales to show as pop-ups: GR, a resistivity, SP (else the first matched)
  const mt = (CURV?.curves || []).filter(c => (HLN?.lines || []).some(L => L.curve === c.id)), out = [];
  for (const re of [/^(GR|CGR|SGR)/i, /^(ILD|RILD|LLD|RT|RES|ILM|RILM|SFL|RLL)/i, /^SP/i]) { const c = mt.find(c => re.test(c.name) && !out.includes(c)); if (c) out.push(c); }
  for (const c of mt) if (out.length < 3 && !out.includes(c)) out.push(c);
  return out;
}
function hdrSummary() {
  if (!HLN || !HLN.n_words) return `<span class="note">No header words were read${INFO.features.ocr.ok ? "" : " (OCR is not installed)"}.</span>`;
  const nm = new Set(HLN.lines.map(L => L.curve).filter(Boolean)).size, mc = HLN.mean_conf;
  return `<div class="hdr-meter"><i style="width:${Math.round(mc)}%;background:${mc >= 90 ? "#16a34a" : mc >= 70 ? "#d97706" : "#dc2626"}"></i></div>
    <b>${HLN.n_words}</b> words read · mean confidence <b>${Math.round(mc)} %</b> · ${HLN.n_low ? `<span class="warnc">${HLN.n_low} below 80 %</span>` : "none below 80 %"} · ${nm} of ${CURV.curves.length} curves matched to a header line`;
}
async function loadHeader() {
  try { HDR = await api("GET", A("/header")); HW = (await api("GET", A("/header/words"))).words; HLN = await api("GET", A("/header/lines")); }
  catch (e) { HW = []; HLN = null; }
}
function renderHdrTbl() {
  const el = $("hdrTbl"); if (!el || !HLN) return;
  el.innerHTML = `<tr><th></th><th style="width:58px">curve</th><th>as read (type to correct)</th><th style="width:40px">conf</th></tr>` + HLN.lines.map((L, k) => {
    const cu = CURV.curves.find(c => c.id === L.curve), cl = L.conf >= 90 ? "" : L.conf >= 70 ? "mid" : "low";
    return `<tr><td><span class="chip" style="background:${cu ? cu.color : "#e2e8f0"}"></span></td><td>${cu ? `<b>${esc(cu.name)}</b>` : '<span class="muted">–</span>'}</td>
      <td><input class="ocr ${cl}" data-k="${k}" value="${esc(L.text)}"></td><td class="${cl ? "warnc" : "okc"}">${Math.round(L.conf)} %</td></tr>`; }).join("");
  el.querySelectorAll("input[data-k]").forEach(inp => inp.onchange = async () => {
    const L = HLN.lines[+inp.dataset.k], parts = inp.value.trim().split(/\s+/).filter(Boolean);
    const edits = L.words.map((i, j) => ({i, text: parts.length === L.words.length ? parts[j] : (j === 0 ? inp.value.trim() : "")}));
    await api("POST", A("/header/words"), {track: L.track, edits}); await loadHeader();
    toast("header line corrected: apply the curve's name or scale in its pop-up"); renderCurvesView(false); });
}
function renderPops() {
  const el = $("scalePops"); el.innerHTML = popCurves().map((c, i) => { const L = HLN.lines.find(q => q.curve === c.id);
    return `<div class="pop" data-id="${c.id}" data-color="${c.color}" style="border-top-color:${c.color}"><div class="pop-h"><span class="chip" style="background:${c.color}"></span><b>${esc(c.name)}</b>
      <span class="note">${esc(c.track)} · read ${Math.round(L.conf)} %</span></div>${scaleForm(c, "pp" + i)}<div class="cp-foot"><span class="note" title="${esc(L.text)}">“${esc(L.text.slice(0, 30))}”</span><button id="pp${i}Apply">Apply</button></div></div>`; }).join("");
  popCurves().forEach((c, i) => bindScaleForm(c, "pp" + i));
}
function renderCurvesView(refit = true) {
  const isCol = !!ST.is_color, f = INFO.features.ocr;
  $("curvePanel").hidden = CVIEW !== "curves"; $("scalePops").innerHTML = "";
  document.querySelectorAll(".seg button").forEach(b => b.classList.toggle("on", b.dataset.v === CVIEW));
  if (CVIEW === "curves") {
    $("cvBody").innerHTML = `<div class="note">Each curve gets its own working image and is traced the same way. Orange dot = name or scale guessed: please check.</div>
      <div class="list clist" id="curveList"></div>
      <h3>Header OCR</h3><div class="card" id="hdrSum">${hdrSummary()}</div><div class="row"><button id="toHdr">Check the header and scales</button></div>
      ${KEYS("Pick a curve (list or the dropdown beside the scan): its ink is shown dark on the grey scan", "Edit name, scale, unit or backup scale beside the scan, then Apply (values re-map, no re-trace)", "Layers (top left): scan, selected curve ink, tracks and grid, traced curves", NAV)}
      ${isCol ? `<details class="more"><summary>Ink colours</summary><table class="t fit" id="palTbl"></table><div class="row">${tool("sample", "c", "Pick colour")}<button id="palApply">Identify again with these colours</button></div></details>` : ""}`;
    renderCurveList(); if (isCol) renderPalette();
    on("toHdr", async () => { CVIEW = "header"; renderCurvesView(); });
    on("palApply", async () => { const pal = readPalette(); await runJob(A("/color"), {palette: pal, curve_color: active()?.cls?.split(" ")[0] in pal ? active().cls.split(" ")[0] : undefined, confirm: true}, "identify curves");
      await loadCurves(); await refresh(); loadHighlight(); render(false); });
  } else {
    $("cvBody").innerHTML = `<div class="note">The header strip as OCR read it; the log body is covered. Each printed line is boxed (in its curve's colour when it names a curve) with its OCR confidence as a bar: green ≥ 90 %, orange ≥ 70 %, red below. Type over a line to correct it; set a curve's scale in its pop-up.</div>
      <div class="card" id="hdrSum">${hdrSummary()}</div>
      <div class="list" style="max-height:330px"><table class="t fit hdr" id="hdrTbl"></table></div>
      <div class="row">${tool("word", "e", "Edit one word on the scan")}<button id="hRerun" ${off(f)}>Re-run header OCR</button></div>${ocrLine()}`;
    renderHdrTbl(); if (HLN) renderPops();
    on("hRerun", async () => { await runJob(A("/header/rerun"), {}, "header OCR"); await loadHeader(); toast("header read again"); renderCurvesView(false); });
  }
  bindTools(); if (CVIEW === "header") setMode("word"); else setMode("pan"); if (refit) focusStep(); draw();
}
R.curves = async () => {
  $("stepLead").textContent = CURV?.kind === "style" ? "Every curve was told apart by its line style (solid, dashed, dotted; thin or thick). The scan is shown in grey and the traces in colour; pick a curve to see its ink."
    : CURV?.kind === "color" ? "Every curve was told apart by its ink colour (and line style where one colour holds two curves). The scan is shown in grey and the traces in colour; pick a curve to see its ink."
    : "One curve on this scan. Its name and scales sit beside the scan.";
  $("stepBody").innerHTML = `<div class="seg"><button data-v="curves" id="segCurves">Curves</button><button data-v="header" id="segHeader">Header and scales</button></div><div id="cvBody"></div>`;
  await loadCurves(); await loadHeader();
  document.querySelectorAll(".seg button").forEach(b => b.onclick = () => { CVIEW = b.dataset.v; renderCurvesView(); });
  renderCurvesView(false);
  main("Curves look right — continue", async () => { CVIEW = "curves"; next(); });
  help(`Grey scan, coloured traces. Pick a curve to show its ink; <b>Header and scales</b> shows what OCR read. ${NAV}`);
};

// Trace and fix ------------------------------------------------------
let UPOS = -1;
function nextUncertain() {
  const u = CHK?.disagree || [];
  if (!u.length) { if ($("uPos")) $("uPos").textContent = "no uncertain spots left"; return; }
  UPOS = (UPOS + 1) % u.length; const [a, b] = u[UPOS], m = (a + b) / 2, h = Math.max(ST.width * 0.3, (b - a) / 2 + 100);
  fitRows(m - h, m + h); $("uPos").textContent = `${UPOS + 1} of ${u.length}`;
}
function renderAnchors() {
  if (!$("ancList")) return;
  $("ancList").innerHTML = (OV?.anchors || []).map((a, i) => `<div>Pin ${i + 1} · row ${a[0].toFixed(0)}<span><a data-go="${a[0]}">go</a><a data-del="${i}" title="remove this pin">✕</a></span></div>`).join("")
    + (OV?.strokes || []).map((st, i) => `<div>Guide ${i + 1} · rows ${st[0][0].toFixed(0)}–${st[st.length - 1][0].toFixed(0)}<span><a data-go="${st[0][0]}">go</a><a data-dels="${i}" title="remove this guide">✕</a></span></div>`).join("")
    || '<span class="note">no pins yet</span>';
  $("ancList").querySelectorAll("[data-go]").forEach(x => x.onclick = () => fitRows(+x.dataset.go - ST.width * 0.3, +x.dataset.go + ST.width * 0.3));
  $("ancList").querySelectorAll("[data-del]").forEach(x => x.onclick = () => fixCall("/fix/unpin", {index: +x.dataset.del}));
  $("ancList").querySelectorAll("[data-dels]").forEach(x => x.onclick = () => fixCall("/fix/unstroke", {index: +x.dataset.dels}));
  $("trInfo").innerHTML = ST.traced ? `<b>${(CHK?.disagree || []).length}</b> place(s) where the line could go another way (yellow) · ${ST.n_anchors} pin(s) · ${OV?.strokes?.length || 0} guide(s)` +
    (ST.qc ? `<br>vs reference LAS: r = <b>${fmt(ST.qc.r, 3)}</b>, RMSE ${fmt(ST.qc.rmse, 2)}` : "") : "not traced";
}
function renderCands() {
  const el = $("candList"); if (!el) return; const cs = OV?.candidates || [];
  el.innerHTML = cs.length ? cs.map(c => `<div class="item"><span style="color:${CAND_COL[c.index % CAND_COL.length]}">■</span> Line ${c.index + 1}${c.picked ? " <b>(digitized)</b>" : ""} <span class="muted">score ${fmt(c.score, 2)}</span>
    ${c.picked ? "" : `<button data-pick="${c.index}">Use this line</button>`}</div>`).join("") : '<span class="muted">one line traced</span>';
  el.querySelectorAll("[data-pick]").forEach(b => b.onclick = async () => { try { await runJob(A("/trace/pick"), {index: +b.dataset.pick}, "pick line"); await refresh(); CHK = null; loadChecks(); renderCands(); } catch (e) {} });
}
async function answer(choice) {
  if (!CUR) return; const row = CUR.row;
  await fixCall("/checks/answer", {id: CUR.id, choice});
  const sp = CHK?.spots || []; CPOS = Math.max(0, sp.findIndex(s => s.row > row)); showSpot();
}
function showSpot(go = true) {
  const sp = CHK?.spots || [], el = $("chkCard"); if (!el) return;
  CUR = sp.length ? sp[Math.min(CPOS, sp.length - 1)] : null;
  if (!CUR) { el.innerHTML = `<div class="okc"><b>No tricky spots left.</b></div><div class="note">${(CHK?.decisions || []).length} answered.</div>`; draw(); return; }
  const cols = ["#2563eb", "#f97316", "#16a34a"];
  el.innerHTML = `<div class="note">Spot ${Math.min(CPOS, sp.length - 1) + 1} of ${sp.length} · ${CUR.kind === "wrap" ? "track edge" : CUR.kind === "hidden" ? "hidden under another curve (null)" : "two lines"} near row ${CUR.row.toFixed(0)}</div>
    <div class="q"><b>${esc(CUR.question)}</b></div>
    ${CUR.choices.map((ch, i) => `<div class="row"><button data-ch="${ch.id}" style="width:100%;border-left:6px solid ${cols[i % 3]}">${i + 1} · ${esc(ch.label)}</button></div>`).join("")}
    <div class="row"><button data-ch="skip">Leave it <span class="kbd">K</span></button><button id="cPrev">◀</button><button id="cNext">▶</button></div>`;
  el.querySelectorAll("[data-ch]").forEach(b => b.onclick = () => answer(b.dataset.ch));
  on("cPrev", () => { CPOS = (CPOS - 1 + sp.length) % sp.length; showSpot(); }); on("cNext", () => { CPOS = (CPOS + 1) % sp.length; showSpot(); });
  if (go) { const h = Math.max(ST.width * 0.3, (CUR.row1 - CUR.row0) / 2 + 150); fitRows(CUR.row - h, CUR.row + h); } draw();
}
R.trace = () => {
  const ml = INFO.features.ml, c = active();
  $("stepLead").textContent = "Click on the real curve wherever its line is wrong: the line is redrawn through your click. Tricky spots are listed below.";
  $("stepBody").innerHTML = `<div class="row"><label>curve</label><select id="trCurve" style="flex:1">${CURV.curves.map(x => `<option value="${x.id}" ${x.id === CURV.active ? "selected" : ""}>${esc(x.name)} · ${esc(x.track)}</option>`).join("")}</select></div>
    <div class="card" id="trInfo"></div>
    <div class="tools">${tool("fix", "f", "Fix: click the curve")}${tool("guide", "g", "Guide: drag along it")}${tool("circle", "c", "Circle: re-trace inside")}${tool("node", "d", "Node: drag / force-click")}</div>
    <div class="row"><label class="muted">Circle radius (px) <input id="circR" type="number" value="60" min="10" max="600" step="10" style="width:5em"></label><button id="tNextU">Next place to look <span class="kbd">N</span></button><span class="note" id="uPos"></span></div>
    <h3>Tricky spots</h3><div class="card" id="chkCard">looking for crossings and edges…</div>
    <h3>How much did that click change?</h3><div class="card" id="prog7"><span class="note">Each click is a pin. When clicks stop changing much, you are done.</span></div>
    <h3>Pins and guides</h3><div class="list" id="ancList"></div>
    <details class="more"><summary>More</summary>
    <h3>Cost map</h3><div class="radio"><label><input type="radio" name="cm" value="conventional" checked> conventional</label>
      <label class="${ml.ok ? "" : "off"}"><input type="radio" name="cm" value="ml" ${ml.ok ? "" : "disabled"}> ML U-Net</label>
      <label class="${ml.ok ? "" : "off"}"><input type="radio" name="cm" value="blend" ${ml.ok ? "" : "disabled"}> blend (conventional + U-Net)</label>${hint(ml)}</div>
    <div class="row"><label><input type="checkbox" id="tMask"> ignore text and stamps</label></div>
    <h3>Lines in this working image</h3><div class="row"><label class="muted">Trace <select id="tNC"><option value="1">1 line</option><option value="2">2 lines</option><option value="3">3 lines</option></select></label></div><div class="list" id="candList"></div>
    <h3>Advanced tools</h3><div class="tools">${tool("exclude", "x", "Not this curve")}${tool("wrap", "w", "Backup interval")}${tool("seed", "s", "Seed trace")}</div>
    <div class="row"><button id="tRetrace">Re-trace all</button><button id="tClrA">Clear pins</button><button id="tClrX">Clear exclusions</button><button id="tClrW">Clear intervals</button><button id="tReset" class="danger">Reset trace</button></div>
    </details>
    ${KEYS("<b>F</b> fix · <b>G</b> guide · drag a green pin to move it", "<b>N</b> next place to look · <b>1</b>/<b>2</b> answer a tricky spot · <b>K</b> leave it", "<b>Z</b> undo · <b>Esc</b> pan · orange = off the track edge")}`;
  $("trCurve").onchange = () => selectCurve($("trCurve").value, false);
  CPOS = 0; renderAnchors(); renderProgress(); loadChecks();
  setTimeout(() => setMode("fix"), 0);
  on("tNextU", nextUncertain);
  const retrace = async b => { try { await runJob(A("/trace"), Object.assign({mode: document.querySelector("input[name=cm]:checked")?.value || "conventional", force: true}, b), "trace"); await refresh(); CHK = null; loadChecks(); renderCands(); } catch (e) {} };
  document.querySelectorAll("input[name=cm]").forEach(r => r.onchange = () => retrace({}));
  api("GET", A("/trace/options")).then(o => { if ($("tMask")) $("tMask").checked = !!o.mask_other; if ($("tNC")) $("tNC").value = String(o.n_curves || 1); renderCands(); }).catch(() => {});
  $("tNC").onchange = () => retrace({n_curves: +$("tNC").value}); $("tMask").onchange = () => retrace({mask_other: $("tMask").checked});
  on("tRetrace", async () => { await runJob(A("/retrace_all"), {}, "re-trace"); await refresh(); loadChecks(); });
  on("tClrA", async () => { await api("POST", E("/anchors/clear")); await refresh(); loadChecks(); });
  on("tClrX", async () => { await api("POST", E("/exclude/clear")); await refresh(); loadChecks(); });
  on("tClrW", async () => { await api("POST", E("/backup_interval/clear")); await refresh(); renderAnchors(); });
  on("tReset", async () => { if (!confirm("Reset to a fresh trace (pins, guides, exclusions and intervals removed)?")) return; await runJob(A("/trace/reset"), {}, "reset trace"); await refresh(); loadChecks(); });
  main("Traces look good — continue", async () => next());
  help(`<b>F</b> fix · <b>G</b> guide · <b>N</b> next place · <b>Z</b> undo · ${NAV}`);
};

// Check and save -----------------------------------------------------
R.save = async () => {
  const f = INFO.features, x = ST.export;
  $("stepLead").textContent = "The scan (left) next to every digitized log (right), row for row. Choose what to plot, then write the figures, the optional CSV and the LAS file.";
  LOGS = await api("GET", A("/logs"));
  const ids = LOGS.curves.map(q => q.id); LOGSEL = LOGSEL ? new Set(ids.filter(i => LOGSEL.has(i))) : new Set(ids); if (!LOGSEL.size) LOGSEL = new Set(ids);
  $("stepBody").innerHTML = (VEC() ? `<div class="card vnote"><b>${VNOTE}:</b> every curve was read from its path in the PDF (page ${(ST.vector?.page ?? 0) + 1}).</div>` : "") + `<h3>Plot</h3>
    <details class="dd" id="logDD"><summary id="logSum"></summary><div class="ddl">${LOGS.curves.map(q => `<label><input type="checkbox" data-log="${q.id}" ${LOGSEL.has(q.id) ? "checked" : ""}><span class="chip" style="background:${q.color}"></span>${esc(q.name)} <span class="muted">${esc(q.track || "")}</span></label>`).join("")}
      <div class="row"><button id="logAll">All</button><button id="logNone">None</button></div></div></details>
    <div class="row"><label style="min-width:0"><input type="checkbox" id="showImg" ${SHOWIMG ? "checked" : ""}> show the original image</label></div>
    <h3>Compare with the reference LAS</h3><div id="cmp"><span class="note">…</span></div>
    <div class="row"><label class="filebtn">load a reference LAS <input type="file" id="refPick" accept=".las"></label></div>
    <h3>Outputs</h3><div class="note">Nothing is written until you press a button.</div>
    <div class="og"><div class="og-h">Figures</div><div class="row"><button data-x="figure" ${off(f.plot)}>Image + logs PNG</button><button data-x="qc_png" ${!f.plot.ok ? off(f.plot) : (!ST.las ? 'disabled title="no reference LAS"' : "")}>QC plot PNG (selected curve)</button></div></div>
    <div class="og"><div class="og-h">Log data (optional)</div><div class="row"><button data-x="logs_csv">CSV of the plotted logs</button></div></div>
    <div class="og"><div class="og-h">LAS file</div><div class="row"><label>folder</label><input id="xDir" value="${esc(x.folder)}" style="flex:1"><button id="xSet">Set</button></div>
      <div class="row"><label>depth step</label><input type="number" id="xStep" min="0" step="any" value="${esc(LASSTEP)}" style="width:6em"> ft<span class="note" id="xStepMsg"></span></div>
      <div class="note">One LAS with every curve (values on the step grid, no smoothing): press <b>Save LAS</b> above.</div></div>
    <details class="more"><summary>More outputs</summary><div class="row"><button data-x="csv">CSV (selected curve)</button><button data-x="session">Session JSON</button><button data-x="training" title="image + corrected trace to training_data/corrections">Training sample</button></div></details>
    <div class="row"><button id="xOpen" ${x.exists ? "" : 'disabled title="created by the first save"'}>Open folder</button></div>
    <h3>Written this session</h3><div class="list" id="xList">${x.written.length ? x.written.map(p => `<div>${esc(p.split(/[\\/]/).pop())}</div>`).join("") : '<span class="note">nothing yet</span>'}</div>`;
  const sum = () => { $("logSum").textContent = `Logs in the plot: ${LOGSEL.size === ids.length ? "all" : LOGSEL.size} of ${ids.length}`; drawLogs(); };
  document.querySelectorAll("input[data-log]").forEach(cb => cb.onchange = () => { cb.checked ? LOGSEL.add(cb.dataset.log) : LOGSEL.delete(cb.dataset.log); sum(); });
  const all = v => { document.querySelectorAll("input[data-log]").forEach(cb => { cb.checked = v; }); LOGSEL = new Set(v ? ids : []); sum(); };
  on("logAll", () => all(true)); on("logNone", () => all(false)); sum();
  $("showImg").onchange = () => { SHOWIMG = $("showImg").checked; applyLayers(); };
  const stepOk = () => { const v = +$("xStep").value, ok = Number.isFinite(v) && v > 0;   // any typed step > 0 ft
    if ($("xStepMsg")) $("xStepMsg").textContent = ok ? "" : " enter a step greater than 0";
    if (!ok) { toast("LAS step must be a number of feet greater than 0", true); throw new Error("bad step"); } return v; };
  $("xStep").oninput = () => { LASSTEP = $("xStep").value; const v = +LASSTEP; $("xStepMsg").textContent = Number.isFinite(v) && v > 0 ? "" : " enter a step greater than 0"; };
  const body = () => ({step: stepOk(), ids: [...LOGSEL], image: SHOWIMG});
  const after = async r => { toast("written: " + r.written.map(p => p.split(/[\\/]/).pop()).join(", ")); await refresh(false); render(false); };
  // the box shows the short folder label; only a typed change is posted (the label is never sent back as a path)
  on("xSet", async () => { if ($("xDir").value === x.folder) return; await api("POST", A("/export/folder"), {path: $("xDir").value}); await refresh(false); render(false); });
  document.querySelectorAll("button[data-x]").forEach(b => b.onclick = async () => { try { await after(await api("POST", A("/export/" + b.dataset.x), body())); } catch (e) {} });
  on("xOpen", async () => api("POST", "/shell/api/open_folder", {path: ST.export.folder_abs}));
  $("refPick").onchange = async () => { const fp = $("refPick").files[0]; if (!fp) return; await api("POST", A("/reference"), {las: await upload(fp)}); await refresh(false); render(false); };
  main("Save LAS", async () => { if ($("xDir").value !== x.folder) await api("POST", A("/export/folder"), {path: $("xDir").value});
    await after(await api("POST", A("/export/las"), {step: stepOk()})); Shell.complete(4, false); });
  help(`Left: the scan as printed. Right: the digitized logs at the same depth (pan and zoom either side). ${NAV}`);
  const q = await api("GET", A("/compare"));
  if (!q.available) { $("cmp").innerHTML = `<div class="card">No reference LAS loaded: you can still save.</div>`; return; }
  $("cmp").innerHTML = `<div class="note">${esc(q.las)}</div><table class="t fit cmp"><tr><th></th><th>curve</th><th>LAS</th><th>r</th><th>RMSE</th><th>n</th></tr>` + q.rows.map(r =>
    `<tr data-id="${r.id}" class="${r.id === q.active ? "sel" : ""}"><td><span class="chip" style="background:${r.color}"></span></td><td>${esc(r.name)}</td><td>${esc(r.ref || "–")}</td><td><b>${fmt(r.r, 3)}</b></td><td>${r.r == null ? "–" : fmt(r.rmse, r.log ? 3 : 2) + (r.log ? " log" : "")}</td><td>${r.n ?? "–"}</td></tr>`).join("") + "</table>";
  $("cmp").querySelectorAll("tr[data-id]").forEach(tr => tr.onclick = async () => { await selectCurve(tr.dataset.id, false); render(false); });
};

(async () => { INFO = await api("GET", "/app/api/info"); Shell.go(0); })();
