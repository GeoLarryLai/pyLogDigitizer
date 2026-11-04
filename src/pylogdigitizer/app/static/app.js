// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
// pyLogDigitizer app: the nine steps of notebook 01. Viewer coordinates = sheet pixels of the
// deskewed working sheet. Editing actions call the editor API (/core/api/sessions/..);
// long tasks are app jobs (/app/api/..) with n / N progress. Files are written only in step 9.
"use strict";
const {$, api, runJob, toast, help} = Shell;
const TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];
const STEPS = [
  {id: "load", title: "Open a scan"}, {id: "dsection", title: "Which depth section?", hidden: true}, {id: "deskew", title: "Check it's straight"}, {id: "section", title: "Check the tracks"},
  {id: "header", title: "Check the curve names"}, {id: "depth", title: "Check the depth numbers"}, {id: "scale", title: "Check the scale"},
  {id: "trace", title: "Fix the trace"}, {id: "checks", title: "Confirm the tricky spots"}, {id: "qc", title: "Compare with reference"}, {id: "export", title: "Save LAS file"}];
let CHK = null, CUR = null, CPOS = 0, STROKE = [], DRAG = null, PROG = null;
let INFO = null, SID = null, ST = null, LAY = null, OV = null, HDR = null, viewer = null, mode = "pan", pts = [], STEP = "load";
let LBL = [], LBLsel = new Set(), HSEL = new Set();
const fmt = (v, d = 1) => v == null || !isFinite(v) ? "–" : (+v).toFixed(d);
const E = p => `/core/api/sessions/${SID}${p}`, A = p => `/app/api/${SID}${p}`;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const off = f => f && !f.ok ? `disabled title="${esc(f.hint)}"` : "";
const hint = f => f && !f.ok ? `<div class="hint">${esc(f.hint)}</div>` : "";
function engineSel() {                      // OCR status line: RapidOCR, or the install hint when it is missing
  const o = INFO.features.ocr;
  return o.ok ? `<div class="hint">OCR: RapidOCR</div>` : `<div class="hint">OCR is not installed - ${esc(o.hint || "")}. Type the values by hand.</div>`;
}
function bindEngine() {}
const KEYS = (...r) => `<div class="controls">${r.map(x => `<div>${x}</div>`).join("")}</div>`;
const NAV = "<kbd>drag</kbd> pan · <kbd>wheel</kbd> zoom · <kbd>double-click</kbd> zoom in · <kbd>Esc</kbd> back to pan";

// ------------------------------------------------------------------ state + viewer
async function refresh(over = true) {
  ST = await api("GET", A("/state")); $("undoBtn").disabled = !ST.undo.length;
  $("undoBtn").title = ST.undo.length ? `Undo: ${ST.undo[ST.undo.length - 1]} (Z)` : "nothing to undo";
  $("doc").textContent = `${ST.well} · ${ST.scan.split("/").pop()} · ${ST.curve}` + (ST.depth_top_base ? ` · ${fmt(ST.depth_top_base[0], 0)}–${fmt(ST.depth_top_base[1], 0)} ft` : "");
  if (over) { OV = await api("GET", E("/overlay")); draw(); }
}
function openViewer() {
  if (viewer) viewer.destroy(); $("empty").hidden = true;
  viewer = OpenSeadragon({id: "viewer", tileSources: E("/image.dzi"), showNavigationControl: false, prefixUrl: "",
    gestureSettingsMouse: {clickToZoom: false, dblClickToZoom: true}, maxZoomPixelRatio: 4, minZoomImageRatio: 0.3, visibilityRatio: 0.2});
  viewer.addHandler("canvas-click", onClick); viewer.addHandler("canvas-press", onPress); viewer.addHandler("canvas-drag", onDrag); viewer.addHandler("canvas-release", onRelease);
  new OpenSeadragon.MouseTracker({element: viewer.canvas, moveHandler: e => { if (mode !== "circle") return; const ip = viewer.viewport.viewerElementToImageCoordinates(e.position); HOVER = {row: ip.y, x: ip.x}; drawSoon(); },
    scrollHandler: e => { if (mode !== "circle" || !e.originalEvent.shiftKey) return; const el = $("circR"); if (!el) return; el.value = Math.max(10, Math.min(600, +el.value * (e.scroll > 0 ? 1.15 : 1 / 1.15))) | 0; e.preventDefault = true; drawSoon(); }}); for (const ev of ["update-viewport", "animation", "animation-finish", "resize", "tile-drawn"]) viewer.addHandler(ev, drawSoon);
  return new Promise(r => viewer.addOnceHandler("open", () => { focusStep(); r(); }));
}
function fitRows(r0, r1, x0 = 0, x1 = ST.width) {
  if (viewer) viewer.viewport.fitBounds(viewer.viewport.imageToViewportRectangle(x0, r0, x1 - x0, Math.max(50, r1 - r0)), true);
}
function focusStep() {
  if (!viewer || !ST || STEP === "checks") return;
  const sec = ST.section, w = ST.width;
  if (STEP === "deskew" || STEP === "dsection") viewer.viewport.goHome(true);
  else if (STEP === "header") fitRows(0, Math.max(LAY?.probe_row || 1500, 800) + 200);
  else if (STEP === "depth" && LAY?.labels?.length) { const r = LAY.labels[0][0]; fitRows(r - 300, r + 1.4 * w); }
  else if (sec) fitRows(sec.rows[0], sec.rows[0] + 1.3 * w);
  else viewer.viewport.goHome(true);
}
let _drawReq = 0;
function drawSoon() { if (!_drawReq) _drawReq = requestAnimationFrame(() => { _drawReq = 0; draw(); }); }
let HOVER = null, HW = [];
function drawWords(c, P) {
  if (STEP !== "header") return; c.lineWidth = 1.5;
  HW.forEach(w => { const a = P(w.x0, w.y0), b = P(w.x1, w.y1);
    c.strokeStyle = w.corrected ? "#16a34a" : (w.conf < 60 ? "#d97706" : "rgba(37,99,235,.7)"); c.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y); });
}
function drawCircle(c, P) {
  if (mode !== "circle" || !HOVER) return; const r = +($("circR")?.value || 60), a = P(HOVER.x, HOVER.row), b = P(HOVER.x + r, HOVER.row);
  c.strokeStyle = "rgba(217,119,6,.9)"; c.lineWidth = 2; c.setLineDash([5, 4]); c.beginPath(); c.arc(a.x, a.y, Math.abs(b.x - a.x), 0, 2 * Math.PI); c.stroke(); c.setLineDash([]);
}
function draw() {
  const cv = $("ov"), c = cv.getContext("2d"); cv.width = cv.clientWidth; cv.height = cv.clientHeight; c.clearRect(0, 0, cv.width, cv.height);
  if (!viewer || !viewer.world.getItemAt(0) || !ST) return;
  const P = (x, y) => viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(x, y));
  const top = viewer.viewport.viewerElementToImageCoordinates(new OpenSeadragon.Point(0, 0)).y;
  const bot = viewer.viewport.viewerElementToImageCoordinates(new OpenSeadragon.Point(0, cv.height)).y;
  const vis = r => r > top - 200 && r < bot + 200;
  if (["section", "deskew", "header"].includes(STEP) && LAY?.available) {     // followed boundaries (notebook colours)
    Object.values(LAY.edges).forEach((pl, j) => { c.strokeStyle = TAB10[j % 10]; c.lineWidth = 1.5; c.beginPath();
      pl.forEach(([r, x], i) => { const p = P(x, r); i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y); }); c.stroke(); });
  }
  drawWords(c, P);
  if (["section", "depth"].includes(STEP)) {                                    // depth labels: red ticks
    c.font = "12px -apple-system,sans-serif";
    LBL.forEach((l, i) => { if (!vis(l.row)) return; const a = P(ST.width * 0.4, l.row), b = P(ST.width * 0.6, l.row);
      c.strokeStyle = LBLsel.has(i) ? "#2563eb" : "#d62728"; c.lineWidth = 3; c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
      if (STEP === "depth") { c.fillStyle = l.value == null ? "#d97706" : "#d62728"; c.fillText(l.value == null ? "?" : l.value, b.x + 6, b.y + 4); } });
  }
  if (STEP === "dsection") {                                                   // log sections: numbered bands to click
    (ST.depth_sections || []).forEach((d, i) => { const a = P(0, d.row0), b = P(ST.width, d.row1), cur = ST.depth_section === d.index;
      c.fillStyle = cur ? "rgba(37,99,235,.22)" : `${TAB10[i % 10]}33`; c.fillRect(a.x, a.y, b.x - a.x, b.y - a.y);
      c.strokeStyle = cur ? "#2563eb" : TAB10[i % 10]; c.lineWidth = cur ? 3 : 2; c.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
      c.font = "bold 16px -apple-system,sans-serif"; c.fillStyle = c.strokeStyle; c.fillText(`Section ${i + 1}: ${fmt(d.top, 0)}–${fmt(d.base, 0)} ft`, a.x + 8, Math.max(a.y, 0) + 20); });
    return;
  }
  drawCircle(c, P);
  const sec = OV?.section;
  if (sec) { c.strokeStyle = "rgba(37,99,235,.8)"; c.lineWidth = 1.5; c.setLineDash([6, 4]); const a = P(0, sec.rows[0]), b = P(0, sec.rows[1]);
    c.beginPath(); c.moveTo(0, a.y); c.lineTo(cv.width, a.y); c.moveTo(0, b.y); c.lineTo(cv.width, b.y); c.stroke();
    if (sec.kind === "box") { const p0 = P(sec.x0, sec.rows[0]), p1 = P(sec.x1, sec.rows[1]); c.strokeRect(p0.x, p0.y, p1.x - p0.x, p1.y - p0.y); }
    c.setLineDash([]); }
  if (["scale", "trace", "checks", "qc", "export"].includes(STEP) && OV) {     // trace: red, off-scale orange, "could go another way" yellow halo
    const dis = (STEP === "trace" || STEP === "checks") ? (CHK?.disagree || []) : [], inDis = r => dis.some(([a, b]) => r >= a && r <= b);
    if (["trace", "checks", "qc", "export"].includes(STEP)) for (const cd of (OV.candidates || [])) { if (cd.picked) continue;    // other curves of the track: own colour, dashed
      c.strokeStyle = CAND_COL[cd.index % CAND_COL.length]; c.globalAlpha = 0.7; c.lineWidth = 2; c.setLineDash([7, 5]); c.beginPath(); let f = true;
      for (const [r, x] of cd.trace) { if (!vis(r)) { f = true; continue; } const p = P(x, r); f ? c.moveTo(p.x, p.y) : c.lineTo(p.x, p.y); f = false; }
      c.stroke(); c.setLineDash([]); c.globalAlpha = 1; }
    let prev = null;
    for (const [r, x, o] of OV.trace) { if (!vis(r)) { prev = null; continue; } const p = P(x, r);
      if (prev) { if (inDis(r)) { c.strokeStyle = "rgba(250,204,21,.55)"; c.lineWidth = 9; c.beginPath(); c.moveTo(prev.x, prev.y); c.lineTo(p.x, p.y); c.stroke(); }
        c.lineWidth = 2; c.strokeStyle = o ? "rgba(255,140,0,.95)" : "rgba(214,39,40,.9)"; c.beginPath(); c.moveTo(prev.x, prev.y); c.lineTo(p.x, p.y); c.stroke(); }
      prev = p; }
    c.strokeStyle = "rgba(37,99,235,.85)"; c.lineWidth = 5; c.lineCap = "round";
    for (const st of (OV.strokes || []).concat(STROKE.length ? [STROKE] : [])) { c.beginPath(); st.forEach(([r, x], i) => { const p = P(x, r); i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y); }); c.stroke(); }
    c.lineCap = "butt";
    if (STEP === "checks" && CUR) { const sp = CUR, cols = ["#2563eb", "#f97316", "#16a34a"];
      const a = P(0, sp.row0), b = P(0, sp.row1); c.fillStyle = "rgba(250,204,21,.12)"; c.fillRect(0, a.y, cv.width, b.y - a.y);
      sp.choices.forEach((ch, i) => { const p = P(ch.x, sp.row); c.strokeStyle = cols[i % 3]; c.lineWidth = 4; c.beginPath(); c.arc(p.x, p.y, 16, 0, 7); c.stroke();
        c.font = "bold 15px -apple-system,sans-serif"; c.fillStyle = cols[i % 3]; c.fillText(String(i + 1), p.x - 4, p.y - 22); }); }
    c.fillStyle = "rgba(255,140,0,.15)"; for (const [a, b] of OV.backup_intervals || []) { const p0 = P(0, a), p1 = P(0, b); c.fillRect(0, p0.y, cv.width, p1.y - p0.y); }
    OV.anchors.forEach((a, i) => { const p = DRAG && DRAG.i === i ? P(DRAG.x, DRAG.row) : P(a[1], a[0]); c.fillStyle = "#16a34a"; c.beginPath(); c.arc(p.x, p.y, 7, 0, 7); c.fill();
      c.strokeStyle = "#fff"; c.lineWidth = 2; c.stroke(); });
    c.strokeStyle = "#e377c2"; c.lineWidth = 2.5; for (const a of OV.excluded || []) { const p = P(a[1], a[0]);
      c.beginPath(); c.moveTo(p.x - 6, p.y - 6); c.lineTo(p.x + 6, p.y + 6); c.moveTo(p.x + 6, p.y - 6); c.lineTo(p.x - 6, p.y + 6); c.stroke(); }
  }
  c.strokeStyle = "#2563eb"; c.lineWidth = 2; for (const q of pts) { const p = P(q[1], q[0]); c.strokeRect(p.x - 5, p.y - 5, 10, 10); }
}
window.addEventListener("resize", draw);

// ------------------------------------------------------------------ clicks + keys
async function onClick(ev) {
  if (!ev.quick || mode === "pan") return; ev.preventDefaultAction = true;
  const ip = viewer.viewport.viewerElementToImageCoordinates(ev.position), row = ip.y, x = ip.x;
  try {
    if (mode === "word") { const w = HW.find(w => row >= w.y0 - 3 && row <= w.y1 + 3 && x >= w.x0 - 3 && x <= w.x1 + 3); if (!w) return;
      const t = prompt(`OCR read "${w.ocr_text || w.text}" (confidence ${Math.round(w.conf)}). Correct text (empty = delete):`, w.text); if (t === null || t === w.text) return;
      HDR = Object.assign(HDR, await api("POST", A("/header/words"), {track: w.track, edits: [{i: w.i, text: t}]}));
      HW = (await api("GET", A("/header/words"))).words; renderHeaderTable(); draw(); return; }
    if (mode === "circle") { await fixCall("/fix/circle", {row, x, radius: +($("circR")?.value || 60), through: ev.originalEvent?.shiftKey || false}); return; }
    if (mode === "fix") { if (Date.now() - (DRAG?.t || 0) < 300) return; await fixCall("/fix/pin", {row, x}); return; }
    if (mode === "anchor") { await api("POST", E("/anchors"), {row, x, snap: true, window_ft: +($("winFt")?.value || 150)}); }
    else if (mode === "exclude") { await api("POST", E("/exclude"), {row, x}); }
    else if (mode === "seed") { await api("POST", E("/trace"), {seed: [row, x]}); setMode("anchor"); }
    else if (mode === "wrap") { pts.push([row, x]); draw(); if (pts.length < 2) return;
      const [a, b] = pts; pts = []; await api("POST", E("/backup_interval"), {rows: [a[0], b[0]]}); }
    else if (mode === "box") { pts.push([row, x]); draw(); if (pts.length < 2) return; const [a, b] = pts; pts = []; setMode("pan");
      await runJob(A("/section"), {box: [a[1], b[1]], rows: [Math.min(a[0], b[0]), Math.max(a[0], b[0])]}, "section from box"); render(); }
    else if (mode === "dsec") { const d = (ST.depth_sections || []).find(d => row >= d.row0 && row <= d.row1);
      if (d) await chooseSection(d.index); return; }
    else if (mode === "label") { const v = prompt(`Depth (ft) of the label at row ${row.toFixed(0)}?`);
      if (v === null || v === "" || !isFinite(+v)) return; LBL.push({row, ocr: null, value: +v, added: true}); LBL.sort((a, b) => a.row - b.row); dirty(true); renderLabels(); draw(); return; }
    await refresh(); if (STEP === "trace") renderAnchors();
  } catch (e) { console.warn(e); }
}
// guided fixing: press near a pin + drag = move it; Guide mode: drag along the curve = guide stroke
const imgPt = ev => viewer.viewport.viewerElementToImageCoordinates(ev.position);
function pinNear(ev) { const ip = imgPt(ev), z = viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true)), tol = 12 / z;
  return (OV?.anchors || []).findIndex(a => Math.hypot(a[1] - ip.x, a[0] - ip.y) < tol); }
function onPress(ev) {
  if (mode === "guide") { const ip = imgPt(ev); STROKE = [[ip.y, ip.x]]; ev.preventDefaultAction = true; return; }
  if (mode === "fix") { const i = pinNear(ev); if (i >= 0) { const ip = imgPt(ev); DRAG = {i, row: ip.y, x: ip.x, moved: false}; viewer.setMouseNavEnabled(false); } }
}
function onDrag(ev) {
  if (mode === "guide" && STROKE.length) { const ip = imgPt(ev); STROKE.push([ip.y, ip.x]); ev.preventDefaultAction = true; drawSoon(); return; }
  if (DRAG) { const ip = imgPt(ev); DRAG.row = ip.y; DRAG.x = ip.x; DRAG.moved = true; ev.preventDefaultAction = true; drawSoon(); }
}
async function onRelease(ev) {
  if (mode === "guide" && STROKE.length) { const st = STROKE; STROKE = [];
    if (st.length > 2) { try { await fixCall("/fix/stroke", {points: st}); } catch (e) { draw(); } } else draw(); return; }
  if (DRAG) { const d = DRAG; viewer.setMouseNavEnabled(true); DRAG = {t: Date.now()}; if (d.moved) await fixCall("/fix/move", {index: d.i, row: d.row, x: d.x}); else { DRAG = null; } }
}
async function fixCall(path, body) {         // one guided edit: re-trace on the server, redraw, update the change meter
  const r = await api("POST", A(path), body); PROG = r.progress; await refresh(); CHK = null; loadChecks();
  if (STEP === "trace") { renderAnchors(); renderProgress(r); renderCands(); } return r;
}
async function loadChecks() { if (!SID || !ST?.traced) return; CHK = await api("GET", A("/checks")); draw(); if (STEP === "trace") renderAnchors(); }
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
  if (STEP === "checks" && CUR && /^[1-3]$/.test(k)) { const ch = CUR.choices[+k - 1]; if (ch) answer(ch.id); return; }
  if (STEP === "checks" && CUR && k === "k") { answer("skip"); return; }
  if (k === "escape") setMode("pan"); else if (k === "z" && !e.metaKey && !e.ctrlKey) undo(); else if (b && !b.disabled) b.click();
  else if ((k === "delete" || k === "backspace") && STEP === "depth") delLabels(); else if ((k === "delete" || k === "backspace") && STEP === "header") delHeader();
});
async function undo() { if (!SID || !ST?.undo.length) return; const r = await api("POST", A("/undo")); toast("undone: " + r.undone);
  if (STEP === "header") await loadHeader(); if (STEP === "depth") await loadLabels(); await refresh(); render(false); }
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

// ------------------------------------------------------------------ steps
const R = {};
async function render(refit = true) { const st = STEPS.findIndex(s => s.id === STEP); setMode("pan"); dirty(false); fig("");
  $("stepTitle").textContent = `${STEPS.slice(0, st + 1).filter(s => !s.hidden).length} · ${STEPS[st].title}`; $("mainBtn").disabled = false;
  try { await R[STEP](); } catch (e) { console.warn(e); } bindTools(); bindEngine(); if (refit) focusStep(); draw(); }
Shell.init({steps: STEPS, onStep: (i, s) => { STEP = s.id; render(); }});
const main = (label, fn) => { $("mainBtn").textContent = label; $("mainBtn").onclick = async () => { $("mainBtn").disabled = true; try { await fn(); } catch (e) { console.warn(e); } finally { $("mainBtn").disabled = false; } }; };
const next = () => Shell.complete(Shell.cur);
const on = (id, fn) => { const b = $(id); if (b) b.onclick = async () => { try { await fn(); } catch (e) { console.warn(e); } }; };

R.load = () => {
  $("stepLead").textContent = "Pick a scanned log. The app then does everything automatically once; the next steps just ask you to check each part.";
  const f = INFO.features;
  $("stepBody").innerHTML = `<div class="row"><select id="cat" style="width:100%">${INFO.catalog.map((c, i) => `<option value="${i}">${esc(c.well)} · ${esc(c.scan.split("/").pop())} · ${esc(c.curve)}${c.las ? " · LAS" : ""}</option>`).join("")}</select></div>
    <div class="card">pyLogDigitizer turns a scanned paper well log into a LAS file. Everything runs on this computer.</div>
    <div class="note">Project: ${esc(INFO.project)}<br>Presets come from <code>dataset_profiles/*_wells.yaml</code>; other TIFFs under <code>example_data/</code> are listed too.</div>
    <div class="row"><label><input type="checkbox" id="autoLayout" checked> automatic layout</label></div>
    <h3>Installed features</h3><div class="card">Tracer backend: <b>${esc(f.backend)}</b>
    <div class="${f.ocr.ok ? "" : "disabled-feature"}">OCR (depth labels, header): ${f.ocr.ok ? `<span class="okc">RapidOCR</span>` : "no"}</div>${hint(f.ocr)}
    <div class="${f.ml.ok ? "" : "disabled-feature"}">ML U-Net cost map: ${f.ml.ok ? '<span class="okc">yes</span>' : "no"}</div>${hint(f.ml)}
    <div class="${f.plot.ok ? "" : "disabled-feature"}">QC plot PNG export: ${f.plot.ok ? '<span class="okc">yes</span>' : "no"}</div>${hint(f.plot)}</div>
    <div id="loaded"></div>`;
  if (ST) $("loaded").innerHTML = `<div class="card">Loaded: <b>${esc(ST.well)}</b> · ${ST.width}×${ST.height} px. Loading another scan replaces it (export first).</div>`;
  main(ST ? "Load this scan (replaces current)" : "Open & analyse", async () => {
    const c = INFO.catalog[+$("cat").value];
    const r = await runJob("/app/api/open", {scan: c.scan, curve: c.curve, well: c.well, las: c.las, settings: c.settings, auto_layout: $("autoLayout").checked}, "loading");
    SID = r.sid; ST = null; Shell.reset(0); await refresh(false);
    STEPS.find(s => s.id === "qc").hidden = !ST.las; STEPS.find(s => s.id === "dsection").hidden = !((ST.depth_sections || []).length > 1); Shell.reset(0); LAY = await api("GET", A("/layout")); await openViewer(); await refresh();
    const errs = Object.entries(ST.errors || {}).filter(([k]) => k !== "depth_section"); if (errs.length) toast(errs.map(([k, v]) => `${k}: ${v}`).join("\n"), true);
    next();
  });
  help(`Choose a scan and press <b>Open & analyse</b>. Progress shows as n / N on the right. ${NAV}`);
};

R.deskew = () => {
  $("stepLead").textContent = "Look at the picture: the grid should be straight and the coloured lines should sit on the track edges.";
  const sk = LAY?.skew || [], t = LAY?.timings || {};
  $("stepBody").innerHTML = LAY?.available ? `<div class="card">Skew per block: ${fmt(Math.min(...sk.map(s => s[1])), 2)}° to ${fmt(Math.max(...sk.map(s => s[1])), 2)}° (${sk.length} blocks)<br>
    Working sheet ${ST.width}×${ST.height} px · load+deskew ${fmt(t.load_deskew_s)} s · follow ${fmt(t.follow_s)} s · labels ${fmt(t.labels_s)} s</div>
    <div class="note">If the sheet is clearly wrong (no tracks found), go back to step 1 and load without automatic layout; you can then draw the track box by hand in step 3.</div>`
    : `<div class="card err">${esc(ST.errors?.layout || "no automatic layout")}</div>`;
  if (LAY?.available) fig(chart({title: "Deskew angle per block", series: [{pts: sk.map(s => [s[0], s[1]]), color: TAB10[0], lw: 1.5}, {pts: sk.map(s => [s[0], s[1]]), color: TAB10[0], dots: true, r: 2.5, alpha: 1}],
    xlabel: "sheet row (px)", ylabel: "skew (deg)", w: 350, h: 190}), true);
  main("Looks right — continue", async () => next());
  help(NAV);
};

R.section = () => {
  $("stepLead").textContent = "Pick the column (track) that holds your curve. If no track is found, draw a box around it.";
  const tr = (ST.tracks || []).filter(t => t.name !== "depth"), cur = ST.section?.track, dtb = ST.depth_top_base || [];
  $("stepBody").innerHTML = `<h3>Track</h3><div class="radio">${tr.map((t, j) => `<label><input type="radio" name="trk" value="${t.name}" ${t.name === cur ? "checked" : ""}>
      <span class="swatch" style="background:${TAB10[Object.keys(LAY?.edges || {}).indexOf(t.name + ".left") % 10]}"></span>${t.name} <span class="note">x ${fmt(t.left, 0)}–${fmt(t.right, 0)} px</span></label>`).join("") || '<div class="note">no tracks detected - draw a box</div>'}</div>
    <div class="row">${tool("box", "b", "Draw box")}</div><div class="note">Box: click two opposite corners of the curve's track.</div>
    <h3>Depth range (ft)</h3><div class="row"><label>top</label><input type="number" id="dTop" value="${dtb[0] != null ? dtb[0].toFixed(0) : ""}"></div>
    <div class="row"><label>base</label><input type="number" id="dBase" value="${dtb[1] != null ? dtb[1].toFixed(0) : ""}"></div>
    <div class="row"><button id="secReset">Reset to detected</button></div>
    <h3>Move the track edges (px)</h3><div class="row"><label>left</label><input type="number" id="eL" value="0" step="1"><label style="min-width:0">right</label><input type="number" id="eR" value="0" step="1"><button id="eApply">Move</button></div>
    <div class="card">Current: <b>${ST.section ? (ST.section.kind === "box" ? `box x ${ST.section.x0}–${ST.section.x1}` : ST.section.track) : "none"}</b>, rows ${ST.section ? ST.section.rows.join("–") : "–"}</div>
    ${KEYS("<b>Draw box</b> (B): click two corners", "Changing the section re-traces it (the trace step opens ready)", NAV)}`;
  document.querySelectorAll("input[name=trk],#dTop,#dBase").forEach(i => i.oninput = () => dirty(true));
  on("eApply", async () => { await runJob(A("/calib/edges"), {left_dx: +$("eL").value || 0, right_dx: +$("eR").value || 0}, "track edges"); await refresh(); render(false); });
  on("secReset", async () => { const t = INFO.catalog.find(c => ST.scan.endsWith(c.scan))?.settings?.track || "track1"; await runJob(A("/section"), {track: t}, "section"); await refresh(); render(); });
  main("Use this section", async () => {
    const t = document.querySelector("input[name=trk]:checked")?.value, d0 = $("dTop").value, d1 = $("dBase").value;
    const same = ST.section && t === ST.section.track && (!dtb.length || (+d0 === +dtb[0].toFixed(0) && +d1 === +dtb[1].toFixed(0)));
    if (!same && (t || ST.section?.kind === "box")) {
      const body = t ? {track: t} : {box: [ST.section.x0, ST.section.x1]}; if (d0 && d1 && dtb.length && !(+d0 === +dtb[0].toFixed(0) && +d1 === +dtb[1].toFixed(0))) body.depth_range = [+d0, +d1];
      await runJob(A("/section"), body, "section"); await refresh(); }
    if (!ST.section) { toast("choose a track or draw a box", true); return; }
    next();
  });
  help(`Coloured lines = followed track boundaries, red ticks = depth labels (as in the notebook). <b>B</b> draw box. ${NAV}`);
};

// header OCR ---------------------------------------------------------
async function loadHeader() { HDR = await api("GET", A("/header")); }
function hdrEdited(r) { const o = (HDR.ocr || []).find(q => q.id === r.id); if (!o) return false; return ["mnemonic", "unit", "left", "right", "type", "backup_left", "backup_right", "track"].some(k => String(o[k] ?? "") !== String(r[k] ?? "")); }
function renderHeaderTable() {
  const cols = [["track", 52], ["mnemonic", 52], ["unit", 46], ["left", 48], ["right", 48], ["type", 56], ["backup_left", 44], ["backup_right", 44]];
  $("hdrTbl").innerHTML = `<tr><th></th><th>track</th><th>curve</th><th>unit</th><th>left</th><th>right</th><th>type</th><th colspan=2>backup</th></tr>` + HDR.rows.map((r, i) =>
    `<tr data-i="${i}" class="${HSEL.has(i) ? "sel" : ""} ${r.origin === "added" ? "added" : hdrEdited(r) ? "edited" : ""}" title="${esc(r.label ? "OCR read: " + r.label : "")}${r.review?.length ? " · needs review: " + r.review.join(", ") : ""}">
     <td><input type="radio" name="huse" value="${r.id}" ${HDR.choice === r.id ? "checked" : ""} title="use this row in the Scale step"></td>` +
    cols.map(([k, w]) => k === "type" ? `<td><select data-k="type">${["linear", "log"].map(t => `<option ${r.type === t ? "selected" : ""}>${t}</option>`).join("")}</select></td>`
      : `<td><input data-k="${k}" value="${esc(r[k] ?? "")}" style="${(r.review || []).some(q => q.startsWith(k.replace("left", "scale_left").replace("right", "scale_right").split("_")[0])) ? "border-color:#d97706" : ""}"></td>`).join("") + "</tr>").join("");
  $("hdrTbl").querySelectorAll("tr[data-i]").forEach(tr => {
    const i = +tr.dataset.i; tr.onclick = ev => { if (ev.target.tagName === "TD") { HSEL.has(i) ? HSEL.delete(i) : HSEL.add(i); renderHeaderTable(); } };
    tr.querySelectorAll("[data-k]").forEach(inp => inp.onchange = () => { HDR.rows[i][inp.dataset.k] = inp.value; dirty(true); renderHeaderTable(); });
  });
  $("hdrTbl").querySelectorAll("input[name=huse]").forEach(r => r.onchange = () => { HDR.choice = r.value; dirty(true); });
}
function delHeader() { if (!HSEL.size) return; HDR.rows = HDR.rows.filter((_, i) => !HSEL.has(i)); HSEL.clear(); dirty(true); renderHeaderTable(); }
R.header = async () => {
  const f = INFO.features.ocr;
  $("stepLead").textContent = "These are the curve names and scale numbers read from the top of the log. Fix anything that looks wrong; orange = please check.";
  $("stepBody").innerHTML = `<div id="hdrInfo" class="note"></div>${hint(f)}<div><table class="t fit" id="hdrTbl"></table></div>
    <div class="row"><button id="hAdd">+ Add row</button><button id="hDel" class="danger">Delete selected</button><button id="hRevert">Revert to OCR</button><button id="hRerun" ${off(f)}>Re-run OCR</button></div>
    <div class="row">${tool("word", "e", "Edit OCR words")}<span class="muted">click a boxed word on the image to correct it; the scale is re-fitted</span></div>${engineSel()}
    ${KEYS("Click a cell's margin to select a row · <kbd>Delete</kbd> removes selected rows", "Type over any value · ✎ = edited, + = added, orange border = OCR flagged for review", "Hover a row to see what OCR read · ◉ = row used in step 6 (Scale)", "Nothing is applied until you press the blue button; <kbd>Z</kbd> undoes")}`;
  await loadHeader();
  $("hdrInfo").innerHTML = !f.ok ? "OCR is not installed: add the curve's scale by hand (+ Add row) or skip to step 6." : HDR.error ? `<span class="err">${esc(HDR.error)}</span>` : `${HDR.rows.length} curve(s) found` + (HDR.rows.filter(r => r.review?.length).length ? ` · <span class="warnc">${HDR.rows.filter(r => r.review?.length).length} need a look (orange)</span>` : "");
  if (!HDR.choice) { const m = HDR.rows.find(r => r.mnemonic === ST.curve && r.track === ST.section?.track) || HDR.rows.find(r => r.track === ST.section?.track); if (m) HDR.choice = m.id; }
  renderHeaderTable(); try { HW = (await api("GET", A("/header/words"))).words; } catch (e) { HW = []; } draw();
  on("hAdd", () => { HDR.rows.push({id: Math.random().toString(36).slice(2, 8), origin: "added", track: ST.section?.track || "track1", mnemonic: ST.curve, unit: "", left: "", right: "", type: "linear", backup_left: "", backup_right: "", review: []}); dirty(true); renderHeaderTable(); });
  on("hDel", delHeader);
  on("hRevert", async () => { HDR = Object.assign(HDR, await api("POST", A("/header/revert"))); HSEL.clear(); renderHeaderTable(); dirty(false); await refresh(false); });
  on("hRerun", async () => { await runJob(A("/header/rerun"), {}, "header OCR"); await loadHeader(); renderHeaderTable(); });
  main("Confirm header — continue", async () => { HDR = Object.assign(HDR, await api("POST", A("/header"), {rows: HDR.rows, choice: HDR.choice || null})); await refresh(false); next(); });
  help(`Viewer shows the header area. ${NAV}`);
};

// depth --------------------------------------------------------------
async function loadLabels() { LAY = await api("GET", A("/layout")); LBL = (LAY.labels || []).map(([row, v]) => ({row, ocr: v, value: v})); LBLsel.clear(); }
function labelDoubt(l) {                 // OCR value missing or off the depth line fitted through the anchors
  if (l.added || l.value !== l.ocr) return "";
  if (l.ocr == null) return "could not be read";
  const a = (ST.calibration?.depth_anchors || []).slice().sort((p, q) => p[0] - q[0]);
  if (a.length < 2) return "";
  let j = a.findIndex(p => p[0] >= l.row); if (j <= 0) j = j === 0 ? 1 : a.length - 1;
  const [r0, d0] = a[j - 1], [r1, d1] = a[j], d = d0 + (d1 - d0) * (l.row - r0) / ((r1 - r0) || 1);
  return Math.abs(d - l.ocr) > 1 ? `read ${l.ocr}, but the others suggest ${d.toFixed(0)}` : "";
}
function renderLabels() {
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
function depthFig() {
  if (!ST.residuals?.length) { fig(""); return; }
  fig(chart({title: "Depth anchors: leave-one-out residuals", series: [{pts: ST.residuals, color: TAB10[0], dots: true, r: 2.5, alpha: 1}], xlabel: "depth (ft)", ylabel: "LOO residual (ft)", w: 350, h: 180}), true);
}
async function chooseSection(i) { await runJob(A("/depth_section"), {index: i}, "depth section"); LAY = await api("GET", A("/layout")); await refresh(); render(false); }
R.dsection = () => {
  const D = ST.depth_sections || [], cur = ST.depth_section;
  $("stepLead").textContent = "This sheet has more than one log section (for example a 2-inch and a 5-inch print). Click the one you want on the scan, or pick it below.";
  $("stepBody").innerHTML = `${ST.errors?.depth_section ? `<div class="card warnc">Not chosen yet - the header does not say which one.</div>` : `<div class="card">Using section ${D.findIndex(d => d.index === (cur ?? ST.calibration?.section_used)) + 1}${cur == null ? " (it matches the depths in the header)" : ""}.</div>`}
    ${D.map((d, i) => `<div class="row"><button id="dsec${d.index}" style="width:100%" ${cur === d.index ? 'class="primary"' : ""}>Section ${i + 1}: ${fmt(d.top, 0)}–${fmt(d.base, 0)} ft · ${fmt(d.px_per_ft, 2)} px/ft</button></div>`).join("")}
    ${KEYS("Click inside a coloured band on the scan to choose it", "More px/ft = a more detailed print", NAV)}`;
  D.forEach(d => on("dsec" + d.index, () => chooseSection(d.index)));
  setTimeout(() => { mode = "dsec"; }, 0);
  main("Use this section — continue", async () => {
    if (ST.errors?.depth_section) { toast("click a section on the scan first", true); return; }
    next(); });
  help(`Coloured bands = log sections found on the sheet. Click one to use it. ${NAV}`);
};

R.depth = async () => {
  const c = ST.calibration || {}, f = INFO.features.ocr;
  $("stepLead").textContent = "These are the depth numbers read from the scan. Fix any orange ones. If none could be read, type one known depth below.";
  $("stepBody").innerHTML = `<div class="card">${c.px_per_ft ? `<b>${esc(c.depth_source)}</b> · ${c.depth_anchors.length} anchors · ${fmt(c.px_per_ft, 3)} px/ft` : '<span class="err">no depth calibration</span>'}
     ${ST.errors?.depth ? `<div class="err">${esc(ST.errors.depth)}</div>` : ""}</div>${f.ok ? "" : hint(f)}
    <div class="note" id="lblDoubt"></div><div class="list" style="max-height:230px"><table class="t" id="lblTbl"></table></div>
    <div class="row">${tool("label", "l", "+ Add label")}<button id="lDel" class="danger">Delete selected</button><button id="lReset">Reset to automatic</button></div>
    <h3>Or type one known depth</h3><div class="note">Depth of the first number below the header, and the step between numbers (often 100).</div><div class="row"><label>first label</label><input type="number" id="fd" placeholder="ft"><label style="min-width:0">step</label><input type="number" id="li" placeholder="100"></div>
    <div class="row"><button id="lAuto">Re-run automatic</button></div>${engineSel()}
    ${KEYS("<b>+ Add label</b> (L): click a printed depth label, type its depth", "Click a row to select and jump to it · <kbd>Delete</kbd> removes selected", "✎ edited, + added · applying edited labels uses them as anchors (piecewise linear)", "Applying re-traces the section; <kbd>Z</kbd> undoes")}`;
  await loadLabels(); renderLabels(); depthFig();
  on("lDel", delLabels);
  on("lReset", async () => { await runJob(A("/depth"), {mode: "reset"}, "depth"); await refresh(); render(); });
  on("lAuto", async () => { await runJob(A("/depth"), {mode: "auto", first_depth: $("fd").value, label_interval: $("li").value}, "depth"); await refresh(); render(); });
  main("Apply depths — continue", async () => {
    const edited = LBL.some(l => l.added || l.value !== l.ocr) || LBL.length !== (LAY.labels || []).length;
    if (edited) { await runJob(A("/depth"), {mode: "labels", labels: LBL.filter(l => l.value != null).map(l => [l.row, l.value])}, "depth from labels"); await refresh(); }
    else if (!c.px_per_ft) { toast("no depth calibration yet: edit/add labels (≥ 2) or use first label + step", true); return; }
    next();
  });
  help(`Red ticks = depth labels with their values (orange ? = OCR could not read). <b>L</b> add label. ${NAV}`);
};

R.scale = () => {
  const s = ST.calibration.scale, h = HDR?.rows?.find(r => r.id === (HDR.choice || ST.header_choice));
  const v = h && h.left !== "" && h.left != null ? {left: h.left, right: h.right, type: h.type, backup: h.backup_left != null && h.backup_left !== "" ? [h.backup_left, h.backup_right] : null} : null;
  const src = v ? `from header row <b>${esc(h.mnemonic)}</b> (${esc(h.track)}) - check it` : "current values";
  const F = v ? Object.assign({}, s, v) : s;
  $("stepLead").textContent = "The values at the left and right edges of the track. They usually come from the header; just check them.";
  $("stepBody").innerHTML = `<div class="note">Filled ${src}.</div>
    <div class="row"><label>left</label><input type="number" id="sL" value="${F.left}"><label style="min-width:0">right</label><input type="number" id="sR" value="${F.right}"></div>
    <div class="row"><label>type</label><select id="sT"><option ${F.type === "linear" ? "selected" : ""}>linear</option><option ${F.type === "log" ? "selected" : ""}>log</option></select></div>
    <h3>Wraparound</h3><div class="row"><label>backup</label><input type="number" id="bL" value="${F.backup ? F.backup[0] : ""}"><input type="number" id="bR" value="${F.backup ? F.backup[1] : ""}"></div>
    <div class="row"><label>wraps on</label><select id="wS"><option ${s.wrap_side === "right" ? "selected" : ""}>right</option><option ${s.wrap_side === "left" ? "selected" : ""}>left</option></select>
      <label style="min-width:0"><input type="checkbox" id="hat" ${s.use_hatch ? "checked" : ""}> hatched backup</label></div>
    <div class="row"><button id="sReset">Reset to preset</button></div>
    <h3>From two grid lines</h3><div class="note">Panel x (px from the left track edge) and the value there. Grid lines at x: <span id="gX">…</span></div>
    <div class="row"><input type="number" id="p1x" placeholder="x1"><input type="number" id="p1v" placeholder="value 1"><input type="number" id="p2x" placeholder="x2"><input type="number" id="p2v" placeholder="value 2"><button id="p2Apply">Fit</button></div>
    <h3>Grid lines</h3><div class="note">Straighten bent grid lines, then every line gets a value from two or more labels. Edit any value and press Fit values; labels that disagree turn red.</div>
    <div class="row"><button id="gDetect">Straighten grid</button><button id="gOff">Undo straightening</button><button id="gVals">Fit values</button><label><input type="checkbox" id="gEdit"> drag grid nodes</label></div>
    <div id="gTable" class="note"></div><div id="gWarn" class="note"></div>
    <div id="gEditor" style="display:none"><canvas id="gCanvas" style="border:1px solid #888;cursor:crosshair"></canvas><img id="gPrev" style="border:1px solid #888;margin-left:6px"></div>
    <div class="card">Applied: ${s.left} → ${s.right} ${s.type}${s.backup ? `, backup ${s.backup[0]} → ${s.backup[1]} (${s.wrap_side})` : ""} · wraps found: ${ST.n_wraps ?? "–"}</div>
    ${KEYS("Applying only re-maps values (no re-trace)", "<b>wraps on</b>: the edge the curve leaves through (GR: right; DT 100→40: left)", "<kbd>Z</kbd> undoes")}`;
  if (v) dirty(true);
  document.querySelectorAll("#stepBody input,#stepBody select").forEach(i => i.oninput = () => dirty(true));
  const apply = async b => { await api("POST", E("/scale"), b); await refresh(); };
  api("GET", A("/calib")).then(c => { const g = $("gX"); if (g) g.textContent = (c.grid_x || []).map(x => x.toFixed(0)).join(", ") || "none found"; }).catch(() => {});
  on("p2Apply", async () => { const r = await api("POST", A("/calib/scale2"), {x1: +$("p1x").value, v1: +$("p1v").value, x2: +$("p2x").value, v2: +$("p2v").value, type: $("sT").value});
    toast(`scale ${fmt(r.left, 3)} → ${fmt(r.right, 3)}`); await refresh(); render(false); });
  const gridShow = g => {
    const t = $("gTable"); if (!t) return;
    const lines = g.lattice ? g.lattice.u : (g.fit ? g.fit.line_x : []);
    const vals = g.fit ? g.fit.values : [];
    const bad = new Set((g.fit?.outliers || []).map(o => Math.round(o.x)));
    t.innerHTML = lines.length ? lines.map((x, k) => `<span style="display:inline-block;margin:2px;${bad.has(Math.round(x)) ? "color:#c00" : ""}">x ${x.toFixed(0)}<br><input class="gv" data-x="${x}" type="number" style="width:5em" value="${vals[k] ?? ""}"></span>`).join("")
      : "No straightened grid yet.";
    $("gWarn").textContent = (g.fit?.warnings || []).join(" · ");
  };
  const gridLoad = async () => { const g = await api("GET", A("/grid")); gridShow(g); return g; };
  gridLoad().catch(() => {});
  on("gDetect", async () => { await runJob(A("/grid/detect"), {}, "straighten grid"); await api("POST", A("/grid/values"), {}); const g = await gridLoad();
    toast(`grid straightened: ${g.lattice ? g.lattice.u.length : 0} lines`); await refresh(); render(false); });
  on("gOff", async () => { await api("POST", A("/grid/off")); await gridLoad(); await refresh(); render(false); });
  on("gVals", async () => {
    const labels = [...document.querySelectorAll(".gv")].filter(i => i.value !== "" && i.dataset.edited).map(i => [+i.dataset.x, +i.value]);
    const all = [...document.querySelectorAll(".gv")].filter(i => i.value !== "").map(i => [+i.dataset.x, +i.value]);
    const r = await api("POST", A("/grid/values"), {labels: labels.length >= 2 ? labels : all, kind: $("sT").value});
    toast(`scale ${fmt(r.left, 3)} → ${fmt(r.right, 3)} ${r.kind}`); await refresh(); render(false); });
  document.addEventListener("input", ev => { if (ev.target.classList?.contains("gv")) ev.target.dataset.edited = "1"; });
  on("gEdit", async () => {                  // node editor: drag a lattice node, see the straightened preview live
    const box = $("gEditor"); box.style.display = $("gEdit").checked ? "block" : "none"; if (!$("gEdit").checked) return;
    const g = await gridLoad(); if (!g.lattice) { toast("straighten the grid first"); return; }
    const L = g.lattice, cv = $("gCanvas"), H = Math.min(g.rows, 1500), sc = Math.min(1, 500 / H);
    cv.width = g.width * sc; cv.height = H * sc; const ctx = cv.getContext("2d");
    let drag = null, X = L.X.map(r => r.slice()), Y = L.Y.map(r => r.slice());
    const draw = () => { ctx.clearRect(0, 0, cv.width, cv.height); ctx.strokeStyle = "#08f";
      for (let i = 0; i < X.length; i++) for (let j = 0; j < X[0].length; j++) if (Y[i][j] < H) { ctx.strokeRect(X[i][j] * sc - 3, Y[i][j] * sc - 3, 6, 6); } };
    const prev = async (i, j, x, y) => { const r = await api("POST", A("/grid/preview"), i == null ? {rows: [0, H]} : {i, j, x, y, radius: 1, rows: [0, H]});
      $("gPrev").src = r.png; $("gPrev").style.height = cv.height + "px"; };
    draw(); prev();
    cv.onmousedown = ev => { const x = ev.offsetX / sc, y = ev.offsetY / sc; let best = 1e9;
      for (let i = 0; i < X.length; i++) for (let j = 0; j < X[0].length; j++) { const d = Math.hypot(X[i][j] - x, Y[i][j] - y); if (d < best) { best = d; drag = [i, j]; } }
      if (best > 15 / sc) drag = null; };
    let pend = null;
    cv.onmousemove = ev => { if (!drag) return; const [i, j] = drag; X[i][j] = ev.offsetX / sc; Y[i][j] = ev.offsetY / sc; draw();
      clearTimeout(pend); pend = setTimeout(() => prev(i, j, X[i][j], Y[i][j]), 80); };
    cv.onmouseup = async () => { if (!drag) return; const [i, j] = drag; drag = null;
      await runJob(A("/grid/node"), {i, j, x: X[i][j], y: Y[i][j], radius: 1}, "grid node"); await refresh(); toast("grid node moved"); };
  });
  on("sReset", async () => { const i = ST.initial_scale; await apply({left: i.left, right: i.right, type: i.type, backup: i.backup, wrap_side: i.wrap_side, use_hatch: i.use_hatch}); render(); });
  main("Apply scale — continue", async () => {
    const bk = $("bL").value !== "" && $("bR").value !== "" ? [+$("bL").value, +$("bR").value] : null;
    await apply({left: +$("sL").value, right: +$("sR").value, type: $("sT").value, backup: bk, wrap_side: $("wS").value, use_hatch: $("hat").checked}); next();
  });
  help(`Red line = current trace. ${NAV}`);
};

let UPOS = -1;
function uncertainSpots() {               // where another path is almost as good (yellow): [[r0, r1], ..]
  return CHK?.disagree || [];
}
function nextUncertain() {
  const u = uncertainSpots();
  if (!u.length) { $("uPos").textContent = "no uncertain spots left"; return; }
  UPOS = (UPOS + 1) % u.length; const [a, b] = u[UPOS], m = (a + b) / 2, h = Math.max(ST.width * 0.6, (b - a) / 2 + 100);
  fitRows(m - h, m + h); $("uPos").textContent = `${UPOS + 1} of ${u.length}`;
}
function renderAnchors() {
  if (!$("ancList")) return;
  $("ancList").innerHTML = (OV?.anchors || []).map((a, i) => `<div>Pin ${i + 1} · row ${a[0].toFixed(0)}<span><a data-go="${a[0]}">go</a><a data-del="${i}" title="remove this pin">✕</a></span></div>`).join("")
    + (OV?.strokes || []).map((st, i) => `<div>Guide ${i + 1} · rows ${st[0][0].toFixed(0)}–${st[st.length - 1][0].toFixed(0)}<span><a data-go="${st[0][0]}">go</a><a data-dels="${i}" title="remove this guide">✕</a></span></div>`).join("")
    || '<span class="note">no pins yet</span>';
  $("ancList").querySelectorAll("[data-go]").forEach(x => x.onclick = () => fitRows(+x.dataset.go - ST.width * 0.6, +x.dataset.go + ST.width * 0.6));
  $("ancList").querySelectorAll("[data-del]").forEach(x => x.onclick = () => fixCall("/fix/unpin", {index: +x.dataset.del}));
  $("ancList").querySelectorAll("[data-dels]").forEach(x => x.onclick = () => fixCall("/fix/unstroke", {index: +x.dataset.dels}));
  $("trInfo").innerHTML = ST.traced ? `<b>${uncertainSpots().length}</b> place(s) where the line could go another way (yellow) · ${ST.n_anchors} pin(s) · ${OV?.strokes?.length || 0} guide(s)` +
    (ST.qc ? `<br>vs paired LAS: r = <b>${fmt(ST.qc.r, 3)}</b>, RMSE ${fmt(ST.qc.rmse, 2)}` : "") : "not traced";
}
function renderCands() {
  const el = $("candList"); if (!el) return; const cs = OV?.candidates || [];
  el.innerHTML = cs.length ? cs.map(c => `<div class="item"><span style="color:${CAND_COL[c.index % CAND_COL.length]}">■</span> Curve ${c.index + 1}${c.picked ? " <b>(digitized)</b>" : ""} <span class="muted">score ${fmt(c.score, 2)}${c.width ? ` · ${c.width >= 4 ? "heavy" : "thin"} ${c.dashed ? "dashed" : "solid"} (${fmt(c.width, 1)} px)` : ""}</span>
    ${c.picked ? "" : `<button data-pick="${c.index}">Use this curve</button>`}</div>`).join("") : '<span class="muted">one curve traced</span>';
  el.querySelectorAll("[data-pick]").forEach(b => b.onclick = async () => { try { await runJob(A("/trace/pick"), {index: +b.dataset.pick}, "pick curve"); await refresh(); CHK = null; loadChecks(); renderCands(); } catch (e) {} });
}
const CAND_COL = ["#d62728", "#2563eb", "#16a34a", "#9333ea"];
R.trace = () => {
  const ml = INFO.features.ml;
  $("stepLead").textContent = "Click on the real curve wherever the red line is wrong - the line is redrawn through your click.";
  $("stepBody").innerHTML = `<div class="card" id="trInfo"></div>
    <h3>How much did that click change?</h3><div class="card" id="prog7"><span class="note">Each click is a pin: the line must pass through it. Fix the worst places first; when clicks stop changing much, you are done.</span></div>
    <div class="tools">${tool("fix", "f", "Fix: click the curve")}${tool("guide", "g", "Guide: drag along it")}${tool("circle", "c", "Circle: re-trace inside")}</div>
    <div class="row"><label class="muted">Circle radius (px) <input id="circR" type="number" value="60" min="10" max="600" step="10" style="width:5em"></label> <span class="muted">scroll with Shift to resize</span></div>
    <div class="note" id="toolHint">Fix: click where the line is wrong · drag a green pin to move it · ✕ in the list removes it.<br>Guide (optional): drag a short stroke along a hard stretch; the line follows it there.</div>
    <div class="row"><button id="tNextU" title="Jump to the next place where another path is almost as good (yellow)">Next place to look <span class="kbd">N</span></button><span class="note" id="uPos"></span></div>
    <h3>Pins and guides</h3><div class="list" id="ancList"></div>
    <details class="more"><summary>More</summary>
    <h3>Cost map</h3><div class="radio"><label><input type="radio" name="cm" value="conventional" checked> conventional</label>
      <label class="${ml.ok ? "" : "off"}"><input type="radio" name="cm" value="ml" ${ml.ok ? "" : "disabled"}> ML U-Net</label>
      <label class="${ml.ok ? "" : "off"}"><input type="radio" name="cm" value="blend" ${ml.ok ? "" : "disabled"}> blend (conventional + U-Net)</label>${hint(ml)}</div>
    <div class="row"><label><input type="checkbox" id="tMask"> ignore text and stamps (masks lettering, stamps and handwriting before tracing)</label></div>
    <h3>Curves in this track</h3><div class="row"><label class="muted">Trace <select id="tNC"><option value="1">1 curve</option><option value="2">2 curves</option><option value="3">3 curves</option></select></label>
      <span class="note">With 2 or more, each curve is traced and drawn dashed; pick the one to digitize (Z undoes).</span></div><div class="list" id="candList"></div>
    <h3>Advanced tools</h3><div class="tools">${tool("exclude", "x", "Not this curve")}${tool("wrap", "w", "Backup interval")}${tool("seed", "s", "Seed trace")}</div>
    <div class="row"><button id="tRetrace">Re-trace all</button><button id="tClrA">Clear pins</button><button id="tClrX">Clear exclusions</button><button id="tClrW">Clear intervals</button><button id="tReset" class="danger">Reset trace</button></div>
    </details>
    ${KEYS("<b>F</b> fix: click on the real curve where the line is wrong (snaps to the ink nearby)", "<b>G</b> guide: drag along a hard stretch of the curve",
      "Drag a green pin to move it · ✕ removes a pin or guide", "<b>N</b> next place to look (yellow = another path is almost as good)",
      "<b>Z</b> undo the last change · <b>Esc</b> pan", "Red = trace, orange = off the track edge, green = your pins, blue = your guides")}`;
  renderAnchors(); renderProgress(); loadChecks();
  setTimeout(() => setMode("fix"), 0);
  on("tNextU", nextUncertain);
  document.querySelectorAll("input[name=cm]").forEach(r => r.onchange = async () => { try { await runJob(A("/trace"), {mode: r.value, force: true}, r.value + " trace"); await refresh(); CHK = null; loadChecks(); } catch (e) {} });
  api("GET", A("/trace/options")).then(o => { const m = $("tMask"); if (m) m.checked = !!o.mask_other; const nc = $("tNC"); if (nc) nc.value = String(o.n_curves || 1); renderCands(); }).catch(() => {});
  const nc = $("tNC"); if (nc) nc.onchange = async () => { try { await runJob(A("/trace"), {mode: document.querySelector("input[name=cm]:checked")?.value || "conventional", force: true, n_curves: +nc.value}, "trace"); await refresh(); CHK = null; loadChecks(); renderCands(); } catch (e) {} };
  const tm = $("tMask"); if (tm) tm.onchange = async () => { try { await runJob(A("/trace"), {mode: document.querySelector("input[name=cm]:checked")?.value || "conventional", force: true, mask_other: tm.checked}, "trace"); await refresh(); CHK = null; loadChecks(); } catch (e) {} };
  on("tRetrace", async () => { await runJob(A("/retrace_all"), {}, "re-trace"); await refresh(); loadChecks(); });
  on("tClrA", async () => { await api("POST", E("/anchors/clear")); await refresh(); loadChecks(); });
  on("tClrX", async () => { await api("POST", E("/exclude/clear")); await refresh(); loadChecks(); });
  on("tClrW", async () => { await api("POST", E("/backup_interval/clear")); await refresh(); renderAnchors(); });
  on("tReset", async () => { if (!confirm("Reset to a fresh conventional trace (pins, guides, exclusions and intervals removed)?")) return; await runJob(A("/trace/reset"), {}, "reset trace"); document.querySelector("input[name=cm][value=conventional]").checked = true; await refresh(); loadChecks(); });
  main("Trace looks good — continue", async () => next());
  help(`<b>F</b> fix · <b>G</b> guide stroke · drag a pin to move it · <b>N</b> next place · <b>Z</b> undo · ${NAV}`);
};

async function answer(choice) {
  if (!CUR) return; const row = CUR.row;
  await fixCall("/checks/answer", {id: CUR.id, choice}); await loadChecks();
  const sp = CHK.spots; CPOS = Math.max(0, sp.findIndex(s => s.row > row)); if (CPOS < 0 || !sp.length) CPOS = 0; showSpot();
}
function showSpot() {
  const sp = CHK?.spots || [], el = $("chkCard"); if (!el) return;
  CUR = sp.length ? sp[Math.min(CPOS, sp.length - 1)] : null;
  if (!CUR) { el.innerHTML = `<div class="okc"><b>Nothing left to confirm.</b></div><div class="note">${(CHK?.decisions || []).length} answered. Continue when the trace looks right.</div>`; draw(); return; }
  const cols = ["#2563eb", "#f97316", "#16a34a"];
  el.innerHTML = `<div class="note">Place ${CPOS + 1} of ${sp.length} · ${CUR.kind === "wrap" ? "edge" : "two lines"} near row ${CUR.row.toFixed(0)}</div>
    <div class="q"><b>${esc(CUR.question)}</b></div>
    ${CUR.choices.map((ch, i) => `<div class="row"><button data-ch="${ch.id}" style="width:100%;border-left:6px solid ${cols[i % 3]}">${i + 1} · ${esc(ch.label)}</button></div>`).join("")}
    <div class="row"><button data-ch="skip">Not sure - leave it <span class="kbd">K</span></button><button id="cPrev">◀ Previous</button><button id="cNext">Next ▶</button></div>`;
  el.querySelectorAll("[data-ch]").forEach(b => b.onclick = () => answer(b.dataset.ch));
  on("cPrev", () => { CPOS = (CPOS - 1 + sp.length) % sp.length; showSpot(); }); on("cNext", () => { CPOS = (CPOS + 1) % sp.length; showSpot(); });
  const h = Math.max(ST.width * 0.5, (CUR.row1 - CUR.row0) / 2 + 150); fitRows(CUR.row - h, CUR.row + h); draw();
}
R.checks = async () => {
  $("stepLead").textContent = "At these places the curve could go two ways; answer each question and the trace follows your answer.";
  $("stepBody").innerHTML = `<div class="card" id="chkCard">looking for crossings and edges…</div><div class="card" id="prog7"></div>
    ${KEYS("<b>1</b> / <b>2</b> pick the numbered circle on the scan · <b>K</b> leave it as it is", "Blue circle = choice 1, orange = choice 2 · yellow band = the place in question", "Each answer pins the trace there (saved with the session) · <b>Z</b> undoes", NAV)}`;
  CPOS = 0; await loadChecks(); showSpot(); renderProgress();
  main("Done checking — continue", async () => next());
  help(`<b>1</b>/<b>2</b> answer · <b>K</b> leave it · <b>Z</b> undo · ${NAV}`);
};

R.qc = async () => {
  $("stepLead").textContent = "Your traced curve next to the reference LAS file for this scan, so you can see how close it is.";
  $("stepBody").innerHTML = `<div id="qcm"></div><div class="note" id="qcn"></div>${KEYS("Go back to step 7 to fix the trace; the QC updates automatically", NAV)}`;
  main("Continue to export", async () => next());
  const q = await api("GET", E("/qc"));
  if (!q.available) { $("qcm").innerHTML = `<div class="card">No QC: ${esc(q.reason)}. You can still export.</div>`; return; }
  const m = q.metrics;
  $("qcm").innerHTML = `<div class="metrics"><div><b>${fmt(m.r, 3)}</b><span>r</span></div><div><b>${fmt(m.rmse, 2)}</b><span>RMSE</span></div><div><b>${fmt(m.bias, 2)}</b><span>bias</span></div><div><b>${m.n}</b><span>n</span></div></div>`;
  $("qcn").textContent = `depth shift ${m.shift_ft} ft · ${m.n_anchors} anchors` + (ST.calibration.scale.type === "log" ? " · log scale compared as log10" : "");
  const lg = q.depth.map((d, i) => [q.ref[i], d]), dg = q.depth.map((d, i) => [q.digitized[i], d]);
  const xs = q.ref.concat(q.digitized).filter(v => v != null && isFinite(v)), lo = Math.min(...xs), hi = Math.max(...xs);
  fig(`<div class="ttl">${esc(ST.well)} ${esc(ST.scan.split("/").pop())} ${esc(ST.curve)}</div><div class="legend"><span><i class="swatch" style="background:#000"></i>reference LAS</span><span><i class="swatch" style="background:#d62728"></i>digitized</span></div>
    <div style="display:flex;gap:6px">${chart({series: [{pts: lg, color: "#000", lw: 0.6}, {pts: dg, color: "#d62728", lw: 0.6, alpha: 0.8}], invertY: true, xlabel: ST.curve, ylabel: "Depth (ft)", w: 260, h: 560})}
    ${chart({title: `r=${fmt(m.r, 3)} RMSE=${fmt(m.rmse, 2)} n=${m.n}`, series: [{pts: q.ref.map((r, i) => [r, q.digitized[i]]), color: TAB10[0], dots: true, r: 1.2, alpha: 0.3}, {pts: [[lo, lo], [hi, hi]], color: "#000", dash: true, lw: 0.8}], xlabel: "reference LAS", ylabel: "digitized", w: 260, h: 260})}</div>`);
  help(NAV);
};

R.export = () => {
  const f = INFO.features, x = ST.export, hasLas = !!ST.las;
  $("stepLead").textContent = "Save the curve as a LAS file (plus CSV and a check picture). Nothing is saved until you press a button.";
  $("stepBody").innerHTML = `<h3>Export folder</h3><div class="row"><input id="xDir" value="${esc(x.folder)}" style="flex:1"><button id="xSet">Set</button></div>
    <div class="note">Files: <code>${esc(ST.name)}.las</code>, <code>.csv</code> (0.5 ft, same values as the LAS), <code>_qc.png</code>, <code>_session.json</code> (replayable).</div>
    <h3>Single files</h3><div class="row"><button data-x="las">LAS</button><button data-x="csv">CSV</button>
      <button data-x="qc_png" ${!f.plot.ok ? off(f.plot) : (!hasLas ? 'disabled title="no paired LAS"' : "")}>QC plot PNG</button><button data-x="session">Session JSON</button><button data-x="training" title="save image + corrected trace to training_data/corrections (for ML training)">Training sample</button></div>${hint(f.plot)}
    <div class="row"><button id="xOpen" ${x.exists ? "" : 'disabled title="created by the first export"'}>Open export folder</button></div>
    <h3>Written this session</h3><div class="list" id="xList">${x.written.length ? x.written.map(p => `<div>${esc(p.split(/[\\/]/).pop())}</div>`).join("") : '<span class="note">nothing yet</span>'}</div>
    ${KEYS("The blue button writes all available files; the single-file buttons write one each", "Re-exporting overwrites files of the same name in this folder")}`;
  const after = async r => { toast("written: " + r.written.map(p => p.split(/[\\/]/).pop()).join(", ")); await refresh(false); render(false); };
  on("xSet", async () => { await api("POST", A("/export/folder"), {path: $("xDir").value}); await refresh(false); render(false); });
  document.querySelectorAll("button[data-x]").forEach(b => b.onclick = async () => { try { await after(await api("POST", A("/export/" + b.dataset.x))); } catch (e) {} });
  on("xOpen", async () => api("POST", "/shell/api/open_folder", {path: ST.export.folder}));
  main("Export all", async () => { if ($("xDir").value !== x.folder) await api("POST", A("/export/folder"), {path: $("xDir").value}); await after(await api("POST", A("/export/all"))); Shell.complete(8, false); });
  help("Exports go to the folder above. " + NAV);
};

(async () => { INFO = await api("GET", "/app/api/info"); Shell.go(0); })();
