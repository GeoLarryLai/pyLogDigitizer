// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
// pyLogDigitizer local editor front end. All coordinates sent to the server are sheet pixels
// (image coordinates of the working, deskewed sheet shown in the viewer).
"use strict";
const $ = id => document.getElementById(id);
let SID = null, ST = null, OV = null, viewer = null, mode = "pan", boxPts = [], depthPending = [], CAT = [];

async function api(method, url, body) {
  setStatus("working: " + url.replace(/^\/api\/sessions\/\w+/, ""));
  const r = await fetch(url, {method, headers: {"Content-Type": "application/json"}, body: body ? JSON.stringify(body) : undefined});
  const j = r.headers.get("content-type")?.includes("json") ? await r.json() : null;
  if (!r.ok) { setStatus("error: " + (j?.detail || r.status), true); throw new Error(j?.detail || r.status); }
  setStatus("mode: " + mode); return j;
}
function setStatus(t, err) { $("status").textContent = t; $("status").style.background = err ? "#a00" : "#333"; }
const S = p => `/api/sessions/${SID}${p}`;
const fmt = (v, d = 3) => v == null ? "–" : (+v).toFixed(d);

async function loadCatalog() {
  CAT = await api("GET", "/api/catalog");
  $("catalog").innerHTML = CAT.map((c, i) => `<option value="${i}">${c.well} · ${c.scan.split("/").pop()} · ${c.curve}${c.las ? " (LAS)" : ""}</option>`).join("");
}

async function openScan() {
  $("headerTbl").innerHTML = ""; $("headerInfo").textContent = "";
  const c = CAT[+$("catalog").value];
  $("openInfo").textContent = "opening (layout + OCR + trace, ~10-60 s)...";
  const st = await api("POST", "/api/sessions", {scan: c.scan, curve: c.curve, well: c.well, las: c.las,
    settings: c.settings, auto_layout: $("autoLayout").checked});
  SID = st.sid; applyState(st);
  if (viewer) viewer.destroy();
  viewer = OpenSeadragon({id: "viewer", tileSources: S("/image.dzi"), showNavigationControl: false,
    gestureSettingsMouse: {clickToZoom: false, dblClickToZoom: true}, maxZoomPixelRatio: 4, minZoomImageRatio: 0.5});
  viewer.addHandler("canvas-click", onClick);
  viewer.addHandler("update-viewport", draw);
  viewer.addHandler("open", () => { const r = ST.section ? ST.section.rows[0] : 0;
    viewer.viewport.fitBoundsWithConstraints(viewer.viewport.imageToViewportRectangle(0, r, ST.width, ST.width * 1.2)); });
  await refreshOverlay();
}

function applyState(st) {
  ST = st;
  const e = Object.entries(st.errors || {}).map(([k, v]) => `${k}: ${v}`).join("\n");
  $("openInfo").innerHTML = `${st.width}×${st.height}px · backend ${st.backend}` + (e ? `\n<span class="err">${e}</span>` : "");
  $("trackSel").innerHTML = (st.tracks || []).filter(t => t.name != "depth").map(t => `<option ${st.section?.track == t.name ? "selected" : ""}>${t.name}</option>`).join("");
  const c = st.calibration || {};
  $("depthInfo").textContent = c.px_per_ft ? `${c.depth_source}: ${c.depth_anchors.length} anchors, ${fmt(c.px_per_ft)} px/ft` +
    (st.depth_top_base ? `\nsection ${fmt(st.depth_top_base[0], 0)}–${fmt(st.depth_top_base[1], 0)} ft` : "") : "no depth calibration";
  if (st.depth_top_base && !$("secTop").value) { $("secTop").value = st.depth_top_base[0].toFixed(0); $("secBase").value = st.depth_top_base[1].toFixed(0); }
  const s = c.scale || {};
  $("scL").value = s.left ?? ""; $("scR").value = s.right ?? ""; $("scT").value = s.type || "linear";
  $("bkL").value = s.backup ? s.backup[0] : ""; $("bkR").value = s.backup ? s.backup[1] : "";
  $("wrapSide").value = s.wrap_side || "right"; $("hatch").checked = !!s.use_hatch;
  const q = st.qc;
  $("qcInfo").textContent = q ? `vs paired LAS: r=${fmt(q.r)} RMSE=${fmt(q.rmse, 2)} bias=${fmt(q.bias, 2)} n=${q.n} shift=${q.shift_ft} ft` :
    (st.traced ? "no paired LAS for QC" : "not traced");
  if (st.traced) $("qcInfo").textContent += `\nwraps ${st.n_wraps} · low-confidence intervals ${st.low_conf_intervals}`;
}

async function refreshOverlay() { OV = await api("GET", S("/overlay")); listAnchors(); draw(); }
function listAnchors() {
  $("anchorList").innerHTML = (OV?.anchors || []).map((a, i) =>
    `#${i} row ${a[0].toFixed(0)} <a href="#" onclick="delAnchor(${i});return false">✕</a> <a href="#" onclick="goRow(${a[0]});return false">go</a>`).join("<br>");
}
window.delAnchor = async i => { applyState(await api("DELETE", S(`/anchors/${i}`))); await refreshOverlay(); };
window.goRow = r => viewer.viewport.fitBoundsWithConstraints(viewer.viewport.imageToViewportRectangle(0, r - ST.width * 0.5, ST.width, ST.width));

function draw() {
  const cv = $("ov"), ctx = cv.getContext("2d");
  cv.width = cv.clientWidth; cv.height = cv.clientHeight; ctx.clearRect(0, 0, cv.width, cv.height);
  if (!viewer || !OV || !viewer.world.getItemAt(0)) return;
  const P = (x, y) => viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(x, y));
  const top = viewer.viewport.viewerElementToImageCoordinates(new OpenSeadragon.Point(0, 0)).y;
  const bot = viewer.viewport.viewerElementToImageCoordinates(new OpenSeadragon.Point(0, cv.height)).y;
  const sec = OV.section;
  if (sec) {   // section outline
    ctx.strokeStyle = "rgba(0,120,255,.7)"; ctx.setLineDash([6, 4]);
    const a = P(0, sec.rows[0]), b = P(0, sec.rows[1]);
    ctx.beginPath(); ctx.moveTo(0, a.y); ctx.lineTo(cv.width, a.y); ctx.moveTo(0, b.y); ctx.lineTo(cv.width, b.y); ctx.stroke();
    if (sec.kind == "box") { const p0 = P(sec.x0, sec.rows[0]), p1 = P(sec.x1, sec.rows[1]); ctx.strokeRect(p0.x, p0.y, p1.x - p0.x, p1.y - p0.y); }
    ctx.setLineDash([]);
  }
  ctx.lineWidth = 2; let prev = null;
  for (const [r, x, off, c] of OV.trace) {
    if (r < top - 50 || r > bot + 50) { prev = null; continue; }
    const p = P(x, r);
    if (prev) { ctx.strokeStyle = off ? "rgba(255,140,0,.9)" : (c < 0.5 ? "rgba(200,0,200,.9)" : "rgba(230,0,0,.85)");
      ctx.beginPath(); ctx.moveTo(prev.x, prev.y); ctx.lineTo(p.x, p.y); ctx.stroke(); }
    prev = p;
  }
  ctx.fillStyle = "#00c040";
  for (const a of OV.anchors) { const p = P(a[1], a[0]); ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, 7); ctx.fill(); }
  ctx.fillStyle = "rgba(255,140,0,.15)";
  for (const [a, b] of (OV.backup_intervals || [])) { const p0 = P(0, a), p1 = P(0, b); ctx.fillRect(0, p0.y, cv.width, p1.y - p0.y); }
  ctx.strokeStyle = "#ff00ff"; ctx.lineWidth = 2;
  for (const a of (OV.excluded || [])) { const p = P(a[1], a[0]); ctx.beginPath(); ctx.moveTo(p.x - 6, p.y - 6); ctx.lineTo(p.x + 6, p.y + 6);
    ctx.moveTo(p.x + 6, p.y - 6); ctx.lineTo(p.x - 6, p.y + 6); ctx.stroke(); }
  ctx.fillStyle = "#0060ff"; ctx.font = "12px sans-serif";
  for (const [r, d] of (OV.depth_anchors || []).concat(depthPending)) { if (r < top || r > bot) continue; const p = P(0, r);
    ctx.fillRect(0, p.y - 1, 18, 3); ctx.fillText(d.toFixed(0), 20, p.y + 4); }
  for (const b of boxPts) { const p = P(b[1], b[0]); ctx.strokeStyle = "#0060ff"; ctx.strokeRect(p.x - 5, p.y - 5, 10, 10); }
}

async function onClick(ev) {
  if (!ev.quick || mode == "pan") return;
  ev.preventDefaultAction = true;
  const ip = viewer.viewport.viewerElementToImageCoordinates(ev.position), row = ip.y, x = ip.x;
  if (mode == "anchor") {
    const r = await api("POST", S("/anchors"), {row, x, snap: $("snap").checked, window_ft: +$("winFt").value || 150});
    applyState(r); await refreshOverlay();
  } else if (mode == "wrap") {
    boxPts.push([row, x]); draw();
    if (boxPts.length == 2) { const [a, b] = boxPts; boxPts = [];
      applyState(await api("POST", S("/backup_interval"), {rows: [a[0], b[0]]})); await refreshOverlay(); }
  } else if (mode == "exclude") {
    applyState(await api("POST", S("/exclude"), {row, x})); await refreshOverlay();
  } else if (mode == "seed") {
    applyState(await api("POST", S("/trace"), {seed: [row, x]})); await refreshOverlay(); setMode("anchor");
  } else if (mode == "box") {
    boxPts.push([row, x]); draw();
    if (boxPts.length == 2) {
      const [a, b] = boxPts; boxPts = [];
      applyState(await api("POST", S("/section"), {box: [a[1], b[1]], rows: [Math.min(a[0], b[0]), Math.max(a[0], b[0])]}));
      await refreshOverlay(); setMode("seed");
    }
  } else if (mode == "depth") {
    const v = prompt(`Depth (ft) of the label at row ${row.toFixed(0)}?`);
    if (v !== null && v !== "" && isFinite(+v)) { depthPending.push([row, +v]); showPending(); draw(); }
  }
}
function showPending() { $("depthPending").textContent = depthPending.map(a => `row ${a[0].toFixed(0)} = ${a[1]} ft`).join("\n"); }

function setMode(m) {
  mode = m; boxPts = [];
  document.querySelectorAll("button.mode").forEach(b => b.classList.toggle("active", b.dataset.mode == m));
  setStatus("mode: " + m + (m == "pan" ? "" : "  (Esc = pan)"));
}
document.querySelectorAll("button.mode").forEach(b => b.onclick = () => setMode(mode == b.dataset.mode ? "pan" : b.dataset.mode));
document.addEventListener("keydown", e => {
  if (e.target.tagName == "INPUT") return;
  if (e.key == "Escape") setMode("pan"); if (e.key == "a") setMode("anchor"); if (e.key == "x") setMode("exclude"); if (e.key == "w") setMode("wrap");
  if (e.key == "z" && OV?.anchors?.length) delAnchor(OV.anchors.length - 1);
});
const act = (id, fn) => $(id).onclick = async () => { try { await fn(); } catch (e) { console.error(e); } };
act("openBtn", openScan);
act("useTrack", async () => { applyState(await api("POST", S("/section"), {track: $("trackSel").value})); await refreshOverlay(); });
act("applyRange", async () => { applyState(await api("POST", S("/section"), {track: ST.section?.kind == "track" ? ST.section.track : undefined,
  box: ST.section?.kind == "box" ? [ST.section.x0, ST.section.x1] : undefined, depth_range: [+$("secTop").value, +$("secBase").value]})); await refreshOverlay(); });
act("applyLabels", async () => { applyState(await api("POST", S("/depth"), {first_depth: $("firstDepth").value ? +$("firstDepth").value : null,
  label_interval: $("labelStep").value ? +$("labelStep").value : null})); await refreshOverlay(); });
act("applyDepth", async () => { if (depthPending.length < 2) { setStatus("click >= 2 depth labels first", true); return; }
  applyState(await api("POST", S("/depth"), {anchors: depthPending})); depthPending = []; showPending(); await refreshOverlay(); });
act("applyScale", async () => { const bk = $("bkL").value !== "" && $("bkR").value !== "" ? [+$("bkL").value, +$("bkR").value] : null;
  applyState(await api("POST", S("/scale"), {left: +$("scL").value, right: +$("scR").value, type: $("scT").value, backup: bk,
    wrap_side: $("wrapSide").value, use_hatch: $("hatch").checked})); });
// header OCR proposals: shown for review; "use" only fills the scale form (Apply scale applies it)
let HDR = null;
const flag = f => f.needs_review ? ` <span class="err" title="needs review (confidence ${fmt(f.confidence, 2)})">?</span>` : "";
const val = f => f.value == null ? "–" : (Array.isArray(f.value) ? f.value.join("…") : f.value);
act("proposeHeader", async () => {
  $("headerInfo").textContent = "reading the header (OCR, a few s)...";
  HDR = await api("GET", S("/proposed_header"));
  $("headerInfo").textContent = `${HDR.curves.length} proposed curve(s) · ${fmt(HDR.runtime_s, 1)} s · unconfirmed (? = needs review)`;
  $("headerTbl").innerHTML = "<tr><th>track</th><th>curve</th><th>unit</th><th>scale</th><th>type</th><th>backup</th><th></th></tr>" +
    HDR.curves.map((c, i) => `<tr><td>${c.track}</td><td title="${c.label.replace(/"/g, "")}">${val(c.mnemonic)}${flag(c.mnemonic)}</td>` +
      `<td>${val(c.unit)}${flag(c.unit)}</td><td>${val(c.scale_left)} → ${val(c.scale_right)}${flag(c.scale_left)}</td>` +
      `<td>${val(c.scale_type)}${flag(c.scale_type)}</td><td>${val(c.backup)}${flag(c.backup)}</td>` +
      `<td><button data-hdr="${i}">use</button></td></tr>`).join("");
  $("headerTbl").querySelectorAll("button[data-hdr]").forEach(b => b.onclick = () => {
    const c = HDR.curves[+b.dataset.hdr];
    if (c.scale_left.value != null) { $("scL").value = c.scale_left.value; $("scR").value = c.scale_right.value; }
    $("scT").value = c.scale_type.value || "linear";
    $("bkL").value = c.backup.value ? c.backup.value[0] : ""; $("bkR").value = c.backup.value ? c.backup.value[1] : "";
    if (c.backup.value && c.scale_left.value != null)
      $("wrapSide").value = Math.abs(c.backup.value[0] - c.scale_right.value) <= Math.abs(c.backup.value[1] - c.scale_left.value) ? "right" : "left";
    setStatus(`filled scale form from proposed ${c.mnemonic.value} (${c.track}) - check it, then Apply scale`);
  });
});
act("autoTrace", async () => { applyState(await api("POST", S("/trace"), {})); await refreshOverlay(); });
act("undoAnchor", async () => { if (OV?.anchors?.length) await delAnchor(OV.anchors.length - 1); });
act("retraceAll", async () => { applyState(await api("POST", S("/retrace_all"))); await refreshOverlay(); });
act("clearWrap", async () => { applyState(await api("POST", S("/backup_interval/clear"))); await refreshOverlay(); });
act("clearExcl", async () => { applyState(await api("POST", S("/exclude/clear"))); await refreshOverlay(); });
act("clearAnchors", async () => { applyState(await api("POST", S("/anchors/clear"))); await refreshOverlay(); });
act("qcPlot", async () => { $("qcImg").src = S("/qc.png?t=" + Date.now()); $("qcPane").hidden = false; });
act("qcClose", async () => { $("qcPane").hidden = true; });
act("exportLas", async () => { const r = await api("POST", S("/export_las")); $("saveInfo").innerHTML = `LAS: ${r.path} <a href="${r.url}">download</a>`; });
act("saveTrain", async () => { const r = await api("POST", S("/save_training")); $("saveInfo").textContent = "training sample: " + r.path; });
act("saveSession", async () => { const r = await api("POST", S("/save_session")); $("saveInfo").textContent = "session: " + r.path; });
window.addEventListener("resize", draw);
loadCatalog();
