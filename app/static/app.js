'use strict';
/* DepthWizard web app - map picker, upload, job progress, 2D views, 3D viewer bridge, downloads. */
const $ = id => document.getElementById(id);
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
const MODES_3D = new Set(['blocks', 'heat', 'surface']);
const MODE_NAMES = { satellite: 'Satellite', height: 'Height map', blocks: '3D blocks', heat: 'Colour by height',
  surface: 'Raw surface', contours: 'Contours', swipe: 'Satellite | Height map' };
const LEGEND_MODES = new Set(['height', 'heat', 'contours', 'swipe']);

// ---------------------------------------------------------------- turbo colour ramp (same as cv2.COLORMAP_TURBO)
const TURBO = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
    const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
    const b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
    lut[i * 3] = r * 255; lut[i * 3 + 1] = g * 255; lut[i * 3 + 2] = b * 255;
  }
  return lut;
})();
const turboCss = t => { const k = Math.round(Math.min(1, Math.max(0, t)) * 255) * 3;
  return `rgb(${TURBO[k]},${TURBO[k + 1]},${TURBO[k + 2]})`; };
document.querySelector('#legend .ramp').style.background =
  `linear-gradient(to right, ${Array.from({ length: 11 }, (_, i) => turboCss(i / 10)).join(',')})`;

const S = { source: 'map', file: null, watchToken: 0, R: null, mode: 'blocks', v3dReady: false, swipe: 0.5, nameAuto: false };
const fmtH = h => (h == null || !isFinite(h)) ? '–' : `${h.toFixed(1)} m`;

// ---------------------------------------------------------------- map picker
const map = L.map('map', { zoomControl: true, attributionControl: true });
L.tileLayer(ESRI, { maxZoom: 20, maxNativeZoom: 19, attribution: 'Esri World Imagery' }).addTo(map);
const areaStyle = { color: '#e0602a', weight: 2, fillOpacity: 0.08 };
const rect = L.rectangle([[0, 0], [0, 0]], areaStyle).addTo(map);
const dot = L.circleMarker([0, 0], { radius: 4, color: '#e0602a', fillColor: '#e0602a', fillOpacity: 1 }).addTo(map);

function areaBounds() {
  const lat = +$('lat').value, lon = +$('lon').value, half = +$('size').value / 2;
  const dLat = half / 111320, dLon = half / (111320 * Math.cos(lat * Math.PI / 180));
  return L.latLngBounds([lat - dLat, lon - dLon], [lat + dLat, lon + dLon]);
}
function updateArea(pan) {
  const lat = +$('lat').value, lon = +$('lon').value;
  if (!isFinite(lat) || !isFinite(lon) || Math.abs(lat) > 85 || Math.abs(lon) > 180) return;
  const b = areaBounds(); rect.setBounds(b); dot.setLatLng([lat, lon]);
  if (pan && !map.getBounds().contains(b)) map.fitBounds(b, { padding: [60, 60], maxZoom: 17 });
}
// a name filled in from a previous result belongs to that area -> clear it when the area moves
function areaMoved() { if (S.nameAuto) { $('name').value = ''; S.nameAuto = false; } }
map.on('click', e => {
  if (S.phase === 'result') return;
  $('lat').value = e.latlng.lat.toFixed(6); $('lon').value = e.latlng.lng.toFixed(6); areaMoved(); updateArea(false);
});
$('lat').addEventListener('change', () => { areaMoved(); updateArea(true); });
$('lon').addEventListener('change', () => { areaMoved(); updateArea(true); });
$('name').addEventListener('input', () => { S.nameAuto = false; });
$('size').addEventListener('input', () => { $('sizev').textContent = `${$('size').value} × ${$('size').value} m`; updateArea(false); });
updateArea(false);
map.fitBounds(areaBounds(), { padding: [60, 60], maxZoom: 17 });

// ---------------------------------------------------------------- source tabs + upload
function setSource(src) {
  S.source = src;
  $('tabMap').classList.toggle('on', src === 'map'); $('tabMap').setAttribute('aria-selected', src === 'map');
  $('tabUpload').classList.toggle('on', src === 'upload'); $('tabUpload').setAttribute('aria-selected', src === 'upload');
  $('paneMap').hidden = src !== 'map'; $('paneUpload').hidden = src !== 'upload';
  $('calib').closest('label').title = src === 'upload' ? 'Uses the city factor fitted in Bengaluru' : '';
  if (S.phase !== 'result') showPickStage();
}
$('tabMap').onclick = () => setSource('map');
$('tabUpload').onclick = () => setSource('upload');

function setFile(f) {
  if (!f || !f.type.startsWith('image/')) { showErr('Please choose an image file (JPG or PNG).'); return; }
  hideErr(); S.file = f; areaMoved();
  $('name').placeholder = f.name.replace(/\.[^.]+$/, '');
  $('dropText').innerHTML = `<b>${escapeHtml(f.name)}</b><br>${(f.size / 1048576).toFixed(1)} MB &middot; click to change`;
  if ($('upImg').src) URL.revokeObjectURL($('upImg').src);
  $('upImg').src = URL.createObjectURL(f); $('upEmpty').hidden = true;
}
$('file').addEventListener('change', e => setFile(e.target.files[0]));
const drop = $('drop');
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, () => drop.classList.remove('over')));
drop.addEventListener('drop', e => { e.preventDefault(); setFile(e.dataTransfer.files[0]); });

// ---------------------------------------------------------------- phases
function showPickStage() {
  $('c2d').hidden = true; $('v3d').hidden = true; $('legend').hidden = true; $('tip2d').hidden = true;
  $('map').hidden = S.source !== 'map'; $('uploadPreview').hidden = S.source !== 'upload';
  if (S.source === 'map') map.invalidateSize();
}
function setPhase(p) {
  S.phase = p;
  $('secSource').hidden = p === 'result'; $('secView').hidden = p !== 'result';
  if (p === 'pick') { showPickStage(); history.replaceState(null, '', location.pathname); }
  else { $('map').hidden = true; $('uploadPreview').hidden = true; }
}
$('newArea').onclick = () => { setPhase('pick'); $('prog').hidden = true; refreshRecent(); };

// ---------------------------------------------------------------- generate + progress
function showErr(msg) { $('err').textContent = msg; $('err').hidden = false; }
function hideErr() { $('err').hidden = true; }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]); }
async function apiError(res) {
  try { const j = await res.json(); const d = j.detail;
    return typeof d === 'string' ? d : Array.isArray(d) ? d.map(x => `${x.loc?.slice(-1)[0] ?? ''}: ${x.msg}`).join('; ') : res.statusText;
  } catch { return `${res.status} ${res.statusText}`; }
}

$('go').onclick = async () => {
  hideErr();
  let res;
  try {
    if (S.source === 'map') {
      const body = { lat: +$('lat').value, lon: +$('lon').value, size_m: +$('size').value, calibrate: $('calib').checked,
        name: $('name').value.trim() };
      res = await fetch('/api/jobs/location', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    } else {
      if (!S.file) { showErr('Choose an image to upload first.'); return; }
      const fd = new FormData();
      fd.append('file', S.file); fd.append('gsd', $('gsd').value); fd.append('calibrate', $('calib').checked);
      fd.append('name', $('name').value.trim());
      setProgress({ pct: 0, stage: 'Uploading image…', log: [] });
      res = await fetch('/api/jobs/upload', { method: 'POST', body: fd });
    }
    if (!res.ok) throw new Error(await apiError(res));
    watch((await res.json()).job_id);
  } catch (e) {
    $('prog').hidden = true; showErr(e.message || String(e)); $('go').disabled = false;
  }
};

function setProgress(s) {
  $('prog').hidden = false; $('go').disabled = true;
  const pct = Math.round(s.pct || 0);
  $('barFill').style.width = pct + '%'; $('bar').setAttribute('aria-valuenow', pct);
  $('stage').textContent = s.stage || ''; $('pct').textContent = pct + '%';
  $('log').textContent = (s.log || []).slice(-4).join('\n');
}
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function watch(id) {
  const token = ++S.watchToken;
  while (token === S.watchToken) {
    let s;
    try { const r = await fetch(`/api/jobs/${id}`); if (!r.ok) throw new Error(await apiError(r)); s = await r.json(); }
    catch (e) { showErr(`Lost contact with the server: ${e.message}`); $('go').disabled = false; return; }
    setProgress(s);
    if (s.state === 'done') { $('go').disabled = false; await openResult(id); $('prog').hidden = true; return; }
    if (s.state === 'error') { $('go').disabled = false; $('prog').hidden = true; showErr(s.error || 'Something went wrong.'); return; }
    await sleep(700);
  }
}

// ---------------------------------------------------------------- result
async function openResult(id) {
  $('busy').hidden = false;
  try {
    const base = `/api/jobs/${id}/files`;
    const [meta, buf, img] = await Promise.all([
      fetch(`${base}/meta.json`).then(r => { if (!r.ok) throw new Error('Result not found'); return r.json(); }),
      fetch(`${base}/height.bin`).then(r => r.arrayBuffer()),
      new Promise((ok, fail) => { const im = new Image(); im.onload = () => ok(im); im.onerror = () => fail(new Error('Could not load image')); im.src = `${base}/image.jpg`; }),
    ]);
    const u16 = new Uint16Array(buf), grid = new Float32Array(u16.length);
    let peak = 0;
    for (let i = 0; i < u16.length; i++) { grid[i] = u16[i] / 100; if (grid[i] > peak) peak = grid[i]; }
    S.R = { id, meta, img, grid, peak, W: meta.W, H: meta.H, gw: meta.gw, gh: meta.gh, hmax: meta.hmax,
      heat: heatCanvas(grid, meta.gw, meta.gh, meta.hmax), contours: {}, cgrid: null, view: null };
    $('interval').value = meta.hmax <= 30 ? '2' : '5';
    $('legMax').textContent = `${meta.hmax}+ m`;
    $('resTitle').textContent = meta.title;
    $('dlZip').href = `/api/jobs/${id}/heightmap.zip`;
    renderStats(meta);
    if (meta.kind === 'location') {                   // "New area" starts from this area (e.g. to rename it)
      const p = meta.params;
      $('lat').value = p.lat; $('lon').value = p.lon; $('size').value = p.size_m;
      $('sizev').textContent = `${p.size_m} × ${p.size_m} m`; $('calib').checked = !!p.calibrate;
      $('name').value = p.name || ''; S.nameAuto = true; updateArea(false);
      if (S.source !== 'map') setSource('map');
    }
    setPhase('result');
    history.replaceState(null, '', `#job=${id}`);
    S.v3dReady = false; $('v3d').src = `/api/jobs/${id}/viewer`;
    setMode(S.mode);
    refreshRecent();
  } catch (e) { showErr(e.message); }
  finally { $('busy').hidden = true; }
}

function renderStats(m) {
  const s = m.stats || {}, c = s.calibration || {};
  let h = `<b>${s.osm + s.auto}</b> buildings` + (m.kind === 'location' ? ` (${s.osm} OpenStreetMap, ${s.auto} auto-detected)` : ' (auto-detected)');
  if (m.osm_timeout) h += '<br><span class="muted">OpenStreetMap did not answer in time. Generate again later to add its outlines.</span>';
  h += `<br>${m.size_m[0]} × ${m.size_m[1]} m &middot; ${m.gsd.toFixed(2)} m/px &middot; tallest ${m.hpeak} m`;
  if (c.mode === 'load') {
    h += `<br>Calibrated: city factor k=${c.k} (fitted on ${escapeHtml(c.fit_on)})`;
    if (c.test_mae_after != null) h += `<br>Check vs OSM floors here: <b>${c.test_mae_after} m</b> avg error on ${c.n} buildings`;
  } else {
    h += '<br>Raw model heights (not calibrated)';
    if (s.validation && s.validation.mae_m != null) h += `<br>Check vs OSM floors: <b>${s.validation.mae_m} m</b> avg error on ${s.validation.n_in_range} buildings`;
  }
  $('stats').innerHTML = h;
}

// ---------------------------------------------------------------- view modes
document.querySelectorAll('#modes button').forEach(b => b.onclick = () => setMode(b.dataset.mode));
function setMode(m) {
  S.mode = m;
  document.querySelectorAll('#modes button').forEach(b => {
    const on = b.dataset.mode === m; b.classList.toggle('on', on); b.setAttribute('aria-pressed', on); });
  const is3d = MODES_3D.has(m);
  $('c2d').hidden = is3d; $('v3d').hidden = !is3d;
  $('opt3d').hidden = !is3d; $('opt2d').hidden = is3d; $('optContour').hidden = m !== 'contours';
  $('legend').hidden = !LEGEND_MODES.has(m); $('swipe').hidden = m !== 'swipe';
  $('tip2d').hidden = true; $('readout').textContent = '–';
  if (is3d) post({ type: 'mode', mode: m });
  else draw();
}

// ---------------------------------------------------------------- 3D viewer bridge (see viewer_bridge.js)
function post(msg) { if (S.v3dReady) $('v3d').contentWindow.postMessage(msg, location.origin); }
$('v3d').addEventListener('load', () => {
  if (!S.R || !$('v3d').src) return;
  S.v3dReady = true;
  post({ type: 'config', hmax: S.R.hmax, lut: Array.from(TURBO) });
  post({ type: 'ex', v: +$('ex').value });
  if (MODES_3D.has(S.mode)) post({ type: 'mode', mode: S.mode });
});
addEventListener('message', e => {
  if (e.origin !== location.origin || e.source !== $('v3d').contentWindow) return;
  const d = e.data || {};
  if (d.type === 'hover') $('readout').textContent = d.h == null ? '–' : (d.capped ? `${fmtH(d.h)}+` : fmtH(d.h));
  else if (d.type === 'shot') { const im = new Image(); im.onload = () => saveShot(im); im.src = d.url; }
});
$('ex').addEventListener('input', () => {
  const v = +$('ex').value;
  $('exv').textContent = v.toFixed(1) + '×' + (v === 1 ? ' (true scale)' : '');
  post({ type: 'ex', v });
});
$('fly').onclick = () => post({ type: 'fly' });
$('reset').onclick = () => post({ type: 'reset' });

// ---------------------------------------------------------------- 2D views (satellite, height map, contours)
const cv = $('c2d'), ctx = cv.getContext('2d');
let dpr = 1;
function sizeCanvas() {
  dpr = Math.min(devicePixelRatio || 1, 2);
  const r = cv.getBoundingClientRect(), w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  return r;
}
function fitView() {
  const r = sizeCanvas(), R = S.R, s = Math.min(r.width / R.W, r.height / R.H) * 0.96;
  R.view = { s, fit: s, tx: (r.width - R.W * s) / 2, ty: (r.height - R.H * s) / 2, cw: r.width, ch: r.height };
}
function heatCanvas(grid, gw, gh, hmax) {
  const c = document.createElement('canvas'); c.width = gw; c.height = gh;
  const x = c.getContext('2d'), im = x.createImageData(gw, gh), d = im.data;
  for (let i = 0; i < grid.length; i++) {
    const k = Math.round(Math.min(1, grid[i] / hmax) * 255) * 3;
    d[i * 4] = TURBO[k]; d[i * 4 + 1] = TURBO[k + 1]; d[i * 4 + 2] = TURBO[k + 2]; d[i * 4 + 3] = 255;
  }
  x.putImageData(im, 0, 0); return c;
}
function draw() {
  const R = S.R; if (!R || MODES_3D.has(S.mode) || $('c2d').hidden) return;
  const r = sizeCanvas();
  if (!R.view || r.width !== R.view.cw || r.height !== R.view.ch) fitView();   // stage was resized
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#16222e'; ctx.fillRect(0, 0, cv.width, cv.height);
  const { s, tx, ty } = R.view;
  ctx.setTransform(dpr * s, 0, 0, dpr * s, dpr * tx, dpr * ty);
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  if (S.mode === 'height') ctx.drawImage(R.heat, 0, 0, R.W, R.H);
  else ctx.drawImage(R.img, 0, 0, R.W, R.H);
  if (S.mode === 'contours') {
    ctx.fillStyle = 'rgba(22,34,46,0.55)'; ctx.fillRect(0, 0, R.W, R.H);
    drawContours(R, s);
  }
  if (S.mode === 'swipe') drawSwipe(R, r);
}

// swipe: satellite left of the divider, height map right. The divider stays put in screen space while you pan/zoom.
function drawSwipe(R, r) {
  const x = Math.round(S.swipe * r.width), { s, tx, ty } = R.view;
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.beginPath(); ctx.rect(x, 0, r.width - x, r.height); ctx.clip();
  ctx.setTransform(dpr * s, 0, 0, dpr * s, dpr * tx, dpr * ty); ctx.drawImage(R.heat, 0, 0, R.W, R.H);
  ctx.restore();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);                    // divider + labels (also end up in screenshots)
  ctx.fillStyle = 'rgba(0,0,0,0.35)'; ctx.fillRect(x - 2, 0, 4, r.height);
  ctx.fillStyle = '#fff'; ctx.fillRect(x - 1, 0, 2, r.height);
  ctx.font = '600 13px Barlow, "Segoe UI", sans-serif'; ctx.textBaseline = 'middle';
  const pill = (text, right) => {
    const w = ctx.measureText(text).width + 18, px = right ? x + 10 : x - 10 - w;
    ctx.fillStyle = 'rgba(238,234,224,0.92)'; ctx.beginPath(); ctx.roundRect(px, 12, w, 24, 4); ctx.fill();
    ctx.fillStyle = '#1d2833'; ctx.fillText(text, px + 9, 24);
  };
  if (x > 90) pill('Satellite', false);
  if (r.width - x > 110) pill('Height map', true);
  const el = $('swipe');
  el.style.left = x + 'px'; el.setAttribute('aria-valuenow', Math.round(S.swipe * 100));
}
function setSwipe(f) { S.swipe = Math.min(0.98, Math.max(0.02, f)); draw(); }
const sw = $('swipe');
sw.addEventListener('pointerdown', e => { sw.setPointerCapture(e.pointerId); sw.classList.add('drag'); e.preventDefault(); });
sw.addEventListener('pointermove', e => {
  if (!sw.hasPointerCapture(e.pointerId)) return;
  const r = cv.getBoundingClientRect(); setSwipe((e.clientX - r.left) / r.width);
});
sw.addEventListener('pointerup', () => sw.classList.remove('drag'));
sw.addEventListener('keydown', e => {
  const step = e.shiftKey ? 0.1 : 0.02;
  const f = { ArrowLeft: S.swipe - step, ArrowRight: S.swipe + step, Home: 0, End: 1 }[e.key];
  if (f !== undefined) { e.preventDefault(); setSwipe(f); }
});

// contours: smoothed, <=512 px copy of the height grid -> d3-contour (marching squares) -> Path2D
function contourGrid(R) {
  if (R.cgrid) return R.cgrid;
  const f = Math.max(1, Math.ceil(Math.max(R.gw, R.gh) / 512)), w = Math.floor(R.gw / f), h = Math.floor(R.gh / f);
  let a = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let j = 0; j < f; j++) for (let i = 0; i < f; i++) sum += R.grid[(y * f + j) * R.gw + x * f + i];
    a[y * w + x] = sum / (f * f);
  }
  for (let pass = 0; pass < 2; pass++) {                      // light 3x3 blur -> cleaner lines
    const b = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let sum = 0, n = 0;
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
        const yy = y + j, xx = x + i; if (yy < 0 || xx < 0 || yy >= h || xx >= w) continue; sum += a[yy * w + xx]; n++; }
      b[y * w + x] = sum / n;
    }
    a = b;
  }
  return (R.cgrid = { a, w, h });
}
function contourSet(R, iv) {
  if (R.contours[iv]) return R.contours[iv];
  const { a, w, h } = contourGrid(R), th = [];
  for (let v = iv; v < R.peak && th.length < 200; v += iv) th.push(v);
  const path = d3.geoPath(), minArea = 12 / (R.meta.gsd * R.W / w) ** 2;   // drop blobs < ~12 m² (tree/noise specks)
  const ringArea = ring => { let s = 0; for (let i = 0, j = ring.length - 1; i < ring.length; j = i++)
    s += (ring[j][0] + ring[i][0]) * (ring[j][1] - ring[i][1]); return Math.abs(s / 2); };
  return (R.contours[iv] = d3.contours().size([w, h]).thresholds(th)(a)
    .map(c => ({ ...c, coordinates: c.coordinates.filter(poly => ringArea(poly[0]) >= minArea) }))
    .filter(c => c.coordinates.length)
    .map(c => ({ value: c.value, major: Math.round(c.value / iv) % 5 === 0, path: new Path2D(path(c)) })));
}
function drawContours(R, s) {
  const iv = +$('interval').value, set = contourSet(R, iv), { w, h } = R.cgrid;
  const k = R.W / w, px = 1 / (s * k);
  ctx.save(); ctx.scale(R.W / w, R.H / h); ctx.lineJoin = 'round';
  for (const c of set) {
    ctx.strokeStyle = turboCss(c.value / R.hmax); ctx.lineWidth = (c.major ? 2 : 0.9) * px; ctx.stroke(c.path);
  }
  ctx.restore();
}
$('interval').addEventListener('change', draw);

// hover, zoom, pan
const tip = $('tip2d');
let drag = null;
function imgPoint(e) {
  const r = cv.getBoundingClientRect(), v = S.R.view;
  return { x: (e.clientX - r.left - v.tx) / v.s, y: (e.clientY - r.top - v.ty) / v.s, cx: e.clientX - r.left, cy: e.clientY - r.top };
}
cv.addEventListener('pointerdown', e => {
  drag = { x: e.clientX, y: e.clientY, tx: S.R.view.tx, ty: S.R.view.ty }; cv.setPointerCapture(e.pointerId); cv.classList.add('drag');
});
cv.addEventListener('pointerup', () => { drag = null; cv.classList.remove('drag'); });
cv.addEventListener('pointermove', e => {
  const R = S.R; if (!R) return;
  if (drag) { R.view.tx = drag.tx + e.clientX - drag.x; R.view.ty = drag.ty + e.clientY - drag.y; draw(); }
  const p = imgPoint(e);
  if (p.x < 0 || p.y < 0 || p.x >= R.W || p.y >= R.H) { tip.hidden = true; $('readout').textContent = '–'; return; }
  const gx = Math.min(R.gw - 1, Math.floor(p.x * R.gw / R.W)), gy = Math.min(R.gh - 1, Math.floor(p.y * R.gh / R.H));
  const h = R.grid[gy * R.gw + gx];
  $('readout').textContent = fmtH(h);
  tip.innerHTML = `<b>${fmtH(h)}</b>`; tip.hidden = false;
  tip.style.left = (p.cx + 14) + 'px'; tip.style.top = (p.cy + 14) + 'px';
});
cv.addEventListener('pointerleave', () => { tip.hidden = true; $('readout').textContent = '–'; });
cv.addEventListener('wheel', e => {
  e.preventDefault();
  const v = S.R.view, p = imgPoint(e);
  const ns = Math.min(v.fit * 24, Math.max(v.fit * 0.5, v.s * Math.exp(-e.deltaY * 0.0015)));
  v.tx = p.cx - p.x * ns; v.ty = p.cy - p.y * ns; v.s = ns; draw();
}, { passive: false });
cv.addEventListener('dblclick', () => { fitView(); draw(); });
addEventListener('resize', () => { if (S.R && !MODES_3D.has(S.mode)) draw(); });

// ---------------------------------------------------------------- downloads
$('shot').onclick = () => { if (MODES_3D.has(S.mode)) post({ type: 'shot' }); else saveShot(cv); };
function saveShot(src) {
  const c = document.createElement('canvas'); c.width = src.width; c.height = src.height;
  const x = c.getContext('2d'); x.drawImage(src, 0, 0);
  const k = Math.max(1, c.width / 1400);                        // scale overlay text with image size
  const pad = 12 * k;
  // caption (bottom-left)
  x.font = `600 ${14 * k}px Barlow, "Segoe UI", sans-serif`;
  const cap = `DepthWizard · ${S.R.meta.title} · ${MODE_NAMES[S.mode]}`;
  const cw = x.measureText(cap).width + 2 * pad;
  x.fillStyle = 'rgba(238,234,224,0.92)'; x.fillRect(pad, c.height - pad - 30 * k, cw, 30 * k);
  x.fillStyle = '#1d2833'; x.textBaseline = 'middle'; x.fillText(cap, 2 * pad, c.height - pad - 15 * k);
  if (LEGEND_MODES.has(S.mode)) {                               // legend (bottom-right)
    const lw = 160 * k, bw = lw + 110 * k, bh = 30 * k, bx = c.width - pad - bw, by = c.height - pad - bh;
    x.fillStyle = 'rgba(238,234,224,0.92)'; x.fillRect(bx, by, bw, bh);
    const g = x.createLinearGradient(bx + 42 * k, 0, bx + 42 * k + lw, 0);
    for (let i = 0; i <= 10; i++) g.addColorStop(i / 10, turboCss(i / 10));
    x.fillStyle = g; x.fillRect(bx + 42 * k, by + 10 * k, lw, 10 * k);
    x.fillStyle = '#1d2833'; x.font = `500 ${13 * k}px Barlow, "Segoe UI", sans-serif`;
    x.fillText('0 m', bx + 10 * k, by + bh / 2); x.fillText(`${S.R.hmax}+ m`, bx + 50 * k + lw, by + bh / 2);
  }
  c.toBlob(b => {
    const a = document.createElement('a'); a.href = URL.createObjectURL(b);
    a.download = `depthwizard_${S.R.meta.title.replace(/[^\w.-]+/g, '_')}_${S.mode}.png`;
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }, 'image/png');
}

// ---------------------------------------------------------------- recent results + startup
async function refreshRecent() {
  try {
    const list = await (await fetch('/api/jobs')).json();
    $('secRecent').hidden = !list.length;
    $('recent').innerHTML = '';
    for (const j of list) {
      const li = document.createElement('li'), b = document.createElement('button');
      const sub = [j.size_m ? `${j.size_m[0]} m` : '', `${j.buildings} bldg`, j.calibrated ? '' : 'raw'].filter(Boolean).join(' · ');
      b.innerHTML = `<span class="t${j.named ? '' : ' coords'}">${escapeHtml(j.title)}</span><small>${sub}</small>`;
      b.title = j.named ? j.title : 'Unnamed area. Open it, then “New area” to give it a name';
      b.classList.toggle('cur', S.phase === 'result' && S.R && S.R.id === j.id);
      b.onclick = () => { hideErr(); openResult(j.id); };
      li.appendChild(b); $('recent').appendChild(li);
    }
  } catch { /* recent list is optional */ }
}
(async function init() {
  const m = location.hash.match(/job=([0-9a-f]{12})/);         // read before setPhase clears the hash
  setPhase('pick');
  try {
    const c = await (await fetch('/api/config')).json();
    if (c.calibration) $('calibK').textContent = `(k = ${c.calibration.k}, fitted on ${c.calibration.fit_on})`;
    else { $('calib').checked = false; $('calib').disabled = true; $('calibK').textContent = '(no runs/city_calib.json)'; }
    if (c.weights) $('weights').textContent = c.weights.replace(/\\/g, '/');
  } catch { /* server offline - Generate will report it */ }
  refreshRecent();
  if (m) openResult(m[1]);
})();
