// OpenReco UI — industry-standard layout: menu bar, Workspace/Reference tree, Model/Photo viewport,
// Console/Photos/Jobs dock, Property pane. Plus CRS picker, layer visibility, measurement, GCP picking.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

let STAGES = {};      // type -> {default_params, params_schema, ...}
let PROJECT = null;   // {name, crs, chunks:[...], layers:[...]}
let WORKFLOWS = [];
let selected = null;
let ACTIVE_CHUNK = "Chunk 1";
const visible = new Set();          // layer ids currently shown in the 3D view
const objects = new Map();          // layer id -> THREE object (cached once loaded)

// industry-standard tree categories: stage type -> tree group label.
const CATEGORY = {
  ingest: "Cameras", sfm: "Tie Points", refine: "Tie Points", markers: "Markers",
  mvs: "Dense Cloud", fuse: "Dense Cloud", merge_chunks: "Dense Cloud", classify: "Point Cloud",
  mesh: "3D Model", texture: "3D Model", splat: "3D Model", tiles: "Tiled Model",
  dsm: "DEM", ortho: "Orthomosaic", contours: "Shapes", indices: "Orthomosaic",
  volume: "Shapes", profile: "Shapes", panorama: "Orthomosaic",
};
const CAT_ORDER = ["Cameras", "Tie Points", "Markers", "Dense Cloud", "Point Cloud",
                   "3D Model", "Tiled Model", "DEM", "Orthomosaic", "Shapes", "Other"];

const $ = (id) => document.getElementById(id);
const log = (m) => { const l = $('log'); l.textContent += `\n${m}`; l.scrollTop = l.scrollHeight; };

// ---- 3D viewport ----------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ canvas: $('c'), antialias: true });
renderer.setPixelRatio(devicePixelRatio);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0e14);
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1e7);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const dl = new THREE.DirectionalLight(0xffffff, 0.7); dl.position.set(1, 1, 1); scene.add(dl);
const measureGroup = new THREE.Group(); scene.add(measureGroup);
function resize() { const w = $('center').clientWidth, h = $('center').clientHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
addEventListener('resize', resize);
(function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();

function frameAll() {
  const box = new THREE.Box3();
  objects.forEach((o, id) => { if (visible.has(id)) box.expandByObject(o); });
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3()).length(), c = box.getCenter(new THREE.Vector3());
  controls.target.copy(c); camera.position.copy(c).add(new THREE.Vector3(size*.6, size*.5, size*.6));
  camera.near = size/1000; camera.far = size*10; camera.updateProjectionMatrix();
}

// load an artifact into a THREE object (mesh / point cloud / gaussian splat as points)
function loadObject(layer) {
  return new Promise((resolve) => {
    const v = viewable(layer);
    if (!v) return resolve(null);
    const url = `/api/file?path=${encodeURIComponent(v.path)}`;
    if (v.kind === 'glb') {
      new GLTFLoader().load(url, g => resolve(g.scene), undefined, () => resolve(null));
      return;
    }
    new PLYLoader().load(url, geo => {
      geo.computeBoundingBox();
      let o;
      if (geo.index) {
        if (!geo.getAttribute('normal')) geo.computeVertexNormals();
        o = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
          vertexColors: !!geo.getAttribute('color'), flatShading: true, side: THREE.DoubleSide }));
      } else {
        // points / gaussian splats (splat .ply has no faces) -> render as colored points
        o = new THREE.Points(geo, new THREE.PointsMaterial({
          size: v.splat ? 2 : 1, sizeAttenuation: false, vertexColors: !!geo.getAttribute('color') }));
      }
      if (!geo.getAttribute('color')) o.material.color.set(0x89b4fa);
      resolve(o);
    }, undefined, () => resolve(null));
  });
}
// pick a viewable artifact (textured glb > mesh ply > splat > points)
function viewable(layer) {
  const a = layer.artifacts || {};
  if (a.splat) return { path: a.splat, kind: 'ply', splat: true };
  for (const [k, kind] of [['glb','glb'],['mesh','ply'],['points','ply'],['merged','ply'],['sparse_ply','ply']])
    if (a[k]) return { path: a[k], kind };
  return null;
}
async function setVisible(layer, on) {
  if (on) {
    if (!objects.has(layer.id)) {
      const o = await loadObject(layer);
      if (!o) { log(`(no viewable geometry for ${layer.id})`); return; }
      objects.set(layer.id, o); scene.add(o);
    }
    objects.get(layer.id).visible = true; visible.add(layer.id); frameAll();
  } else {
    visible.delete(layer.id);
    if (objects.has(layer.id)) objects.get(layer.id).visible = false;
  }
  renderWorkspace();
}

// ---- measurement (distance / area) ----------------------------------------
let measureMode = null;           // null | 'dist' | 'area'
let measurePts = [];
const raycaster = new THREE.Raycaster(); raycaster.params.Points.threshold = 0.5;
function setMeasure(mode) {
  measureMode = (measureMode === mode) ? null : mode;
  measurePts = []; measureGroup.clear();
  $('distBtn').classList.toggle('on', measureMode === 'dist');
  $('areaBtn').classList.toggle('on', measureMode === 'area');
  $('measure').classList.toggle('show', !!measureMode);
  $('measure').textContent = measureMode ? `Click points on the model (${measureMode})` : '';
}
$('distBtn').onclick = () => setMeasure('dist');
$('areaBtn').onclick = () => setMeasure('area');
$('clearMeasBtn').onclick = () => { measurePts = []; measureGroup.clear(); $('measure').textContent = ''; };
renderer.domElement.addEventListener('pointerdown', (e) => {
  if (!measureMode || e.button !== 0) return;
  const r = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX-r.left)/r.width)*2-1, -((e.clientY-r.top)/r.height)*2+1);
  raycaster.setFromCamera(ndc, camera);
  const targets = [...objects.entries()].filter(([id]) => visible.has(id)).map(([, o]) => o);
  const hit = raycaster.intersectObjects(targets, true)[0];
  if (!hit) return;
  measurePts.push(hit.point.clone());
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0), new THREE.MeshBasicMaterial());
  measureGroup.add(new THREE.Points(new THREE.BufferGeometry().setFromPoints([hit.point]),
    new THREE.PointsMaterial({ size: 8, sizeAttenuation: false, color: 0xf9e2af })));
  redrawMeasure();
});
function redrawMeasure() {
  // keep only the marker points; rebuild the connecting line + readout
  [...measureGroup.children].filter(c => c.isLine).forEach(c => measureGroup.remove(c));
  if (measurePts.length >= 2) {
    const pts = measureMode === 'area' ? [...measurePts, measurePts[0]] : measurePts;
    measureGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0xf9e2af })));
  }
  let txt = '';
  if (measureMode === 'dist') {
    let d = 0; for (let i = 1; i < measurePts.length; i++) d += measurePts[i].distanceTo(measurePts[i-1]);
    txt = `distance: ${d.toFixed(3)} m  (${measurePts.length} pts)`;
  } else if (measureMode === 'area' && measurePts.length >= 3) {
    txt = `area: ${polygonArea(measurePts).toFixed(3)} m²  ·  perimeter: ${perimeter(measurePts).toFixed(3)} m`;
  } else {
    txt = `picked ${measurePts.length} point(s)`;
  }
  $('measure').textContent = txt;
}
function perimeter(p) { let s = 0; for (let i = 0; i < p.length; i++) s += p[i].distanceTo(p[(i+1)%p.length]); return s; }
function polygonArea(p) {            // 3D polygon area via the cross-product (Newell) method
  const n = new THREE.Vector3();
  for (let i = 0; i < p.length; i++) n.add(new THREE.Vector3().crossVectors(p[i], p[(i+1)%p.length]));
  return Math.abs(n.length()) / 2;
}

// ---- data + workspace tree ------------------------------------------------
async function loadStages() {
  STAGES = {}; for (const s of await (await fetch('/api/stages')).json()) STAGES[s.type] = s;
  const sel = $('newType'); sel.innerHTML = '';
  Object.keys(STAGES).filter(t => !t.startsWith('dummy')).sort().forEach(t => {
    const o = document.createElement('option'); o.value = t; o.textContent = t; sel.appendChild(o); });
}
async function loadProject() {
  PROJECT = await (await fetch('/api/project')).json();
  if ($('pname')) $('pname').textContent = PROJECT.name || '';
  document.title = `${PROJECT.name || 'OpenReco'} — OpenReco`;
  $('crsLabel').textContent = PROJECT.crs || 'CRS';
  $('refCrs').textContent = PROJECT.crs ? `Project CRS: ${PROJECT.crs}` : 'No CRS set (local frame).';
  if (!PROJECT.chunks.includes(ACTIVE_CHUNK)) ACTIVE_CHUNK = PROJECT.chunks[0] || "Chunk 1";
  renderWorkspace();
}
// industry-standard icons per category / item
const CAT_ICON = { Cameras:'📷', "Tie Points":'·:·', "Dense Cloud":'☁', "Point Cloud":'⛰',
  "3D Model":'△', "Tiled Model":'▦', DEM:'▒', Orthomosaic:'🗺', Shapes:'⬡', Markers:'📍', Other:'◇' };
const collapsed = new Set();           // node ids that are collapsed (everything expanded by default)
const isOpen = (id) => !collapsed.has(id);
function toggle(id) { if (collapsed.has(id)) collapsed.delete(id); else collapsed.add(id); renderWorkspace(); }

// a short metric badge for a layer item (points / faces / images / area …)
function layerMetric(L) {
  const m = L.metrics || {};
  const num = (x) => Number(x).toLocaleString();
  if (m.num_points != null) return `${num(m.num_points)} pts`;
  if (m.total_points != null) return `${num(m.total_points)} pts`;
  if (m.faces != null) return `${num(m.faces)} faces`;
  if (m.kept != null) return `${num(m.kept)} imgs`;
  if (m.total != null) return `${num(m.total)} imgs`;
  if (m.num_registered != null) return `${num(m.num_registered)} cams`;
  return '';
}

function row({ depth, id, hasKids, icon, label, count, cls = '', sel = false, disabled = false,
              chk = null, eye = null, dot = null, onClick, onDbl, onCtx, drag = null, drop = null }) {
  const d = document.createElement('div');
  d.className = 'tnode ' + cls + (sel ? ' sel' : '') + (disabled ? ' off' : '');
  d.style.paddingLeft = (depth * 14 + 6) + 'px';
  const car = hasKids ? (isOpen(id) ? '▾' : '▸') : '';
  let html = `<span class="car">${car}</span>`;
  if (chk !== null) html += `<input type="checkbox" class="en" ${chk.checked ? 'checked' : ''}>`;
  if (eye !== null) html += `<span class="eye ${eye ? 'on' : ''}">${eye === undefined ? '·' : '👁'}</span>`;
  html += `<span class="ico">${icon || ''}</span><span class="lbl">${label}</span>`;
  if (dot !== null) html = html.replace('<span class="ico">', `<span class="dot ${dot||''}"></span><span class="ico">`);
  if (count) html += `<span class="cnt">${count}</span>`;
  d.innerHTML = html;
  const caret = d.querySelector('.car');
  if (hasKids) caret.onclick = (e) => { e.stopPropagation(); toggle(id); };
  const cb = d.querySelector('.en');
  if (cb && chk && chk.onToggle) cb.onclick = (e) => { e.stopPropagation(); chk.onToggle(cb.checked); };
  if (onClick) d.onclick = onClick;
  if (onDbl) d.ondblclick = onDbl;
  if (onCtx) d.oncontextmenu = (e) => { e.preventDefault(); onCtx(e); };
  if (drag) { d.draggable = true; d.ondragstart = (e) => { e.dataTransfer.setData('text/plain', drag);
    e.dataTransfer.effectAllowed = 'move'; }; }
  if (drop) {
    d.ondragover = (e) => { e.preventDefault(); d.classList.add('dragover'); };
    d.ondragleave = () => d.classList.remove('dragover');
    d.ondrop = (e) => { e.preventDefault(); d.classList.remove('dragover'); drop(e.dataTransfer.getData('text/plain')); };
  }
  return d;
}

function renderWorkspace() {
  const el = $('workspace'); el.innerHTML = '';
  $('activeChunk').textContent = `active: ${ACTIVE_CHUNK}`;
  const byChunk = {}; PROJECT.chunks.forEach(c => byChunk[c] = []);
  PROJECT.layers.forEach(L => { (byChunk[L.chunk] = byChunk[L.chunk] || []).push(L); });

  // root: Workspace
  el.appendChild(row({ depth: 0, id: 'root', hasKids: true, icon: '🗂',
    label: 'Workspace', count: `${PROJECT.chunks.length} chunk(s)`,
    onClick: () => toggle('root'),
    onCtx: (e) => showCtx(e, [{ label: '＋ Add chunk…', fn: addChunk }]) }));
  if (!isOpen('root')) return;

  Object.keys(byChunk).forEach(chunk => {
    const cid = 'chunk:' + chunk;
    const layers = byChunk[chunk];
    const chunkOn = layers.length === 0 || layers.some(L => L.enabled !== false);
    el.appendChild(row({ depth: 1, id: cid, hasKids: true, cls: 'chunk' + (chunk === ACTIVE_CHUNK ? ' active' : ''),
      icon: '📦', label: chunk, count: `${layers.length}`,
      chk: { checked: chunkOn, onToggle: (v) => chunkAction({ action: 'set_enabled', name: chunk, enabled: v }) },
      onClick: () => { ACTIVE_CHUNK = chunk; renderWorkspace(); loadPhotos(); },
      onDbl: () => { ACTIVE_CHUNK = chunk; renderWorkspace(); },
      drop: (id) => layerAction({ action: 'move', id, to: chunk }),    // drop a layer onto a chunk
      onCtx: (e) => showCtx(e, [
        { label: 'Set as active chunk', fn: () => { ACTIVE_CHUNK = chunk; renderWorkspace(); } },
        { label: '＋ Add Photos…', fn: () => { ACTIVE_CHUNK = chunk; openBrowse(); } },
        { sep: true },
        { label: 'Rename chunk…', fn: () => renameChunk(chunk) },
        { label: 'Remove chunk', danger: true, fn: () => removeChunk(chunk) },
      ]) }));
    if (!isOpen(cid)) return;

    const cats = {}; byChunk[chunk].forEach(L => { const c = CATEGORY[L.type] || 'Other'; (cats[c] = cats[c] || []).push(L); });
    CAT_ORDER.filter(c => cats[c]).forEach(cat => {
      const catId = `cat:${chunk}:${cat}`;
      el.appendChild(row({ depth: 2, id: catId, hasKids: true, cls: 'cat', icon: CAT_ICON[cat] || '◇',
        label: cat, count: `${cats[cat].length}` }));
      if (!isOpen(catId)) return;
      cats[cat].forEach(L => {
        const canView = !!viewable(L);
        const badge = layerMetric(L);
        el.appendChild(row({ depth: 3, id: L.id, hasKids: false, sel: selected === L.id,
          disabled: L.enabled === false,
          icon: '', label: `${L.id} <span class="cnt">${L.type}</span>`, count: badge,
          dot: L.status || '', eye: canView ? true : undefined,
          chk: { checked: L.enabled !== false, onToggle: (v) => layerAction({ action: 'set_enabled', id: L.id, enabled: v }) },
          drag: L.id,                                   // drag a layer to another chunk
          onClick: () => selectLayer(L.id),
          onDbl: () => openLayer(L),                    // double-click opens in the proper view
          onCtx: (e) => showCtx(e, layerCtx(L, canView)) }));
        // wire the eye toggle (last rendered row)
        const node = el.lastChild, eyeEl = node.querySelector('.eye');
        if (eyeEl) { eyeEl.classList.toggle('on', visible.has(L.id));
          eyeEl.onclick = (ev) => { ev.stopPropagation(); if (canView) setVisible(L, !visible.has(L.id)); }; }
      });
    });
  });
}

// ---- context menu ----------------------------------------------------------
function showCtx(e, items) {
  const m = $('ctxMenu'); m.innerHTML = '';
  items.forEach(it => {
    if (it.sep) { m.appendChild(document.createElement('hr')); return; }
    const d = document.createElement('div'); d.textContent = it.label; if (it.danger) d.className = 'danger';
    d.onclick = () => { hideCtx(); it.fn(); };
    m.appendChild(d);
  });
  m.style.left = Math.min(e.clientX, innerWidth - 200) + 'px';
  m.style.top = Math.min(e.clientY, innerHeight - items.length * 30 - 10) + 'px';
  m.classList.remove('hidden');
}
function hideCtx() { $('ctxMenu').classList.add('hidden'); }
document.addEventListener('click', hideCtx);
document.addEventListener('contextmenu', (e) => { if (!e.target.closest('.tnode')) hideCtx(); });

function layerCtx(L, canView) {
  const items = [];
  if (canView) items.push({ label: visible.has(L.id) ? 'Hide in view' : 'Show in view',
    fn: () => setVisible(L, !visible.has(L.id)) });
  items.push({ label: 'Rename layer…', fn: () => renameLayer(L) });
  PROJECT.chunks.filter(c => c !== L.chunk).forEach(c =>
    items.push({ label: `Move to "${c}"`, fn: () => layerAction({ action: 'move', id: L.id, to: c }) }));
  items.push({ sep: true });
  items.push({ label: 'Remove layer', danger: true, fn: () => {
    if (confirm(`Remove layer ${L.id}?`)) layerAction({ action: 'remove', id: L.id }); } });
  return items;
}

// ---- chunk / layer actions -------------------------------------------------
async function chunkAction(body) {
  const r = await fetch('/api/chunk', { method: 'POST', body: JSON.stringify(body) });
  const j = await r.json(); if (!r.ok) { log('chunk error: ' + j.error); return false; }
  await loadProject(); return true;
}
async function layerAction(body) {
  const r = await fetch('/api/layer', { method: 'POST', body: JSON.stringify(body) });
  const j = await r.json(); if (!r.ok) { log('layer error: ' + j.error); return; }
  if (selected === body.id) selected = body.action === 'rename' ? body.to : null;
  log(`${body.action} ${body.id}` + (body.to ? ` -> ${body.to}` : '')); await loadProject();
}
async function addChunk() {
  const name = prompt('New chunk name:', `Chunk ${PROJECT.chunks.length + 1}`); if (!name) return;
  if (await chunkAction({ action: 'add', name })) { ACTIVE_CHUNK = name; renderWorkspace(); }
}
async function renameChunk(name) {
  const to = prompt('Rename chunk:', name); if (!to || to === name) return;
  if (await chunkAction({ action: 'rename', name, to })) { if (ACTIVE_CHUNK === name) ACTIVE_CHUNK = to; renderWorkspace(); }
}
async function removeChunk(name) {
  if (!confirm(`Remove chunk "${name}" and all its layers?`)) return;
  if (await chunkAction({ action: 'remove', name })) { if (ACTIVE_CHUNK === name) ACTIVE_CHUNK = PROJECT.chunks[0] || 'Chunk 1'; renderWorkspace(); }
}
function renameLayer(L) {
  const to = prompt('Rename layer:', L.id); if (!to || to === L.id) return;
  layerAction({ action: 'rename', id: L.id, to });
}
$('newChunk').onclick = addChunk;
function selectLayer(id) {
  selected = id; renderWorkspace();
  const L = PROJECT.layers.find(x => x.id === id);
  renderParams(L);
  if (L) rasterView(L);          // raster products (ortho/DSM/index) open in the 2D Ortho view
}
// double-click: open a layer in whichever view fits it best
function openLayer(L) {
  selected = L.id; renderParams(L);
  if (rasterArtifact(L)) { rasterView(L); return; }            // ortho / DEM / index -> 2D
  if (viewable(L)) { setVisible(L, true); selectVtab('model'); frameAll(); return; }  // mesh/cloud -> 3D
  if (L.type === 'ingest') { showCameras(L); return; }         // cameras -> 3D positions
  selectDock('console'); log(`${L.id}: nothing to display yet (run it first)`);
}
function showCameras(_L) { log('camera positions view: coming next'); }   // implemented with /api/cameras

// ---- properties / params --------------------------------------------------
function renderParams(L) {
  const info = STAGES[L.type] || { default_params: {} };
  const cur = { ...(info.default_params || {}), ...L.params };
  const box = $('params'); box.innerHTML = `<div class="muted">${L.id} — ${L.type} · ${L.chunk}</div>`;
  for (const [k, v] of Object.entries(cur)) {
    const lab = document.createElement('label'); lab.textContent = k; box.appendChild(lab);
    const inp = document.createElement('input'); inp.dataset.k = k;
    if (typeof v === 'boolean') { inp.type = 'checkbox'; inp.checked = v; }
    else if (typeof v === 'number') { inp.type = 'number'; inp.value = v; inp.step = 'any'; }
    else { inp.type = 'text'; inp.value = Array.isArray(v) ? v.join(',') : v; }
    box.appendChild(inp);
  }
  const btn = document.createElement('button'); btn.textContent = 'Update layer'; btn.style.marginTop = '10px';
  btn.onclick = () => updateStage(L); box.appendChild(btn);
  if (viewable(L)) {
    const vb = document.createElement('button'); vb.textContent = visible.has(L.id) ? 'Hide in view' : 'Show in view';
    vb.style.margin = '10px 0 0 6px'; vb.onclick = () => setVisible(L, !visible.has(L.id)); box.appendChild(vb);
  }
  buildExport(box, L);
  if (Object.keys(L.metrics || {}).length) {
    const m = document.createElement('div'); m.className = 'muted'; m.style.marginTop = '8px';
    m.textContent = Object.entries(L.metrics).map(([k, v]) => `${k}=${v}`).join('  ');
    box.appendChild(m);
  }
}
function buildExport(box, L) {
  const arts = Object.entries(L.artifacts || {}).filter(
    ([, v]) => typeof v === 'string' && /\.(ply|las|tif|tiff|geojson|obj|glb)$/i.test(v));
  if (!arts.length) return;
  const h = document.createElement('label'); h.textContent = 'Export'; box.appendChild(h);
  const asel = document.createElement('select');
  arts.forEach(([k, v]) => { const o = document.createElement('option'); o.value = v; o.textContent = k; asel.appendChild(o); });
  const fsel = document.createElement('select');
  const refresh = async () => {
    fsel.innerHTML = '';
    const { formats } = await (await fetch('/api/formats?path=' + encodeURIComponent(asel.value))).json();
    (formats || []).forEach(f => { const o = document.createElement('option'); o.value = f; o.textContent = f; fsel.appendChild(o); });
  };
  asel.onchange = refresh; box.appendChild(asel); box.appendChild(fsel); refresh();
  const eb = document.createElement('button'); eb.textContent = 'Export as…'; eb.style.marginTop = '6px';
  eb.onclick = async () => {
    const j = await (await fetch('/api/export', { method:'POST',
      body: JSON.stringify({ path: asel.value, fmt: fsel.value }) })).json();
    log(j.out ? `exported -> ${j.out}` : `export error: ${j.error}`);
  };
  box.appendChild(eb);
}
function collectParams(L) {
  const defaults = (STAGES[L.type] || {}).default_params || {};
  const out = {};
  $('params').querySelectorAll('input[data-k]').forEach(inp => {
    const k = inp.dataset.k, d = defaults[k];
    if (inp.type === 'checkbox') out[k] = inp.checked;
    else if (inp.type === 'number') out[k] = parseFloat(inp.value);
    else out[k] = Array.isArray(d) ? inp.value.split(',').map(s => s.trim()).filter(Boolean) : inp.value;
  });
  return out;
}
async function updateStage(L) {
  await fetch('/api/stage', { method:'POST', body: JSON.stringify(
    { id: L.id, type: L.type, inputs: L.inputs, params: collectParams(L), chunk: L.chunk }) });
  log(`updated ${L.id}`); await loadProject(); selectLayer(L.id);
}
$('addBtn').onclick = async () => {
  const id = $('newId').value.trim(), type = $('newType').value;
  if (!id) return;
  const r = await fetch('/api/stage', { method:'POST', body: JSON.stringify({ id, type, inputs: [], params: {}, chunk: ACTIVE_CHUNK }) });
  if (r.ok) { $('newId').value = ''; log(`added ${id} (${type})`); await loadProject(); selectLayer(id); }
};

// ---- run + live progress (SSE) --------------------------------------------
$('runBtn').onclick = async () => {
  const r = await fetch('/api/run', { method:'POST', body: '{}' });
  if (r.status === 409) { log('already running'); return; }
  log('--- run started ---'); $('status').textContent = 'running…'; selectDock('console');
  const es = new EventSource('/api/events');
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.event === 'stage_start') { setDot(ev.id, 'running'); log(`▶ ${ev.id} (${ev.type})`); }
    else if (ev.event === 'progress') $('status').textContent = `${ev.id}: ${Math.round(ev.frac*100)}% ${ev.message||''}`;
    else if (ev.event === 'stage_done') { setDot(ev.id, ev.status); log(`✓ ${ev.id} [${ev.status}]`); }
    else if (ev.event === 'stage_skipped') { setDot(ev.id, 'failed'); log(`⨯ ${ev.id} skipped`); }
    else if (ev.event === 'run_done') log(`--- run ${ev.ok ? 'OK' : 'FAILED'} ---`);
    else if (ev.event === 'run_error') log(`error: ${ev.error}`);
  };
  es.addEventListener('eof', async () => { es.close(); $('status').textContent = 'done';
    for (const id of [...visible]) objects.delete(id);   // force reload of refreshed artifacts
    objects.forEach(o => scene.remove(o)); objects.clear();
    const reshow = [...visible]; visible.clear();
    await loadProject(); await loadPhotos();
    for (const id of reshow) { const L = PROJECT.layers.find(x => x.id === id); if (L) await setVisible(L, true); }
    if (selected) selectLayer(selected); });
};
function setDot(id, cls) { const L = PROJECT.layers.find(x => x.id === id); if (L) L.status = cls; renderWorkspace(); }

// ---- menus ----------------------------------------------------------------
function closeMenus() { document.querySelectorAll('.mMenu').forEach(m => m.classList.add('hidden')); }
document.querySelectorAll('.mItem').forEach(it => {
  it.onclick = (e) => { e.stopPropagation();
    const m = it.querySelector('.mMenu'); const wasOpen = !m.classList.contains('hidden');
    closeMenus(); if (!wasOpen) m.classList.remove('hidden'); };
});
document.addEventListener('click', () => { closeMenus(); });
function menuEntry(menu, label, fn, desc) {
  const d = document.createElement('div');
  d.innerHTML = desc ? `${label}<div class="t">${desc}</div>` : label;
  d.onclick = (e) => { e.stopPropagation(); closeMenus(); fn(); };
  $(menu).appendChild(d);
}
function menuSep(menu) { $(menu).appendChild(document.createElement('hr')); }
async function loadWorkflows() {
  WORKFLOWS = await (await fetch('/api/workflows')).json();
  // File menu
  $('m-file').innerHTML = '';
  menuEntry('m-file', '📄 New project…', newProject);
  menuEntry('m-file', '💾 Save project', saveProject);
  menuSep('m-file');
  menuEntry('m-file', '＋ New chunk', addChunk);
  menuEntry('m-file', '🌐 Set coordinate system…', openCrsPicker);
  // Workflow menu = the familiar operations
  $('m-workflow').innerHTML = '';
  WORKFLOWS.forEach(op => menuEntry('m-workflow', op.op, () => openOp(op), op.desc));
  // Model menu = view helpers
  $('m-model').innerHTML = '';
  menuEntry('m-model', 'Frame all', frameAll);
  menuEntry('m-model', 'Hide all layers', () => { visible.forEach(id => { const o = objects.get(id); if (o) o.visible = false; });
    visible.clear(); renderWorkspace(); });
  // Tools menu
  $('m-tools').innerHTML = '';
  menuEntry('m-tools', '📏 Measure distance', () => setMeasure('dist'));
  menuEntry('m-tools', '▱ Measure area', () => setMeasure('area'));
  menuSep('m-tools');
  menuEntry('m-tools', '📍 Markers / GCPs', () => { selectLeft('reference'); loadMarkers(); });
  // Help
  $('m-help').innerHTML = '';
  menuEntry('m-help', 'About OpenReco', () => log('OpenReco — open, reproducible photogrammetry. Clean-room; permissive OSS.'));
}

// ---- project: new / save --------------------------------------------------
async function newProject() {
  const path = prompt('New project folder (full path):', '');
  if (!path) return;
  const name = prompt('Project name:', path.split(/[\\/]/).filter(Boolean).pop()) || undefined;
  const r = await fetch('/api/new_project', { method:'POST', body: JSON.stringify({ path, name }) });
  const j = await r.json();
  if (!r.ok) { log('new project error: ' + j.error); return; }
  // reset all client state for the freshly loaded project
  objects.forEach(o => scene.remove(o)); objects.clear(); visible.clear(); selected = null;
  ACTIVE_CHUNK = 'Chunk 1'; MARKERS = []; activeMarker = null;
  await loadProject(); await loadMarkers(); await loadPhotos();
  $('pname') && ($('pname').textContent = j.name);
  log(`new project: ${j.name} @ ${j.project_dir}`);
}
async function saveProject() {
  const j = await (await fetch('/api/save_project', { method:'POST', body: '{}' })).json();
  log(j.ok ? `saved ${j.path}` : `save error: ${j.error}`);
}

// ---- tabs (left pane / dock / viewport) -----------------------------------
function selectLeft(name) {
  document.querySelectorAll('[data-ltab]').forEach(b => b.classList.toggle('on', b.dataset.ltab === name));
  $('lt-workspace').classList.toggle('hidden', name !== 'workspace');
  $('lt-reference').classList.toggle('hidden', name !== 'reference');
}
document.querySelectorAll('[data-ltab]').forEach(b => b.onclick = () => selectLeft(b.dataset.ltab));
function selectDock(name) {
  document.querySelectorAll('[data-dtab]').forEach(b => b.classList.toggle('on', b.dataset.dtab === name));
  ['console','photos','jobs'].forEach(n => $('dt-' + n).classList.toggle('hidden', n !== name));
  if (name === 'photos') loadPhotos();
}
document.querySelectorAll('[data-dtab]').forEach(b => b.onclick = () => selectDock(b.dataset.dtab));
function selectVtab(name) {
  document.querySelectorAll('[data-vtab]').forEach(b => b.classList.toggle('on', b.dataset.vtab === name));
  $('imgview').classList.toggle('show', name === 'photo');
  $('orthoview').classList.toggle('show', name === 'ortho');
  $('c').style.display = name === 'model' ? 'block' : 'none';
}
document.querySelectorAll('[data-vtab]').forEach(b => b.onclick = () => selectVtab(b.dataset.vtab));

// ---- Ortho 2D raster view (pan/zoom a server-rendered PNG of any GeoTIFF) --
// raster products that render as 2D layers: prefer a .tif (ortho/DSM), else an index .tif
function rasterArtifact(layer) {
  const a = layer.artifacts || {};
  for (const k of ['ortho', 'dsm'])
    if (a[k] && /\.tif/i.test(a[k])) return a[k];
  for (const v of Object.values(a))
    if (typeof v === 'string' && /\.tif$/i.test(v)) return v;       // e.g. a vegetation index
  return null;
}
let oz = { s: 1, tx: 0, ty: 0 };
function applyOrtho() { $('orthoimg').style.transform = `translate(${oz.tx}px,${oz.ty}px) scale(${oz.s})`; }
function rasterView(layer) {
  const tif = rasterArtifact(layer); if (!tif) return false;
  selectVtab('ortho');
  const img = $('orthoimg');
  img.onload = () => {                                   // fit to viewport
    const w = $('center').clientWidth, h = $('center').clientHeight;
    oz.s = Math.min(w / img.naturalWidth, h / img.naturalHeight) * 0.95;
    oz.tx = (w - img.naturalWidth * oz.s) / 2; oz.ty = (h - img.naturalHeight * oz.s) / 2;
    applyOrtho();
  };
  img.src = `/api/raster_png?path=${encodeURIComponent(tif)}`;
  $('orthohint').textContent = `${layer.id} · ${tif.split(/[\\/]/).pop()} · scroll to zoom, drag to pan`;
  return true;
}
(function orthoNav() {
  const view = $('orthoview');
  view.addEventListener('wheel', (e) => { e.preventDefault();
    const r = view.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const f = e.deltaY < 0 ? 1.15 : 1/1.15;
    oz.tx = mx - (mx - oz.tx) * f; oz.ty = my - (my - oz.ty) * f; oz.s *= f; applyOrtho();
  }, { passive: false });
  let drag = null;
  view.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, tx: oz.tx, ty: oz.ty };
    view.classList.add('drag'); view.setPointerCapture(e.pointerId); });
  view.addEventListener('pointermove', (e) => { if (!drag) return;
    oz.tx = drag.tx + (e.clientX - drag.x); oz.ty = drag.ty + (e.clientY - drag.y); applyOrtho(); });
  view.addEventListener('pointerup', () => { drag = null; view.classList.remove('drag'); });
})();

// ---- photos pane + GCP picking --------------------------------------------
let PHOTOS = { images: [] };
async function loadPhotos() {
  PHOTOS = await (await fetch('/api/images?chunk=' + encodeURIComponent(ACTIVE_CHUNK))).json();
  const el = $('photos'); el.innerHTML = '';
  if (!PHOTOS.images.length) { el.textContent = 'Run ingest in this chunk to list source photos.'; return; }
  PHOTOS.images.forEach(im => {
    const t = document.createElement('div'); t.className = 'th' + (im.excluded ? ' exc' : '');
    const url = `/api/file?path=${encodeURIComponent(im.path)}`;
    t.innerHTML = `<button class="rm" title="Remove from chunk">✕</button>`
                + `<img loading="lazy" src="${url}"><div>${im.name}</div>`;
    t.querySelector('img').onclick = () => openPhoto(im);
    t.querySelector('.rm').onclick = (e) => { e.stopPropagation(); removePhoto(im); };
    el.appendChild(t);
  });
}
async function removePhoto(im) {
  if (!confirm(`Remove ${im.name} from this chunk? (source file is not deleted)`)) return;
  const r = await fetch('/api/remove_photo', { method:'POST',
    body: JSON.stringify({ layer: im.layer, name: im.name }) });
  const j = await r.json();
  if (r.ok) { log(`removed ${im.name} (${j.remaining} left in ${im.layer})`); await loadPhotos(); await loadProject(); }
  else log('remove error: ' + j.error);
}
let curPhoto = null;
function openPhoto(im) {
  curPhoto = im; selectVtab('photo');
  const wrap = $('imgwrap'); wrap.innerHTML = '';
  const img = document.createElement('img'); img.src = `/api/file?path=${encodeURIComponent(im.path)}`;
  img.onload = () => { drawPins(); };
  wrap.appendChild(img);
  wrap.onclick = (e) => {
    if (e.target.classList.contains('pin')) return;
    if (activeMarker == null) { log('select a marker in Reference > Markers first'); return; }
    const r = img.getBoundingClientRect();
    const u = Math.round((e.clientX - r.left) * (im.width || img.naturalWidth) / r.width || (e.clientX - r.left));
    const v = Math.round((e.clientY - r.top) * (im.height || img.naturalHeight) / r.height || (e.clientY - r.top));
    // store in natural image pixels
    const uu = Math.round((e.clientX - r.left) / r.width * img.naturalWidth);
    const vv = Math.round((e.clientY - r.top) / r.height * img.naturalHeight);
    MARKERS[activeMarker].observations = MARKERS[activeMarker].observations.filter(o => o.image !== im.name);
    MARKERS[activeMarker].observations.push({ image: im.name, u: uu, v: vv });
    log(`marker ${MARKERS[activeMarker].name}: observed in ${im.name} @ (${uu},${vv})`);
    renderMarkers(); drawPins();
  };
}
function drawPins() {
  const wrap = $('imgwrap'); const img = wrap.querySelector('img'); if (!img) return;
  [...wrap.querySelectorAll('.pin')].forEach(p => p.remove());
  const r = img.getBoundingClientRect(), wrapR = wrap.getBoundingClientRect();
  MARKERS.forEach(mk => mk.observations.forEach(o => {
    if (!curPhoto || o.image !== curPhoto.name) return;
    const pin = document.createElement('div'); pin.className = 'pin';
    pin.style.left = (o.u / img.naturalWidth * img.clientWidth) + 'px';
    pin.style.top = (o.v / img.naturalHeight * img.clientHeight) + 'px';
    pin.innerHTML = `<span>${mk.name}</span>`;
    wrap.appendChild(pin);
  }));
}

// ---- markers / GCP reference table ----------------------------------------
let MARKERS = [];           // [{name, world:[x,y,z]|null, observations:[{image,u,v}]}]
let activeMarker = null;
async function loadMarkers() {
  const j = await (await fetch('/api/markers')).json();
  MARKERS = (j.markers || []).map(m => ({ name: m.name, world: m.world || null,
    observations: m.observations || [] }));
  if (MARKERS.length && activeMarker == null) activeMarker = 0;
  renderMarkers();
}
function renderMarkers() {
  const box = $('markerTable');
  if (!MARKERS.length) { box.innerHTML = '<div class="muted">No markers. Add one, then pick it in photos.</div>'; return; }
  const t = document.createElement('table');
  t.innerHTML = '<tr><th>name</th><th>X</th><th>Y</th><th>Z</th><th>obs</th></tr>';
  MARKERS.forEach((m, i) => {
    const tr = document.createElement('tr'); tr.className = (i === activeMarker ? 'sel' : '');
    const w = m.world || ['','',''];
    tr.innerHTML = `<td>${m.name}</td>`
      + [0,1,2].map(k => `<td><input data-mi="${i}" data-wk="${k}" value="${w[k]}" style="width:64px"></td>`).join('')
      + `<td>${m.observations.length}</td>`;
    tr.onclick = (e) => { if (e.target.tagName !== 'INPUT') { activeMarker = i; renderMarkers(); } };
    t.appendChild(tr);
  });
  box.innerHTML = ''; box.appendChild(t);
  box.querySelectorAll('input[data-mi]').forEach(inp => inp.onchange = () => {
    const m = MARKERS[+inp.dataset.mi]; m.world = m.world || [0,0,0];
    m.world[+inp.dataset.wk] = parseFloat(inp.value) || 0;
  });
}
$('addMarker').onclick = () => {
  const name = prompt('Marker name:', `GCP${MARKERS.length + 1}`); if (!name) return;
  MARKERS.push({ name, world: null, observations: [] }); activeMarker = MARKERS.length - 1; renderMarkers();
};
$('saveMarkers').onclick = async () => {
  const j = await (await fetch('/api/markers', { method:'POST', body: JSON.stringify({ markers: MARKERS }) })).json();
  log(j.ok ? `saved ${j.count} marker(s) -> ${j.gcp_csv}` : `marker save error: ${j.error}`);
};
$('useGcps').onclick = async () => {
  await $('saveMarkers').onclick();            // persist current picks first
  let epsg = (PROJECT.crs || '').replace(/^EPSG:/i, '');
  if (!/^\d+$/.test(epsg)) {
    epsg = prompt('GCP coordinate system EPSG code (e.g. 32613):', '');
    if (!epsg) return;
  }
  const r = await fetch('/api/use_gcps', { method:'POST',
    body: JSON.stringify({ chunk: ACTIVE_CHUNK, crs_epsg: parseInt(epsg) }) });
  const j = await r.json();
  if (r.ok) { log(`georef now uses GCPs: ${j.updated.join(', ')} (EPSG:${j.gcp_crs_epsg})`); await loadProject(); }
  else log(`use GCPs error: ${j.error} — add a Georeference step to this chunk first`);
};
$('refCrsBtn').onclick = openCrsPicker;

// ---- CRS picker -----------------------------------------------------------
let crsChoice = null;
function openCrsPicker() {
  crsChoice = null; $('crsOk').disabled = true; $('crsResults').innerHTML = '';
  $('crsInfo').textContent = ''; $('crsSearch').value = ''; $('crsModal').classList.remove('hidden');
  $('crsSearch').focus();
}
$('crsCancel').onclick = () => $('crsModal').classList.add('hidden');
let crsTimer = null;
$('crsSearch').oninput = () => {
  clearTimeout(crsTimer);
  crsTimer = setTimeout(async () => {
    const q = $('crsSearch').value.trim(); if (q.length < 2) return;
    const j = await (await fetch('/api/crs?search=' + encodeURIComponent(q))).json();
    const box = $('crsResults'); box.innerHTML = '';
    (j.results || []).slice(0, 40).forEach(r => {
      // search_crs returns code already prefixed, e.g. "EPSG:32613"
      const code = String(r.code || r.id || '').replace(/^EPSG:/i, '');
      const name = r.name || r.title || '';
      const d = document.createElement('div'); d.className = 'crsrow';
      d.textContent = `EPSG:${code} — ${name}`;
      d.onclick = () => { crsChoice = `EPSG:${code}`; $('crsOk').disabled = false;
        [...box.children].forEach(c => c.style.background = ''); d.style.background = '#1d2738';
        $('crsInfo').textContent = `selected ${crsChoice}`; };
      box.appendChild(d);
    });
    if (!box.children.length) box.innerHTML = '<div class="muted">no matches</div>';
  }, 250);
};
$('crsOk').onclick = async () => {
  if (!crsChoice) return;
  const j = await (await fetch('/api/project', { method:'POST', body: JSON.stringify({ crs: crsChoice }) })).json();
  $('crsModal').classList.add('hidden'); log(`project CRS set to ${j.crs}`); await loadProject();
};
$('crsBtn').onclick = openCrsPicker;

// ---- build dialog (workflow op) -------------------------------------------
function openOp(op) {
  $('mTitle').textContent = op.op; $('mDesc').textContent = op.desc;
  const base = op.stage; let n = 1; const ids = new Set(PROJECT.layers.map(l => l.id));
  while (ids.has(base + n)) n++;
  $('mId').value = base + n;
  const inb = $('mInputs'); inb.innerHTML = PROJECT.layers.length ? '' : '<span class="muted">none yet</span>';
  PROJECT.layers.forEach(l => {
    const w = document.createElement('label'); w.className = 'chk';
    w.innerHTML = `<input type="checkbox" value="${l.id}"> ${l.id} <span class="muted">(${l.type})</span>`;
    inb.appendChild(w);
  });
  const fb = $('mFields'); fb.innerHTML = '';
  op.fields.forEach(f => {
    const lab = document.createElement('label'); lab.textContent = f.label; fb.appendChild(lab);
    let inp;
    if (f.type === 'enum') {
      inp = document.createElement('select');
      Object.keys(f.options).forEach(k => { const o = document.createElement('option'); o.value = k; o.textContent = k; inp.appendChild(o); });
      inp.value = f.default;
    } else if (f.type === 'bool') { inp = document.createElement('input'); inp.type = 'checkbox'; inp.checked = !!f.default; }
    else if (f.type === 'path' || f.type === 'string') {
      inp = document.createElement('input'); inp.type = 'text'; inp.value = f.default;
      if (f.type === 'path') inp.placeholder = 'folder path (e.g. D:\\data\\flight1 or "images")';
    }
    else { inp = document.createElement('input'); inp.type = 'number'; inp.step = 'any'; inp.value = f.default; }
    inp.dataset.label = f.label; inp.dataset.type = f.type; fb.appendChild(inp);
  });
  // Add Photos (ingest) gets a file picker that selects specific images across folders
  $('mBrowse').classList.toggle('hidden', op.stage !== 'ingest');
  $('mBrowse').onclick = () => openBrowse($('mId').value.trim());
  $('mOk').onclick = () => submitOp(op);
  $('modal').classList.remove('hidden');
}
$('mCancel').onclick = () => $('modal').classList.add('hidden');

// ---- file picker (Add Photos: navigate folders, multi-select images) ------
let brSelected = new Map();    // path -> name
let brCurrent = null;
async function openBrowse(layerId) {
  $('browseModal').dataset.layerId = layerId || '';
  brSelected = new Map();
  $('browseModal').classList.remove('hidden');
  await browseTo(brCurrent);             // resume last dir, or drives/root
}
async function browseTo(path) {
  const url = '/api/browse' + (path ? '?path=' + encodeURIComponent(path) : '');
  const d = await (await fetch(url)).json();
  if (d.error) { log('browse: ' + d.error); return; }
  brCurrent = d.path; $('brPath').value = d.path || '';
  const dirs = $('brDirs'); dirs.innerHTML = '';
  if (d.parent !== null && d.parent !== undefined) {
    // up handled by button; list child folders
  }
  d.dirs.forEach(p => {
    const row = document.createElement('div'); row.className = 'brdir';
    row.textContent = '📁 ' + p.split(/[\\/]/).filter(Boolean).pop();
    row.onclick = () => browseTo(p); dirs.appendChild(row);
  });
  if (!d.dirs.length) dirs.innerHTML = '<div class="muted" style="padding:6px">no sub-folders</div>';
  const grid = $('brImages'); grid.innerHTML = '';
  d.images.forEach(im => {
    const t = document.createElement('div'); t.className = 'brth' + (brSelected.has(im.path) ? ' sel' : '');
    t.innerHTML = `<img loading="lazy" src="/api/thumb?path=${encodeURIComponent(im.path)}"><div>${im.name}</div>`;
    t.onclick = () => { if (brSelected.has(im.path)) brSelected.delete(im.path); else brSelected.set(im.path, im.name);
      t.classList.toggle('sel'); updateBrCount(); };
    grid.appendChild(t);
  });
  $('brImages').dataset.all = JSON.stringify(d.images.map(i => i.path));
  $('brAll').checked = false;
  if (!d.images.length) grid.innerHTML = '<div class="muted" style="padding:6px">no images in this folder</div>';
  updateBrCount();
}
function updateBrCount() { $('brCount').textContent = `${brSelected.size} selected`; }
$('brUp').onclick = async () => {
  const d = await (await fetch('/api/browse' + (brCurrent ? '?path=' + encodeURIComponent(brCurrent) : ''))).json();
  browseTo(d.parent);                    // null parent -> drives/root
};
$('brGo').onclick = () => browseTo($('brPath').value.trim() || null);
$('brPath').onkeydown = (e) => { if (e.key === 'Enter') $('brGo').onclick(); };
$('brAll').onchange = (e) => {
  const all = JSON.parse($('brImages').dataset.all || '[]');
  if (e.target.checked) all.forEach(p => brSelected.set(p, p.split(/[\\/]/).pop()));
  else all.forEach(p => brSelected.delete(p));
  browseTo(brCurrent);                    // re-render selection state
};
$('brCancel').onclick = () => $('browseModal').classList.add('hidden');
$('brAdd').onclick = async () => {
  if (!brSelected.size) { log('select at least one image'); return; }
  const id = $('browseModal').dataset.layerId || undefined;
  const r = await fetch('/api/add_photos', { method:'POST',
    body: JSON.stringify({ paths: [...brSelected.keys()], chunk: ACTIVE_CHUNK, id }) });
  const j = await r.json();
  if (!r.ok) { log('add photos error: ' + j.error); return; }
  $('browseModal').classList.add('hidden'); $('modal').classList.add('hidden');
  log(`added ${j.count} photo(s) to ${ACTIVE_CHUNK} as ${j.id}` + (j.staged ? ' (copied into project)' : ''));
  let toSelect = j.id;
  if ($('brAlign').checked) {                 // auto-chain an Align Photos step on these images
    const ids = new Set([j.id]); let n = 1; while (ids.has('sfm' + n)) n++;
    const ar = await fetch('/api/operation', { method:'POST', body: JSON.stringify(
      { op: 'Align Photos', id: 'sfm' + n, inputs: [j.id], values: {}, chunk: ACTIVE_CHUNK }) });
    if (ar.ok) { toSelect = 'sfm' + n; log(`+ Align Photos (sfm${n}) wired to ${j.id}`); }
  }
  await loadProject(); selectLayer(toSelect); selectDock('photos');
};
async function submitOp(op) {
  const id = $('mId').value.trim(); if (!id) return;
  const inputs = [...$('mInputs').querySelectorAll('input:checked')].map(c => c.value);
  const values = {};
  $('mFields').querySelectorAll('[data-label]').forEach(inp => {
    const t = inp.dataset.type;
    values[inp.dataset.label] = t === 'bool' ? inp.checked
      : (t === 'enum' || t === 'path' || t === 'string') ? inp.value : parseFloat(inp.value);
  });
  const r = await fetch('/api/operation', { method:'POST',
    body: JSON.stringify({ op: op.op, id, inputs, values, chunk: ACTIVE_CHUNK }) });
  const j = await r.json();
  if (r.ok) { $('modal').classList.add('hidden'); log(`built ${id} (${op.op})`); await loadProject(); selectLayer(id); }
  else log(`build error: ${j.error}`);
}

// ---- boot ----
(async () => { resize(); await loadStages(); await loadWorkflows(); await loadProject();
  await loadMarkers(); })();
