// OpenReco UI — layer tree, schema-driven parameter panels, run+SSE, three.js viewport.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

let STAGES = {};      // type -> {default_params, params_schema, ...}
let PROJECT = null;   // {name, chunks:[...], layers:[...]}
let selected = null;
let ACTIVE_CHUNK = "Chunk 1";

const $ = (id) => document.getElementById(id);
const log = (m, cls) => { const l = $('log'); l.innerHTML += `\n${m}`; l.scrollTop = l.scrollHeight; };

// ---- 3D viewport ----------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ canvas: $('c'), antialias: true });
renderer.setPixelRatio(devicePixelRatio);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0e14);
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1e7);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const dl = new THREE.DirectionalLight(0xffffff, 0.7); dl.position.set(1, 1, 1); scene.add(dl);
let current = null;
function resize() { const w = $('center').clientWidth, h = $('center').clientHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
addEventListener('resize', resize);
(function loop(){ requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();

function frame(obj) {
  const box = new THREE.Box3().setFromObject(obj), size = box.getSize(new THREE.Vector3()).length();
  const c = box.getCenter(new THREE.Vector3());
  controls.target.copy(c); camera.position.copy(c).add(new THREE.Vector3(size*.6, size*.5, size*.6));
  camera.near = size/1000; camera.far = size*10; camera.updateProjectionMatrix();
}
function clearView() { if (current) { scene.remove(current); current = null; } }
function viewFile(url, kind) {
  clearView();
  if (kind === 'glb') new GLTFLoader().load(url, g => { current = g.scene; scene.add(current); frame(current); });
  else new PLYLoader().load(url, geo => {
    geo.computeBoundingBox();
    let o;
    if (geo.index) { if (!geo.getAttribute('normal')) geo.computeVertexNormals();
      o = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: !!geo.getAttribute('color'),
        flatShading:true, side:THREE.DoubleSide })); }
    else o = new THREE.Points(geo, new THREE.PointsMaterial({ size:1, sizeAttenuation:false,
        vertexColors: !!geo.getAttribute('color') }));
    if (!geo.getAttribute('color')) o.material.color.set(0x89b4fa);
    current = o; scene.add(o); frame(o);
  });
}
// pick a viewable artifact from a layer (textured glb > mesh ply > points ply)
function viewable(layer) {
  const a = layer.artifacts || {};
  for (const [k, kind] of [['glb','glb'],['mesh','ply'],['points','ply'],['sparse_ply','ply']])
    if (a[k]) return { path: a[k], kind };
  return null;
}

// ---- data + rendering -----------------------------------------------------
async function loadStages() {
  STAGES = {}; for (const s of await (await fetch('/api/stages')).json()) STAGES[s.type] = s;
  const sel = $('newType'); sel.innerHTML = '';
  Object.keys(STAGES).filter(t => !t.startsWith('dummy')).sort().forEach(t => {
    const o = document.createElement('option'); o.value = t; o.textContent = t; sel.appendChild(o); });
}
async function loadProject() {
  PROJECT = await (await fetch('/api/project')).json();
  $('pname').textContent = `${PROJECT.name} · ${PROJECT.crs || 'local'}`;
  if (!PROJECT.chunks.includes(ACTIVE_CHUNK)) ACTIVE_CHUNK = PROJECT.chunks[0] || "Chunk 1";
  renderWorkspace();
}
function renderWorkspace() {
  const el = $('workspace'); el.innerHTML = '';
  $('activeChunk').textContent = `active: ${ACTIVE_CHUNK}`;
  const byChunk = {}; PROJECT.chunks.forEach(c => byChunk[c] = []);
  PROJECT.layers.forEach(L => { (byChunk[L.chunk] = byChunk[L.chunk] || []).push(L); });
  Object.keys(byChunk).forEach(chunk => {
    const h = document.createElement('div'); h.className = 'chunk' + (chunk === ACTIVE_CHUNK ? ' active' : '');
    h.innerHTML = `<span>▸ ${chunk}</span><span class="cnt">${byChunk[chunk].length}</span>`;
    h.onclick = () => { ACTIVE_CHUNK = chunk; renderWorkspace(); };
    el.appendChild(h);
    byChunk[chunk].forEach(L => {
      const d = document.createElement('div'); d.className = 'layer' + (selected === L.id ? ' sel' : '');
      d.innerHTML = `<span class="dot ${L.status||''}"></span><span class="id">${L.id}</span>`
                  + `<span class="t">${L.type}</span>`;
      d.onclick = () => selectLayer(L.id); el.appendChild(d);
    });
  });
}
$('newChunk').onclick = async () => {
  const name = prompt('New chunk name:', `Chunk ${PROJECT.chunks.length + 1}`);
  if (!name) return;
  await fetch('/api/chunk', { method:'POST', body: JSON.stringify({ name }) });
  ACTIVE_CHUNK = name; await loadProject();
};
function selectLayer(id) {
  selected = id; renderWorkspace();
  const L = PROJECT.layers.find(x => x.id === id);
  renderParams(L);
  const v = viewable(L);
  if (v) viewFile(`/api/file?path=${encodeURIComponent(v.path)}`, v.kind); else clearView();
}
function renderParams(L) {
  const info = STAGES[L.type] || { default_params: {} };
  const defaults = info.default_params || {};
  const cur = { ...defaults, ...L.params };
  const box = $('params'); box.innerHTML = `<div class="muted">${L.id} — ${L.type}</div>`;
  for (const [k, v] of Object.entries(cur)) {
    const lab = document.createElement('label'); lab.textContent = k; box.appendChild(lab);
    const inp = document.createElement('input'); inp.dataset.k = k;
    if (typeof v === 'boolean') { inp.type = 'checkbox'; inp.checked = v; }
    else if (typeof v === 'number') { inp.type = 'number'; inp.value = v; inp.step = 'any'; }
    else { inp.type = 'text'; inp.value = Array.isArray(v) ? v.join(',') : v; }
    box.appendChild(inp);
  }
  const btn = document.createElement('button'); btn.textContent = 'Update layer'; btn.style.marginTop='10px';
  btn.onclick = () => updateStage(L);
  box.appendChild(btn);
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
  log('--- run started ---'); $('status').textContent = 'running…';
  const es = new EventSource('/api/events');
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.event === 'stage_start') setDot(ev.id, 'running'), log(`▶ ${ev.id} (${ev.type})`);
    else if (ev.event === 'progress') $('status').textContent = `${ev.id}: ${Math.round(ev.frac*100)}% ${ev.message||''}`;
    else if (ev.event === 'stage_done') { setDot(ev.id, ev.status); log(`✓ ${ev.id} [${ev.status}]`); }
    else if (ev.event === 'stage_skipped') { setDot(ev.id, 'failed'); log(`⨯ ${ev.id} skipped`); }
    else if (ev.event === 'run_done') log(`--- run ${ev.ok ? 'OK' : 'FAILED'} ---`);
    else if (ev.event === 'run_error') log(`error: ${ev.error}`);
  };
  es.addEventListener('eof', async () => { es.close(); $('status').textContent = 'done';
    await loadProject(); if (selected) selectLayer(selected); });
};
function setDot(id, cls) {
  const L = PROJECT.layers.find(x => x.id === id); if (L) L.status = cls; renderWorkspace();
}

// ---- workflow menu + build dialog -----------------------------------------
let WORKFLOWS = [];
async function loadWorkflows() {
  WORKFLOWS = await (await fetch('/api/workflows')).json();
  const drop = $('wfDrop'); drop.innerHTML = '';
  WORKFLOWS.forEach(op => {
    const d = document.createElement('div');
    d.innerHTML = `${op.op}<div class="t">${op.desc}</div>`;
    d.onclick = () => { drop.classList.add('hidden'); openOp(op); };
    drop.appendChild(d);
  });
}
$('wfBtn').onclick = () => $('wfDrop').classList.toggle('hidden');
document.addEventListener('click', e => {
  if (!e.target.closest('.menu')) $('wfDrop').classList.add('hidden');
});

function openOp(op) {
  $('mTitle').textContent = op.op; $('mDesc').textContent = op.desc;
  // default id: <stagetype><n>
  const base = op.stage; let n = 1;
  const ids = new Set(PROJECT.layers.map(l => l.id));
  while (ids.has(base + n)) n++;
  $('mId').value = base + n;
  // inputs: checkboxes of existing layers
  const inb = $('mInputs'); inb.innerHTML = PROJECT.layers.length ? '' : '<span class="muted">none yet</span>';
  PROJECT.layers.forEach(l => {
    const w = document.createElement('label'); w.className = 'chk';
    w.innerHTML = `<input type="checkbox" value="${l.id}"> ${l.id} <span class="muted">(${l.type})</span>`;
    inb.appendChild(w);
  });
  // fields
  const fb = $('mFields'); fb.innerHTML = '';
  op.fields.forEach(f => {
    const lab = document.createElement('label'); lab.textContent = f.label; fb.appendChild(lab);
    let inp;
    if (f.type === 'enum') {
      inp = document.createElement('select');
      Object.keys(f.options).forEach(k => { const o = document.createElement('option'); o.value = k; o.textContent = k; inp.appendChild(o); });
      inp.value = f.default;
    } else if (f.type === 'bool') {
      inp = document.createElement('input'); inp.type = 'checkbox'; inp.checked = !!f.default;
    } else { inp = document.createElement('input'); inp.type = 'number'; inp.step = 'any'; inp.value = f.default; }
    inp.dataset.label = f.label; inp.dataset.type = f.type; fb.appendChild(inp);
  });
  $('mOk').onclick = () => submitOp(op);
  $('modal').classList.remove('hidden');
}
$('mCancel').onclick = () => $('modal').classList.add('hidden');

async function submitOp(op) {
  const id = $('mId').value.trim(); if (!id) return;
  const inputs = [...$('mInputs').querySelectorAll('input:checked')].map(c => c.value);
  const values = {};
  $('mFields').querySelectorAll('[data-label]').forEach(inp => {
    values[inp.dataset.label] = inp.dataset.type === 'bool' ? inp.checked
      : inp.dataset.type === 'enum' ? inp.value : parseFloat(inp.value);
  });
  const r = await fetch('/api/operation', { method:'POST',
    body: JSON.stringify({ op: op.op, id, inputs, values, chunk: ACTIVE_CHUNK }) });
  const j = await r.json();
  if (r.ok) { $('modal').classList.add('hidden'); log(`built ${id} (${op.op})`);
    await loadProject(); selectLayer(id); }
  else log(`build error: ${j.error}`);
}

// ---- boot ----
(async () => { resize(); await loadStages(); await loadWorkflows(); await loadProject(); })();
