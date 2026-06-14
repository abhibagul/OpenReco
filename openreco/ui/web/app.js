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
  clean: "Dense Cloud", import_cloud: "Dense Cloud",
  mesh: "3D Model", texture: "3D Model", splat: "3D Model", tiles: "Tiled Model",
  dsm: "DEM", ortho: "Orthomosaic", contours: "Shapes", indices: "Orthomosaic",
  volume: "Shapes", profile: "Shapes", panorama: "Orthomosaic",
};
const CAT_ORDER = ["Cameras", "Tie Points", "Markers", "Dense Cloud", "Point Cloud",
                   "3D Model", "Tiled Model", "DEM", "Orthomosaic", "Shapes", "Other"];

const $ = (id) => document.getElementById(id);
// styled console: each entry gets a timestamp + colored level chip
function classifyLog(m) {
  if (/^E\d{8}|\berror\b|\bfail|✗|⨯/i.test(m)) return 'err';
  if (/^W\d{8}|\bwarn/i.test(m)) return 'warn';
  if (/✓|\bOK\b|\bdone\b|\bsaved\b|\bexported\b|registered/i.test(m)) return 'ok';
  if (/^▶|^---|\brun\b|^built|^added|^updated/i.test(m)) return 'run';
  if (/^I\d{8}/.test(m)) return 'debug';                 // native COLMAP/glog info line
  return 'info';
}
function log(m, level) {
  const el = $('log'); if (!el) return;
  const lvl = level || classifyLog(m);
  const near = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
  const row = document.createElement('div'); row.className = 'logrow ' + lvl;
  row.innerHTML = `<span class="lt">${new Date().toTimeString().slice(0, 8)}</span>`
                + `<span class="ll">${lvl}</span>`;
  const msg = document.createElement('span'); msg.className = 'lm'; msg.textContent = m;
  row.appendChild(msg); el.appendChild(row);
  while (el.childElementCount > 1200) el.removeChild(el.firstChild);
  if (near) el.scrollTop = el.scrollHeight;              // autoscroll only if already at bottom
}

// ---- 3D viewport ----------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ canvas: $('c'), antialias: true });
renderer.setPixelRatio(devicePixelRatio);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0e14);
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1e7);
camera.up.set(0, 0, 1);                 // Z-up world (matches survey/ENU data: X east, Y north, Z up)
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const dl = new THREE.DirectionalLight(0xffffff, 0.7); dl.position.set(1, 1, 1); scene.add(dl);
const measureGroup = new THREE.Group(); scene.add(measureGroup);
// ---- infinite ground grid (shader plane in the XY world plane, Z-up) -------
const grid = new THREE.Mesh(
  new THREE.PlaneGeometry(2e6, 2e6),
  new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
    extensions: { derivatives: true },          // enable fwidth() for the grid AA
    uniforms: { uCam: { value: new THREE.Vector3() }, uFade: { value: 100.0 },
                uMinor: { value: new THREE.Color(0x3a4252) }, uMajor: { value: new THREE.Color(0x5a657c) } },
    vertexShader: `varying vec3 vW;
      void main(){ vec4 wp = modelMatrix * vec4(position,1.0); vW = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp; }`,
    fragmentShader: `precision highp float; varying vec3 vW;
      uniform vec3 uCam; uniform float uFade; uniform vec3 uMinor; uniform vec3 uMajor;
      float gridline(float size){ vec2 c = vW.xy / size;
        vec2 g = abs(fract(c - 0.5) - 0.5) / fwidth(c);
        return 1.0 - min(min(g.x, g.y), 1.0); }
      void main(){
        float d = distance(uCam, vW);
        float fade = 1.0 - clamp(d / uFade, 0.0, 1.0);
        if (fade <= 0.0) discard;
        float minor = gridline(1.0), major = gridline(10.0), huge = gridline(100.0);
        float a = max(max(minor * 0.5, major * 0.85), huge);   // LOD: fine lines fade when sub-pixel
        if (a < 0.012) discard;
        vec3 col = mix(uMinor, uMajor, step(0.5, max(major, huge)));
        gl_FragColor = vec4(col, a * fade * 0.7); }`,
  }));
grid.renderOrder = -1;
scene.add(grid);


const axes = new THREE.AxesHelper(1); scene.add(axes);
let helpers = true;
function resize() {
  const w = $('center').clientWidth, h = $('center').clientHeight;
  if (!w || !h) return;
  // canvas is position:absolute filling #center via CSS, so only update the draw buffer (no inline px)
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
// track the actual size of the viewport pane — fires on pane-splitter drags, dock resize, window, etc.
new ResizeObserver(() => resize()).observe($('center'));

// ---- navigation cube (CAD-style navigation cube) -----------------------------
const gizmoR = new THREE.WebGLRenderer({ canvas: $('gizmo'), antialias: true, alpha: true });
gizmoR.setPixelRatio(devicePixelRatio); gizmoR.setSize(96, 96);
const gizmoScene = new THREE.Scene();
const gizmoCam = new THREE.OrthographicCamera(-1.7, 1.7, 1.7, -1.7, 0.1, 100);
gizmoCam.position.set(0, 0, 5);
gizmoScene.add(new THREE.AmbientLight(0xffffff, 0.9));
const gizmoLight = new THREE.DirectionalLight(0xffffff, 0.6); gizmoLight.position.set(2, 3, 4); gizmoScene.add(gizmoLight);
function faceTex(label) {            // CAD-style: light face, dark centered label
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const x = c.getContext('2d');
  x.fillStyle = '#dfe4ec'; x.fillRect(0, 0, 128, 128);
  x.strokeStyle = '#9aa6b8'; x.lineWidth = 4; x.strokeRect(2, 2, 124, 124);
  x.fillStyle = '#2a3344'; x.font = 'bold 22px system-ui'; x.textAlign = 'center'; x.textBaseline = 'middle';
  x.fillText(label, 64, 66);
  return new THREE.CanvasTexture(c);
}
// BoxGeometry face order: +X,-X,+Y,-Y,+Z,-Z. Z-up world, CAD labels.
const gizmoFaces = ['RIGHT', 'LEFT', 'BACK', 'FRONT', 'TOP', 'BOTTOM'];
const gizmoCube = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2),
  gizmoFaces.map(l => new THREE.MeshBasicMaterial({ map: faceTex(l) })));
gizmoScene.add(gizmoCube);
gizmoScene.add(new THREE.LineSegments(new THREE.EdgesGeometry(gizmoCube.geometry),
  new THREE.LineBasicMaterial({ color: 0x6b7689 })));

// per-region hover highlight (faces / edges / corners), child of the cube so it rotates with it
const GZ_M = 0.66;                   // face half-extent; outer band [M,1] = edges/corners
const gizmoHi = new THREE.Mesh(new THREE.BufferGeometry(),
  new THREE.MeshBasicMaterial({ color: 0x1e88ff, transparent: true, opacity: 0.5, depthTest: false }));
gizmoHi.renderOrder = 2; gizmoHi.visible = false; gizmoCube.add(gizmoHi);
function gizmoZone(p) {              // p in cube-local [-1,1]^3 -> region normal (-1/0/1 per axis)
  const s = (v) => (Math.abs(v) > GZ_M ? Math.sign(v) : 0);
  const n = [s(p.x), s(p.y), s(p.z)];
  // the face axis (|coord|~1) is always part of the zone even if exactly on M
  [0, 1, 2].forEach(i => { if (Math.abs([p.x, p.y, p.z][i]) > 0.999) n[i] = Math.sign([p.x, p.y, p.z][i]); });
  return n;
}
function gizmoHighlight(n) {         // build highlight quads on the cube surface for zone n
  const verts = [];
  const rng = (k) => (n[k] === 0 ? [-GZ_M, GZ_M] : (n[k] > 0 ? [GZ_M, 1] : [-1, -GZ_M]));
  for (let fa = 0; fa < 3; fa++) {
    if (n[fa] === 0) continue;
    const o = [0, 1, 2].filter(i => i !== fa);
    const [a0, a1] = rng(o[0]), [b0, b1] = rng(o[1]);
    const mk = (av, bv) => { const q = [0, 0, 0]; q[fa] = n[fa] * 1.02; q[o[0]] = av; q[o[1]] = bv; return q; };
    const c = [mk(a0, b0), mk(a1, b0), mk(a1, b1), mk(a0, b1)];
    verts.push(...c[0], ...c[1], ...c[2], ...c[0], ...c[2], ...c[3]);
  }
  gizmoHi.geometry.dispose();
  gizmoHi.geometry = new THREE.BufferGeometry();
  gizmoHi.geometry.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  gizmoHi.visible = verts.length > 0;
}
const gizmoRay = new THREE.Raycaster();
let gizmoHoverN = null;
function gizmoPick(e) {
  const r = $('gizmo').getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX-r.left)/r.width)*2-1, -((e.clientY-r.top)/r.height)*2+1);
  gizmoRay.setFromCamera(ndc, gizmoCam);
  const hit = gizmoRay.intersectObject(gizmoCube)[0];
  if (!hit) return null;
  return gizmoZone(gizmoCube.worldToLocal(hit.point.clone()));
}
$('gizmo').addEventListener('pointermove', (e) => {
  const n = gizmoPick(e);
  gizmoHoverN = n;
  if (n && (n[0] || n[1] || n[2])) gizmoHighlight(n); else gizmoHi.visible = false;
});
$('gizmo').addEventListener('pointerleave', () => { gizmoHi.visible = false; gizmoHoverN = null; });
$('gizmo').addEventListener('pointerdown', (e) => {
  const n = gizmoPick(e);
  if (n && (n[0] || n[1] || n[2])) snapView(new THREE.Vector3(n[0], n[1], n[2]));
});
function snapView(n) {
  const d = camera.position.distanceTo(controls.target) || 10;
  n = n.clone().normalize();
  camera.up.set(0, 0, 1);                                          // Z-up world
  if (Math.abs(n.z) > 0.9) camera.up.set(0, 1, 0);                 // straight top/bottom: north up
  camera.position.copy(controls.target).add(n.multiplyScalar(d));
  camera.lookAt(controls.target); controls.update();
}
// 90-degree rotations (CAD nav arrows): orbit around an axis, or roll the view
function orbitView(axis, deg) {
  const off = camera.position.clone().sub(controls.target);
  const q = new THREE.Quaternion().setFromAxisAngle(axis, deg * Math.PI / 180);
  off.applyQuaternion(q); camera.up.applyQuaternion(q);
  camera.position.copy(controls.target).add(off);
  camera.lookAt(controls.target); controls.update();
}
function rotateGizmo(kind) {
  const fwd = controls.target.clone().sub(camera.position).normalize();
  const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
  if (kind === 'left') orbitView(new THREE.Vector3(0, 0, 1), 90);
  else if (kind === 'right') orbitView(new THREE.Vector3(0, 0, 1), -90);
  else if (kind === 'up') orbitView(right, -90);
  else if (kind === 'down') orbitView(right, 90);
  else if (kind === 'rollL') orbitView(fwd, -90);
  else if (kind === 'rollR') orbitView(fwd, 90);
  else if (kind === 'home') snapView(new THREE.Vector3(1, -1, 0.8));
}
document.querySelectorAll('#gizmoNav .gn').forEach(b => b.onclick = () => rotateGizmo(b.dataset.rot));

(function loop(){ requestAnimationFrame(loop); controls.update();
  if (grid.visible) {                    // keep the infinite grid under the view + scale fade to zoom
    const u = grid.material.uniforms;
    u.uCam.value.copy(camera.position);
    u.uFade.value = Math.max(30, camera.position.distanceTo(controls.target) * 4.5);
    grid.position.x = controls.target.x; grid.position.y = controls.target.y;
  }
  renderer.render(scene, camera);
  gizmoCube.quaternion.copy(camera.quaternion).invert();   // cube mirrors the main camera
  gizmoR.render(gizmoScene, gizmoCam);
})();

function frameAll() {
  const box = new THREE.Box3();
  objects.forEach((o, id) => { if (visible.has(id)) box.expandByObject(o); });
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3()).length(), c = box.getCenter(new THREE.Vector3());
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(size*.6, -size*.6, size*.45));  // oblique, Z-up
  camera.near = Math.max(size/1000, 1e-3); camera.far = Math.max(size*1000, 1e6);
  camera.updateProjectionMatrix();
  // ground grid sits at the content's lowest Z (the infinite plane spans everything); axes at corner
  grid.position.z = box.min.z;
  const g = size || 1;
  axes.scale.setScalar(g * 0.4); axes.position.copy(box.min);
}
function toggleHelpers() { helpers = !helpers; grid.visible = helpers; axes.visible = helpers; }

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

// ---- 3D point-cloud editing: box-select + delete (non-destructive) ---------
let selMode = false, selLayer = null, selObj = null, selOrig = null;
let selRemoved = [], selSet = new Set(), selHi = null, selDrag = null;
function setSelMode(on) {
  selMode = on; $('selBtn').classList.toggle('on', on);
  controls.enabled = !on; $('center').classList.toggle('selecting', on);
  if (on) beginEdit();
}
function beginEdit() {
  let L = PROJECT.layers.find(x => x.id === selected);
  let obj = L && objects.get(L.id);
  if (!(obj && obj.isPoints)) {                 // else any visible point cloud
    for (const [id, o] of objects) if (visible.has(id) && o.isPoints) {
      obj = o; L = PROJECT.layers.find(x => x.id === id) || { id, chunk: ACTIVE_CHUNK }; break; }
  }
  if (!(obj && obj.isPoints)) { log('show a point-cloud layer first, then enable Select', 'warn'); setSelMode(false); return; }
  if (!selLayer || selLayer.id !== L.id) {       // fresh edit session for this layer
    selLayer = L; selRemoved = [];
    const n = obj.geometry.getAttribute('position').count;
    selOrig = new Int32Array(n); for (let i = 0; i < n; i++) selOrig[i] = i;
  }
  selObj = obj;
  log(`editing ${selLayer.id}: drag a box to select points (shift to add)`);
}
function clearSel() { selSet.clear(); updateSelHi(); }
function updateSelHi() {
  if (selHi) { scene.remove(selHi); selHi.geometry.dispose(); selHi = null; }
  if (!selSet.size || !selObj) return;
  const pos = selObj.geometry.getAttribute('position');
  const arr = new Float32Array(selSet.size * 3); let k = 0;
  selSet.forEach(i => { arr[k++] = pos.getX(i); arr[k++] = pos.getY(i); arr[k++] = pos.getZ(i); });
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(arr, 3));
  selHi = new THREE.Points(g, new THREE.PointsMaterial({ color: 0xff3b30, size: 4, sizeAttenuation: false, depthTest: false }));
  selHi.applyMatrix4(selObj.matrixWorld); scene.add(selHi);
}
function selectInRect(x0, y0, x1, y1, add) {
  if (!add) selSet.clear();
  const pos = selObj.geometry.getAttribute('position'); selObj.updateMatrixWorld();
  const m = selObj.matrixWorld, w = renderer.domElement.clientWidth, h = renderer.domElement.clientHeight;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).applyMatrix4(m).project(camera);
    if (v.z > 1) continue;                       // behind camera / clipped
    const sx = (v.x * 0.5 + 0.5) * w, sy = (-v.y * 0.5 + 0.5) * h;
    if (sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1) selSet.add(i);
  }
  updateSelHi(); log(`${selSet.size.toLocaleString()} point(s) selected`);
}
function deleteSelected() {
  if (!selObj || !selSet.size) { log('nothing selected', 'warn'); return; }
  const geo = selObj.geometry, pos = geo.getAttribute('position');
  const col = geo.getAttribute('color'), nor = geo.getAttribute('normal');
  selSet.forEach(i => selRemoved.push(selOrig[i]));
  const keep = []; for (let i = 0; i < pos.count; i++) if (!selSet.has(i)) keep.push(i);
  const np = new Float32Array(keep.length * 3), nc = col ? new Float32Array(keep.length * 3) : null;
  const nn = nor ? new Float32Array(keep.length * 3) : null, no = new Int32Array(keep.length);
  keep.forEach((i, j) => {
    np[j*3] = pos.getX(i); np[j*3+1] = pos.getY(i); np[j*3+2] = pos.getZ(i);
    if (nc) { nc[j*3] = col.getX(i); nc[j*3+1] = col.getY(i); nc[j*3+2] = col.getZ(i); }
    if (nn) { nn[j*3] = nor.getX(i); nn[j*3+1] = nor.getY(i); nn[j*3+2] = nor.getZ(i); }
    no[j] = selOrig[i];
  });
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(np, 3));
  if (nc) g.setAttribute('color', new THREE.BufferAttribute(nc, 3));
  if (nn) g.setAttribute('normal', new THREE.BufferAttribute(nn, 3));
  const removedNow = pos.count - keep.length;
  geo.dispose(); selObj.geometry = g; selOrig = no; selSet.clear(); updateSelHi();
  log(`deleted ${removedNow.toLocaleString()} point(s) — ${selRemoved.length.toLocaleString()} total; Save to persist`);
}
async function saveEdits() {
  if (!selLayer || !selRemoved.length) { log('no edits to save', 'warn'); return; }
  const j = await (await fetch('/api/edit_cloud', { method:'POST',
    body: JSON.stringify({ layer: selLayer.id, removed: selRemoved, chunk: selLayer.chunk }) })).json();
  if (!j.ok) { log('edit save error: ' + (j.error || 'failed'), 'err'); return; }
  log(`saved edited cloud as ${j.id} (kept ${j.kept.toLocaleString()}, removed ${j.removed.toLocaleString()})`, 'ok');
  selRemoved = []; selSet.clear(); updateSelHi(); selLayer = null; selObj = null; selOrig = null; setSelMode(false);
  await loadProject(); await runPipeline({ targets: [j.id] });
}
$('selBtn').onclick = () => setSelMode(!selMode);
$('delSelBtn').onclick = deleteSelected;
$('clearSelBtn').onclick = clearSel;
$('saveEditBtn').onclick = saveEdits;
renderer.domElement.addEventListener('pointerdown', (e) => {
  if (!selMode || e.button !== 0) return;
  selDrag = { x0: e.clientX, y0: e.clientY, r: renderer.domElement.getBoundingClientRect(), shift: e.shiftKey };
  $('selrect').style.display = 'block';
});
addEventListener('pointermove', (e) => {
  if (!selDrag) return; const r = selDrag.r, el = $('selrect');
  el.style.left = (Math.min(e.clientX, selDrag.x0) - r.left) + 'px';
  el.style.top = (Math.min(e.clientY, selDrag.y0) - r.top) + 'px';
  el.style.width = Math.abs(e.clientX - selDrag.x0) + 'px';
  el.style.height = Math.abs(e.clientY - selDrag.y0) + 'px';
});
addEventListener('pointerup', (e) => {
  if (!selDrag) return; const d = selDrag; selDrag = null; $('selrect').style.display = 'none';
  if (!selObj) return; const r = d.r;
  const x0 = Math.min(d.x0, e.clientX) - r.left, x1 = Math.max(d.x0, e.clientX) - r.left;
  const y0 = Math.min(d.y0, e.clientY) - r.top, y1 = Math.max(d.y0, e.clientY) - r.top;
  if (x1 - x0 < 3 && y1 - y0 < 3) return;
  selectInRect(x0, y0, x1, y1, d.shift);
});

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
const ic = (n) => `<svg class="ic"><use href="#i-${n}"/></svg>`;   // line-icon helper
const CAT_ICON = { Cameras:'camera', "Tie Points":'dots', "Dense Cloud":'cloud', "Point Cloud":'mountain',
  "3D Model":'triangle', "Tiled Model":'grid', DEM:'layers', Orthomosaic:'map', Shapes:'hex',
  Markers:'pin', Other:'box' };
const OP_ICON = { ingest:'image', sfm:'camera', georef:'globe', mvs:'cloud', merge_chunks:'box',
  mesh:'triangle', texture:'image', dsm:'layers', ortho:'map', classify:'mountain', contours:'hex',
  tiles:'grid', clean:'eraser' };
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
  if (eye !== null) html += `<span class="eye ${eye ? 'on' : ''}">${eye === undefined ? '' : ic('eye')}</span>`;
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
  el.appendChild(row({ depth: 0, id: 'root', hasKids: true, icon: ic('folders'),
    label: 'Workspace', count: `${PROJECT.chunks.length} chunk(s)`,
    onClick: () => toggle('root'),
    onCtx: (e) => showCtx(e, [{ label: 'Add chunk…', icon: 'folder-plus', fn: addChunk }]) }));
  if (!isOpen('root')) return;

  Object.keys(byChunk).forEach(chunk => {
    const cid = 'chunk:' + chunk;
    const layers = byChunk[chunk];
    const chunkOn = layers.length === 0 || layers.some(L => L.enabled !== false);
    el.appendChild(row({ depth: 1, id: cid, hasKids: true, cls: 'chunk' + (chunk === ACTIVE_CHUNK ? ' active' : ''),
      icon: ic('box'), label: chunk, count: `${layers.length}`,
      chk: { checked: chunkOn, onToggle: (v) => chunkAction({ action: 'set_enabled', name: chunk, enabled: v }) },
      onClick: () => { ACTIVE_CHUNK = chunk; renderWorkspace(); loadPhotos(); },
      onDbl: () => { ACTIVE_CHUNK = chunk; renderWorkspace(); },
      drop: (id) => layerAction({ action: 'move', id, to: chunk }),    // drop a layer onto a chunk
      onCtx: (e) => showCtx(e, [
        { label: 'Set as active chunk', icon: 'box', fn: () => { ACTIVE_CHUNK = chunk; renderWorkspace(); } },
        { label: `Run chunk "${chunk}"`, icon: 'play', fn: () => runPipeline({ targets: layers.map(l => l.id) }) },
        { label: 'Add Photos…', icon: 'image', fn: () => { ACTIVE_CHUNK = chunk; openBrowse(); } },
        { sep: true },
        { label: 'Rename chunk…', icon: 'edit', fn: () => renameChunk(chunk) },
        { label: 'Remove chunk', icon: 'trash', danger: true, fn: () => removeChunk(chunk) },
      ]) }));
    if (!isOpen(cid)) return;

    const cats = {}; byChunk[chunk].forEach(L => { const c = CATEGORY[L.type] || 'Other'; (cats[c] = cats[c] || []).push(L); });
    CAT_ORDER.filter(c => cats[c]).forEach(cat => {
      const catId = `cat:${chunk}:${cat}`;
      el.appendChild(row({ depth: 2, id: catId, hasKids: true, cls: 'cat', icon: ic(CAT_ICON[cat] || 'box'),
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
    const d = document.createElement('div'); if (it.danger) d.className = 'danger';
    if (it.icon) d.innerHTML = ic(it.icon);
    const sp = document.createElement('span'); sp.textContent = it.label; d.appendChild(sp);
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
  items.push({ label: `Run "${L.id}" (recompute)`, icon: 'play', fn: () => runPipeline({ targets: [L.id], force: [L.id] }) });
  items.push({ label: 'Run up to here', icon: 'play', fn: () => runPipeline({ targets: [L.id] }) });
  items.push({ sep: true });
  if (canView) items.push({ label: visible.has(L.id) ? 'Hide in view' : 'Show in view', icon: 'eye',
    fn: () => setVisible(L, !visible.has(L.id)) });
  items.push({ label: 'Rename layer…', icon: 'edit', fn: () => renameLayer(L) });
  PROJECT.chunks.filter(c => c !== L.chunk).forEach(c =>
    items.push({ label: `Move to "${c}"`, icon: 'move', fn: () => layerAction({ action: 'move', id: L.id, to: c }) }));
  items.push({ sep: true });
  items.push({ label: 'Remove layer', icon: 'trash', danger: true, fn: () => {
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
  if (L && L.type === 'contours') contourView(L);
  else if (L) rasterView(L);     // raster products (ortho/DSM/index) open in the 2D Ortho view
}
// double-click: open a layer in whichever view fits it best
function openLayer(L) {
  selected = L.id; renderParams(L);
  if (L.type === 'contours') { contourView(L); return; }       // contour lines over the DSM
  if (rasterArtifact(L)) { rasterView(L); return; }            // ortho / DEM / index -> 2D
  if (viewable(L)) { setVisible(L, true); selectVtab('model'); frameAll(); return; }  // mesh/cloud -> 3D
  if (L.type === 'ingest') { showCameras(L); return; }         // cameras -> 3D positions
  selectDock('console'); log(`${L.id}: nothing to display yet (run it first)`);
}
// show/hide camera positions (frustums) in the 3D view, like the reference tool "Show Cameras"
async function showCameras(_L) {
  const key = 'cameras:' + ACTIVE_CHUNK;
  if (objects.has(key)) {                       // toggle off if already shown
    scene.remove(objects.get(key)); objects.delete(key); visible.delete(key); return;
  }
  const data = await (await fetch('/api/cameras?chunk=' + encodeURIComponent(ACTIVE_CHUNK))).json();
  if (!data.cameras.length) { selectDock('console');
    log('no camera positions yet — run Align Photos, or add photos with GPS'); return; }
  const grp = buildCameras(data.cameras);
  objects.set(key, grp); scene.add(grp); visible.add(key);
  selectVtab('model'); frameAll();
  log(`showing ${data.cameras.length} camera(s) — ${data.frame === 'gps' ? 'EXIF GPS (pre-alignment)' : 'solved poses'}`);
}
function buildCameras(cams) {
  const grp = new THREE.Group();
  const pts = cams.map(c => new THREE.Vector3(c.c[0], c.c[1], c.c[2]));
  grp.add(new THREE.Points(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.PointsMaterial({ size: 6, sizeAttenuation: false, color: 0x89dceb })));
  const bb = new THREE.Box3().setFromPoints(pts);
  const sz = bb.getSize(new THREE.Vector3());
  const diag = sz.length() || 1;
  const horiz = Math.max(sz.x, sz.y) || 1;
  const s = Math.max(diag * 0.02, 1e-4);
  // a ground plane below the cameras + vertical drop-lines (so the capture height is visible)
  const groundZ = bb.min.z - horiz * 0.35;
  const drop = [];
  pts.forEach(C => drop.push(C, new THREE.Vector3(C.x, C.y, groundZ)));
  grp.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(drop),
    new THREE.LineBasicMaterial({ color: 0x45506a })));
  const mat = new THREE.LineBasicMaterial({ color: 0x89dceb });
  const segs = [];
  cams.forEach(c => {
    if (!c.fwd || !c.up) return;                // no orientation -> just the position point
    const C = new THREE.Vector3(...c.c);
    const f = new THREE.Vector3(...c.fwd).normalize();
    const up = new THREE.Vector3(...c.up).normalize();
    const r = new THREE.Vector3().crossVectors(f, up).normalize();
    const ctr = C.clone().add(f.clone().multiplyScalar(s * 2));
    const corner = (sx, sy) => ctr.clone().add(r.clone().multiplyScalar(sx * s)).add(up.clone().multiplyScalar(sy * s * 0.75));
    const a = corner(1, 1), b = corner(-1, 1), d = corner(-1, -1), e = corner(1, -1);
    segs.push(C, a, C, b, C, d, C, e, a, b, b, d, d, e, e, a);
  });
  if (segs.length) grp.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segs), mat));
  return grp;
}

// ---- properties / params --------------------------------------------------
function section(title) {
  const s = document.createElement('div'); s.className = 'section';
  if (title) { const h = document.createElement('div'); h.className = 'stitle'; h.textContent = title; s.appendChild(h); }
  return s;
}
function mkbtn(label, fn, icon, primary) {
  const b = document.createElement('button'); if (primary) b.className = 'run';
  b.innerHTML = (icon ? ic(icon) + ' ' : '') + `<span>${label}</span>`;
  b.onclick = fn; return b;
}
function renderParams(L) {
  const box = $('params'); box.innerHTML = '';
  if (!L) { box.innerHTML = '<div class="empty">Select a layer to see its properties.</div>'; return; }
  const info = STAGES[L.type] || { default_params: {} };
  const cur = { ...(info.default_params || {}), ...L.params };

  const head = document.createElement('div'); head.className = 'phead';
  head.innerHTML = `<span class="ico">${ic(CAT_ICON[CATEGORY[L.type]] || 'box')}</span>`
    + `<span class="pid">${L.id}</span><span class="chip">${L.type}</span>`
    + `<span class="sp"></span><span class="muted" style="font-size:11px">${L.chunk}</span>`;
  box.appendChild(head);

  // parameters as a label|control grid
  const psec = section('Parameters');
  const grid = document.createElement('div'); grid.className = 'prop';
  for (const [k, v] of Object.entries(cur)) {
    const lab = document.createElement('label'); lab.textContent = k; lab.title = k; grid.appendChild(lab);
    const inp = document.createElement('input'); inp.dataset.k = k;
    if (typeof v === 'boolean') {
      inp.type = 'checkbox'; inp.checked = v;
      const cell = document.createElement('div'); cell.className = 'chkcell'; cell.appendChild(inp); grid.appendChild(cell);
    } else {
      if (typeof v === 'number') { inp.type = 'number'; inp.value = v; inp.step = 'any'; }
      else { inp.type = 'text'; inp.value = Array.isArray(v) ? v.join(',') : v; }
      grid.appendChild(inp);
    }
  }
  if (!Object.keys(cur).length) grid.innerHTML = '<span class="empty">no parameters</span>';
  psec.appendChild(grid); box.appendChild(psec);

  const act = document.createElement('div'); act.className = 'actions';
  act.appendChild(mkbtn('Run', () => runPipeline({ targets: [L.id], force: [L.id] }), 'play', true));
  act.appendChild(mkbtn('Update', () => updateStage(L), 'edit'));
  if (viewable(L)) act.appendChild(mkbtn(visible.has(L.id) ? 'Hide' : 'Show', () => setVisible(L, !visible.has(L.id)), 'eye'));
  box.appendChild(act);

  if (Object.keys(L.metrics || {}).length) {
    const msec = section('Results');
    const mg = document.createElement('div'); mg.className = 'metrics';
    for (const [k, v] of Object.entries(L.metrics)) {
      const c = document.createElement('div'); c.className = 'metric';
      c.innerHTML = `<span class="mk">${k}</span><span class="mv" title="${v}">${v}</span>`;
      mg.appendChild(c);
    }
    msec.appendChild(mg); box.appendChild(msec);
  }
  buildExport(box, L);
}
function buildExport(box, L) {
  const arts = Object.entries(L.artifacts || {}).filter(
    ([, v]) => typeof v === 'string' && /\.(ply|las|tif|tiff|geojson|obj|glb)$/i.test(v));
  if (!arts.length) return;
  const sec = section('Export');
  const g = document.createElement('div'); g.className = 'prop';
  const asel = document.createElement('select');
  arts.forEach(([k, v]) => { const o = document.createElement('option'); o.value = v; o.textContent = k; asel.appendChild(o); });
  const fsel = document.createElement('select');
  const refresh = async () => {
    fsel.innerHTML = '';
    const { formats } = await (await fetch('/api/formats?path=' + encodeURIComponent(asel.value))).json();
    (formats || []).forEach(f => { const o = document.createElement('option'); o.value = f; o.textContent = f; fsel.appendChild(o); });
  };
  asel.onchange = refresh;
  g.append(Object.assign(document.createElement('label'), { textContent: 'artifact' }), asel,
           Object.assign(document.createElement('label'), { textContent: 'format' }), fsel);
  sec.appendChild(g);
  const eb = mkbtn('Export as…', async () => {
    const j = await (await fetch('/api/export', { method:'POST',
      body: JSON.stringify({ path: asel.value, fmt: fsel.value }) })).json();
    log(j.out ? `exported → ${j.out}` : `export error: ${j.error}`, j.out ? 'ok' : 'err');
  }, 'save');
  eb.style.marginTop = '8px'; sec.appendChild(eb); box.appendChild(sec); refresh();
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

// ---- run progress popup (industry-standard, minimizable) --------------------
function progShow(title) {
  $('progTitle').textContent = title || 'Processing…';
  $('progStage').textContent = ''; $('progPct').textContent = '';
  $('progLog').textContent = ''; setBar(null);
  $('progress').classList.remove('hidden', 'min');
}
function progHide() { $('progress').classList.add('hidden'); }
function setBar(frac) {
  const b = $('progBar');
  if (frac == null || isNaN(frac)) { b.classList.add('indet'); b.style.width = '35%'; }
  else { b.classList.remove('indet'); b.style.width = Math.round(frac * 100) + '%'; }
}
function progLog(msg, cls) {
  const el = $('progLog'); const line = document.createElement('div');
  if (cls) line.className = cls; line.textContent = msg; el.appendChild(line);
  el.scrollTop = el.scrollHeight;
  while (el.childNodes.length > 500) el.removeChild(el.firstChild);
}
$('progMin').onclick = () => $('progress').classList.toggle('min');
$('progHead').ondblclick = () => $('progress').classList.toggle('min');
$('progCancel').onclick = async () => {
  $('progCancel').disabled = true;
  await fetch('/api/cancel', { method:'POST', body:'{}' });
  log('cancel requested'); progLog('cancel requested — stopping after the current stage…', 'warn');
};

// ---- run + live progress (SSE) --------------------------------------------
$('runBtn').onclick = () => runPipeline();
async function runPipeline(body = {}) {
  const r = await fetch('/api/run', { method:'POST', body: JSON.stringify(body) });
  if (r.status === 409) { log('already running'); return; }
  log('--- run started ---'); $('status').textContent = 'running…';
  $('progCancel').disabled = false; progShow('Processing…');
  const es = new EventSource('/api/events');
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.event === 'log') {
      const cls = ev.level === 'ERROR' ? 'err' : (ev.level === 'WARNING' ? 'warn' : '');
      log(ev.msg, cls || undefined); progLog(ev.msg, cls);
    } else if (ev.event === 'stage_start') { setDot(ev.id, 'running');
      $('progTitle').textContent = `Processing: ${ev.id} (${ev.type})`; $('progStage').textContent = `running ${ev.id}…`;
      setBar(null); log(`▶ ${ev.id} (${ev.type})`); progLog(`▶ ${ev.id} (${ev.type})`);
    } else if (ev.event === 'progress') {
      const pct = `${ev.id}: ${Math.round(ev.frac*100)}% ${ev.message||''}`;
      $('status').textContent = pct; $('progStage').textContent = pct;
      $('progPct').textContent = `${Math.round(ev.frac*100)}%`; setBar(ev.frac);
    } else if (ev.event === 'stage_done') { setDot(ev.id, ev.status);
      const m = `${ev.status === 'failed' ? '✗' : '✓'} ${ev.id} [${ev.status}]` + (ev.error ? ` — ${ev.error}` : '');
      log(m); progLog(m, ev.status === 'failed' ? 'err' : '');
    } else if (ev.event === 'stage_skipped') { setDot(ev.id, 'failed'); log(`⨯ ${ev.id} skipped`); progLog(`⨯ ${ev.id} skipped`, 'warn'); }
    else if (ev.event === 'run_done') { log(`--- run ${ev.ok ? 'OK' : 'FAILED'} ---`);
      $('progTitle').textContent = ev.ok ? 'Done' : 'Failed'; $('progStage').textContent = ev.ok ? 'completed' : 'failed';
      setBar(1); progLog(`--- run ${ev.ok ? 'OK' : 'FAILED'} ---`, ev.ok ? '' : 'err'); }
    else if (ev.event === 'run_error') { log(`error: ${ev.error}`); progLog(`error: ${ev.error}`, 'err'); }
  };
  es.addEventListener('eof', async () => { es.close(); $('status').textContent = 'done';
    setTimeout(progHide, 1600);
    const reshow = [...visible];
    const camChunks = reshow.filter(id => id.startsWith('cameras:')).map(id => id.slice(8));
    reshow.forEach(id => { if (objects.has(id)) { scene.remove(objects.get(id)); objects.delete(id); } });
    visible.clear();
    await loadProject(); await loadPhotos();
    for (const id of reshow) { const L = PROJECT.layers.find(x => x.id === id); if (L) await setVisible(L, true); }
    for (const ch of camChunks) {                 // re-show camera overlays (not layers) after a run
      const prev = ACTIVE_CHUNK; ACTIVE_CHUNK = ch; await showCameras(); ACTIVE_CHUNK = prev;
    }
    if (selected) selectLayer(selected);
    showGcpAccuracy(); });
}
function setDot(id, cls) { const L = PROJECT.layers.find(x => x.id === id); if (L) L.status = cls; renderWorkspace(); }

// ---- menus ----------------------------------------------------------------
function closeMenus() { document.querySelectorAll('.mMenu').forEach(m => m.classList.add('hidden')); }
document.querySelectorAll('.mItem').forEach(it => {
  it.onclick = (e) => { e.stopPropagation();
    const m = it.querySelector('.mMenu'); const wasOpen = !m.classList.contains('hidden');
    closeMenus(); if (!wasOpen) m.classList.remove('hidden'); };
});
document.addEventListener('click', () => { closeMenus(); });
function menuEntry(menu, label, fn, desc, icon) {
  const d = document.createElement('div');
  const i = icon ? ic(icon) + ' ' : '';
  d.innerHTML = `${i}<span>${label}</span>` + (desc ? `<div class="t">${desc}</div>` : '');
  d.onclick = (e) => { e.stopPropagation(); closeMenus(); fn(); };
  $(menu).appendChild(d);
}
function menuSep(menu) { $(menu).appendChild(document.createElement('hr')); }
async function loadWorkflows() {
  WORKFLOWS = await (await fetch('/api/workflows')).json();
  // File menu
  $('m-file').innerHTML = '';
  menuEntry('m-file', 'New project…', newProject, null, 'file-plus');
  menuEntry('m-file', 'Save project', saveProject, null, 'save');
  menuSep('m-file');
  menuEntry('m-file', 'New chunk', addChunk, null, 'folder-plus');
  menuEntry('m-file', 'Set coordinate system…', openCrsPicker, null, 'globe');
  menuSep('m-file');
  menuEntry('m-file', 'Processing report', () => window.open('/api/report', '_blank'), null, 'chart');
  // Workflow menu = the familiar operations
  $('m-workflow').innerHTML = '';
  WORKFLOWS.forEach(op => menuEntry('m-workflow', op.op, () => openOp(op), op.desc, OP_ICON[op.stage] || 'play'));
  // Model menu = view helpers
  $('m-model').innerHTML = '';
  menuEntry('m-model', 'Frame all', frameAll, null, 'maximize');
  menuEntry('m-model', 'Show / hide grid + axes', toggleHelpers, null, 'grid');
  menuEntry('m-model', 'Show / hide cameras', () => showCameras(), 'camera positions of the active chunk', 'camera');
  menuEntry('m-model', 'Hide all layers', () => { visible.forEach(id => { const o = objects.get(id); if (o) o.visible = false; });
    visible.clear(); renderWorkspace(); }, null, 'eye');
  // Tools menu
  $('m-tools').innerHTML = '';
  menuEntry('m-tools', 'Measure distance', () => setMeasure('dist'), null, 'ruler');
  menuEntry('m-tools', 'Measure area', () => setMeasure('area'), null, 'square');
  menuSep('m-tools');
  menuEntry('m-tools', 'Markers / GCPs', () => { selectLeft('reference'); loadMarkers(); }, null, 'pin');
  // Help
  $('m-help').innerHTML = '';
  menuEntry('m-help', 'About OpenReco', () => log('OpenReco — open, reproducible photogrammetry. Clean-room; permissive OSS.'), null, 'info');
}

// ---- quality / speed presets ----------------------------------------------
async function loadPresets() {
  const ps = await (await fetch('/api/presets')).json();
  const sel = $('presetSel'); sel.innerHTML = '<option value="">Quality preset…</option>';
  ps.forEach(p => { const o = document.createElement('option'); o.value = p.name;
    o.textContent = `${p.name} · ${p.speed}`; sel.appendChild(o); });
  sel.onchange = async () => {
    if (!sel.value) return;
    const j = await (await fetch('/api/preset', { method:'POST', body: JSON.stringify({ name: sel.value }) })).json();
    if (j.ok) { log(`preset "${j.preset}" applied to ${j.updated} layer(s); new layers will use it too`, 'ok');
      await loadProject(); if (selected) selectLayer(selected); }
    else log('preset error: ' + j.error, 'err');
  };
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
  if (name === 'reference') { loadMarkers(); showGcpAccuracy(); }
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
  $('mapview').classList.toggle('show', name === 'map');
  $('c').style.display = name === 'model' ? 'block' : 'none';
  // orientation cube only belongs to the 3D view
  const showGz = name === 'model' ? 'block' : 'none';
  $('gizmo').style.display = showGz; $('gizmoNav').style.display = showGz;
  if (name === 'map') showOnMap(mapRaster);
}
document.querySelectorAll('[data-vtab]').forEach(b => b.onclick = () => selectVtab(b.dataset.vtab));

// ---- web map: place a georeferenced ortho/DEM on an OpenStreetMap basemap ----
let lmap = null, lover = null, mapRaster = null;
function initMap() {
  if (lmap) return;
  lmap = L.map('leaflet', { zoomControl: false, attributionControl: false }).setView([0, 0], 2);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 22 }).addTo(lmap);
  L.control.zoom({ position: 'bottomright' }).addTo(lmap);
}
async function showOnMap(layer) {
  initMap(); setTimeout(() => lmap.invalidateSize(), 30);
  if (!layer) { log('select a georeferenced raster (ortho / DEM), then open the Map tab'); return; }
  const tif = rasterArtifact(layer);
  if (!tif) { log(`${layer.id} has no raster to map — pick an ortho or DEM layer`, 'warn'); return; }
  log(`loading ${layer.id} onto the map…`);
  const j = await (await fetch('/api/geo_overlay?path=' + encodeURIComponent(tif))).json();
  if (!j.ok) { log(`map overlay error: ${j.error || 'failed'} — georeference the layer first`, 'err'); return; }
  if (lover) lmap.removeLayer(lover);
  lover = L.imageOverlay(j.image, j.bounds, { opacity: parseFloat($('mapOpacity').value) }).addTo(lmap);
  lmap.fitBounds(j.bounds);
  log(`${layer.id} placed on the map`, 'ok');
}
$('mapOpacity').oninput = () => { if (lover) lover.setOpacity(parseFloat($('mapOpacity').value)); };

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
function applyOrtho() {
  const t = `translate(${oz.tx}px,${oz.ty}px) scale(${oz.s})`;
  $('orthoimg').style.transform = t; $('orthocanvas').style.transform = t;
}
function rasterView(layer, tifOverride, onReady) {
  const tif = tifOverride || rasterArtifact(layer); if (!tif) return false;
  if (rasterArtifact(layer)) mapRaster = layer;   // remember for the Map tab
  selectVtab('ortho');
  const img = $('orthoimg'), cv = $('orthocanvas');
  cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);   // clear any prior overlay
  img.onload = () => {                                   // fit to viewport
    const w = $('center').clientWidth, h = $('center').clientHeight;
    oz.s = Math.min(w / img.naturalWidth, h / img.naturalHeight) * 0.95;
    oz.tx = (w - img.naturalWidth * oz.s) / 2; oz.ty = (h - img.naturalHeight * oz.s) / 2;
    cv.width = img.naturalWidth; cv.height = img.naturalHeight;
    applyOrtho();
    if (onReady) onReady(img);
  };
  img.src = `/api/raster_png?path=${encodeURIComponent(tif)}`;
  $('orthohint').textContent = `${layer.id} · ${tif.split(/[\\/]/).pop()} · scroll to zoom, drag to pan`;
  return true;
}
// contours: draw pixel-space lines over their input DSM raster in the 2D Ortho view
async function contourView(L) {
  const dsmLayer = (L.inputs || []).map(id => PROJECT.layers.find(x => x.id === id))
    .find(x => x && (x.artifacts || {}).dsm);
  const linesPath = (L.artifacts || {}).lines;
  if (!dsmLayer || !linesPath) { selectDock('console'); log(`${L.id}: run it (needs a DSM input) to view contours`); return; }
  rasterView(L, dsmLayer.artifacts.dsm, async (img) => {
    const data = await (await fetch(`/api/file?path=${encodeURIComponent(linesPath)}`)).json();
    const cv = $('orthocanvas'), g = cv.getContext('2d');
    const sx = img.naturalWidth / data.width, sy = img.naturalHeight / data.height;
    g.clearRect(0, 0, cv.width, cv.height);
    g.strokeStyle = '#f9e2af'; g.lineWidth = Math.max(1, img.naturalWidth / 1200);
    g.beginPath();
    for (const [c0, r0, c1, r1] of data.segments) {
      g.moveTo(c0 * sx, r0 * sy); g.lineTo(c1 * sx, r1 * sy);
    }
    g.stroke();
    $('orthohint').textContent = `${L.id} · ${data.segments.length} contour segments over ${dsmLayer.id}`;
  });
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
    t.innerHTML = `<button class="rm" title="Remove from chunk">${ic('x')}</button>`
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
const iz = { s: 1, tx: 0, ty: 0 };       // photo view pan/zoom
function applyImg() { $('imgwrap').style.transform = `translate(${iz.tx}px,${iz.ty}px) scale(${iz.s})`; }
function openPhoto(im) {
  curPhoto = im; selectVtab('photo');
  const wrap = $('imgwrap'); wrap.innerHTML = '';
  const img = document.createElement('img'); img.src = `/api/file?path=${encodeURIComponent(im.path)}`;
  img.onload = () => {                    // fit to viewport
    const w = $('center').clientWidth, h = $('center').clientHeight;
    iz.s = Math.min(w / img.naturalWidth, h / img.naturalHeight) * 0.96;
    iz.tx = (w - img.naturalWidth * iz.s) / 2; iz.ty = (h - img.naturalHeight * iz.s) / 2;
    applyImg(); drawPins();
  };
  wrap.appendChild(img);
  $('imghint').textContent = `${im.name} · scroll to zoom, drag to pan, click to place the selected marker`;
}
function drawPins() {
  const wrap = $('imgwrap'); const img = wrap.querySelector('img'); if (!img) return;
  [...wrap.querySelectorAll('.pin')].forEach(p => p.remove());
  MARKERS.forEach(mk => mk.observations.forEach(o => {
    if (!curPhoto || o.image !== curPhoto.name) return;
    const pin = document.createElement('div'); pin.className = 'pin';
    pin.style.left = o.u + 'px'; pin.style.top = o.v + 'px';   // natural-pixel coords (wrap is scaled)
    pin.style.transform = `scale(${1 / iz.s})`; pin.style.transformOrigin = '0 0';   // keep pin size constant
    pin.innerHTML = `<span>${mk.name}</span>`;
    wrap.appendChild(pin);
  }));
}
function placeObservation(e) {
  const img = $('imgwrap').querySelector('img'); if (!img || !curPhoto) return;
  if (activeMarker == null) { log('select a marker in Reference ▸ Markers first'); return; }
  const r = img.getBoundingClientRect();
  if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) return;
  const uu = Math.round((e.clientX - r.left) / r.width * img.naturalWidth);
  const vv = Math.round((e.clientY - r.top) / r.height * img.naturalHeight);
  MARKERS[activeMarker].observations = MARKERS[activeMarker].observations.filter(o => o.image !== curPhoto.name);
  MARKERS[activeMarker].observations.push({ image: curPhoto.name, u: uu, v: vv });
  log(`marker ${MARKERS[activeMarker].name}: observed in ${curPhoto.name} @ (${uu},${vv})`);
  renderMarkers(); drawPins();
}
(function photoNav() {
  const view = $('imgview'); let drag = null, moved = false;
  view.addEventListener('wheel', (e) => { e.preventDefault();
    const r = view.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    iz.tx = mx - (mx - iz.tx) * f; iz.ty = my - (my - iz.ty) * f; iz.s *= f; applyImg(); drawPins();
  }, { passive: false });
  view.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, tx: iz.tx, ty: iz.ty };
    moved = false; view.classList.add('drag'); view.setPointerCapture(e.pointerId); });
  view.addEventListener('pointermove', (e) => { if (!drag) return;
    if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3) moved = true;
    iz.tx = drag.tx + (e.clientX - drag.x); iz.ty = drag.ty + (e.clientY - drag.y); applyImg(); });
  view.addEventListener('pointerup', (e) => { view.classList.remove('drag'); const wasDrag = moved; drag = null;
    if (!wasDrag) placeObservation(e); });
})();

// ---- markers / GCP reference table ----------------------------------------
let MARKERS = [];           // [{name, world:[x,y,z]|null, observations:[{image,u,v}]}]
let activeMarker = null;
async function loadMarkers() {
  const j = await (await fetch('/api/markers')).json();
  MARKERS = (j.markers || []).map(m => ({ name: m.name, world: m.world || null,
    observations: m.observations || [], type: m.type === 'check' ? 'check' : 'control' }));
  if (MARKERS.length && activeMarker == null) activeMarker = 0;
  renderMarkers();
}
function renderMarkers() {
  const box = $('markerTable');
  if (!MARKERS.length) { box.innerHTML = '<div class="muted">No markers. Add one, then pick it in photos.</div>'; return; }
  const t = document.createElement('table');
  t.innerHTML = '<tr><th>name</th><th>X</th><th>Y</th><th>Z</th><th>obs</th><th>type</th><th></th></tr>';
  MARKERS.forEach((m, i) => {
    const tr = document.createElement('tr'); tr.className = (i === activeMarker ? 'sel' : '');
    const w = m.world || ['','',''];
    tr.innerHTML = `<td>${m.name}</td>`
      + [0,1,2].map(k => `<td><input data-mi="${i}" data-wk="${k}" value="${w[k]}" style="width:54px"></td>`).join('')
      + `<td>${m.observations.length}</td>`
      + `<td><select data-ti="${i}" style="width:72px"><option value="control"${m.type!=='check'?' selected':''}>control</option>`
      + `<option value="check"${m.type==='check'?' selected':''}>check</option></select></td>`
      + `<td><button class="iconbtn mrm" data-rm="${i}" title="Remove marker">${ic('trash')}</button></td>`;
    tr.onclick = (e) => { if (!['INPUT', 'SELECT', 'BUTTON'].includes(e.target.tagName) && !e.target.closest('button'))
      { activeMarker = i; renderMarkers(); } };
    t.appendChild(tr);
  });
  box.innerHTML = ''; box.appendChild(t);
  box.querySelectorAll('input[data-mi]').forEach(inp => inp.onchange = () => {
    const m = MARKERS[+inp.dataset.mi]; m.world = m.world || [0,0,0];
    m.world[+inp.dataset.wk] = parseFloat(inp.value) || 0;
  });
  box.querySelectorAll('select[data-ti]').forEach(sel => sel.onchange = () => {
    MARKERS[+sel.dataset.ti].type = sel.value;
  });
  box.querySelectorAll('button[data-rm]').forEach(b => b.onclick = (e) => {
    e.stopPropagation(); const i = +b.dataset.rm;
    log(`removed marker ${MARKERS[i].name}`);
    MARKERS.splice(i, 1);
    if (activeMarker === i) activeMarker = MARKERS.length ? 0 : null;
    else if (activeMarker > i) activeMarker--;
    renderMarkers(); drawPins();
  });
}
$('addMarker').onclick = () => {
  const name = prompt('Marker name:', `GCP${MARKERS.length + 1}`); if (!name) return;
  MARKERS.push({ name, world: null, observations: [], type: 'control' });
  activeMarker = MARKERS.length - 1; renderMarkers();
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

// auto coded-target (ArUco) detection + printable template
['4x4_50', '5x5_100', '6x6_250', 'apriltag_36h11', 'aruco_original'].forEach(dn => {
  const o = document.createElement('option'); o.value = dn; o.textContent = dn; $('mkDict').appendChild(o);
});
$('markerSheet').onclick = () => window.open(`/api/marker_template?dictionary=${$('mkDict').value}&count=24`, '_blank');
$('detectMarkers').onclick = async () => {
  log(`detecting coded targets (${$('mkDict').value}) in ${ACTIVE_CHUNK}…`);
  const j = await (await fetch('/api/detect_markers', { method:'POST',
    body: JSON.stringify({ chunk: ACTIVE_CHUNK, dictionary: $('mkDict').value }) })).json();
  if (!j.ok) { log('detect error: ' + (j.error || 'failed'), 'err'); return; }
  j.markers.forEach(nm => {                          // merge: update observations, keep any world coords
    const ex = MARKERS.find(m => m.name === nm.name);
    if (ex) ex.observations = nm.observations; else MARKERS.push({ ...nm });
  });
  if (MARKERS.length && activeMarker == null) activeMarker = 0;
  renderMarkers();
  log(`detected ${j.markers.length} marker(s), ${j.detections} observation(s) across ${j.images_scanned} photo(s) — enter their world X/Y/Z, then Save`,
      j.markers.length ? 'ok' : 'warn');
};

// GCP accuracy: read the chunk's georef.json (per-GCP residuals + control/check RMSE)
async function showGcpAccuracy() {
  const box = $('gcpAccuracy');
  const gl = PROJECT && PROJECT.layers.find(L => L.type === 'georef' && L.chunk === ACTIVE_CHUNK
    && (L.artifacts || {}).georef);
  if (!gl) { box.innerHTML = '<span class="muted">Run Georeference with GCPs to see residuals.</span>'; return; }
  let info;
  try { info = await (await fetch(`/api/file?path=${encodeURIComponent(gl.artifacts.georef)}`)).json(); }
  catch (_e) { box.innerHTML = '<span class="muted">no georef report yet</span>'; return; }
  if (info.method !== 'gcp' || !info.gcps) {
    box.innerHTML = `<span class="muted">georef method: ${info.method} (RMS ${info.rms_residual_m ?? '—'} m). Use GCPs for control/check accuracy.</span>`;
    return;
  }
  let html = `<div>control RMSE: <b>${info.control_rms_m ?? '—'} m</b> (${info.num_control} pts)`
    + (info.check_rms_m != null ? ` · check RMSE: <b>${info.check_rms_m} m</b> (${info.num_check})` : '') + '</div>';
  html += '<table style="margin-top:4px"><tr><th>GCP</th><th>type</th><th>err(m)</th><th>dx</th><th>dy</th><th>dz</th></tr>';
  for (const g of info.gcps) {
    const hi = g.error_m > 0.5 ? ' style="color:var(--warn)"' : '';
    html += `<tr${hi}><td>${g.name}</td><td>${g.type}</td><td>${g.error_m}</td><td>${g.dx}</td><td>${g.dy}</td><td>${g.dz}</td></tr>`;
  }
  box.innerHTML = html + '</table>';
}
$('refreshAcc').onclick = showGcpAccuracy;

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
  // auto-wire: pick the latest layer (prefer same chunk) that provides each artifact the op needs
  const needs = op.needs || [];
  const auto = new Set();
  needs.forEach(art => {
    const providers = PROJECT.layers.filter(l => (l.provides || []).includes(art));
    const pref = providers.filter(l => l.chunk === ACTIVE_CHUNK);
    const pick = (pref.length ? pref : providers).slice(-1)[0];
    if (pick) auto.add(pick.id);
  });
  const inb = $('mInputs'); inb.innerHTML = PROJECT.layers.length ? '' : '<span class="muted">none yet</span>';
  if (needs.length) { const h = document.createElement('div'); h.className = 'muted';
    h.style.marginBottom = '4px'; h.textContent = `needs: ${needs.join(', ')} (auto-selected below)`; inb.appendChild(h); }
  PROJECT.layers.forEach(l => {
    const w = document.createElement('label'); w.className = 'chk';
    const on = auto.has(l.id) ? 'checked' : '';
    w.innerHTML = `<input type="checkbox" value="${l.id}" ${on}> ${l.id} <span class="muted">(${l.type})</span>`;
    inb.appendChild(w);
  });
  const fb = $('mFields'); fb.innerHTML = '';
  const fg = document.createElement('div'); fg.className = 'prop'; fb.appendChild(fg);
  op.fields.forEach(f => {
    const lab = document.createElement('label'); lab.textContent = f.label; lab.title = f.label;
    let inp, cell;
    if (f.type === 'enum') {
      inp = document.createElement('select');
      Object.keys(f.options).forEach(k => { const o = document.createElement('option'); o.value = k; o.textContent = k; inp.appendChild(o); });
      inp.value = f.default;
    } else if (f.type === 'bool') { inp = document.createElement('input'); inp.type = 'checkbox'; inp.checked = !!f.default;
      cell = document.createElement('div'); cell.className = 'chkcell'; cell.appendChild(inp); }
    else if (f.type === 'path' || f.type === 'string') {
      inp = document.createElement('input'); inp.type = 'text'; inp.value = f.default;
      if (f.type === 'path') inp.placeholder = 'folder path (e.g. D:\\data\\flight1 or "images")';
    }
    else { inp = document.createElement('input'); inp.type = 'number'; inp.step = 'any'; inp.value = f.default; }
    inp.dataset.label = f.label; inp.dataset.type = f.type;
    fg.append(lab, cell || inp);
  });
  // Add Photos (ingest) gets a file picker that selects specific images across folders
  $('mBrowse').classList.toggle('hidden', op.stage !== 'ingest');
  $('mBrowse').onclick = () => openBrowse($('mId').value.trim());
  $('mOk').onclick = () => submitOp(op, true);
  $('mAddOnly').onclick = () => submitOp(op, false);
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
    row.innerHTML = ic('folders');
    const s = document.createElement('span'); s.textContent = p.split(/[\\/]/).filter(Boolean).pop(); row.appendChild(s);
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
async function submitOp(op, run) {
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
  if (!r.ok) { log(`build error: ${j.error}`); return; }
  $('modal').classList.add('hidden'); log(`built ${id} (${op.op})`);
  await loadProject(); selectLayer(id);
  if (run) await runPipeline();           // industry-standard: process the step right away (cache-aware)
}

// ---- resizable panes (drag the splitters; persisted in localStorage) ------
const PANES = { left: 300, right: 330, dock: 190 };
function layoutPanes() {
  const app = $('app');
  app.style.gridTemplateColumns = `${PANES.left}px 1fr ${PANES.right}px`;
  app.style.gridTemplateRows = `28px 34px 1fr ${PANES.dock}px`;
  // reposition the drag handles to match
  $('splitL').style.left = (PANES.left - 5) + 'px';
  $('splitR').style.right = (PANES.right - 5) + 'px';
  $('splitD').style.bottom = (PANES.dock - 5) + 'px';
  $('splitL').style.bottom = $('splitR').style.bottom = PANES.dock + 'px';
  resize();   // keep the 3D canvas in sync
}
function setupSplitters() {
  const saved = localStorage.getItem('orePanes');
  if (saved) { try { Object.assign(PANES, JSON.parse(saved)); } catch (_e) { /* ignore */ } }
  const drag = (el, onMove) => {
    el.addEventListener('pointerdown', (e) => {
      e.preventDefault(); el.classList.add('drag'); el.setPointerCapture(e.pointerId);
      const move = (ev) => { onMove(ev); layoutPanes(); };
      const up = (ev) => { el.classList.remove('drag');
        try { el.releasePointerCapture(ev.pointerId); } catch (_e) { /* already released */ }
        el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', up);
        localStorage.setItem('orePanes', JSON.stringify(PANES));
      };
      el.addEventListener('pointermove', move); el.addEventListener('pointerup', up);
    });
  };
  drag($('splitL'), (e) => { PANES.left = Math.max(160, Math.min(e.clientX, innerWidth - 420)); });
  drag($('splitR'), (e) => { PANES.right = Math.max(180, Math.min(innerWidth - e.clientX, innerWidth - 420)); });
  drag($('splitD'), (e) => { PANES.dock = Math.max(60, Math.min(innerHeight - e.clientY, innerHeight - 200)); });
  layoutPanes();
}

// ---- boot ----
(async () => { setupSplitters(); resize(); await loadStages(); await loadWorkflows(); await loadPresets();
  await loadProject(); await loadMarkers();
  log(`OpenReco ready · project "${PROJECT.name}" · ${PROJECT.layers.length} layer(s)`, 'ok'); })();
