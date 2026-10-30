// The live demo: C vector field -> cimo (in Pyodide) -> WebAssembly -> particles.
// Pyodide runs cimo; each edit recompiles the C. Every frame JS calls the compiled step()
// directly on a trail ring buffer in Pyodide's memory, which three.js draws from in place.
(function () {
  'use strict';

  const N = 16000;          // particles
  const T = 18;             // trail length (ring slots)
  const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
  const LIBS = [
    'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.js',
  ];
  const CM_MODE = 'https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/mode/clike/clike.min.js';
  const CM_CSS = 'https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.css';

  const INTEGRATOR = `
/* ---- integrator: RK4 over every particle (edit this too) ---- */
static unsigned long long rng = 88172645463325252ULL;
static float frand(void)  /* uniform in [-1, 1) */
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)(rng >> 40) * (2.0f / 16777216.0f) - 1.0f;
}

static void put(int n, int T, float *ring, int i, float x, float y, float z)
{
    for (int t = 0; t < T; t++) {
        float *p = ring + 3 * (t * n + i);
        p[0] = x; p[1] = y; p[2] = z;
    }
}

void seed(int n, int T, float *ring)
{
    for (int i = 0; i < n; i++)
        put(n, T, ring, i, SPAWN * frand(), SPAWN * frand(), SPAWN * frand());
}

/* one frame: read ring slot head, write the next slot;
   returns how many particles escaped and were respawned */
int step(int n, int T, int head, float *ring)
{
    const float *cur = ring + 3 * head * n;
    float *next = ring + 3 * ((head + 1) % T) * n;
    const float h = DT / SUBSTEPS;
    int respawned = 0;
    for (int i = 0; i < n; i++) {
        float x = cur[3 * i], y = cur[3 * i + 1], z = cur[3 * i + 2];
        for (int s = 0; s < SUBSTEPS; s++) {
            float ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz;
            float g = h / 2;
            field(x, y, z, &ax, &ay, &az);
            field(x + g * ax, y + g * ay, z + g * az, &bx, &by, &bz);
            field(x + g * bx, y + g * by, z + g * bz, &cx, &cy, &cz);
            field(x + h * cx, y + h * cy, z + h * cz, &dx, &dy, &dz);
            x += h / 6 * (ax + 2 * bx + 2 * cx + dx);
            y += h / 6 * (ay + 2 * by + 2 * cy + dy);
            z += h / 6 * (az + 2 * bz + 2 * cz + dz);
        }
        if (!(fabsf(x) + fabsf(y) + fabsf(z) < 1e4f)) {  /* escaped or NaN */
            x = SPAWN * frand(); y = SPAWN * frand(); z = SPAWN * frand();
            put(n, T, ring, i, x, y, z);
            respawned++;
        }
        next[3 * i] = x; next[3 * i + 1] = y; next[3 * i + 2] = z;
    }
    return respawned;
}
`;

  const PRESETS = {
    'Aizawa': {
      spawn: '1.0f', dt: '0.012f', sub: 3, body: `
    const float a = 0.95f, b = 0.7f, c = 0.6f;
    const float d = 3.5f, e = 0.25f, f = 0.1f;
    *dx = (z - b) * x - d * y;
    *dy = d * x + (z - b) * y;
    *dz = c + a * z - z * z * z / 3
        - (x * x + y * y) * (1 + e * z) + f * z * x * x * x;`},
    'Lorenz': {
      spawn: '20.0f', dt: '0.0035f', sub: 3, body: `
    const float s = 10, r = 28, b = 8.0f / 3;
    *dx = s * (y - x);
    *dy = x * (r - z) - y;
    *dz = x * y - b * z;`},
    'Thomas': {
      spawn: '4.0f', dt: '0.06f', sub: 3, body: `
    const float b = 0.208186f;
    *dx = sinf(y) - b * x;
    *dy = sinf(z) - b * y;
    *dz = sinf(x) - b * z;`},
    'Halvorsen': {
      spawn: '2.0f', dt: '0.006f', sub: 3, body: `
    const float a = 1.89f;
    *dx = -a * x - 4 * y - 4 * z - y * y;
    *dy = -a * y - 4 * z - 4 * x - z * z;
    *dz = -a * z - 4 * x - 4 * y - x * x;`},
    'Dadras': {
      spawn: '3.0f', dt: '0.006f', sub: 3, body: `
    const float a = 3, b = 2.7f, c = 1.7f, d = 2, e = 9;
    *dx = y - a * x + b * y * z;
    *dy = c * y - x * z + z;
    *dz = d * x * y - e * z;`},
    'Rössler': {
      spawn: '6.0f', dt: '0.03f', sub: 3, body: `
    const float a = 0.2f, b = 0.2f, c = 5.7f;
    *dx = -y - z;
    *dy = x + a * y;
    *dz = b + z * (x - c);`},
    'Chen–Lee': {
      spawn: '8.0f', dt: '0.004f', sub: 3, body: `
    const float a = 5, b = -10, c = -0.38f;
    *dx = a * x - y * z;
    *dy = b * y + x * z;
    *dz = c * z + x * y / 3;`},
  };

  function source(name) {
    const p = PRESETS[name];
    return `/* ${name} attractor. Edit the vector field below:
   it recompiles as you type. */
#include <math.h>

#define SPAWN ${p.spawn}   /* initial spread of the particles */
#define DT ${p.dt}      /* time per frame */
#define SUBSTEPS ${p.sub}     /* RK4 steps per frame */

static void field(float x, float y, float z,
                  float *dx, float *dy, float *dz)
{${p.body}
}
` + INTEGRATOR;
  }

  const PY = `
import ctypes, time
import cimo

N, T = ${N}, ${T}
ring = (ctypes.c_float * (N * T * 3))()
RING = ctypes.addressof(ring)
_live = None

def build(src):
    global _live
    t = time.perf_counter()
    try:
        m = cimo.compile(src)
    except cimo.CompileError as e:
        return {'ok': False, 'error': str(e)}
    ms = (time.perf_counter() - t) * 1000
    names = dir(m)
    for name, sig in (('step', ['w', 'w', 'w', 'l']), ('seed', ['w', 'w', 'l'])):
        if name not in names or getattr(m, name)._params != sig:
            return {'ok': False, 'error': 'the demo calls %s(int n, int T, %sfloat *ring); keep '
                    'its signature' % (name, 'int head, ' if name == 'step' else '')}
    _live = m
    e = m.exports
    return {'ok': True, 'ms': ms, 'step': m.step.function, 'seed': m.seed.function,
            'getsp': getattr(e, '__cimo_get_sp'), 'setsp': getattr(e, '__cimo_set_sp')}
`;

  globalThis.cimoDemo = { source, presets: Object.keys(PRESETS), N, T, PY };  // for tests
  const $ = (s, r) => (r || document).querySelector(s);
  const root = $('#demo');
  if (!root) return;
  const ui = {
    start: $('.demo-start', root), status: $('.demo-status', root), stats: $('.demo-stats', root),
    err: $('.demo-err', root), code: $('.demo-code', root), view: $('.demo-view', root),
    preset: $('.demo-preset', root), reseed: $('.demo-reseed', root), out: $('.demo-out', root),
  };
  // before starting: the default source as static text
  ui.preset.innerHTML = Object.keys(PRESETS).map(n => `<option>${n}</option>`).join('');
  const want = /[?&]preset=([^&#]+)/.exec(location.search);
  if (want && PRESETS[decodeURIComponent(want[1])]) ui.preset.value = decodeURIComponent(want[1]);
  ui.code.querySelector('pre').textContent = source(ui.preset.value);

  function script(src) {
    return new Promise((ok, fail) => {
      const s = document.createElement('script');
      s.src = src; s.onload = ok; s.onerror = () => fail(new Error('could not load ' + src));
      document.head.appendChild(s);
    });
  }
  function status(msg) { ui.status.textContent = msg; }

  let py, buildPy, editor, kernel = null, head = 0, frameMs = 0, paused = false;
  let fullUpload = true, mark = null;

  ui.start.addEventListener('click', async () => {
    ui.start.disabled = true;
    root.classList.add('loading');
    try {
      const css = document.createElement('link');
      css.rel = 'stylesheet'; css.href = CM_CSS; document.head.appendChild(css);
      status('Loading three.js and CodeMirror…');
      await Promise.all(LIBS.map(script));
      await script(CM_MODE);
      status('Loading Pyodide (CPython in WebAssembly)…');
      await script(PYODIDE + 'pyodide.js');
      py = await loadPyodide({ indexURL: PYODIDE });
      py.setStdout({ batched: s => { ui.out.textContent = s; } });
      status('Loading cimo (one 1.1 MB Python file)…');
      const text = await (await fetch('cimo.py')).text();
      py.FS.writeFile('/home/pyodide/cimo.py', text);
      py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
      status('Importing cimo…');
      await py.runPythonAsync(PY);
      buildPy = py.globals.get('build');
      setupEditor();
      setupView();
      status('Compiling the C (first time: also the parts of the C library it uses)…');
      await new Promise(r => setTimeout(r, 30));
      if (compile(true)) seed();
      const warm = /[?&]warm=(\d+)/.exec(location.search);  // (for screenshots)
      for (let f = 0; warm && f < +warm[1]; f++) {
        kernel.step(N, T, head, BigInt(py.globals.get('RING')));
        head = (head + 1) % T;
      }
      if (warm) { resetIndex(); fullUpload = true; }
      root.classList.remove('loading');
      root.classList.add('running');
      requestAnimationFrame(frame);
    } catch (e) {
      status('Could not start the demo: ' + (e && e.message || e));
      console.error(e);
      ui.start.disabled = false;
      root.classList.remove('loading');
    }
  });

  if (/[?&]autostart\b/.test(location.search)) ui.start.click();

  function setupEditor() {
    ui.code.innerHTML = '';
    editor = CodeMirror(ui.code, {
      value: source(ui.preset.value), mode: 'text/x-csrc', theme: 'cimo', lineNumbers: true,
      indentUnit: 4, viewportMargin: Infinity, scrollbarStyle: 'null',
    });
    let timer = null;
    editor.on('change', () => {
      clearTimeout(timer);
      timer = setTimeout(() => compile(false), 250);
    });
    ui.preset.addEventListener('change', () => {
      editor.setValue(source(ui.preset.value));
      clearTimeout(timer);
      if (compile(true)) seed();
    });
    ui.reseed.addEventListener('click', seed);
  }

  function compile(first) {
    const proxy = buildPy(editor.getValue());
    const r = proxy.toJs({ dict_converter: Object.fromEntries });
    proxy.destroy();
    if (mark) { editor.removeLineClass(mark, 'background', 'cm-err-line'); mark = null; }
    if (!r.ok) {
      const m = /^[^:]*:(\d+):(\d+): (.*)$/m.exec(r.error);
      ui.err.textContent = m ? `line ${m[1]}: ${m[3]}` : r.error;
      ui.err.hidden = false;
      if (m) mark = editor.addLineClass(+m[1] - 1, 'background', 'cm-err-line');
      return false;
    }
    ui.err.hidden = true;
    kernel = r;
    paused = false;
    ui.stats.dataset.compile = `compiled in ${r.ms.toFixed(0)} ms`;
    return true;
  }

  // ---- three.js view

  let renderer, scene, camera, group, lines, points, posAttr, idxAttr, mat;
  let yaw = 0.6, pitch = 0.35, dist = 3.6, auto = true;
  let center = new Float32Array(3), scale = 1, fitTimer = 0, fitReady = false;

  function heapView() {
    const heap = py._module.HEAPF32;
    const ring = py.globals.get('RING');
    return heap.subarray(ring >> 2, (ring >> 2) + N * T * 3);
  }

  function setupView() {
    const canvas = ui.view.querySelector('canvas');
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setClearColor(0x080b16);
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    group = new THREE.Group();
    scene.add(group);

    const geo = new THREE.BufferGeometry();
    posAttr = new THREE.BufferAttribute(heapView(), 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', posAttr);
    const slot = new Float32Array(N * T), hue = new Float32Array(N * T);
    for (let t = 0; t < T; t++)
      for (let i = 0; i < N; i++) { slot[t * N + i] = t; hue[t * N + i] = (i * 0.618034) % 1; }
    geo.setAttribute('slot', new THREE.BufferAttribute(slot, 1));
    geo.setAttribute('hue', new THREE.BufferAttribute(hue, 1));
    // segment s joins slot s and slot s-1 of every particle; the one joining the newest and
    // the oldest slot is collapsed each frame
    const idx = new Uint32Array(T * N * 2);
    for (let s = 0; s < T; s++)
      for (let i = 0; i < N; i++) {
        idx[(s * N + i) * 2] = s * N + i;
        idx[(s * N + i) * 2 + 1] = ((s + T - 1) % T) * N + i;
      }
    idxAttr = new THREE.BufferAttribute(idx, 1);
    idxAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setIndex(idxAttr);
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e9);

    mat = new THREE.ShaderMaterial({
      uniforms: { head: { value: 0 }, T: { value: T } },
      vertexShader: `
        attribute float slot; attribute float hue;
        uniform float head; uniform float T;
        varying vec3 vCol; varying float vA;
        void main() {
          float age = mod(head - slot + T, T) / (T - 1.0);
          vA = pow(1.0 - age, 1.6) * 0.32;
          vec3 a = vec3(0.39, 0.50, 1.0), b = vec3(0.13, 0.83, 0.93), c = vec3(0.75, 0.52, 0.98);
          vCol = hue < 0.5 ? mix(a, b, hue * 2.0) : mix(b, c, hue * 2.0 - 1.0);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: `
        varying vec3 vCol; varying float vA;
        void main() { gl_FragColor = vec4(vCol * vA, 1.0); }`,
      blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
    });
    lines = new THREE.LineSegments(geo, mat);
    lines.frustumCulled = false;
    group.add(lines);
    const pgeo = new THREE.BufferGeometry();
    pgeo.setAttribute('position', posAttr);
    points = new THREE.Points(pgeo, new THREE.PointsMaterial({
      color: 0xdbe5ff, size: 1.6, sizeAttenuation: false, transparent: true, opacity: 0.85,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    points.frustumCulled = false;
    group.add(points);

    // orbit: drag to rotate, wheel or pinch to zoom; auto-rotates until touched
    let drag = null;
    canvas.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; auto = false; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', e => {
      if (!drag) return;
      yaw -= (e.clientX - drag[0]) * 0.006;
      pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - drag[1]) * 0.006));
      drag = [e.clientX, e.clientY];
    });
    canvas.addEventListener('pointerup', () => { drag = null; });
    canvas.addEventListener('wheel', e => { e.preventDefault(); dist = Math.max(1.2, Math.min(9, dist * Math.exp(e.deltaY * 0.001))); }, { passive: false });
    new ResizeObserver(resize).observe(ui.view);
    resize();
  }

  function resize() {
    const w = ui.view.clientWidth, h = ui.view.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function resetIndex() {
    const ix = idxAttr.array;
    for (let s = 0; s < T; s++)
      for (let i = 0; i < N; i++) ix[(s * N + i) * 2 + 1] = ((s + T - 1) % T) * N + i;
    const gap = (head + 1) % T;
    for (let i = 0; i < N; i++) ix[(gap * N + i) * 2 + 1] = ix[(gap * N + i) * 2];
    idxAttr.needsUpdate = true;
  }

  function seed() {
    if (!kernel) return;
    const sp = kernel.getsp();
    try { kernel.seed(N, T, BigInt(py.globals.get('RING'))); } catch (e) { kernel.setsp(sp); }
    head = 0;
    resetIndex();
    fullUpload = true;
    fitReady = false;
    fitTimer = 0;
  }

  // keep the attractor framed: robust bounds (5th..95th percentile) of a sample of particles
  function fit(arr, base, snap) {
    const k = 600, xs = [[], [], []];
    for (let j = 0; j < k; j++) {
      const i = (j * 26.7 | 0) % N, o = base + 3 * i;
      for (let d = 0; d < 3; d++) xs[d].push(arr[o + d]);
    }
    let r = 0;
    const c = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      const v = xs[d].filter(Number.isFinite).sort((a, b) => a - b);
      if (!v.length) return;
      const lo = v[Math.floor(v.length * 0.03)], hi = v[Math.floor(v.length * 0.97)];
      c[d] = (lo + hi) / 2;
      r = Math.max(r, (hi - lo) / 2);
    }
    const s = 1 / Math.max(r * 1.25, 1e-6), q = snap ? 1 : 0.08;
    for (let d = 0; d < 3; d++) center[d] += (c[d] - center[d]) * q;
    scale += (s - scale) * q;
  }

  let last = performance.now(), fpsAcc = 0, fpsN = 0, fps = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    if (!kernel || !renderer) return;
    const dt = now - last; last = now;
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 500) { fps = 1000 * fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; }
    let arr = posAttr.array;
    if (arr.buffer !== py._module.HEAPF32.buffer) {  // Pyodide's memory grew: a new view
      arr = heapView();
      posAttr.array = arr;
      fullUpload = true;
    }
    if (!paused) {
      const t0 = performance.now();
      const sp = kernel.getsp();
      let respawned = 0;
      try {
        respawned = kernel.step(N, T, head, BigInt(py.globals.get('RING')));
      } catch (e) {
        kernel.setsp(sp);
        paused = true;
        ui.err.textContent = 'runtime error: ' + String(e.message || e).split('\n')[0];
        ui.err.hidden = false;
      }
      frameMs = frameMs * 0.9 + (performance.now() - t0) * 0.1;
      if (!paused) {
        const prev = head;
        head = (head + 1) % T;
        if (respawned) fullUpload = true;
        // segment slot head+1 joins the newest and oldest slots: collapse it; restore the old one
        const ix = idxAttr.array, gap = (head + 1) % T, was = (prev + 1) % T;
        for (let i = 0; i < N; i++) {
          const g = (gap * N + i) * 2, w = (was * N + i) * 2;
          ix[g + 1] = ix[g];
          ix[w + 1] = ((was + T - 1) % T) * N + i;
        }
        idxAttr.needsUpdate = true;
        if (fullUpload) {
          posAttr.updateRange = { offset: 0, count: -1 };
          fullUpload = false;
        } else {
          posAttr.updateRange = { offset: head * N * 3, count: N * 3 };
        }
        posAttr.needsUpdate = true;
      }
    }
    if (--fitTimer <= 0) { fit(arr, head * N * 3, !fitReady); fitReady = true; fitTimer = 8; }
    mat.uniforms.head.value = head;
    points.geometry.setDrawRange(head * N, N);
    if (auto) yaw += dt * 0.00012;
    group.scale.setScalar(scale);
    group.position.set(-center[0] * scale, -center[1] * scale, -center[2] * scale);
    camera.position.set(dist * Math.cos(pitch) * Math.sin(yaw), dist * Math.sin(pitch), dist * Math.cos(pitch) * Math.cos(yaw));
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
    ui.stats.textContent = `${ui.stats.dataset.compile || ''} · ${N.toLocaleString()} particles · step ${frameMs.toFixed(1)} ms/frame in WebAssembly · ${fps.toFixed(0)} fps`;
  }
})();
