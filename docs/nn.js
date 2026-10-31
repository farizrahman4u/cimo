// The training demo: a network you draw -> JSON -> C (nn/codegen.py, in Pyodide) -> cimo ->
// WebAssembly -> trained here, with every stage on screen. Shares Pyodide with the banner demo
// (runtime.js). Each frame JS calls the compiled train/predict/acts directly on buffers in
// Pyodide's memory.
(function () {
  'use strict';
  const root = document.getElementById('nn');
  if (!root) return;
  const $ = s => root.querySelector(s);

  const N = 300;          // data points
  const G = 64;           // decision-boundary grid (G*G <= NMAX in codegen.py)
  const TH = 20;          // grid of each neuron thumbnail
  const LIM = 1.25;       // the plane shown: [-LIM, LIM]^2
  const STEPS = 3000, PER_FRAME = 20;

  // ---- datasets: two classes in the plane
  function rng(seed) { let s = seed >>> 0 || 1; return () => ((s = Math.imul(s ^ (s >>> 15), 2246822507) + 0x6D2B79F5 >>> 0) / 4294967296); }
  function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()); }
  const DATA = {
    'Two spirals': r => { const P = []; for (let i = 0; i < N; i++) { const c = i % 2, t = (i >> 1) / (N / 2) * 2.8 * Math.PI + 0.4, a = t + c * Math.PI, d = t / (3.2 * Math.PI); P.push([d * Math.cos(a) + gauss(r) * .025, d * Math.sin(a) + gauss(r) * .025, c]); } return P; },
    'Circles': r => { const P = []; for (let i = 0; i < N; i++) { const c = i % 2, a = r() * 2 * Math.PI, d = (c ? .75 : .32) + gauss(r) * .07; P.push([d * Math.cos(a), d * Math.sin(a), c]); } return P; },
    'Moons': r => { const P = []; for (let i = 0; i < N; i++) { const c = i % 2, t = r() * Math.PI; P.push(c ? [.9 * Math.cos(t) - .45 + gauss(r) * .06, .9 * Math.sin(t) - .25 + gauss(r) * .06, 0] : [.9 * Math.cos(t + Math.PI) + .45 + gauss(r) * .06, .9 * Math.sin(t + Math.PI) + .25 + gauss(r) * .06, 1]); } return P; },
    'XOR': r => { const P = []; for (let i = 0; i < N; i++) { const x = r() * 2 - 1, y = r() * 2 - 1; P.push([x, y, (x > 0) !== (y > 0) ? 1 : 0]); } return P; },
  };

  // ---- graphs
  const OPS = {
    input: { label: 'Input', ins: 0 }, output: { label: 'Output', ins: 1 },
    dense: { label: 'Dense', ins: 1 }, relu: { label: 'ReLU', ins: 1 }, tanh: { label: 'Tanh', ins: 1 },
    sigmoid: { label: 'Sigmoid', ins: 1 }, sin: { label: 'Sin', ins: 1 }, square: { label: 'Square', ins: 1 },
    add: { label: 'Add', ins: 2 }, concat: { label: 'Concat', ins: 2 },
  };
  const PALETTE = ['dense', 'tanh', 'relu', 'sin', 'sigmoid', 'square', 'add', 'concat'];
  // [id, op, x, y, inputs, units]: positions on a 0..1 by 0..1 board
  const PRESETS = {
    'Tanh MLP': [['in', 'input', .015, .45, []], ['d1', 'dense', .155, .45, ['in'], 8], ['t1', 'tanh', .295, .45, ['d1']], ['d2', 'dense', .435, .45, ['t1'], 8], ['t2', 'tanh', .575, .45, ['d2']], ['d3', 'dense', .715, .45, ['t2'], 1], ['out', 'output', .86, .45, ['d3']]],
    'Sine features': [['in', 'input', .015, .45, []], ['d1', 'dense', .155, .45, ['in'], 12], ['s1', 'sin', .295, .45, ['d1']], ['d2', 'dense', .435, .45, ['s1'], 8], ['t2', 'tanh', .575, .45, ['d2']], ['d3', 'dense', .715, .45, ['t2'], 1], ['out', 'output', .86, .45, ['d3']]],
    'Skip connection': [['in', 'input', .015, .45, []], ['d1', 'dense', .145, .25, ['in'], 8], ['t1', 'tanh', .285, .25, ['d1']], ['d2', 'dense', .425, .25, ['t1'], 8], ['t2', 'tanh', .565, .25, ['d2']], ['a', 'add', .6, .7, ['t1', 't2']], ['d3', 'dense', .73, .7, ['a'], 1], ['out', 'output', .86, .7, ['d3']]],
    'Linear (too simple)': [['in', 'input', .015, .45, []], ['d1', 'dense', .43, .45, ['in'], 1], ['out', 'output', .86, .45, ['d1']]],
  };

  let graph = [];   // [{id, op, x, y, inputs: [id|null], units}]
  let uid = 0;
  function loadPreset(name) {
    graph = PRESETS[name].map(([id, op, x, y, inputs, units]) => ({ id, op, x, y, inputs: inputs.slice(), units }));
    uid = 0;
    // a phone-width board fits four nodes across: wrap the chain onto two rows, left to right
    if (board.clientWidth && board.clientWidth < 560) {
      const s = [...graph].sort((a, b) => a.x - b.x), rows = s.length > 4 ? 2 : 1, per = Math.ceil(s.length / rows);
      const span = 1 - 84 / board.clientWidth;
      s.forEach((n, k) => {
        const r = Math.floor(k / per), c = k % per;
        n.x = .015 + c * (span - .015) / Math.max(1, per - 1);
        n.y = rows === 1 ? .45 : [.29, .73][r];
      });
    }
  }

  // ---- DOM
  const board = $('.nn-board'), wires = $('.nn-wires'), boundary = $('.nn-boundary'), lossCv = $('.nn-loss');
  const ui = {
    start: $('.nn-start'), status: $('.nn-status'), data: $('.nn-data'), preset: $('.nn-preset'),
    lr: $('.nn-lr'), retrain: $('.nn-retrain'), step: $('.nn-step'), lossT: $('.nn-loss-t'),
    acc: $('.nn-acc'), err: $('.nn-err'), panel: $('.nn-panel pre'), palette: $('.nn-palette'),
    form: $('.nn-infer'), q: $('.nn-q'), qx: $('.nn-qx'), qerr: $('.nn-qerr'), res: $('.nn-res'), resBox: $('.nn-res-box'), resT: $('.nn-res-t'),
  };
  ui.data.innerHTML = Object.keys(DATA).map(n => `<option>${n}</option>`).join('');
  ui.preset.innerHTML = Object.keys(PRESETS).map(n => `<option>${n}</option>`).join('');
  // small pictures of each op (16x16, stroked): what the layer computes
  const ICONS = {
    dense: '<circle cx="3" cy="4" r="1.3"/><circle cx="3" cy="12" r="1.3"/><circle cx="13" cy="3" r="1.3"/><circle cx="13" cy="8" r="1.3"/><circle cx="13" cy="13" r="1.3"/><path d="M4.3 4L11.7 3M4.3 4L11.7 8M4.3 4L11.7 13M4.3 12L11.7 3M4.3 12L11.7 8M4.3 12L11.7 13" stroke-width=".9"/>',
    tanh: '<path d="M1 8h14M8 1v14" stroke-width=".7" opacity=".45"/><path d="M1 13.5C5 13.5 5.5 13 8 8s3-5.5 7-5.5"/>',
    relu: '<path d="M1 12h14M8 1v14" stroke-width=".7" opacity=".45"/><path d="M1 12h7l7-9"/>',
    sin: '<path d="M1 8h14" stroke-width=".7" opacity=".45"/><path d="M1 8c2-6 4.5-6 6.5 0s4.5 6 6.5 0"/>',
    sigmoid: '<path d="M1 14h14" stroke-width=".7" opacity=".45"/><path d="M1 13.5C6 13.5 6.5 3 8 3h0c-1.5 0 2 0 7 0" stroke-width="0"/><path d="M1 13.5c4.5 0 4.5-.5 7-5.5S11 2.5 15 2.5"/><path d="M1 2.5h14" stroke-width=".7" stroke-dasharray="1.5 1.5" opacity=".45"/>',
    square: '<path d="M1 14h14M8 1v14" stroke-width=".7" opacity=".45"/><path d="M2 2.5C4 11 6 14 8 14s4-3 6-11.5"/>',
    add: '<circle cx="8" cy="8" r="6"/><path d="M8 5v6M5 8h6"/>',
    concat: '<rect x="1.5" y="4" width="5" height="8" rx="1"/><rect x="9.5" y="4" width="5" height="8" rx="1"/><path d="M6.5 8h3" />',
  };
  const icon = op => `<svg viewBox="0 0 16 16" aria-hidden="true">${ICONS[op]}</svg>`;
  ui.palette.innerHTML = '<span class="nn-add">add</span>' + PALETTE.map(op => `<button type="button" data-op="${op}" title="add a ${OPS[op].label} node">${icon(op)}${OPS[op].label}</button>`).join('');
  loadPreset(ui.preset.value);

  let widths = {}, usedSet = new Set(), thumbs = {};
  function render() {
    board.querySelectorAll('.nn-node').forEach(e => e.remove());
    thumbs = {};
    for (const n of graph) {
      const el = document.createElement('div');
      el.className = 'nn-node op-' + n.op + (usedSet.size && !usedSet.has(n.id) ? ' unused' : '');
      el.dataset.id = n.id;
      el.style.left = (n.x * 100) + '%';
      el.style.top = (n.y * 100) + '%';
      const fixed = n.op === 'input' || n.op === 'output';
      let body = '';
      if (n.op === 'dense') body += `<div class="nn-units"><button type="button" data-d="-1" aria-label="fewer units">−</button><b>${n.units}</b><button type="button" data-d="1" aria-label="more units">+</button></div>`;
      if (!fixed) body += '<canvas class="nn-th" width="1" height="1"></canvas><div class="nn-vals"></div>';
      if (n.op === 'input') body += '<div class="nn-note">x, y</div>';
      if (n.op === 'output') body += '<div class="nn-note">score</div>';
      el.innerHTML = `<div class="nn-h">${OPS[n.op].label}${fixed ? '' : '<button type="button" class="nn-x" aria-label="remove">×</button>'}</div>${body}` +
        (OPS[n.op].ins ? Array.from({ length: OPS[n.op].ins }, (_, k) => `<i class="nn-in" data-k="${k}" style="top:${OPS[n.op].ins === 2 ? 34 + k * 40 : 50}%"></i>`).join('') : '') +
        (n.op !== 'output' ? '<i class="nn-out"></i>' : '');
      board.appendChild(el);
      if (!fixed) thumbs[n.id] = el.querySelector('.nn-th');
    }
    drawWires();
  }

  function portPos(el, sel) {
    const p = el.querySelector(sel).getBoundingClientRect(), b = board.getBoundingClientRect();
    return [p.left + p.width / 2 - b.left, p.top + p.height / 2 - b.top];
  }
  function curve(a, b) {
    const dx = Math.max(30, Math.abs(b[0] - a[0]) / 2);
    return `M${a[0]},${a[1]} C${a[0] + dx},${a[1]} ${b[0] - dx},${b[1]} ${b[0]},${b[1]}`;
  }
  let temp = null, flowing = new Set();
  function drawWires() {
    let s = '';
    const node = id => board.querySelector(`.nn-node[data-id="${id}"]`);
    for (const n of graph) n.inputs.forEach((src, k) => {
      if (!src || !node(src)) return;
      const a = portPos(node(src), '.nn-out'), b = portPos(node(n.id), `.nn-in[data-k="${k}"]`);
      const d = curve(a, b), w = widths[src];
      s += `<path class="w${flowing.has(n.id) ? ' flow' : ''}" data-to="${n.id}" d="${d}"/><path class="hit" d="${d}" data-to="${n.id}" data-k="${k}"><title>click to disconnect</title></path>`;
      if (w) s += `<text x="${(a[0] + b[0]) / 2}" y="${Math.min(a[1], b[1]) - 9}">${w}</text>`;
    });
    if (temp) s += `<path class="w temp" d="${curve(temp.a, temp.b)}"/>`;
    wires.innerHTML = s;
  }

  // dragging nodes, drawing wires, editing
  board.addEventListener('pointerdown', e => {
    const el = e.target.closest('.nn-node');
    if (e.target.closest('button')) return;
    if (e.target.classList.contains('nn-out') && el) {
      temp = { from: el.dataset.id, a: portPos(el, '.nn-out'), b: portPos(el, '.nn-out') };
      board.setPointerCapture(e.pointerId);
      e.preventDefault();
      return;
    }
    if (el && e.target.closest('.nn-h')) {
      const n = graph.find(q => q.id === el.dataset.id), b = board.getBoundingClientRect();
      const ox = e.clientX - (n.x * b.width), oy = e.clientY - (n.y * b.height);
      const move = ev => {
        n.x = Math.max(0, Math.min(1 - el.offsetWidth / b.width, (ev.clientX - ox) / b.width));
        n.y = Math.max(0, Math.min(.86, (ev.clientY - oy) / b.height));
        el.style.left = (n.x * 100) + '%'; el.style.top = (n.y * 100) + '%';
        drawWires();
      };
      const up = () => { removeEventListener('pointermove', move); removeEventListener('pointerup', up); };
      addEventListener('pointermove', move); addEventListener('pointerup', up);
      e.preventDefault();
    }
  });
  board.addEventListener('pointermove', e => {
    if (!temp) return;
    const b = board.getBoundingClientRect();
    temp.b = [e.clientX - b.left, e.clientY - b.top];
    drawWires();
  });
  board.addEventListener('pointerup', e => {
    if (!temp) return;
    const t = document.elementFromPoint(e.clientX, e.clientY), from = temp.from;
    temp = null;
    if (t && t.classList.contains('nn-in')) {
      const to = graph.find(q => q.id === t.closest('.nn-node').dataset.id);
      if (to.id !== from) { to.inputs[+t.dataset.k] = from; changed(); return; }
    }
    drawWires();
  });
  board.addEventListener('click', e => {
    const hit = e.target.closest('.hit');
    if (hit) { graph.find(q => q.id === hit.dataset.to).inputs[+hit.dataset.k] = null; changed(); return; }
    const el = e.target.closest('.nn-node');
    if (!el) return;
    const n = graph.find(q => q.id === el.dataset.id);
    if (e.target.classList.contains('nn-x')) {
      graph = graph.filter(q => q !== n);
      graph.forEach(q => { q.inputs = q.inputs.map(s => s === n.id ? null : s); });
      changed();
    } else if (e.target.dataset.d) {
      n.units = Math.max(1, Math.min(32, n.units + +e.target.dataset.d));
      changed();
    }
  });
  ui.palette.addEventListener('click', e => {
    const op = e.target.dataset && e.target.dataset.op;
    if (!op) return;
    const n = { id: 'n' + (++uid), op, x: .4 + (uid % 5) * .03, y: .74 - (uid % 3) * .05, inputs: Array(OPS[op].ins).fill(null) };
    if (op === 'dense') n.units = 8;
    graph.push(n);
    changed();
  });
  ui.preset.addEventListener('change', () => { loadPreset(ui.preset.value); changed(); });
  ui.data.addEventListener('change', () => { setData(); restart(); });
  ui.lr.addEventListener('change', restart);
  ui.retrain.addEventListener('click', () => { seed++; restart(); });
  addEventListener('resize', drawWires);
  new ResizeObserver(drawWires).observe(board);

  // ---- stages
  const stage = { json: '', c: '', wasm: '' };
  let openStage = (/[?&]nnstage=(\w+)/.exec(location.search) || [0, 'c'])[1];  // (?nnstage=: for screenshots)
  root.querySelectorAll('.nn-chip[data-s]').forEach(ch => ch.addEventListener('click', () => {
    openStage = openStage === ch.dataset.s ? null : ch.dataset.s;
    showStage();
  }));
  const MAXLINES = 1500;  // a disassembly runs to ~10k lines: show the start, quickly
  const html = {};       // highlighted HTML per stage, built once per build
  function showStage() {
    root.querySelectorAll('.nn-chip[data-s]').forEach(ch => ch.classList.toggle('open', ch.dataset.s === openStage));
    const want = openStage;
    if (want === 'wasm' && !stage.wasm && net) {  // switch now, disassemble on the next tick
      ui.panel.className = 'hl-wasm';
      ui.panel.textContent = ';; disassembling the module…';
      root.classList.add('panel-open');
      setTimeout(() => {
        try { stage.wasm = ns.get('nn_wat')(); } catch (e) { stage.wasm = ';; could not disassemble: ' + e.message; }
        if (openStage === want) showStage();
      }, 30);
      return;
    }
    const text = want ? stage[want] : '';
    if (text && !html[want]) {
      const lines = text.split('\n');
      const cut = lines.length > MAXLINES;
      html[want] = numbered(HL[want](cut ? lines.slice(0, MAXLINES).join('\n') : text)) +
        (cut ? `<span class="more t-cm">${want === 'wasm' ? ';;' : '/*'} … ${(lines.length - MAXLINES).toLocaleString()} more lines${want === 'wasm' ? '' : ' */'}</span>` : '');
    }
    ui.panel.className = 'hl-' + (want || '') + ' numbered';
    ui.panel.innerHTML = text ? html[want] : '';
    ui.panel.scrollTop = 0;
    root.classList.toggle('panel-open', !!text);
  }

  // one block per source line (CSS numbers them); a highlight span that crosses a line break
  // (a multi-line comment) is closed at the end of the line and reopened on the next
  function numbered(h) {
    let open = null;
    return h.replace(/\n$/, '').split('\n').map(line => {
      const head = open ? `<span class="${open}">` : '';
      const re = /<span class="([^"]+)">|<\/span>/g;
      let m;
      while ((m = re.exec(line))) open = m[1] || null;
      return `<span class="ln">${head}${line}${open ? '</span>' : ''}</span>`;
    }).join('');
  }

  // ---- syntax highlighting (JSON, C, WebAssembly text)
  const esc = t => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  function tokens(text, re, cls) {
    let out = '', last = 0;
    text.replace(re, (m, ...g) => {
      const at = g[g.length - 2];
      out += esc(text.slice(last, at));
      const k = g.findIndex((x, i) => i < cls.length && x !== undefined);
      out += k < 0 || !cls[k] ? esc(m) : `<span class="t-${cls[k]}">${esc(m)}</span>`;
      last = at + m.length;
      return m;
    });
    return out + esc(text.slice(last));
  }
  const HL = {
    json: t => tokens(t, /("(?:[^"\\]|\\.)*")(?=\s*:)|("(?:[^"\\]|\\.)*")|(-?\b\d+(?:\.\d+)?\b)|\b(true|false|null)\b/g, ['fn', 'str', 'num', 'kw']),
    c: t => tokens(t, /(\/\*[\s\S]*?\*\/|\/\/[^\n]*)|(^[ \t]*#[^\n]*)|\b(for|if|else|return|while|static|const|unsigned)\b|\b(void|int|float|double|char|long)\b|(\b\d+(?:\.\d+)?(?:e[-+]?\d+)?[fu]?\b)|\b([A-Za-z_]\w*)(?=\s*\()/gm, ['cm', 'pp', 'kw', 'ty', 'num', 'fn']),
    wasm: t => tokens(t, /(;;[^\n]*)|\b((?:i32|i64|f32|f64)\.[a-z0-9_]+|local\.(?:get|set|tee)|global\.(?:get|set)|memory\.\w+)\b|\b(func|param|result|local|export|type|offset)\b|(\$[\w.]+)|("[^"]*")|\b(block|loop|if|else|end|br_if|br_table|br|return|call_indirect|call|unreachable|drop|select)\b|(-?\b\d+(?:\.\d+)?(?:e[-+]?\d+)?\b)/g, ['cm', 'op', 'ty', 'fn', 'str', 'kw', 'num']),
  };
  function chip(s, text) { const el = root.querySelector(`.nn-chip[data-n="${s}"] small`); if (el) el.textContent = text; }

  function toJSON() {
    return JSON.stringify({ nodes: graph.map(n => Object.assign({ id: n.id, op: n.op }, n.op === 'dense' ? { units: n.units } : {}, OPS[n.op].ins ? { inputs: n.inputs } : {})) });
  }

  // ---- Python side
  const PY = `
import json, time, ctypes, struct
import cimo, codegen, wat
import cimo.wasm as _cw
F = ctypes.c_float
N, G, TH = ${N}, ${G}, ${TH}
X = (F * (N * 2))(); Y = (F * N)(); PX = (F * N)(); QX = (F * 2)(); QO = (F * 1)()
GX = (F * (G * G * 2))(); GO = (F * (G * G))(); TX = (F * (TH * TH * 2))()
ADDR = {k: ctypes.addressof(v) for k, v in dict(X=X, Y=Y, PX=PX, GX=GX, GO=GO, TX=TX, QX=QX, QO=QO).items()}
_live = {}

def nn_wat():
    """The network's own functions as WebAssembly text, disassembled from the built module."""
    mod = _live['mod']
    text, sec, others = wat.disassemble(mod, {'init', 'train', 'predict', 'acts', 'views',
                                              'forward', 'backward'})
    head = [';; WebAssembly text of the network, disassembled from the %.1f KB module cimo built' % (len(mod) / 1024),
            ';; sections: ' + ' \u00b7 '.join('%s %d B' % (n.replace('custom ', ''), z) for n, z in sec),
            ';; also in the module: %d runtime functions (%s)' % (len(others), ', '.join(o for o in others if not o.startswith('__'))[:160]),
            '']
    return '\\n'.join(head) + text

def nn_build(graph_json):
    t0 = time.perf_counter()
    pretty = json.dumps(json.loads(graph_json), indent=1)
    try:
        r = codegen.generate(graph_json)
    except codegen.GraphError as e:
        return {'ok': False, 'stage': 'graph', 'error': str(e), 'json': pretty}
    t1 = time.perf_counter()
    try:
        m = cimo.compile(r['c'])
        mod, _ = _cw.build(r['c'])
    except cimo.CompileError as e:
        return {'ok': False, 'stage': 'c', 'error': str(e), 'json': pretty, 'c': r['c']}
    t2 = time.perf_counter()
    np_ = max(r['nparams'], 1)
    na = sum(a['width'] for a in r['acts'])
    bufs = dict(P=(F * np_)(), M=(F * np_)(), V=(F * np_)(), WS=(F * r['ws'])(), AO=(F * max(1, na * TH * TH))(), QA=(F * max(1, na))())
    _live.clear(); _live.update(bufs); _live['m'] = m
    _live['mod'] = mod
    e = m.exports
    return {'ok': True, 'json': pretty, 'c': r['c'], 'nparams': r['nparams'],
            'acts': r['acts'], 'widths': r['widths'], 'used': r['used'],
            'gen_ms': (t1 - t0) * 1000, 'cc_ms': (t2 - t1) * 1000, 'size': len(mod),
            'addr': dict(ADDR, **{k: ctypes.addressof(v) for k, v in bufs.items()}),
            'init': m.init.function, 'train': m.train.function, 'predict': m.predict.function,
            'acts_fn': m.acts.function,
            'getsp': getattr(e, '__cimo_get_sp'), 'setsp': getattr(e, '__cimo_set_sp')}
`;

  let py = null, ns = null, buildPy = null, net = null, seed = 1, step = 0, t = 0, running = false, losses = [];
  function heap() { return py._module.HEAPF32; }
  function put(addr, arr) { heap().set(arr, addr >> 2); }

  let points = [];
  function setData() {
    points = DATA[ui.data.value](rng(7 + ui.data.selectedIndex));
    if (!py) { drawBoundary(); return; }
    const A = ns.get('ADDR').toJs();
    const X = new Float32Array(N * 2), Y = new Float32Array(N);
    points.forEach((p, i) => { X[2 * i] = p[0]; X[2 * i + 1] = p[1]; Y[i] = p[2]; });
    put(A.get('X'), X); put(A.get('Y'), Y);
    const g = new Float32Array(G * G * 2), th = new Float32Array(TH * TH * 2);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { g[2 * (j * G + i)] = -LIM + (i + .5) / G * 2 * LIM; g[2 * (j * G + i) + 1] = LIM - (j + .5) / G * 2 * LIM; }
    for (let j = 0; j < TH; j++) for (let i = 0; i < TH; i++) { th[2 * (j * TH + i)] = -LIM + (i + .5) / TH * 2 * LIM; th[2 * (j * TH + i) + 1] = LIM - (j + .5) / TH * 2 * LIM; }
    put(A.get('GX'), g); put(A.get('TX'), th);
  }

  let timer = null;
  function changed() {
    render();
    if (!py) return;
    clearTimeout(timer);
    timer = setTimeout(build, 250);
  }

  function build() {
    const json = toJSON();
    const proxy = buildPy(json);
    const r = proxy.toJs({ dict_converter: Object.fromEntries });
    proxy.destroy();
    stage.json = r.json || json;
    delete html.json;
    chip('graph', `${graph.length} nodes`);
    chip('json', `${(json.length / 1024).toFixed(1)} KB`);
    if (!r.ok) {
      ui.err.textContent = (r.stage === 'graph' ? 'graph: ' : 'C: ') + r.error;
      ui.err.hidden = false;
      root.classList.add('broken');
      if (r.c) { stage.c = r.c; delete html.c; }
      showStage();
      return;
    }
    ui.err.hidden = true;
    root.classList.remove('broken');
    stage.c = r.c; stage.wasm = '';
    for (const k in html) delete html[k];
    widths = r.widths instanceof Map ? Object.fromEntries(r.widths) : r.widths;
    usedSet = new Set(r.used);
    chip('graph', `${r.nparams} params`);
    chip('c', `${r.c.split('\n').length} lines`);
    chip('wasm', `${(r.size / 1024).toFixed(1)} KB · ${r.cc_ms.toFixed(0)} ms`);
    net = r;
    net.addr = r.addr instanceof Map ? Object.fromEntries(r.addr) : r.addr;
    render();
    showStage();
    restart();
    if (query) infer(false);
  }

  function call(f, ...args) {  // pointer arguments (BigInt) are marked by the caller
    const sp = net.getsp();
    try { return f(...args); } catch (e) { net.setsp(sp); throw e; }
  }

  function restart() {
    if (!net) return;
    const A = net.addr;
    heap().fill(0, A.M >> 2, (A.M >> 2) + Math.max(1, net.nparams));
    heap().fill(0, A.V >> 2, (A.V >> 2) + Math.max(1, net.nparams));
    call(net.init, seed, BigInt(A.P));
    step = 0; t = 0; losses = [];
    running = true;
  }

  // ---- drawing
  const C0 = [63, 92, 246], C1 = [245, 158, 11];   // class 0 blue, class 1 amber
  function drawBoundary(prob) {
    const ctx = boundary.getContext('2d'), W = boundary.width;
    if (prob) {
      const img = ctx.createImageData(G, G);
      for (let k = 0; k < G * G; k++) {
        const p = prob[k], c = p > .5 ? C1 : C0, a = Math.min(1, Math.abs(p - .5) * 2);
        img.data[4 * k] = 255 + (c[0] - 255) * a * .55;
        img.data[4 * k + 1] = 255 + (c[1] - 255) * a * .55;
        img.data[4 * k + 2] = 255 + (c[2] - 255) * a * .55;
        img.data[4 * k + 3] = 255;
      }
      const off = document.createElement('canvas'); off.width = off.height = G;
      off.getContext('2d').putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, W, W);
    } else {
      ctx.fillStyle = '#f5f6fa'; ctx.fillRect(0, 0, W, W);
    }
    // the plot's coordinates: a light grid every 0.5, axes through 0, ticks along two edges
    const at = v => (v + LIM) / (2 * LIM) * W;
    ctx.save();
    ctx.font = `${Math.round(W / 40)}px "DM Mono", ui-monospace, monospace`;
    for (const v of [-1, -0.5, 0, 0.5, 1]) {
      ctx.strokeStyle = v === 0 ? 'rgba(15,23,42,.28)' : 'rgba(15,23,42,.1)';
      ctx.lineWidth = v === 0 ? 1.2 : 1;
      ctx.beginPath(); ctx.moveTo(at(v), 0); ctx.lineTo(at(v), W); ctx.moveTo(0, W - at(v)); ctx.lineTo(W, W - at(v)); ctx.stroke();
      ctx.fillStyle = 'rgba(15,23,42,.5)';
      const t = String(v);
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText(t, at(v), W - 4);
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(t, 5, W - at(v));
    }
    ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.fillStyle = 'rgba(15,23,42,.38)';
    ctx.fillText('x', W - 6, W - 4);
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('y', 5, 4);
    ctx.restore();
    for (const [x, y, c] of points) {
      const px = (x + LIM) / (2 * LIM) * W, py_ = (LIM - y) / (2 * LIM) * W;
      ctx.beginPath(); ctx.arc(px, py_, W / 110, 0, 7);
      ctx.fillStyle = c ? 'rgb(217,119,6)' : 'rgb(39,55,189)';
      ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = '#fff'; ctx.stroke();
    }
    if (query) {  // the inference point: a red laser dot
      const qx = (query[0] + LIM) / (2 * LIM) * W, qy = (LIM - query[1]) / (2 * LIM) * W;
      const g = ctx.createRadialGradient(qx, qy, 0, qx, qy, W / 22);
      g.addColorStop(0, 'rgba(239,68,68,.55)'); g.addColorStop(1, 'rgba(239,68,68,0)');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(qx, qy, W / 22, 0, 7); ctx.fill();
      ctx.beginPath(); ctx.arc(qx, qy, W / 85, 0, 7);
      ctx.fillStyle = '#ef4444'; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = '#fff'; ctx.stroke();
    }
  }
  function drawLoss() {
    const ctx = lossCv.getContext('2d'), W = lossCv.width, H = lossCv.height;
    ctx.clearRect(0, 0, W, H);
    if (losses.length < 2) return;
    const mx = Math.max(...losses.slice(0, 5), .1);
    ctx.beginPath();
    losses.forEach((l, i) => { const x = i / (STEPS / PER_FRAME) * W, y = H - 2 - Math.min(1, l / mx) * (H - 4); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.strokeStyle = '#3f5cf6'; ctx.lineWidth = 1.5; ctx.stroke();
  }
  function drawThumbs() {
    const A = net.addr, h = heap(), n = TH * TH;
    call(net.acts_fn, n, BigInt(A.TX), BigInt(A.P), BigInt(A.WS), BigInt(A.AO));
    for (const a of net.acts) {
      const cv = thumbs[a.id];
      if (!cv) continue;
      const show = Math.min(a.width, 8), cols = Math.min(show, 4), rows = Math.ceil(show / cols);
      if (cv.width !== cols * (TH + 1) - 1) { cv.width = cols * (TH + 1) - 1; cv.height = rows * (TH + 1) - 1; }
      const ctx = cv.getContext('2d'), img = ctx.createImageData(cv.width, cv.height);
      const base = (A.AO >> 2) + a.offset * n;
      for (let u = 0; u < show; u++) {
        let lo = Infinity, hi = -Infinity;
        for (let k = 0; k < n; k++) { const v = h[base + k * a.width + u]; if (v < lo) lo = v; if (v > hi) hi = v; }
        const ox = (u % cols) * (TH + 1), oy = Math.floor(u / cols) * (TH + 1), sc = hi - lo > 1e-6 ? 1 / (hi - lo) : 0;
        for (let k = 0; k < n; k++) {
          const v = (h[base + k * a.width + u] - lo) * sc, c = v > .5 ? C1 : C0, s = Math.abs(v - .5) * 2;
          const q = 4 * ((oy + Math.floor(k / TH)) * cv.width + ox + k % TH);
          img.data[q] = 255 + (c[0] - 255) * s; img.data[q + 1] = 255 + (c[1] - 255) * s; img.data[q + 2] = 255 + (c[2] - 255) * s; img.data[q + 3] = 255;
        }
      }
      if (query) {  // where the point falls in each unit's map
        const ix = Math.min(TH - 1, Math.max(0, Math.floor((query[0] + LIM) / (2 * LIM) * TH)));
        const iy = Math.min(TH - 1, Math.max(0, Math.floor((LIM - query[1]) / (2 * LIM) * TH)));
        for (let u = 0; u < show; u++) {
          const q = 4 * ((Math.floor(u / cols) * (TH + 1) + iy) * cv.width + (u % cols) * (TH + 1) + ix);
          img.data[q] = 239; img.data[q + 1] = 68; img.data[q + 2] = 68; img.data[q + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
    }
  }

  // ---- inference on one point
  let query = null;
  function parsePoint(text) {
    const parts = text.split(',').map(t => t.trim());
    if (parts.length !== 2 || parts.some(t => t === ''))
      throw new Error('Enter two numbers separated by a comma, e.g. 0.3, -0.5');
    const v = parts.map(t => (/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(t) ? Number(t) : NaN));
    const bad = parts.find((t, i) => !Number.isFinite(v[i]));
    if (bad !== undefined) throw new Error(`"${bad}" is not a number`);
    if (v.some(x => Math.abs(x) > LIM)) throw new Error(`x and y must be between -${LIM} and ${LIM} (the plot's range)`);
    return v;
  }
  function infer(animate) {
    if (!net || !query) return;
    const A = net.addr, h = heap();
    h[A.QX >> 2] = query[0]; h[(A.QX >> 2) + 1] = query[1];
    call(net.predict, 1, BigInt(A.QX), BigInt(A.P), BigInt(A.WS), BigInt(A.QO));
    const p = h[A.QO >> 2], c = p > .5 ? C1 : C0, a = Math.min(1, Math.abs(p - .5) * 2);
    ui.resBox.style.background = `rgb(${c.map(v => Math.round(255 + (v - 255) * (.25 + .75 * a))).join(',')})`;
    ui.resT.innerHTML = `${p > .5 ? 'amber' : 'blue'} class, <b>${(100 * Math.max(p, 1 - p)).toFixed(1)}%</b> <span style="color:var(--ink4)">(p = ${p.toFixed(3)})</span>`;
    ui.res.hidden = false;
    call(net.acts_fn, 1, BigInt(A.QX), BigInt(A.P), BigInt(A.WS), BigInt(A.QA));
    for (const n of net.acts) {  // each node's activations for this point, as colored cells
      const el = board.querySelector(`.nn-node[data-id="${n.id}"] .nn-vals`);
      if (!el) continue;
      const vals = Array.from({ length: n.width }, (_, j) => h[(A.QA >> 2) + n.offset + j]);
      const m = Math.max(1e-6, ...vals.map(Math.abs));
      el.innerHTML = vals.slice(0, 16).map(v => {
        const s = Math.abs(v) / m, c = v > 0 ? C1 : C0;
        return `<i title="${v.toFixed(3)}" style="background:rgb(${c.map(q => Math.round(255 + (q - 255) * s)).join(',')})"></i>`;
      }).join('');
    }
    if (!running) { drawBoundary(h.subarray(A.GO >> 2, (A.GO >> 2) + G * G)); drawThumbs(); }
    if (animate) flow();
  }
  let flowTimers = [];
  function flow() {  // light up the path the point takes, node by node
    flowTimers.forEach(clearTimeout);
    flowTimers = [];
    board.querySelectorAll('.nn-node.lit').forEach(e => e.classList.remove('lit'));
    flowing.clear();
    drawWires();
    (net.used || []).forEach((id, k) => {
      flowTimers.push(setTimeout(() => {
        const el = board.querySelector(`.nn-node[data-id="${id}"]`);
        if (el) el.classList.add('lit');
        flowing.add(id);
        drawWires();
      }, 140 * k));
    });
    flowTimers.push(setTimeout(() => {
      flowing.clear(); drawWires();
      board.querySelectorAll('.nn-node.lit').forEach(e => e.classList.remove('lit'));
    }, 140 * (net.used || []).length + 1400));
  }
  ui.form.addEventListener('submit', e => {
    e.preventDefault();
    try {
      query = parsePoint(ui.q.value);
      ui.qerr.hidden = true; ui.q.classList.remove('bad');
      if (!net) { ui.qerr.textContent = 'Start the demo first (Train it live).'; ui.qerr.hidden = false; return; }
      ui.qx.hidden = false;
      infer(true);
    } catch (err) {
      ui.qerr.textContent = err.message; ui.qerr.hidden = false; ui.q.classList.add('bad');
      ui.res.hidden = true;
    }
  });
  // an empty field (or the x) clears the point from everything, as if it was never entered
  function clearQuery() {
    query = null;
    ui.qx.hidden = true;
    ui.res.hidden = true; ui.qerr.hidden = true; ui.q.classList.remove('bad');
    flowTimers.forEach(clearTimeout); flowTimers = [];
    flowing.clear();
    board.querySelectorAll('.nn-node.lit').forEach(e => e.classList.remove('lit'));
    board.querySelectorAll('.nn-vals').forEach(e => { e.innerHTML = ''; });
    drawWires();
    if (net && !running) {
      const A = net.addr;
      drawBoundary(heap().subarray(A.GO >> 2, (A.GO >> 2) + G * G));
      drawThumbs();
    } else if (!net) drawBoundary();
  }
  ui.q.addEventListener('input', () => {
    ui.q.classList.remove('bad'); ui.qerr.hidden = true;
    if (!ui.q.value.trim()) clearQuery();
  });
  ui.qx.addEventListener('click', () => { ui.q.value = ''; clearQuery(); ui.q.focus(); });
  boundary.addEventListener('click', e => {
    const r = boundary.getBoundingClientRect();
    const x = -LIM + (e.clientX - r.left) / r.width * 2 * LIM, y = LIM - (e.clientY - r.top) / r.height * 2 * LIM;
    ui.q.value = x.toFixed(2) + ', ' + y.toFixed(2);
    ui.form.requestSubmit();
  });

  let frameN = 0;
  function frame() {
    requestAnimationFrame(frame);
    if (!net || !running) return;
    const A = net.addr;
    try {
      const k = Math.min(PER_FRAME, STEPS - step);
      const loss = call(net.train, k, N, BigInt(A.X), BigInt(A.Y), BigInt(A.P), BigInt(A.M), BigInt(A.V), BigInt(A.WS), +ui.lr.value, t);
      step += k; t += k; losses.push(loss);
      call(net.predict, G * G, BigInt(A.GX), BigInt(A.P), BigInt(A.WS), BigInt(A.GO));
      drawBoundary(heap().subarray(A.GO >> 2, (A.GO >> 2) + G * G));
      if (frameN++ % 3 === 0 || step >= STEPS) {
        drawThumbs();
        call(net.predict, N, BigInt(A.X), BigInt(A.P), BigInt(A.WS), BigInt(A.PX));
        const px = heap().subarray(A.PX >> 2, (A.PX >> 2) + N);
        let ok = 0;
        points.forEach((p, i) => { if ((px[i] > .5) === (p[2] === 1)) ok++; });
        ui.acc.textContent = (100 * ok / N).toFixed(0) + '%';
        if (query) infer(false);
      }
      ui.step.textContent = step;
      ui.lossT.textContent = loss.toFixed(3);
      drawLoss();
      if (step >= STEPS) running = false;
    } catch (e) {
      running = false;
      ui.err.textContent = 'training stopped: ' + String(e.message || e).split('\n')[0];
      ui.err.hidden = false;
    }
  }

  // ---- start: with the banner's Pyodide when it is already there, else on request
  async function start() {
    if (py) return;
    ui.start.disabled = true;
    root.classList.add('loading');
    try {
      py = await window.cimoRuntime.load(m => { ui.status.textContent = m; });
      ui.status.textContent = 'Loading the code generator…';
      for (const f of ['codegen.py', 'wat.py'])
        py.FS.writeFile('/home/pyodide/' + f, await (await fetch('nn/' + f)).text());
      ns = py.toPy({});  // its own namespace: the banner demo's Python shares globals
      await py.runPythonAsync(PY, { globals: ns });
      buildPy = ns.get('nn_build');
      setData();
      root.classList.remove('loading');
      root.classList.add('running');
      build();
      requestAnimationFrame(frame);
      const q = /[?&]nnq=([^&#]+)/.exec(location.search);  // (for screenshots: ?nnq=0.3,-0.5)
      if (q) { ui.q.value = decodeURIComponent(q[1]); setTimeout(() => ui.form.requestSubmit(), 600); }
    } catch (e) {
      py = null;
      ui.status.textContent = 'Could not start: ' + (e && e.message || e);
      ui.start.disabled = false;
      root.classList.remove('loading');
    }
  }
  ui.start.addEventListener('click', start);
  // start by itself once on screen if Pyodide is already loading (the banner demo started it)
  let onScreen = false;
  new IntersectionObserver(es => {
    onScreen = es[0].isIntersecting;
    if (onScreen && window.cimoRuntime && window.cimoRuntime.started()) start();
  }, { threshold: 0.2 }).observe(root);
  addEventListener('cimo-runtime', () => { if (onScreen) start(); });

  render();
  setData();
  showStage();
})();
