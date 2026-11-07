# Run by site_demos.mjs inside Pyodide, with the demos' own Python and inputs set as globals.
import ctypes
import math

from pyodide.code import run_js

# calls f as the page does, in JavaScript: the arguments at the indexes in ptrs are pointers, i32 Numbers (as Emscripten's
# code passes them)
call = run_js('(f, ptrs, ...a) => f(...a.map((x, i) => ptrs.includes(i) ? Number(x) : x))')

# the attractor banner (docs/demo.js): build() every preset, then seed() and step() as the page calls them
ns = {}
exec(ATTRACTOR_PY, ns)
N, T, RING = ns['N'], ns['T'], ns['RING']
for name, src in ATTRACTOR_SOURCES.items():
    r = ns['build'](src)
    assert r['ok'], (name, r.get('error'))
    sp = r['getsp']()
    call(r['seed'], [2], N, T, RING)
    for head in range(3):
        n = call(r['step'], [3], N, T, head, RING)
        assert 0 <= n <= N, (name, n)
    r['setsp'](sp)
    ring = (ctypes.c_float * (N * T * 3)).from_address(RING)
    nxt = [ring[3 * 3 * N + k] for k in range(0, 3 * N, 97)]  # slot 3, written by the third step
    assert all(math.isfinite(v) for v in nxt) and any(v != 0 for v in nxt), name
    print('attractor', name, 'ok')

# the training demo (docs/nn.js): nn_build() every preset graph, then init, train, predict as the page calls them
ns = {}
exec(NN_PY, ns)
N, G = ns['N'], ns['G']
for name, graph in NN_GRAPHS.items():
    r = ns['nn_build'](graph)
    assert r['ok'], (name, r.get('stage'), r.get('error'))
    A = r['addr'].to_py() if hasattr(r['addr'], 'to_py') else r['addr']
    X = (ctypes.c_float * (2 * N)).from_address(A['X'])
    Y = (ctypes.c_float * N).from_address(A['Y'])
    for i in range(N):
        X[2 * i], X[2 * i + 1] = math.cos(i), math.sin(2 * i)
        Y[i] = float(i % 2)
    sp = r['getsp']()
    call(r['init'], [1], 1, A['P'])
    p = lambda k: A[k]
    losses = [call(r['train'], [2, 3, 4, 5, 6, 7], 1, N, p('X'), p('Y'), p('P'), p('M'), p('V'), p('WS'), 0.01, t + 1)
              for t in range(20)]
    assert all(math.isfinite(v) for v in losses), (name, losses)
    call(r['predict'], [1, 2, 3, 4], N, p('X'), p('P'), p('WS'), p('PX'))
    PX = (ctypes.c_float * N).from_address(A['PX'])
    assert all(math.isfinite(PX[i]) for i in range(N)), name
    r['setsp'](sp)
    assert 'func' in ns['nn_wat']()
    print('nn', name, 'ok')
print('SITE DEMOS OK')
