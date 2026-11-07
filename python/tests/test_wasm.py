"""The wasm backend (cimo.backend.wasm, cimo.wasm): modules run under Node in a stand-in for
Pyodide's loader (tests/wasm/host.mjs) and must match the native build's results. With
PYODIDE_JS set (path to the `pyodide` npm package's pyodide.mjs), also inside real Pyodide.
"""

import ctypes
import os
import platform
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'wasm'))
from host import Host, KernelFailure, available  # noqa: E402

import cimo  # noqa: E402
import cimo.wasm as W  # noqa: E402

pytestmark = pytest.mark.skipif(not available(), reason='needs node')
ARM64 = platform.machine().lower() in ('arm64', 'aarch64')
RT = os.path.join(HERE, 'runtime')
# Pyodide for the tests that run in it: PYODIDE_JS, else the checkout's scratch/spikes/pyodide when installed there
# (npm install pyodide@<the version docs/runtime.js loads>), so the full suite runs the site's demos too
_LOCAL = os.path.join(HERE, '..', '..', 'scratch', 'spikes', 'pyodide', 'node_modules', 'pyodide', 'pyodide.mjs')
if not os.environ.get('PYODIDE_JS') and os.path.exists(_LOCAL):
    os.environ['PYODIDE_JS'] = os.path.abspath(_LOCAL)


@pytest.fixture(scope='module')
def host():
    h = Host()
    yield h
    h.close()


@pytest.fixture(scope='module', autouse=True)
def _no_fma():
    from cimo import driver
    old = driver.FMA
    driver.FMA = False
    yield
    driver.FMA = old


def build(host, src):
    mod, sigs = W.build(src)
    host.load(mod)
    host.sigs = dict(line.split(' ', 1) for line in sigs.splitlines())
    return host.sigs


def test_kernels(host):
    sigs = build(host, r'''
long sum(const long *p, long n) { long s = 0; for (long i = 0; i < n; i++) s += p[i]; return s; }
double dot(int n, const double *a, const double *b) {
    double s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
#include <math.h>
float fmix(float a, float b) { return fminf(a, b) + fmaxf(a, b); }
double rnd(double x) { return rint(x) + 2 * nearbyint(x); }
float rndf(float x) { return rintf(x) + 2 * nearbyintf(x); }
unsigned char sat(int x) { return x < 0 ? 0 : x > 255 ? 255 : x; }
int fact(int n) { return n < 2 ? 1 : n * fact(n - 1); }
''')
    p = host.array('q', list(range(1000)))
    assert host.call('sum', 'l,l', p, 1000)[0] == 499500
    a, b = host.array('d', [1.5] * 37), host.array('d', [2.0] * 37)
    assert host.call('dot', 'w,l,l', 37, a, b)[0] == 111.0
    assert host.call('sat', 'w', 300)[0] == 255 and host.call('sat', 'w', -5)[0] == 0
    assert host.call('fact', 'w', 10)[0] == 3628800
    assert host.call('fmix', 's,s', float('nan'), 2.0)[0] == 4.0  # C fmin/fmax ignore a NaN
    for x, want in ((2.5, 6.0), (-3.5, -12.0), (1.25, 3.0), (-0.5, -0.0)):  # ties to even
        assert host.call('rnd', 'd', x)[0] == want and host.call('rndf', 's', x)[0] == want


def test_unsigned_results_in_the_signatures(host):
    sigs = build(host, 'unsigned a(void) { return 1; }\nunsigned long b(void) { return 2; }\n'
                       'int c(void) { return 3; }\nunsigned char d(void) { return 4; }\n')
    assert sigs['a'].startswith('W') and sigs['b'].startswith('L')
    assert sigs['c'].startswith('w') and sigs['d'].startswith('w')


def test_control_flow_structs_and_pointers(host):
    build(host, r'''
#include <stdlib.h>
#include <string.h>
struct pt { double x, y; int tag; };
static struct pt mid(struct pt a, struct pt b) {
    struct pt r = { (a.x + b.x) / 2, (a.y + b.y) / 2, a.tag | b.tag }; return r; }
static int cmp(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }
static const char *names[] = { "zero", "one", "two" };
static int (*ops[2])(int, int);
static int add(int a, int b) { return a + b; }
static int mul(int a, int b) { return a * b; }
long test(int sel) {
    switch (sel) {
    case 0: { struct pt m = mid((struct pt){1, 2, 1}, (struct pt){3, 6, 2}); return m.x * 100 + m.y * 10 + m.tag; }
    case 1: { int v[] = { 5, 3, 9, 1, 7 }; qsort(v, 5, sizeof *v, cmp); return v[0] * 10000 + v[1] * 1000 + v[2] * 100 + v[3] * 10 + v[4]; }
    case 2: return strlen(names[2]) * 10 + names[1][1];
    case 3: ops[0] = add; ops[1] = mul; return ops[0](6, 7) * 100 + ops[1](6, 7);
    case 4: { int n = 0; for (int i = 0; i < 10; i++) { if (i == 3) continue; if (i == 8) break; n += i; } return n; }
    case 5: { int i = 0; again: i++; if (i < 5) goto again; return i; }
    case 6: { char *s = malloc(100); strcpy(s, "hello"); long r = s[4]; free(s); return r; }
    default: return -1;
    }
}
''')
    want = [2 * 100 + 4 * 10 + 3, 13579, 3 * 10 + ord('n'), 13 * 100 + 42, 25, 5, ord('o')]
    for sel, w in enumerate(want):
        assert host.call('test', 'w', sel)[0] == w, sel


def test_vlas_freed_and_volatile(host):
    # a VLA per iteration (freed at the end of each, else the shadow stack overflows) and
    # volatile accesses (getsp/setsp and vload/vstore in the wasm emitter)
    build(host, r'''
void *volatile p;
long loops(long n) {
    long s = 0;
    for (long i = 0; i < n; i++) { int x[i % 1000 + 1]; x[i % 1000] = 2; p = x; s += x[i % 1000]; }
    int k = 0;
again:;
    int y[k % 500 + 1];
    y[0] = 1; p = y;
    if (++k < 100000) goto again;
    return s + k;
}
''')
    assert host.call('loops', 'l', 200000)[0] == 400000 + 100000


def test_printf_and_errors(host):
    build(host, r'''
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
int hello(int n) { printf("hi %d %s %.3f\n", n, "there", 1.5); return n; }
void check(int x) { assert(x > 0); }
void stop(void) { abort(); }
''')
    assert host.call('hello', 'w', 7) == (7, 'hi 7 there 1.500\n')
    with pytest.raises(KernelFailure) as e:
        host.call('check', 'w', -1)
    assert e.value.err['code'] == 2 and e.value.err['expr'] == 'x > 0'
    with pytest.raises(KernelFailure) as e:
        host.call('stop', '')
    assert e.value.err['code'] == 1
    assert host.call('hello', 'w', 1)[0] == 1  # the stack pointer was restored


def test_irreducible_control_flow_is_a_clear_error():
    with pytest.raises(cimo.CompileError, match='irreducible'):
        W.build(r'''
int f(int n) { int i = 0; if (n) goto inside; for (; i < 10; i++) { inside: n++; } return n; }
''')


# the runtime test programs (tests/runtime): the wasm build's output must equal the native one
# known differences: musl's pow and glibc's hypot algorithm take their fma paths natively
# (__FP_FAST_FMA) and the plain ones on wasm (no scalar fma), which round differently (each under
# 1 ulp): pow a few hard cases; hypot's fma kernel is correctly rounded 87% of the time and its
# correction step 99%, so about 6% of its 600 cases (glibc's bits on each target)
_FMA_DIFFS_OK = {'pow': 8, 'hypot': 60}


@pytest.mark.skipif(not ARM64, reason='the native build is the reference (arm64)')
@pytest.mark.parametrize('t', ['t_string', 't_ctype', 't_stdlib', 't_printf', 't_math'])
def test_runtime_programs_match_native(host, t):
    path = os.path.join(RT, t + '.c')
    with open(path) as f:
        src = f.read()
    mod, sigs = W.build(src, path)
    host.load(mod)
    host.sigs = dict(line.split(' ', 1) for line in sigs.splitlines())
    cap = 8 << 20
    p = host.alloc(cap)
    n, _ = host.call('run', 'l,l', p, cap)
    got = host.read(p, n).decode('latin-1')
    m = cimo.load(path)
    buf = ctypes.create_string_buffer(cap)
    n = m.run(buf, cap)
    ref = buf.raw[:n].decode('latin-1')
    diffs = [(x, y) for x, y in zip(got.splitlines(), ref.splitlines()) if x != y]
    assert len(got.splitlines()) == len(ref.splitlines())
    if t == 't_math':
        names = [x.split()[0] for x, _ in diffs]
        assert all(names.count(n) <= _FMA_DIFFS_OK.get(n, 0) for n in names), (len(diffs), diffs[:5])
    else:
        assert not diffs, diffs[:5]


@pytest.mark.skipif(not os.environ.get('PYODIDE_JS'), reason='set PYODIDE_JS to run in Pyodide')
def test_in_pyodide(tmp_path):
    script = tmp_path / 't.py'
    script.write_text(r'''
import ctypes, cimo
k = cimo.compile(r"""
#include <stdio.h>
#include <assert.h>
long sum(const long *p, long n) { long s = 0; for (long i = 0; i < n; i++) s += p[i]; return s; }
int shout(int n) { printf("n=%d\n", n); return n * 2; }
void check(int x) { assert(x > 0); }
""")
a = (ctypes.c_int64 * 100)(*range(100))
assert k.sum(a, 100) == 4950
assert k.shout(21) == 42
try:
    k.check(0)
    raise SystemExit('no error')
except cimo.KernelError as e:
    assert e.kind == 'assert' and 'x > 0' in str(e), e
assert k.sum(a, 10) == 45
f = cimo.compile(r"""
int rd(long a) { return *(int *)a; }
int dv(int a, int b) { return a / b; }
int rec(int n) { return n < 0 ? 0 : rec(n + 1) + 1; }
int ok(int a) { return a + 1; }
""")
for call, kind in ((lambda: f.rd(0x7fffff00), 'memory fault'), (lambda: f.dv(1, 0), 'division fault'),
                   (lambda: f.rec(1), 'stack overflow')):
    try:
        call()
        raise SystemExit('no error: ' + kind)
    except cimo.KernelError as e:
        assert e.kind == kind and str(e).startswith(kind), (kind, e)
    assert f.ok(1) == 2
import contextlib, io
u = cimo.compile(r"""
#include <stdio.h>
void byte(int c) { putchar(c); }
void fill(int n) { for (int i = 0; i < n; i++) putchar('a'); putchar(0xc3); putchar(0xa9); }
""")
f = io.StringIO()
with contextlib.redirect_stdout(f):
    u.byte(0xc3)  # a character split between two kernel calls
    u.byte(0xa9)
    u.fill(4087)  # and one split by a full buffer
    u.byte(10)
assert f.getvalue() == 'é' + 'a' * 4087 + 'é\n', repr(f.getvalue()[:8] + f.getvalue()[-8:])
h = cimo.compile(r"""
#include <stdlib.h>
int big(int w) { void *p = w == 0 ? malloc((1UL << 32) + 16) : w == 1 ? calloc(1, 1UL << 43) : malloc(0x7fff0001UL);
                 return p == 0; }
long mk(int n) { long *p = malloc(n * sizeof *p); for (int i = 0; i < n; i++) p[i] = i; return (long)p; }
long rd(long p, int i) { return ((long *)p)[i]; }
""")
for w in range(3):  # more than wasm32 can have: NULL, not a small block or a trap (2026-11-07)
    assert h.big(w) == 1, w
q = h.mk(1000)
assert h.rd(q, 999) == 999 and h.heap_bytes() > 0
h.free(q)
h.reset_heap()
assert h.rd(h.mk(10), 9) == 9
libm = ctypes.CDLL(None)  # Pyodide's own frexp, called by address: its pointer parameter is an i32 (2026-11-07)
fx = cimo.compile(r"""
typedef double (*F)(double, int *);
double call(long f, double x) { int e; double m = ((F)f)(x, &e); return m * 1000 + e; }
static double mine(double x, int *e) { *e = 7; return x / 2; }
double own(double x) { volatile F g = mine; int e; return g(x, &e) * 10 + e; }
""")
assert fx.call(ctypes.cast(libm.frexp, ctypes.c_void_p).value, 12.0) == 754.0
assert fx.own(4.0) == 27.0
print('PYODIDE OK')
''')
    r = subprocess.run(['node', os.path.join(HERE, 'wasm', 'pyodide_run.mjs'), str(script)],
                       capture_output=True, universal_newlines=True, timeout=600)
    assert 'n=21' in r.stdout and 'PYODIDE OK' in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


@pytest.mark.skipif(not os.environ.get('PYODIDE_JS'), reason='set PYODIDE_JS to run in Pyodide')
def test_site_demos_in_pyodide():
    """The website's live demos (docs/demo.js, docs/nn.js) run their own Python in Pyodide, against this cimo: every
    preset compiled and its kernels called as the page calls them. A change to cimo once broke the attractor demo
    (pointer parameters became 'p' in signatures, and its check still wanted 'l'); the site deploys only after this."""
    r = subprocess.run(['node', os.path.join(HERE, 'wasm', 'site_demos.mjs')],
                       capture_output=True, universal_newlines=True, timeout=900)
    assert r.returncode == 0 and 'SITE DEMOS OK' in r.stdout, r.stdout[-3000:] + r.stderr[-1000:]


def test_large_malloc_is_null(host):
    """More than wasm32 can have is NULL from malloc and calloc (a 4 GiB request wrapped to a small block before, and
    one just under 2 GiB reached the host as a negative size), and the heap still works after."""
    build(host, r'''
#include <stdlib.h>
int big(int which) {
    void *p = which == 0 ? malloc((1UL << 32) + 16) : which == 1 ? calloc(1, 1UL << 43) : which == 2 ? malloc(0x7fff0001UL)
              : realloc(malloc(16), 1UL << 40);
    return p == 0;
}
long small(void) { long *p = malloc(100 * sizeof *p); for (int i = 0; i < 100; i++) p[i] = i; long s = 0;
    for (int i = 0; i < 100; i++) s += p[i]; free(p); return s; }
''')
    for which in range(4):
        assert host.call('big', 'w', which)[0] == 1, which
    assert host.call('small', '')[0] == 4950
