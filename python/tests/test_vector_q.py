"""GNU vector types in vector registers (class q, frontend vecq=): every operator on every element type, one q value
and two, against the same code lane by lane (vecq=0); and every q operation the frontend emits compiles for each
native target."""

import ctypes
import random

import pytest

import cimo
from cimo import driver

TYPES = ['signed char', 'unsigned char', 'short', 'unsigned short', 'int', 'unsigned', 'long', 'unsigned long',
         'float', 'double']
BINARY = ['+', '-', '*', '/', '&', '|', '^', '==', '!=', '<', '<=', '>', '>=']
UNARY = ['-', '~']


def _src(ty, size):
    fl = ty in ('float', 'double')
    ops = [o for o in BINARY if not (fl and o in '&|^')]
    lines = ['typedef %s V __attribute__((vector_size(%d)));' % (ty, size),
             'typedef __typeof__((V){0} == (V){0}) M;',
             'void f(const V *a, const V *b, V *out, M *cmp, %s s) {' % ty]
    k = c = 0
    for o in ops:
        if o in ('==', '!=', '<', '<=', '>', '>='):
            lines.append('  cmp[%d] = a[0] %s b[0]; cmp[%d] = a[0] %s s; cmp[%d] = s %s b[0];' % (c, o, c + 1, o, c + 2, o))
            c += 3
        else:
            lines.append('  out[%d] = a[0] %s b[0]; out[%d] = a[0] %s s; out[%d] = s %s b[0];' % (k, o, k + 1, o, k + 2, o))
            k += 3
    for o in UNARY:
        if not (fl and o == '~'):
            lines.append('  out[%d] = %sa[0];' % (k, o))
            k += 1
    if not fl:
        for sh in (0, 1, 3):
            lines.append('  out[%d] = a[0] << %d; out[%d] = a[0] >> %d;' % (k, sh, k + 1, sh))
            k += 2
    lines.append('  out[%d] = a[0] * b[0] + a[1];' % k)
    lines.append('}')
    return '\n'.join(lines) + '\n', k + 1, c


def _values(ty, n, rnd):
    if ty == 'float' or ty == 'double':
        pool = [0.0, -0.0, 1.0, -1.5, 3.25, 1e30, -1e-30, float('inf'), float('nan')]
        return [rnd.choice(pool) if rnd.random() < 0.5 else rnd.uniform(-100, 100) for _ in range(n)]
    bits = 8 * {'char': 1, 'short': 2, 'int': 4, 'unsigned': 4, 'long': 8}[ty.split()[-1]]
    lo, hi = (0, 2 ** bits - 1) if ty.startswith('unsigned') else (-2 ** (bits - 1), 2 ** (bits - 1) - 1)
    # no 0 and no -1: x / 0 and INT_MIN / -1 are undefined (x86-64 traps)
    return [v if v not in (0, -1) else 3 for v in (rnd.choice([lo, hi, 1, -2 if lo else 1, 2, 7, rnd.randint(lo, hi)])
                                                   for _ in range(n))]


_CT = {'signed char': ctypes.c_int8, 'unsigned char': ctypes.c_uint8, 'short': ctypes.c_int16,
       'unsigned short': ctypes.c_uint16, 'int': ctypes.c_int32, 'unsigned': ctypes.c_uint32, 'long': ctypes.c_int64,
       'unsigned long': ctypes.c_uint64, 'float': ctypes.c_float, 'double': ctypes.c_double}


@pytest.mark.parametrize('size', [16, 32, 64])
@pytest.mark.parametrize('ty', TYPES)
def test_operators_match_lane_by_lane(ty, size, monkeypatch):
    if size == 64 and ty not in ('signed char', 'unsigned', 'double'):
        pytest.skip('four q values: a few element types are enough')
    src, nout, ncmp = _src(ty, size)
    q = cimo.compile(src, cache=False)
    monkeypatch.setattr(driver, '_vecq', lambda target=None: 0)
    lanes = cimo.compile(src, cache=False)
    ct = _CT[ty]
    n = size // ctypes.sizeof(ct)
    rnd = random.Random(hash((ty, size)) & 0xffff)
    for _ in range(20):
        a = (ct * (2 * n))(*_values(ty, 2 * n, rnd))
        b = (ct * (2 * n))(*_values(ty, 2 * n, rnd))
        s = _values(ty, 1, rnd)[0]
        outs = []
        for m in (q, lanes):
            out = (ctypes.c_char * (nout * size))()
            cmp = (ctypes.c_char * (max(ncmp, 1) * size))()
            m.f(a, b, out, cmp, s)
            outs.append((bytes(out), bytes(cmp)))
        assert outs[0] == outs[1], (ty, size, list(a), list(b), s)


@pytest.mark.parametrize('target', ['arm64', 'amd64_sysv', 'amd64_avx2'])
def test_every_q_operation_compiles(target, monkeypatch):
    feats = frozenset(('fma', 'popcnt', 'lzcnt', 'bmi1', 'avx', 'avx2', 'bmi2')) if target == 'amd64_avx2' else frozenset()
    t = 'amd64_sysv' if target.startswith('amd64') else target
    monkeypatch.setattr(driver, '_x86', lambda tg=None: feats if (tg or driver.TARGET).startswith('amd64') else frozenset())
    monkeypatch.setattr(driver, 'TARGET', t)
    for ty in TYPES:
        for size in [z for z in (16, 32, 64, 128) if driver._vecq(t) <= z <= 4 * driver._vecq(t)] + [0]:
            src = _src(ty, size)[0] if size else _lanesrc(ty)
            pp = driver._preprocess(src, 'x.c', target=t)
            ssa, params = driver._ir(pp, 'x.c')
            if size and size == driver._vecq(t):
                assert ' =q q' in ssa, (ty, size)
            driver._qbe(ssa, params, target=t)


def _lanesrc(ty):
    return ('typedef %s V __attribute__((vector_size(16)));\n'
            'typedef %s W __attribute__((vector_size(32)));\n'
            'double f(const V *a, const W *b) { V x = a[0] * a[0]; W y = b[0] + b[0]; double s = 0;\n'
            '  for (int i = 0; i < 16 / sizeof x[0]; i++) s = s * 3 + x[0] + x[1] - y[1];\n'
            '  return s + x[sizeof x / sizeof x[0] - 1] + y[sizeof y / sizeof y[0] - 1] + y[sizeof y / sizeof y[0] / 2]; }\n'
            ) % (ty, ty)


@pytest.mark.parametrize('ty', TYPES)
def test_lane_reads_match(ty, monkeypatch):
    src = _lanesrc(ty)
    q = cimo.compile(src, cache=False)
    monkeypatch.setattr(driver, '_vecq', lambda target=None: 0)
    lanes = cimo.compile(src, cache=False)
    ct = _CT[ty]
    rnd = random.Random(7)
    for _ in range(20):
        a = (ct * (16 // ctypes.sizeof(ct)))(*_values(ty, 16 // ctypes.sizeof(ct), rnd))
        b = (ct * (32 // ctypes.sizeof(ct)))(*_values(ty, 32 // ctypes.sizeof(ct), rnd))
        r1, r2 = q.f(a, b), lanes.f(a, b)
        assert r1 == r2 or (r1 != r1 and r2 != r2), (ty, list(a), list(b))


def test_bytes_of_a_vector_after_a_vector_store():
    """A vector store (qstore) into a slot whose bytes were loaded before: load forwarding must not reuse the old bytes
    (it took the store for 8 bytes wide; gcc's va-arg-25.c)."""
    src = ('typedef unsigned V __attribute__((vector_size(16)));\n'
           'long f(const V *p) { V v; v = p[0]; long a, b; __builtin_memcpy(&a, (char *)&v + 8, 8); v = p[1];\n'
           '  __builtin_memcpy(&b, (char *)&v + 8, 8); return a * 1000 + b; }\n')
    a = (ctypes.c_uint * 8)(0, 0, 1, 0, 0, 0, 2, 0)
    assert cimo.compile(src, cache=False).f(a) == 1002


def _pairs():
    out = []
    size = {t: ctypes.sizeof(_CT[t]) for t in TYPES}
    for s in TYPES:
        for d in TYPES:
            for n in (2, 4, 8, 16):
                if n * size[s] in (16, 32, 64) and n * size[d] in (16, 32, 64):
                    out.append((s, d, n))
    return out


@pytest.mark.parametrize('s, d, n', _pairs())
def test_convertvector_matches_lane_by_lane(s, d, n, monkeypatch):
    src = ('typedef %s S __attribute__((vector_size(%d)));\ntypedef %s D __attribute__((vector_size(%d)));\n'
           'void f(const S *a, D *r) { *r = __builtin_convertvector(*a, D); }\n'
           % (s, n * ctypes.sizeof(_CT[s]), d, n * ctypes.sizeof(_CT[d])))
    q = cimo.compile(src, cache=False)
    monkeypatch.setattr(driver, '_vecq', lambda target=None: 0)
    lanes = cimo.compile(src, cache=False)
    rnd = random.Random(n)
    for _ in range(10):
        if s in ('float', 'double'):  # in the destination's range: out of it is undefined
            lo = 0 if d.startswith('unsigned') else -100
            vals = [rnd.uniform(lo, 100) for _ in range(n)] if d not in ('float', 'double') else _values(s, n, rnd)
        else:
            vals = _values(s, n, rnd)
        a = (_CT[s] * n)(*vals)
        r1, r2 = (_CT[d] * n)(), (_CT[d] * n)()
        q.f(a, r1)
        lanes.f(a, r2)
        assert bytes(r1) == bytes(r2), (s, d, n, vals, list(r1), list(r2))


@pytest.mark.parametrize('ty', TYPES)
def test_shufflevector_matches_lane_by_lane(ty, monkeypatch):
    n = 16 // ctypes.sizeof(_CT[ty])
    rnd = random.Random(n)
    shuffles = [[rnd.choice([-1] + list(range(2 * n))) for _ in range(n)] for _ in range(6)]
    shuffles += [list(range(n)), list(range(n, 2 * n)), [n - 1 - i for i in range(n)], [-1] * n]
    body = ''.join('  r[%d] = __builtin_shufflevector(a[0], b[0], %s);\n' % (k, ', '.join(map(str, s)))
                   for k, s in enumerate(shuffles))
    src = 'typedef %s V __attribute__((vector_size(16)));\nvoid f(const V *a, const V *b, V *r) {\n%s}\n' % (ty, body)
    q = cimo.compile(src, cache=False)
    monkeypatch.setattr(driver, '_vecq', lambda target=None: 0)
    lanes = cimo.compile(src, cache=False)
    a = (_CT[ty] * n)(*_values(ty, n, rnd))
    b = (_CT[ty] * n)(*_values(ty, n, rnd))
    r1, r2 = (ctypes.c_char * (16 * len(shuffles)))(), (ctypes.c_char * (16 * len(shuffles)))()
    q.f(a, b, r1)
    lanes.f(a, b, r2)
    for k in range(len(shuffles)):  # an index of -1 leaves the lane unspecified: compare the others
        for i, x in enumerate(shuffles[k]):
            if x >= 0:
                es = 16 // n
                o = 16 * k + es * i
                assert bytes(r1)[o:o + es] == bytes(r2)[o:o + es], (ty, shuffles[k])
