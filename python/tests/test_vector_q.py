"""GNU vector types in vector registers (class q, frontend vecq=): every operator on every element type, one q value
and two, against the same code lane by lane (vecq=0); and every q operation the frontend emits compiles for each
native target."""

import ctypes
import math
import random
import struct

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
    q = cimo.compile(src, cache=False, fma=False)  # the lanes don't contract a*b + c (test_vector_fma)
    monkeypatch.setattr(driver, '_vecq', lambda target=None: 0)
    lanes = cimo.compile(src + "\nstatic int lane_by_lane;\n", cache=False, fma=False)  # compiles are kept per source
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
        assert outs[0][1] == outs[1][1], (ty, size, list(a), list(b), s)  # comparison masks: every bit
        if ty in ('float', 'double'):  # C leaves a NaN's sign and payload open (x86-64 picks by operand order)
            ct2 = ctypes.c_float if ty == 'float' else ctypes.c_double
            v1, v2 = [list((ct2 * (len(o[0]) // ctypes.sizeof(ct2))).from_buffer_copy(o[0])) for o in outs]
            assert all(x == y and math.copysign(1, x) == math.copysign(1, y) or (x != x and y != y)
                       for x, y in zip(v1, v2)), (ty, size, list(a), list(b), s)
        else:
            assert outs[0][0] == outs[1][0], (ty, size, list(a), list(b), s)


@pytest.mark.parametrize('target', ['arm64', 'amd64_sysv', 'amd64_avx2'])
def test_every_q_operation_compiles(target, monkeypatch):
    feats = frozenset(('fma', 'popcnt', 'lzcnt', 'bmi1', 'avx', 'avx2', 'bmi2')) if target == 'amd64_avx2' else frozenset()
    t = 'amd64_sysv' if target.startswith('amd64') else target
    monkeypatch.setattr(driver, '_x86', lambda tg=None: feats if (tg or driver.TARGET).startswith('amd64') else frozenset())
    monkeypatch.setattr(driver, 'TARGET', t)
    for ty in TYPES:
        n = 16 // ctypes.sizeof(_CT[ty])
        extra = ('typedef %s V __attribute__((vector_size(16)));\nvoid g(const V *a, const V *b, int s, V *r) {\n'
                 '  r[0] = __builtin_shufflevector(a[0], b[0], %s); r[1] = __builtin_shufflevector(a[0], b[0], %s);%s }\n'
                 % (ty, ', '.join(str(3 * i % (2 * n)) for i in range(n)), ', '.join(str(i // 2 + n * (i % 2)) for i in range(n)),
                    '' if ty in ('float', 'double') else ' r[2] = a[0] << s; r[3] = a[0] >> b[0];'))
        for size in [z for z in (16, 32, 64, 128) if driver._vecq(t) <= z <= 4 * driver._vecq(t) or z == 16] + [0, -1]:
            src = _src(ty, size)[0] if size > 0 else _lanesrc(ty) if size == 0 else extra
            pp = driver._preprocess(src, 'x.c', target=t)
            ssa, params = driver._ir(pp, 'x.c')
            if size in (16, driver._vecq(t)):  # 16 bytes with AVX2: its lower half (qload16)
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
    lanes = cimo.compile(src + "\nstatic int lane_by_lane;\n", cache=False)  # another source: compiles are kept per process
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
    lanes = cimo.compile(src + "\nstatic int lane_by_lane;\n", cache=False)  # another source: compiles are kept per process
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
    h = n // 2
    shuffles += [list(range(n)), list(range(n, 2 * n)), [n - 1 - i for i in range(n)], [-1] * n,
                 [(i // 2) + (n if i % 2 else 0) for i in range(n)], [h + (i // 2) + (n if i % 2 else 0) for i in range(n)],
                 [2 * i for i in range(n)], [2 * i + 1 for i in range(n)], [(i & ~1) + (n if i % 2 else 0) for i in range(n)],
                 [((i & ~1) + 1 + (0 if i % 2 else n)) for i in range(n)], [(i + 1) % (2 * n) for i in range(n)],
                 [i + n - 1 for i in range(n)], [n + 1] * n, [1, -1] * h, [(i + n + 1) for i in range(n - 1)] + [0]]
    body = ''.join('  r[%d] = __builtin_shufflevector(a[0], b[0], %s);\n' % (k, ', '.join(map(str, s)))
                   for k, s in enumerate(shuffles))
    src = 'typedef %s V __attribute__((vector_size(16)));\nvoid f(const V *a, const V *b, V *r) {\n%s}\n' % (ty, body)
    q = cimo.compile(src, cache=False)
    monkeypatch.setattr(driver, '_vecq', lambda target=None: 0)
    lanes = cimo.compile(src + "\nstatic int lane_by_lane;\n", cache=False)  # another source: compiles are kept per process
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


@pytest.mark.parametrize('ty', ['float', 'double'])
def test_vector_fma(ty):
    """a*b + c, a*b - c, c - a*b and with a scalar factor, contracted as for scalars: one rounding, as fma() lane by
    lane gives."""
    import math
    src = ('typedef %s V __attribute__((vector_size(16)));\n'
           'void f(const V *a, const V *b, const V *c, %s s, V *r) { r[0] = a[0] * b[0] + c[0]; r[1] = a[0] * b[0] - c[0];\n'
           '  r[2] = c[0] - a[0] * b[0]; r[3] = c[0] + s * a[0]; }\n') % (ty, ty)
    if driver._vecq() != 16 or not driver._contract():
        pytest.skip('16-byte vectors are lanes here (AVX2), or nothing contracts (no FMA)')
    pp = driver._preprocess(src, 'x.c')
    assert driver._ir(pp, 'x.c')[0].count('qfma1') == 4
    m = cimo.compile(src, cache=False, fma=True)
    ct = _CT[ty]
    n = 16 // ctypes.sizeof(ct)
    rnd = random.Random(3)
    rd = (lambda x: struct.unpack('f', struct.pack('f', x))[0]) if ty == 'float' else (lambda x: x)
    for _ in range(20):
        a, b, c = ([rd(rnd.uniform(-100, 100)) for _ in range(n)] for _ in range(3))
        s = rd(rnd.uniform(-3, 3))
        r = (ct * (4 * n))()
        m.f((ct * n)(*a), (ct * n)(*b), (ct * n)(*c), s, r)
        for i in range(n):
            want = [_fma(a[i], b[i], c[i]), _fma(a[i], b[i], -c[i]), _fma(-a[i], b[i], c[i]), _fma(s, a[i], c[i])]
            assert [r[k * n + i] for k in range(4)] == [rd(w) for w in want], (ty, i)


def _fma(x, y, z):
    """x*y + z rounded once (exact rationals, then to the nearest double; float results then round again, which for
    a float x, y, z is the same as rounding once: 2 * 24 + 2 bits fit in a double's 53)."""
    from fractions import Fraction
    return float(Fraction(x) * Fraction(y) + Fraction(z))


@pytest.mark.parametrize('ty', TYPES)
def test_lane_writes_and_variable_lanes(ty, monkeypatch):
    """v[i] = x and v[i] with a variable i, through a copy of the vector (which stays a register)."""
    n = 16 // ctypes.sizeof(_CT[ty])
    src = ('typedef %s V __attribute__((vector_size(16)));\n'
           'void f(const V *a, V *r, int k, %s x) { V v = a[0]; v[0] = x; v[k] = v[k] + x; v[%d] = v[k];\n'
           '  for (int i = 1; i < %d; i++) v[i] = v[i - 1] + v[i]; r[0] = v; r[1] = v + v; }\n') % (ty, ty, n - 1, n)
    q = cimo.compile(src, cache=False, fma=False)
    monkeypatch.setattr(driver, '_vecq', lambda target=None: 0)
    lanes = cimo.compile(src + '\nstatic int lane_by_lane;\n', cache=False, fma=False)
    rnd = random.Random(11)
    for k in range(n):
        a = (_CT[ty] * n)(*_values(ty, n, rnd))
        x = _values(ty, 1, rnd)[0]
        r1, r2 = (ctypes.c_char * 32)(), (ctypes.c_char * 32)()
        q.f(a, r1, k, x)
        lanes.f(a, r2, k, x)
        assert bytes(r1) == bytes(r2), (ty, k)


@pytest.mark.parametrize('ty', [t for t in TYPES if t not in ('float', 'double')])
def test_shifts_by_a_variable(ty, monkeypatch):
    bits = 8 * ctypes.sizeof(_CT[ty])
    n = 16 // ctypes.sizeof(_CT[ty])
    src = ('typedef %s V __attribute__((vector_size(16)));\n'
           'void f(const V *a, const V *c, int s, long t, V *r) { r[0] = a[0] << s; r[1] = a[0] >> s; r[2] = a[0] << c[0];\n'
           '  r[3] = a[0] >> c[0]; r[4] = a[0] >> t; r[5] = a[0] << t; }\n') % ty
    q = cimo.compile(src, cache=False)
    monkeypatch.setattr(driver, '_vecq', lambda target=None: 0)
    lanes = cimo.compile(src + '\nstatic int lane_by_lane;\n', cache=False)
    rnd = random.Random(bits)
    for _ in range(20):
        a = (_CT[ty] * n)(*_values(ty, n, rnd))
        c = (_CT[ty] * n)(*[rnd.randrange(bits) for _ in range(n)])
        s, t = rnd.randrange(bits), rnd.randrange(bits)
        r1, r2 = (ctypes.c_char * 96)(), (ctypes.c_char * 96)()
        q.f(a, c, s, t, r1)
        lanes.f(a, c, s, t, r2)
        assert bytes(r1) == bytes(r2), (ty, s, t, list(c))


def test_a_vector_read_before_it_is_written():
    """Lanes set in a loop, the vector then passed on: its register is read before the first write on the loop's
    first way round (SSA's undefined value, which has no q constant; it is 0). gcc's vector-div-neg-pow2.c."""
    src = ('typedef int V __attribute__((vector_size(16)));\n__attribute__((noipa)) static V g(V x) { return x + 1; }\n'
           'int f(int n) { int s = 0; for (int j = 0; j < n; j++) { V x, q; for (int i = 0; i < 4; i++) x[i] = i + j;\n'
           '  q = g(x); for (int i = 0; i < 4; i++) s += q[i]; } return s; }\n')
    assert cimo.compile(src, cache=False).f(3) == sum(i + j + 1 for j in range(3) for i in range(4))


@pytest.mark.parametrize('ty', TYPES)
def test_shuffle_with_a_variable_mask(ty, monkeypatch):
    """__builtin_shuffle(a, m) and (a, b, m): each index modulo the lanes (all of a's and b's), as byte tables."""
    n = 16 // ctypes.sizeof(_CT[ty])
    it = {1: 'signed char', 2: 'short', 4: 'int', 8: 'long'}[16 // n]
    src = ('typedef %s V __attribute__((vector_size(16)));\ntypedef %s M __attribute__((vector_size(16)));\n'
           'void f(const V *a, const V *b, const M *m, V *r) { r[0] = __builtin_shuffle(a[0], m[0]);\n'
           '  r[1] = __builtin_shuffle(a[0], b[0], m[0]); }\n') % (ty, it)
    q = cimo.compile(src, cache=False)
    monkeypatch.setattr(driver, '_vecq', lambda target=None: 0)
    lanes = cimo.compile(src + '\nstatic int lane_by_lane;\n', cache=False)
    rnd = random.Random(n)
    mt = _CT[it]
    for _ in range(30):
        a = (_CT[ty] * n)(*_values(ty, n, rnd))
        b = (_CT[ty] * n)(*_values(ty, n, rnd))
        m = (mt * n)(*[rnd.choice([rnd.randrange(4 * n), -rnd.randrange(1, 100), rnd.randrange(2 * n)]) for _ in range(n)])
        r1, r2 = (ctypes.c_char * 32)(), (ctypes.c_char * 32)()
        q.f(a, b, m, r1)
        lanes.f(a, b, m, r2)
        assert bytes(r1) == bytes(r2), (ty, list(m))


@pytest.mark.parametrize('size', [32, 64])
@pytest.mark.parametrize('ty', ['signed char', 'short', 'int', 'long', 'float', 'double'])
def test_wide_shuffles(ty, size, monkeypatch):
    """Shuffles of 32 and 64 bytes in 16-byte pieces: constant (including a result of another length) and variable."""
    es = ctypes.sizeof(_CT[ty])
    n = size // es
    it = {1: 'signed char', 2: 'short', 4: 'int', 8: 'long'}[es]
    rnd = random.Random(size + es)
    consts = [[rnd.choice([-1] + list(range(2 * n))) for _ in range(n)] for _ in range(3)]
    consts.append([2 * n - 1 - i for i in range(n)])
    half = [rnd.randrange(2 * n) for _ in range(max(1, (16 // es)))]  # a 16-byte result from wide sources
    src = ('typedef %s V __attribute__((vector_size(%d)));\ntypedef %s M __attribute__((vector_size(%d)));\n'
           'typedef %s H __attribute__((vector_size(16)));\n'
           'void f(const V *a, const V *b, const M *m, V *r, H *h) {\n' % (ty, size, it, size, ty) +
           ''.join('  r[%d] = __builtin_shufflevector(a[0], b[0], %s);\n' % (k, ', '.join(map(str, c))) for k, c in enumerate(consts)) +
           '  r[%d] = __builtin_shuffle(a[0], m[0]); r[%d] = __builtin_shuffle(a[0], b[0], m[0]);\n' % (len(consts), len(consts) + 1) +
           '  h[0] = __builtin_shufflevector(a[0], b[0], %s); }\n' % ', '.join(map(str, half)))
    assert 'qtbl' in driver._ir(driver._preprocess(src, 'x.c'), 'x.c')[0]
    q = cimo.compile(src, cache=False)
    monkeypatch.setattr(driver, '_vecq', lambda target=None: 0)
    lanes = cimo.compile(src + '\nstatic int lane_by_lane;\n', cache=False)
    nr = len(consts) + 2
    for _ in range(10):
        a = (_CT[ty] * n)(*_values(ty, n, rnd))
        b = (_CT[ty] * n)(*_values(ty, n, rnd))
        m = (_CT[it] * n)(*[rnd.choice([rnd.randrange(4 * n), -rnd.randrange(1, 99)]) for _ in range(n)])
        outs = []
        for mod in (q, lanes):
            r = (ctypes.c_char * (size * nr))()
            h = (ctypes.c_char * 16)()
            mod.f(a, b, m, r, h)
            outs.append((bytes(r), bytes(h)))
        for k, c in enumerate(consts):  # unspecified (-1) lanes aside
            for i, x in enumerate(c):
                if x >= 0:
                    o = size * k + es * i
                    assert outs[0][0][o:o + es] == outs[1][0][o:o + es], (ty, size, c)
        tail = size * len(consts)
        assert outs[0][0][tail:] == outs[1][0][tail:], (ty, size, list(m))
        assert outs[0][1] == outs[1][1], (ty, size, half)
