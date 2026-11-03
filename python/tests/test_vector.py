"""128-bit values (ext.vector): IR text through arm64 and x86-64, each instruction against a model in Python, and what register
allocation has to get right: copies, spills, phis and values live across calls."""

import array
import ctypes
import math
import re
import struct

import pytest

import cimo
from cimo import driver
import hosts

pytestmark = pytest.mark.skipif(not hosts.RUNS, reason='needs a target with q')

_SIG = driver._build('void f(void *d, void *a, void *b) {}')[1] if hosts.RUNS else None


def run(body, a, b, tail=''):
    """Runs `function $f(l %d, l %a, l %b)`, which has the 16 bytes at a and b; returns the 16 bytes at d."""
    ssa = 'export function $f(l %%d, l %%a, l %%b) {\n@s\n%s\n\tret\n}\n%s' % (body, tail)
    text = driver._qbe(ssa)
    m = driver.Module(text, _SIG, (), snapshot=False)
    d = ctypes.create_string_buffer(16)
    x = ctypes.create_string_buffer(a, 16)
    y = ctypes.create_string_buffer(b, 16)
    m.f(ctypes.addressof(d), ctypes.addressof(x), ctypes.addressof(y))
    return d.raw


LOADS = '\t%x =q qload %a\n\t%y =q qload %b\n'
FMT = {'b': 'b', 'h': 'h', 'w': 'i', 'l': 'q', 's': 'f', 'd': 'd'}
COUNT = {'b': 16, 'h': 8, 'w': 4, 'l': 2, 's': 4, 'd': 2}
BITS = {'b': 8, 'h': 16, 'w': 32, 'l': 64}


def pack(x, vals):
    return struct.pack('<%d%s' % (COUNT[x], FMT[x]), *vals)


def unpack(x, raw):
    return struct.unpack('<%d%s' % (COUNT[x], FMT[x]), raw)


def wrap(x, v):
    n = BITS[x]
    v &= (1 << n) - 1
    return v - (1 << n) if v >> (n - 1) else v


def uns(x, v):
    return v & ((1 << BITS[x]) - 1)


def f32(v):
    return struct.unpack('f', struct.pack('f', v))[0]


def ints(x, seed):
    n = COUNT[x]
    base = [1, -1, 0, 5, -7, 100, -100, 3, 127, -128, 42, -42, 9, -9, 2, -2]
    return [wrap(x, base[(i + seed) % 16] * (seed * 31 + 1)) for i in range(n)]


INTOPS = {'qadd': lambda a, b, x: wrap(x, a + b), 'qsub': lambda a, b, x: wrap(x, a - b), 'qmul': lambda a, b, x: wrap(x, a * b),
          'qmin': lambda a, b, x: min(a, b), 'qmax': lambda a, b, x: max(a, b),
          'qumin': lambda a, b, x: wrap(x, min(uns(x, a), uns(x, b))), 'qumax': lambda a, b, x: wrap(x, max(uns(x, a), uns(x, b))),
          'qceq': lambda a, b, x: -1 if a == b else 0, 'qcgt': lambda a, b, x: -1 if a > b else 0, 'qcge': lambda a, b, x: -1 if a >= b else 0,
          'qcugt': lambda a, b, x: -1 if uns(x, a) > uns(x, b) else 0, 'qcuge': lambda a, b, x: -1 if uns(x, a) >= uns(x, b) else 0}
LANES = {'qadd': 'bhwl', 'qsub': 'bhwl', 'qmul': 'bhw', 'qmin': 'bhw', 'qmax': 'bhw', 'qumin': 'bhw', 'qumax': 'bhw',
         'qceq': 'bhwl', 'qcgt': 'bhwl', 'qcge': 'bhwl', 'qcugt': 'bhwl', 'qcuge': 'bhwl'}


@pytest.mark.parametrize('op,x', [(o, x) for o in INTOPS for x in LANES[o]])
def test_integer(op, x):
    a, b = ints(x, 1), ints(x, 6)
    got = unpack(x, run(LOADS + '\t%%r =q %s%s %%x, %%y\n\tqstore %%r, %%d' % (op, x), pack(x, a), pack(x, b)))
    assert list(got) == [INTOPS[op](p, q, x) for p, q in zip(a, b)]


FPOPS = {'qadd': lambda a, b: a + b, 'qsub': lambda a, b: a - b, 'qmul': lambda a, b: a * b, 'qdiv': lambda a, b: a / b,
         'qmin': min, 'qmax': max}


@pytest.mark.parametrize('op,x', [(o, x) for o in FPOPS for x in 'sd'])
def test_float(op, x):
    a = [1.5, -2.25, 8.0, 0.125][:COUNT[x]]
    b = [0.5, 4.0, -3.0, 16.0][:COUNT[x]]
    got = unpack(x, run(LOADS + '\t%%r =q %s%s %%x, %%y\n\tqstore %%r, %%d' % (op, x), pack(x, a), pack(x, b)))
    assert list(got) == [(f32 if x == 's' else float)(FPOPS[op](p, q)) for p, q in zip(a, b)]


@pytest.mark.parametrize('x', 'sd')
def test_float_compare_is_a_mask(x):
    a = [1.0, 2.0, float('nan'), 4.0][:COUNT[x]]
    b = [1.0, 3.0, 1.0, 0.0][:COUNT[x]]
    ix = {'s': 'w', 'd': 'l'}[x]
    for op, want in (('qceq', lambda p, q: p == q), ('qcgt', lambda p, q: p > q), ('qcge', lambda p, q: p >= q)):
        got = unpack(ix, run(LOADS + '\t%%r =q %s%s %%x, %%y\n\tqstore %%r, %%d' % (op, x), pack(x, a), pack(x, b)))
        assert list(got) == [-1 if want(p, q) else 0 for p, q in zip(a, b)], op


@pytest.mark.parametrize('x', 'sd')
def test_one_argument_float(x):
    a = [4.0, -9.0, 16.0, 0.25][:COUNT[x]]
    for op, want in (('qneg', lambda v: -v), ('qabs', abs), ('qsqrt', lambda v: math.sqrt(abs(v)))):
        body = '\t%x =q qload %a\n' + ('\t%%x =q qabs%s %%x\n' % x if False else '')
        src = a if op != 'qsqrt' else [abs(v) for v in a]
        got = unpack(x, run('\t%%x =q qload %%a\n\t%%r =q %s%s %%x\n\tqstore %%r, %%d' % (op, x), pack(x, src), bytes(16)))
        assert list(got) == [want(v) for v in src], op


@pytest.mark.parametrize('x', 'bhwl')
def test_neg_and_shifts(x):
    a = ints(x, 3)
    got = unpack(x, run('\t%%x =q qload %%a\n\t%%r =q qneg%s %%x\n\tqstore %%r, %%d' % x, pack(x, a), bytes(16)))
    assert list(got) == [wrap(x, -v) for v in a]
    n = {'b': 3, 'h': 5, 'w': 17, 'l': 33}[x]
    for op, want in (('qshl', lambda v: wrap(x, v << n)), ('qshr', lambda v: wrap(x, uns(x, v) >> n)), ('qsar', lambda v: v >> n)):
        got = unpack(x, run('\t%%x =q qload %%a\n\t%%r =q %s%s %%x, %d\n\tqstore %%r, %%d' % (op, x, n), pack(x, a), bytes(16)))
        assert list(got) == [want(v) for v in a], op


def test_logic_and_select():
    a, b = bytes(range(16)), bytes(range(100, 116))
    x, y = int.from_bytes(a, 'little'), int.from_bytes(b, 'little')
    full = (1 << 128) - 1
    for op, want in (('qand', x & y), ('qor', x | y), ('qxor', x ^ y)):
        got = run(LOADS + '\t%%r =q %s %%x, %%y\n\tqstore %%r, %%d' % op, a, b)
        assert int.from_bytes(got, 'little') == want, op
    got = run('\t%x =q qload %a\n\t%r =q qnot %x\n\tqstore %r, %d', a, b)
    assert int.from_bytes(got, 'little') == x ^ full
    # qsel1 %a, %b, %mask: where %a and %b aren't the destination, and where they are
    got = run(LOADS + '\t%m =q qcgtb %x, %y\n\t%r =q qsel1 %x, %y, %m\n\tqstore %r, %d', a, b)
    assert got == bytes(p if struct.unpack('b', bytes([p]))[0] > struct.unpack('b', bytes([q]))[0] else q for p, q in zip(a, b))
    got = run(LOADS + '\t%m =q qxor %x, %x\n\t%x =q qsel1 %x, %y, %m\n\tqstore %x, %d'.replace('%x =q qsel1', '%z =q qsel1').replace('qstore %x', 'qstore %z'), a, b)
    assert got == b


def _fma_here():
    from cimo import cpu
    return hosts.ARM64 or 'fma' in cpu.x86_features()


@pytest.mark.skipif(not _fma_here(), reason='an x86-64 CPU without FMA')
@pytest.mark.parametrize('x', 'sd')
def test_fused_multiply_add_rounds_once(x):
    # a*b+c where the product alone rounds away what the sum keeps
    e = 1.0 + 2 ** (-13 if x == 's' else -30)
    a = [e] * COUNT[x]
    c = [-(1.0 + 2 ** (-12 if x == 's' else -29))] * COUNT[x]
    body = '\t%%x =q qload %%a\n\t%%c =q qload %%b\n\t%%r =q qfma1%s %%x, %%x, %%c\n\tqstore %%r, %%d' % x
    got = unpack(x, run(body, pack(x, a), pack(x, c)))
    want = 2.0 ** (-26 if x == 's' else -60)
    assert all(v == want for v in got)
    # the destination may be any of the three
    body = '\t%%x =q qload %%a\n\t%%c =q qload %%b\n\t%%y =q qfma1%s %%c, %%x, %%x\n\tqstore %%y, %%d' % x
    assert unpack(x, run(body, pack(x, [2.0] * COUNT[x]), pack(x, [3.0] * COUNT[x]))) == (8.0,) * COUNT[x]


@pytest.mark.parametrize('x,k,val', [('b', 'w', -3), ('h', 'w', -300), ('w', 'w', -70000), ('l', 'l', -2 ** 40)])
def test_dup_and_get_integer(x, k, val):
    got = unpack(x, run('\t%%v =%s copy %d\n\t%%r =q qdup%s %%v\n\tqstore %%r, %%d' % (k, val, x), bytes(16), bytes(16)))
    assert got == (val,) * COUNT[x]
    a = ints(x, 2)
    for lane in (0, COUNT[x] - 1):
        store = 'storel' if k == 'l' else 'storew'
        d = run('\t%%x =q qload %%a\n\t%%v =%s qget%s %%x, %d\n\t%s %%v, %%d' % (k, x, lane, store), pack(x, a), bytes(16))
        n = BITS[x]
        low = int.from_bytes(d[:8 if k == 'l' else 4], 'little')
        assert low & ((1 << n) - 1) == uns(x, a[lane])


@pytest.mark.parametrize('x,k', [('s', 's'), ('d', 'd')])
def test_dup_and_get_float(x, k):
    a = [1.5, 2.5, 3.5, 4.5][:COUNT[x]]
    d = run('\t%%x =q qload %%a\n\t%%v =%s qget%s %%x, %d\n\t%%r =q qdup%s %%v\n\tqstore %%r, %%d' % (k, x, COUNT[x] - 1, x), pack(x, a), bytes(16))
    assert unpack(x, d) == (a[-1],) * COUNT[x]


def test_all_registers_and_spills():
    """More live values than registers: spills of 16 bytes, including one next to a 4-byte and an 8-byte slot."""
    n = 40
    lines = ['\t%x =q qload %a', '\t%y =q qload %b', '\t%w =w loadw %a', '\t%l =l loadl %b', '\t%s =s loads %a']
    lines += ['\t%%v%d =q qaddw %%%s, %%y' % (i, 'x' if i == 0 else 'v%d' % (i - 1)) for i in range(n)]
    lines += ['\t%s0 =q qaddw %v0, %v1']
    lines += ['\t%%s%d =q qaddw %%s%d, %%v%d' % (i, i - 1, i + 1) for i in range(1, n - 1)]
    lines += ['\tqstore %%s%d, %%d' % (n - 2), '\t%m =w add %w, 1', '\t%n =l add %l, 1', '\tstorew %m, %a', '\tstorel %n, %b', '\tstores %s, %b']
    a, b = array.array('i', [1, 2, 3, 4]), array.array('i', [10, 20, 30, 40])
    got = array.array('i', run('\n'.join(lines), a.tobytes(), b.tobytes()))
    want = []
    for p, q in zip(a, b):
        vs = []
        cur = p
        for i in range(n):
            cur += q
            vs.append(cur)
        want.append(sum(vs))
    assert list(got) == want


def test_loop_with_a_phi():
    """Sums 4 x i32 in a vector phi round a loop, and the values rotate (a swap in register allocation)."""
    body = '''	%x =q qload %a
	%y =q qload %b
	jmp @h
@h
	%i =w phi @s 0, @l %j
	%p =q phi @s %x, @l %q
	%r =q phi @s %y, @l %p
	%c =w csltw %i, 5
	jnz %c, @l, @e
@l
	%q =q qaddw %r, %y
	%j =w add %i, 1
	jmp @h
@e
	qstore %p, %d'''
    a, b = [1, 2, 3, 4], [10, 20, 30, 40]
    got = struct.unpack('<4i', run(body, struct.pack('<4i', *a), struct.pack('<4i', *b)))
    want = []
    for p0, y in zip(a, b):
        p, r = p0, y
        for _ in range(5):
            p, r = r + y, p
        want.append(p)
    assert list(got) == want


@pytest.mark.parametrize('rotate', [2, 5, 40])
def test_phis_under_pressure(rotate):
    """Vector phis that rotate, more of them than registers: the moves between them go through stack slots."""
    names = ['p%d' % k for k in range(rotate)]
    turns = 2 * rotate + 3  # round more than once: the values have to reach the ones in stack slots
    phis = ''.join('\t%%%s =q phi @s %%x, @l %%%s\n' % (names[k], ('n' if k == 0 else names[k - 1])) for k in range(rotate))
    summed = ['\t%%t0 =q qaddw %%y, %%%s\n' % names[0]]
    summed += ['\t%%t%d =q qaddw %%t%d, %%%s\n' % (k, k - 1, names[k]) for k in range(1, rotate)]
    body = LOADS + '\tjmp @h\n@h\n\t%i =w phi @s 0, @l %j\n' + phis + '\t%%c =w csltw %%i, %d\n\tjnz %%c, @l, @e\n@l\n' % turns \
        + ''.join(summed) + '\t%%n =q qaddw %%t%d, %%y\n\t%%j =w add %%i, 1\n\tjmp @h\n@e\n\t%%r =q qaddw %%%s, %%%s\n\tqstore %%r, %%d' % (rotate - 1, names[0], names[-1])
    a, b = [1, 2, 3, 4], [10, 20, 30, 40]
    got = struct.unpack('<4i', run(body, struct.pack('<4i', *a), struct.pack('<4i', *b)))
    want = []
    for x, y in zip(a, b):
        ps = [x] * rotate
        for _ in range(turns):
            ps = [wrap('w', y + sum(ps) + y)] + ps[:-1]
        want.append(wrap('w', ps[0] + ps[-1]))
    assert list(got) == want


@pytest.mark.parametrize('live', [1, 3, 12])
def test_live_across_a_call(live):
    """A call keeps at most the low 64 bits of an FP register. The callee here wipes all of v0 to v31."""
    # every register in use at once, and all of them kept by storing them: nothing here can be deleted
    n = 36
    wipe = 'data $sink = align 16 { z %d }\nexport function $wipe() {\n@s\n\t%%v =l copy -1\n\t%%z =q qdupl %%v\n' % (16 * n)
    wipe += ''.join('\t%%z%d =q qaddw %%%s, %%z\n' % (k, 'z' if k == 0 else 'z%d' % (k - 1)) for k in range(n))
    wipe += ''.join('\t%%p%d =l add $sink, %d\n\tqstore %%z%d, %%p%d\n' % (k, 16 * k, k, k) for k in range(n))
    wipe += '\tret\n}\n'
    lines = ['\t%%x%d =q qaddw %%%s, %%y' % (n, 'x' if n == 0 else 'x%d' % (n - 1)) for n in range(live)]
    lines = LOADS.rstrip('\n').split('\n') + lines + ['\tcall $wipe()']
    lines += ['\t%s0 =q qaddw %x0, %y'] + ['\t%%s%d =q qaddw %%s%d, %%x%d' % (n, n - 1, n) for n in range(1, live)]
    lines += ['\tqstore %%s%d, %%d' % (live - 1)]
    a, b = [1, 2, 3, 4], [10, 20, 30, 40]
    got = struct.unpack('<4i', run('\n'.join(lines), struct.pack('<4i', *a), struct.pack('<4i', *b), wipe))
    want = []
    for p, y in zip(a, b):
        xs = [p + y * (n + 1) for n in range(live)]
        want.append(xs[0] + y + sum(xs[1:]))
    assert list(got) == want


def test_callee_saved_low_halves_stay_kept():
    """A function that uses q registers under pressure still keeps d8 to d15 for its caller."""
    many = ''.join('\t%%v%d =q qaddw %%%s, %%x\n' % (n, 'x' if n == 0 else 'v%d' % (n - 1)) for n in range(32))
    use = '\t%r =q qaddw %v0, %v1\n' + ''.join('\t%%r%d =q qaddw %%%s, %%v%d\n' % (n, 'r' if n == 2 else 'r%d' % (n - 1), n) for n in range(2, 32))
    callee = 'function $c(l %%a) {\n@s\n\t%%x =q qload %%a\n%s%s\tqstore %%r31, %%a\n\tret\n}\n' % (many, use)
    caller = '''	%v =d loadd %a
	%x1 =d add %v, d_1
	call $c(l %b)
	%y1 =d add %x1, d_1
	stored %y1, %d'''
    d = run(caller, struct.pack('<2d', 1.0, 0.0), bytes(16), callee)
    assert struct.unpack('<d', d[:8])[0] == 3.0


@pytest.mark.skipif(not _fma_here(), reason='an x86-64 CPU without FMA')
def test_text_form():
    from cimo.backend import ext, parse
    import io
    ssa = 'export function $f(l %a, l %d) {\n@s\n\t%x =q qload %a\n\t%r =q qfma1s %x, %x, %x\n\tqstore %r, %d\n\tret\n}\n'
    assert re.search(r'fmla|vfmadd231p', driver._qbe(ssa))


@pytest.mark.parametrize('x', 'sd')
@pytest.mark.parametrize('op', ['qmin', 'qmax'])
def test_float_min_max_is_fminnm(op, x):
    """A number beats a quiet NaN whichever side it's on, and -0 is below +0 whichever comes first."""
    nan, big = float('nan'), 5.0
    cases = [(nan, 1.5), (1.5, nan), (nan, nan), (-0.0, 0.0), (0.0, -0.0), (1.0, big), (big, 1.0), (-big, -1.0)]
    for n in range(0, len(cases), COUNT[x]):
        part = cases[n:n + COUNT[x]]
        part += [(1.0, 2.0)] * (COUNT[x] - len(part))
        body = LOADS + '\t%%r =q %s%s %%x, %%y\n\tqstore %%r, %%d' % (op, x)
        got = unpack(x, run(body, pack(x, [p[0] for p in part]), pack(x, [p[1] for p in part])))
        for (a, b), r in zip(part, got):
            if a != a or b != b:
                want = b if a != a else a
            elif a == b:
                want = (-0.0 if op == 'qmin' else 0.0) if a == 0 else a
            else:
                want = min(a, b) if op == 'qmin' else max(a, b)
            assert (r != r and want != want) or (r == want and math.copysign(1, r) == math.copysign(1, want)), (a, b, r)


@pytest.mark.parametrize('x,k', [('b', 'w'), ('h', 'w'), ('w', 'w'), ('l', 'l')])
def test_get_every_integer_lane(x, k):
    vals = [wrap(x, (n + 1) * 0x0123456789abcdef) for n in range(COUNT[x])]
    bits = BITS[x]
    for n in range(COUNT[x]):
        store = 'storel' if k == 'l' else 'storew'
        body = '\t%%x =q qload %%a\n\t%%r =%s qget%s %%x, %d\n\t%s %%r, %%d' % (k, x, n, store)
        got = int.from_bytes(run(body, pack(x, vals), bytes(16))[:8 if k == 'l' else 4], 'little')
        assert got == vals[n] & ((1 << bits) - 1)  # zero-extended


@pytest.mark.parametrize('x', 'sd')
def test_get_every_float_lane(x):
    vals = [1.5 + n for n in range(COUNT[x])]
    for n in range(COUNT[x]):
        body = '\t%%x =q qload %%a\n\t%%r =%s qget%s %%x, %d\n\tstore%s %%r, %%d' % (x, x, n, x)
        got = struct.unpack('<' + FMT[x], run(body, pack(x, vals), bytes(16))[:struct.calcsize(FMT[x])])[0]
        assert got == vals[n]
