"""M5: cimo's bundled runtime library (python/cimo/runtime) and how modules link with it.

The C test programs in tests/runtime/t_*.c each export `long run(char *out, long cap)`, which
writes results as text. Each is built three ways and the outputs compared:

- cimo:  cimo.load(), i.e. cimo's runtime compiled by cimo;
- musl:  gcc -O2 against the same runtime sources (musl as gcc builds it): must be identical,
         so this checks the compiler on the runtime as much as the runtime;
- glibc: gcc against the host libc: identical where the C standard pins the result (string,
         ctype, strto*, qsort, printf except documented implementation-defined cases), within
         a few ulps for math.

Plus: printf ordering with Python's output and capture by redirect_stdout, the math builtins
(single instructions, exact C semantics), link= and the unknown-symbol error, and a check that
the default build resolves nothing from the process.
"""

import contextlib
import ctypes
import io
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys

import pytest

import cimo
from cimo import linker

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.join(HERE, 'runtime')
CIMO_PKG = os.path.dirname(os.path.abspath(cimo.__file__))
CC = os.environ.get('CC', 'cc')
LINUX_ARM64 = sys.platform.startswith('linux') and platform.machine().lower() in ('aarch64', 'arm64')

pytestmark = pytest.mark.skipif(platform.machine().lower() not in ('arm64', 'aarch64'),
                                reason='arm64 only')
need_cc = pytest.mark.skipif(not (shutil.which(CC) and LINUX_ARM64),
                             reason='needs a C compiler on Linux arm64 for the references')

TESTS = ['t_string', 't_ctype', 't_stdlib', 't_printf', 't_math']
CAP = 8 << 20
# runtime files that need cimo's module glue (heap context, unwinding) or the stdout hook
NOT_IN_REF = ('heap.c', 'abort.c', os.path.join('stdio', 'stdout.c'),
              os.path.join('string', 'strdup.c'), os.path.join('string', 'strndup.c'))


def _run(f):
    buf = ctypes.create_string_buffer(CAP)
    n = f(buf, CAP)
    return buf.raw[:n].decode('latin-1')


@pytest.fixture(scope='module')
def cimo_out():
    from cimo import driver
    out = {}
    old = driver.FMA
    driver.FMA = False  # the test programs' own arithmetic (inputs) as gcc -ffp-contract=off
    try:
        for t in TESTS:
            m = cimo.load(os.path.join(RT, t + '.c'))
            out[t] = _run(m.run)
    finally:
        driver.FMA = old
    return out


@pytest.fixture(scope='module')
def refs(tmp_path_factory):
    """{'musl': {test: output}, 'glibc': {test: output}} from gcc builds."""
    tmp = tmp_path_factory.mktemp('rtref')
    inc = ['-nostdinc', '-isystem', os.path.join(CIMO_PKG, 'include')]
    # -fno-math-errno: else gcc's __builtin_sqrt calls sqrt for negative arguments, which in
    # the runtime's sqrt.c is itself
    common = ['-O2', '-fPIC', '-ffp-contract=off', '-fno-builtin', '-fno-math-errno', '-w',
              '-std=gnu11', '-U_FORTIFY_SOURCE']
    objs = []
    for rel in linker.runtime_sources():
        if rel in NOT_IN_REF:
            continue
        o = str(tmp / (rel.replace(os.sep, '_') + '.o'))
        objs.append((os.path.join(linker.RUNTIME_DIR, rel), o))
    # one gcc per directory keeps this to a few seconds
    for src, o in objs:
        subprocess.run([CC, '-c'] + common + inc + ['-I', linker.RUNTIME_INCLUDE, src, '-o', o],
                       check=True)
    lib = str(tmp / 'libcimort.a')
    subprocess.run(['ar', 'rcs', lib] + [o for _, o in objs], check=True)
    out = {'musl': {}, 'glibc': {}}
    for t in TESTS:
        src = os.path.join(RT, t + '.c')
        so = str(tmp / (t + '-musl.so'))
        subprocess.run([CC, '-shared'] + common + inc + ['-I', RT, src, lib, '-o', so,
                                                         '-Wl,-Bsymbolic'], check=True)
        out['musl'][t] = _run(ctypes.CDLL(so).run)
        so = str(tmp / (t + '-glibc.so'))
        subprocess.run([CC, '-shared'] + common + ['-I', RT, src, '-o', so, '-lm'], check=True)
        out['glibc'][t] = _run(ctypes.CDLL(so).run)
    return out


# ---------------------------------------------------------------------------------------------
# differential tests

def _diff(a, b, n=8):
    la, lb = a.splitlines(), b.splitlines()
    d = [(i, x, y) for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    if len(la) != len(lb):
        d.append(('lines', len(la), len(lb)))
    return d[:n]


@need_cc
@pytest.mark.parametrize('t', TESTS)
def test_same_as_musl_built_by_gcc(t, cimo_out, refs):
    assert len(cimo_out[t]) > 1000
    if cimo_out[t] != refs['musl'][t]:  # (not assert ==: pytest would diff megabytes)
        pytest.fail(repr(_diff(cimo_out[t], refs['musl'][t])))


@need_cc
@pytest.mark.parametrize('t', ['t_string', 't_ctype', 't_stdlib'])
def test_same_as_glibc(t, cimo_out, refs):
    if cimo_out[t] != refs['glibc'][t]:
        pytest.fail(repr(_diff(cimo_out[t], refs['glibc'][t])))


def _printf_known_difference(ours, theirs):
    """Formats where C leaves the output to the implementation and musl and glibc differ."""
    fmt = ours.split(' => ')[0]
    if fmt in ('%5%',):  # undefined: a conversion spec with flags around %
        return True
    if fmt in ('[%ls] [%lc] [%5ls]',):  # wide characters: glibc converts by the locale
        return True
    if '#' in fmt and fmt[-1] in 'gG' and ours.rstrip().endswith(('e+03', 'E+03', 'e+05', 'E+05')):
        # musl 1.2.5 bug: %#.Ng drops the trailing zeros when rounding carries into a new
        # digit (999.9995 -> "1.e+03", glibc and C: "1.00e+03"); gcc-built musl does the same
        return True
    if 'a' in fmt.lower() and re.search(r'%[-+ #0-9.]*[aA]', fmt):
        # %a: the leading digit of a hex float is unspecified for .0 precision rounding
        # (glibc may print 0x2p+0 where musl prints 0x1p+1)
        return True
    return False


@need_cc
def test_printf_matches_glibc(cimo_out, refs):
    ours, theirs = cimo_out['t_printf'].splitlines(), refs['glibc']['t_printf'].splitlines()
    assert len(ours) == len(theirs) > 20000
    bad = [(a, b) for a, b in zip(ours, theirs) if a != b and not _printf_known_difference(a, b)]
    if bad:
        pytest.fail('%d differences, first: %r' % (len(bad), bad[:10]))


def _ulps(a, b, single):
    """Distance in units in the last place between two results given as bit patterns."""
    fmt, ifmt, bits = ('<f', '<i', 32) if single else ('<d', '<q', 64)
    x = struct.unpack(fmt, struct.pack('<Q' if not single else '<I', a))[0]
    y = struct.unpack(fmt, struct.pack('<Q' if not single else '<I', b))[0]
    if math.isnan(x) or math.isnan(y):
        return 0 if math.isnan(x) and math.isnan(y) else float('inf')
    if x == y:
        return 0

    def ordered(v):
        i = struct.unpack(ifmt, struct.pack(fmt, v))[0]
        return i if i >= 0 else -(1 << (bits - 1)) - i
    return abs(ordered(x) - ordered(y))


# Functions the C standard (Annex F) pins exactly, or that musl and glibc both compute
# correctly rounded: must agree bit for bit (NaN payloads aside).
EXACT = set('''fabs sqrt ceil floor nearbyint rint round trunc fmod remainder copysign nextafter
fdim fmax fmin fma frexp modf ldexp scalbn remquo lrint ilogb logb scalbln'''.split())
# Allowed ulps between musl and glibc otherwise (sum of both libraries' documented errors,
# rounded up); lgamma/tgamma near poles and zeros of the gamma function are worse.
ULPS = {'lgamma': 1 << 20, 'tgamma': 16, 'erfc': 8, 'erf': 4, 'acosh': 4, 'asinh': 4,
        'atanh': 4, 'cbrt': 4, 'pow': 4, 'expm1': 4, 'log1p': 4, 'sinh': 4, 'cosh': 4,
        'tanh': 4, 'hypot': 2}


@need_cc
def test_math_matches_glibc_within_ulps(cimo_out, refs):
    ours = cimo_out['t_math'].splitlines()
    theirs = refs['glibc']['t_math'].splitlines()
    assert len(ours) == len(theirs) > 20000
    bad = []
    for a, b in zip(ours, theirs):
        if a == b:
            continue
        fa, fb = a.split(), b.split()
        name = fa[0]
        base = name[:-1] if name.endswith('f') and name not in ('modf', 'erf') else name
        single = base != name or name in ('modff', 'erff')
        if name in ('erff', 'modff'):
            base = name[:-1]
        if name in ('lrint', 'lrintf', 'ilogb', 'nan'):
            bad.append((a, b))  # integer results: exact
            continue
        for x, y in zip(fa[2:], fb[2:]):
            if len(x) > 8 or len(y) > 8 or single:
                u = _ulps(int(x, 16), int(y, 16), single)
            else:
                u = abs(int(x, 16) - int(y, 16))
            limit = 0 if base in EXACT else ULPS.get(base, 2)
            if u > limit:
                bad.append((a, b, u))
    if bad:
        pytest.fail('%d differences, first: %r' % (len(bad), bad[:10]))


# ---------------------------------------------------------------------------------------------
# stdout

def test_printf_order_and_redirect():
    m = cimo.compile(r'''
#include <stdio.h>
void hello(int n) { printf("hello %d\n", n); }
void partial(void) { printf("no newline;"); putchar('x'); }
void many(int n) { for (int i = 0; i < n; i++) printf("%d,", i); puts("end"); }
void chars(void) { putchar('a'); putchar('\n'); puts("p"); printf("%s", ""); }
''')
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        print('py 1')
        m.hello(1)
        print('py 2')
        m.partial()       # no newline: still flushed when the kernel returns
        print('py 3')
        m.many(3000)      # more than the 4 KB buffer
        m.chars()
        m.hello(2)
    big = ','.join(str(i) for i in range(3000)) + ',end\n'
    assert f.getvalue() == ('py 1\nhello 1\npy 2\nno newline;xpy 3\n' + big +
                            'a\np\nhello 2\n')


def test_printf_utf8_and_return_values():
    m = cimo.compile(r'''
#include <stdio.h>
int p(void) { return printf("%s|%5.1f|%x\n", "\xc3\xa9t\xc3\xa9", 2.25, 255); }
int q(void) { return puts("done"); }
''')
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        assert m.p() == len('été|  2.2|ff\n'.encode())
        assert m.q() >= 0
    assert f.getvalue() == 'été|  2.2|ff\ndone\n'


def test_printf_from_threads():
    """Concurrent printing must not crash or corrupt memory (lines may interleave)."""
    import threading
    m = cimo.compile('#include <stdio.h>\nvoid p(int t) { for (int i = 0; i < 200; i++) '
                     'printf("t%d %d\\n", t, i); }')
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        ts = [threading.Thread(target=m.p, args=(t,)) for t in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert f.getvalue().count('\n') >= 1


# ---------------------------------------------------------------------------------------------
# math builtins

BUILTIN_SRC = r'''
#include <math.h>
double b_sqrt(double x) { return sqrt(x); }
double b_fabs(double x) { return fabs(x); }
double b_floor(double x) { return floor(x); }
double b_ceil(double x) { return ceil(x); }
double b_trunc(double x) { return trunc(x); }
double b_round(double x) { return round(x); }
double b_fmin(double x, double y) { return fmin(x, y); }
double b_fmax(double x, double y) { return fmax(x, y); }
double b_fma(double x, double y, double z) { return fma(x, y, z); }
float b_sqrtf(float x) { return sqrtf(x); }
float b_roundf(float x) { return roundf(x); }
float b_fminf(float x, float y) { return fminf(x, y); }
float b_floorf(float x) { return floorf(x); }
/* through a function pointer: the runtime's out-of-line definitions */
static double (*volatile p1[])(double) = { sqrt, fabs, floor, ceil, trunc, round };
static double (*volatile p2[])(double, double) = { fmin, fmax };
double o1(int k, double x) { return p1[k](x); }
double o2(int k, double x, double y) { return p2[k](x, y); }
'''

SPECIALS = [0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.49999999999999994, -0.49999999999999994,
            4503599627370495.5, -4503599627370495.5, 1e300, -1e-300, 5e-324, 3.7, -3.7, 1e16 + 1,
            float('inf'), float('-inf'), float('nan'), 2.0, 1e-310]


def _bits(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def _c_round(x):
    """C round(): halfway cases away from zero."""
    if math.isnan(x) or math.isinf(x) or x == 0:
        return x
    r = math.floor(abs(x) + 0.5) if abs(x) < 2 ** 52 else abs(x)
    if abs(x) < 2 ** 52 and r - abs(x) == 0.5 and (abs(x) + 0.5) != r:
        pass
    # exact: floor(|x| + 0.5) can round up in the addition for 0.49999999999999994
    if abs(x) < 2 ** 52:
        t = math.floor(abs(x))
        r = t + 1 if abs(x) - t >= 0.5 else t
    return math.copysign(r, x)


def _c_fmin(x, y, mx=False):
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    if x == y == 0:
        neg = (math.copysign(1, x) < 0, math.copysign(1, y) < 0)
        return (0.0 if not all(neg) else -0.0) if mx else (-0.0 if any(neg) else 0.0)
    return max(x, y) if mx else min(x, y)


def test_math_builtins_are_instructions():
    from cimo import driver
    text, _ = driver._build(BUILTIN_SRC)
    body = {}
    for name in ('b_sqrt', 'b_fabs', 'b_floor', 'b_ceil', 'b_trunc', 'b_round', 'b_fmin',
                 'b_fmax', 'b_fma', 'b_sqrtf', 'b_roundf', 'b_fminf', 'b_floorf'):
        m = re.search(r'^%s:\n(.*?)\n\s*ret\n' % name, text, re.S | re.M)
        assert m, name
        body[name] = m.group(1)
        assert 'bl\t' not in body[name] and 'blr' not in body[name], (name, body[name])
    for name, insn in (('b_sqrt', 'fsqrt\td'), ('b_fabs', 'fabs\td'), ('b_floor', 'frintm\td'),
                       ('b_ceil', 'frintp\td'), ('b_trunc', 'frintz\td'),
                       ('b_round', 'frinta\td'), ('b_fmin', 'fminnm\td'),
                       ('b_fmax', 'fmaxnm\td'), ('b_fma', 'fmadd\td'), ('b_sqrtf', 'fsqrt\ts'),
                       ('b_roundf', 'frinta\ts'), ('b_fminf', 'fminnm\ts'),
                       ('b_floorf', 'frintm\ts')):
        assert insn in body[name], (name, body[name])


def test_math_builtins_semantics():
    m = cimo.compile(BUILTIN_SRC)
    ops = [('sqrt', m.b_sqrt, lambda x: math.sqrt(x) if x >= 0 or math.isnan(x) else float('nan'), 0),
           ('fabs', m.b_fabs, abs, 1), ('floor', m.b_floor, lambda x: x if not math.isfinite(x) else math.copysign(math.floor(x), x), 2),
           ('ceil', m.b_ceil, lambda x: x if not math.isfinite(x) else math.copysign(math.ceil(x), x), 3),
           ('trunc', m.b_trunc, lambda x: x if not math.isfinite(x) else math.copysign(math.trunc(x), x), 4),
           ('round', m.b_round, _c_round, 5)]
    for name, f, ref, k in ops:
        for x in SPECIALS + [-s for s in SPECIALS]:
            want = ref(x)
            for got in (f(x), m.o1(k, x)):
                if math.isnan(want):
                    assert math.isnan(got), (name, x, got)
                else:
                    assert _bits(got) == _bits(want), (name, x, got, want)
    vals = [0.0, -0.0, 1.0, -1.0, float('nan'), float('inf'), float('-inf'), 5e-324]
    for x in vals:
        for y in vals:
            for k, (f, mx) in enumerate(((m.b_fmin, False), (m.b_fmax, True))):
                want = _c_fmin(x, y, mx)
                for got in (f(x, y), m.o2(k, x, y)):
                    if math.isnan(want):
                        assert math.isnan(got)
                    else:
                        assert _bits(got) == _bits(want), (k, x, y, got, want)
    assert m.b_fma(0.1, 10.0, -1.0) == 5.551115123125783e-17  # fused: one rounding
    assert m.b_roundf(2.5) == 3.0 and m.b_roundf(-0.5) == -1.0
    assert m.b_fminf(float('nan'), 2.0) == 2.0
    assert m.b_sqrtf(2.0) == struct.unpack('<f', struct.pack('<f', math.sqrt(2)))[0]


def test_user_function_named_like_a_builtin():
    """A static function (or one with another prototype) is not the library function."""
    m = cimo.compile('static double sqrt(double x) { return x + 1; }\n'
                     'double f(double x) { return sqrt(x); }\n')
    assert m.f(4.0) == 5.0


def test_math_in_loops_is_optimized():
    """sqrt in a loop no longer blocks the middle end (nbody_acc's inner loop)."""
    from cimo import driver
    text, _ = driver._build(r'''
#include <math.h>
void f(int n, const double *restrict x, double *restrict y) {
    for (int i = 0; i < n; i++) y[i] = 1.0 / sqrt(x[i] * x[i] + 1.0);
}''')
    assert text.count('fsqrt') >= 4  # unrolled by 4
    assert '\tbl\t' not in re.split(r'^f:$', text, flags=re.M)[1]


# ---------------------------------------------------------------------------------------------
# linking

def test_default_build_resolves_nothing_from_the_process(monkeypatch):
    """A module using the whole runtime links with no symbol left for the host: the merged
    object has no undefined symbols at all, and the loader's dlsym is never asked."""
    from cimo import loader

    def no_dlsym(name):
        raise AssertionError('looked up %s in the process' % name)
    monkeypatch.setattr(loader, '_dlsym', no_dlsym)
    seen = []
    real_link = linker.link

    def spy(obj, hooks=None, libs=()):
        merged, resolve = real_link(obj, hooks, libs)
        seen.append(merged)
        return merged, resolve
    monkeypatch.setattr(linker, 'link', spy)
    m = cimo.compile(r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>
static int cmp(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }
double all(int n) {
    char buf[64];
    int *v = malloc(n * sizeof *v);
    for (int i = 0; i < n; i++) v[i] = (i * 7919) % n;
    qsort(v, n, sizeof *v, cmp);
    snprintf(buf, sizeof buf, "%d %s %g", v[n - 1], "x", sin(1.0));
    double r = strtod("2.5", 0) + strlen(buf) + toupper('a') + atoi("12") + pow(2, 0.5)
               + exp(1) + log(2) + tgamma(4.5) + atan2(1, 2) + fmod(7, 3) + cbrt(27);
    free(v);
    assert(r > 0);
    printf("%s\n", buf);
    return r;
}''')
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        r = m.all(100)
    assert f.getvalue().startswith('99 x 0.841471\n')
    assert r > 0
    merged = seen[-1]
    undefined = [y.name for y in merged.symbols.values() if not y.defined and not y.common]
    assert undefined == []


def test_unknown_symbol_is_a_compile_error():
    src = 'int getpid(void);\nint f(void) { return getpid(); }\n'
    with pytest.raises(cimo.CompileError) as e:
        cimo.compile(src)
    msg = str(e.value)
    assert "'getpid'" in msg and 'link=' in msg
    m = cimo.compile(src, link=['libc'])
    assert m.f() == os.getpid()


def test_link_library_by_soname_and_path():
    src = 'unsigned long zlibCompileFlags(void);\nunsigned long f(void) { return zlibCompileFlags(); }\n'
    import ctypes.util
    so = ctypes.util.find_library('z')
    if not so:
        pytest.skip('no zlib')
    z = ctypes.CDLL(so)
    z.zlibCompileFlags.restype = ctypes.c_ulong
    for spec in (so, 'z'):
        assert cimo.compile(src, link=[spec]).f() == z.zlibCompileFlags()
    with pytest.raises(cimo.CompileError) as e:
        cimo.compile(src, link=['libc'])
    assert 'zlibCompileFlags' in str(e.value)
    with pytest.raises(cimo.CompileError):
        cimo.compile(src, link=['libdoesnotexist.so.9'])


def test_runtime_wins_over_linked_libc():
    """With link=['libc'], functions the runtime provides still come from the runtime (malloc
    from the module heap, printf to sys.stdout); the rest from libc."""
    m = cimo.compile(r'''
#include <stdio.h>
int getpid(void);
int f(void) { return printf("pid %d\n", getpid()); }''', link=['libc'])
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        m.f()
    assert f.getvalue() == 'pid %d\n' % os.getpid()


@pytest.mark.skipif(not LINUX_ARM64 or not os.path.exists('/usr/include/unistd.h'),
                    reason='needs glibc headers')
def test_system_headers():
    m = cimo.compile(r'''
#include <unistd.h>
#include <string.h>
long f(void) { return getpid() + strlen("abc"); }''', link=['libc'], system_headers=True)
    assert m.f() == os.getpid() + 3


def test_user_definition_beats_runtime_and_clash_is_reported():
    m = cimo.compile('#include <string.h>\nsize_t strlen(const char *s) { return 42; }\n'
                     'long f(void) { return strlen("x"); }\n')
    assert m.f() == 42
    # defining a function the runtime also defines in an object that gets pulled in
    # defining puts and calling printf: printf's object (runtime/stdio/stdout.c) defines both
    with pytest.raises(cimo.CompileError) as e:
        cimo.compile('#include <stdio.h>\nint puts(const char *s) { return 0; }\n'
                     'int f(void) { return printf("x"); }\n')
    assert "'puts'" in str(e.value) and 'stdout.c' in str(e.value)


def test_runtime_build_is_cached():
    rt = linker.runtime()
    assert len(rt.index) == len(linker.runtime_sources())
    assert linker.runtime() is rt
    # every exported C library function of the headers is provided
    names = set(s for e in rt.index.values() for s in e['defs'])
    pre = '_' if sys.platform == 'darwin' else ''
    for hdr in ('string.h', 'stdlib.h', 'ctype.h', 'stdio.h', 'math.h'):
        with open(os.path.join(CIMO_PKG, 'include', hdr)) as f:
            text = f.read()
        for fn in re.findall(r'^[a-z_][\w \*]*?\b(\w+)\(', text, re.M):
            if fn.startswith('__cimo'):
                continue
            assert pre + fn in names, (hdr, fn)
