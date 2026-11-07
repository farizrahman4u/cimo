"""Calling convention against a reference compiler, with random signatures (abi_gen.py): gcc on
Linux, clang (Visual Studio's, or $CIMO_WIN_CLANG) on Windows arm64, MinGW-w64 gcc ($CIMO_WIN_GCC)
on Windows x64, Apple clang (Xcode or the Command Line Tools) on macOS. For each signature, cimo calling
the reference, the reference calling cimo, and cimo calling itself must give what the reference
gives calling itself.

    python tests/test_abi.py [SEED [COUNT]]      # one more seed, from the command line
"""

import ctypes
import os
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run as a script

import abi_gen  # noqa: E402
import cimo  # noqa: E402
from cimo import driver  # noqa: E402

SEEDS = [1, 2, 3, 4]
COUNT = 60


def reference():
    """(compiler command, shared library suffix), or None."""
    if driver.TARGET == 'amd64_windows':
        # MinGW-w64 gcc follows the Windows x64 convention (and LLP64, like MSVC)
        cc = os.environ.get('CIMO_WIN_GCC', 'gcc')
        return ([cc, '-O1', '-shared', '-w'], '.dll') if _have(cc) else None
    if sys.platform == 'win32':
        from test_windows import _clang
        c = _clang()
        if not c or driver.TARGET != 'arm64_win':
            return None
        return [c, '--target=aarch64-pc-windows-msvc', '-O1', '-shared', '-nostdlib',
                '-fuse-ld=lld', '-Wl,/noentry', '-w'], '.dll'
    if driver.TARGET in ('arm64', 'amd64_sysv') and _have('gcc'):
        return ['gcc', '-O1', '-shared', '-fPIC', '-w'], '.so'
    if driver.TARGET == 'arm64_apple' and _have('clang'):
        return ['clang', '-arch', 'arm64', '-O1', '-dynamiclib', '-w'], '.dylib'
    return None


def _have(cmd):
    try:
        subprocess.run([cmd, '--version'], capture_output=True)
        return True
    except OSError:
        return False


def check(seed, count, d):
    """Mismatches as (signature number, what, expected, got)."""
    ref_src, cimo_src = abi_gen.sources(seed, count, sys.platform == 'win32')
    cmd, ext = reference()
    src = os.path.join(d, 'ref%d.c' % seed)
    lib = os.path.join(d, 'ref%d%s' % (seed, ext))
    with open(src, 'w') as f:
        f.write(ref_src)
    subprocess.check_call(cmd + ['-o', lib, src])
    ref = ctypes.CDLL(lib)
    m = cimo.compile(cimo_src, link=[lib])
    bad = []
    for n in range(count):
        call = getattr(ref, 'ref_call%d' % n)
        call.restype = ctypes.c_uint64
        call.argtypes = [ctypes.c_void_p]
        want = call(ctypes.cast(getattr(ref, 'ref_f%d' % n), ctypes.c_void_p).value)
        for what in 'xyz':
            got = getattr(m, '%s%d' % (what, n))() & 0xffffffffffffffff
            if got != want:
                bad.append((n, {'x': 'cimo calls reference', 'y': 'reference calls cimo',
                                'z': 'cimo calls cimo'}[what], want, got))
    return bad


@pytest.mark.skipif(reference() is None, reason='needs gcc (Linux, Windows x64: MinGW), clang (Windows arm64) or Apple clang (macOS)')
@pytest.mark.parametrize('seed', SEEDS)
def test_random_signatures(seed):
    # in a child process: an ABI mismatch often crashes
    p = subprocess.run([sys.executable, os.path.abspath(__file__), str(seed), str(COUNT)],
                       capture_output=True, universal_newlines=True, timeout=600)
    assert p.returncode == 0 and ' 0 mismatches' in p.stdout, (
        'exit status %d (the C source: python tests/abi_gen.py %d %d)\n%s%s'
        % (p.returncode, seed, COUNT, p.stdout, p.stderr[-2000:]))


# packed structs by value: (declaration of struct s, initializer, expression over v). QBE aligns
# every field, so members at offsets it wouldn't choose must not change the struct's size; 7 and 15
# bytes are the sizes where that took more registers, or a pointer, instead of what C's size takes.
PACKED = [
    ('struct __attribute__((packed)) s { char a; int b; short c; };', '{1, 2, 3}', 'v.a + v.b * 10 + v.c * 100'),
    ('struct __attribute__((packed)) s { char a; short b; };', '{1, 2}', 'v.a + v.b * 10'),
    ('struct __attribute__((packed)) s { char a; int b; };', '{1, 2}', 'v.a + v.b * 10'),
    ('struct __attribute__((packed)) s { short a; int b; char c; };', '{1, 2, 3}', 'v.a + v.b * 10 + v.c * 100'),
    ('struct __attribute__((packed)) s { char a; long long b; short c; int d; };', '{1, 2, 3, 4}',
     'v.a + v.b * 10 + v.c * 100 + v.d * 1000'),
    ('struct __attribute__((packed)) s { char a; long long b; long long c; };', '{1, 2, 3}',
     'v.a + v.b * 10 + v.c * 100'),
    ('struct __attribute__((packed)) s { char a; int b[2]; short c; };', '{1, {2, 3}, 4}',
     'v.a + v.b[1] * 10 + v.c * 100'),
    ('struct __attribute__((packed)) i { int x; }; struct s { char a; struct __attribute__((packed)) { int x; short y; } b; };',
     '{1, {2, 3}}', 'v.a + v.b.x * 10 + v.b.y * 100'),
    ('union __attribute__((packed)) u { char c; int i; }; struct __attribute__((packed)) s { char a; union u b; short c; };',
     '{1, {.i = 2}, 3}', 'v.a + v.b.i * 10 + v.c * 100'),
    ('struct __attribute__((packed)) s { float a; float b; };', '{1, 2}', 'v.a + v.b * 10'),
    ('struct __attribute__((packed)) s { char p; float a; float b; };', '{1, 2, 3}', 'v.p + v.a * 10 + v.b * 100'),
    ('struct __attribute__((packed)) s { char p; double a; };', '{1, 2}', 'v.p + v.a * 10'),
    # after the body, which `aligned(N)` on the struct itself doesn't change how it is passed on Linux
    ('struct s { char a; int b; short c; } __attribute__((packed));', '{1, 2, 3}', 'v.a + v.b * 10 + v.c * 100'),
    ('struct s { long long a; } __attribute__((aligned(16)));', '{7}', 'v.a'),
    ('struct s { long long a; long long b; } __attribute__((aligned(16)));', '{7, 8}', 'v.a + v.b * 10'),
    ('struct s { char a; long long b; } __attribute__((packed, aligned(16)));', '{7, 8}', 'v.a + v.b * 10'),
    ('struct s { float a; } __attribute__((aligned(16)));', '{7}', 'v.a'),
    ('struct s { _Alignas(16) long long a; } __attribute__((aligned(16)));', '{7}', 'v.a'),
    ('struct s { _Alignas(8) char a; char b; } __attribute__((packed));', '{1, 2}', 'v.a + v.b * 10'),
]


def check_packed(d):
    """As check(), for PACKED; each struct gets its own translation of `s`."""
    tests = []
    for n, (decl, init, expr) in enumerate(PACKED):
        decl = re.sub(r'\b(s|i|u)\b(?= \{| b;)', lambda m: '%s%d' % (m.group(1), n), decl)
        decl = re.sub(r'\bunion u\b', 'union u%d' % n, decl)
        tests.append(decl + '''
API double @f%(n)d(int p, struct s%(n)d v, int q) { return (%(expr)s) + p * 1e6 + q * 1e7; }
API double @call%(n)d(double (*f)(int, struct s%(n)d, int)) { struct s%(n)d v = %(init)s; return f(5, v, 6); }
''' % {'n': n, 'expr': expr, 'init': init})
    code = abi_gen.HEADER + '\n'.join(tests)
    glue = '\n'.join('''double ref_f%(n)d(int, struct s%(n)d, int);
double ref_call%(n)d(double (*)(int, struct s%(n)d, int));
double x%(n)d(void) { return call%(n)d(ref_f%(n)d); }
double y%(n)d(void) { return ref_call%(n)d(f%(n)d); }
double z%(n)d(void) { return call%(n)d(f%(n)d); }''' % {'n': n} for n in range(len(PACKED)))
    cmd, ext = reference()
    src = os.path.join(d, 'packed.c')
    lib = os.path.join(d, 'packed' + ext)
    with open(src, 'w') as f:
        f.write(code.replace('@', 'ref_'))
    subprocess.check_call(cmd + ['-o', lib, src])
    ref = ctypes.CDLL(lib)
    m = cimo.compile(code.replace('@', '') + glue, link=[lib])
    bad = []
    for n in range(len(PACKED)):
        call = getattr(ref, 'ref_call%d' % n)
        call.restype = ctypes.c_double
        call.argtypes = [ctypes.c_void_p]
        want = call(ctypes.cast(getattr(ref, 'ref_f%d' % n), ctypes.c_void_p).value)
        for what in 'xyz':
            got = getattr(m, '%s%d' % (what, n))()
            if got != want:
                bad.append((n, {'x': 'cimo calls reference', 'y': 'reference calls cimo',
                                'z': 'cimo calls cimo'}[what], want, got))
    return bad


@pytest.mark.skipif(reference() is None, reason='needs gcc (Linux, Windows x64: MinGW), clang (Windows arm64) or Apple clang (macOS)')
def test_packed_structs_by_value():
    p = subprocess.run([sys.executable, os.path.abspath(__file__), 'packed'],
                       capture_output=True, universal_newlines=True, timeout=600)
    assert p.returncode == 0 and ' 0 mismatches' in p.stdout, (
        'exit status %d\n%s%s' % (p.returncode, p.stdout, p.stderr[-2000:]))


if __name__ == '__main__':
    import shutil
    import tempfile
    if sys.argv[1:] == ['packed']:
        d = tempfile.mkdtemp()
        try:
            bad = check_packed(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        for n, what, w, g in bad:
            print('%s: %s: want %r, got %r' % (PACKED[n][0], what, w, g))
        print('packed: %d structs, %d mismatches' % (len(PACKED), len(bad)))
        sys.exit(1 if bad else 0)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    count = int(sys.argv[2]) if len(sys.argv) > 2 else COUNT
    d = tempfile.mkdtemp()
    try:
        bad = check(seed, count, d)
    finally:
        shutil.rmtree(d, ignore_errors=True)  # Windows: the loaded DLL cannot be deleted
    for n, what, w, g in bad:
        print('f%d: %s: want %#x, got %#x' % (n, what, w, g))
    print('seed %d: %d signatures, %d mismatches' % (seed, count, len(bad)))
    sys.exit(1 if bad else 0)


# GNU vector types by value (AAPCS64: 8 and 16 bytes in one V register; more than 8 of them go on the stack), against
# the reference compiler in both directions. arm64 only: x86-64 still passes them as structs of their lanes.
VECTORS = r'''
typedef unsigned long long u64;
typedef float v4f __attribute__((vector_size(16)));
typedef double v2d __attribute__((vector_size(16)));
typedef int v4i __attribute__((vector_size(16)));
typedef signed char v16b __attribute__((vector_size(16)));
typedef short v4h __attribute__((vector_size(8)));
typedef float v2f __attribute__((vector_size(8)));
struct hfa { float a, b; };
#define H(v) h = (h ^ hv(&v, sizeof v)) * 0x100000001b3ULL
static u64 hv(const void *p, int n) { u64 h = 1469598103934665603ULL; const unsigned char *c = p;
  for (int i = 0; i < n; i++) h = (h ^ c[i]) * 0x100000001b3ULL; return h; }
P v4f f0(v4f a, int k, v2d b, v4i c, double d, v16b e, v4h f, struct hfa g, v2f x) {
  u64 h = 0; H(a); H(k); H(b); H(c); H(d); H(e); H(f); H(g); H(x); v4f r = { (float)(h & 0xffff), 2, 3, (float)(h >> 48) }; return r; }
P v2d f1(v4f a0, v4f a1, v4f a2, v4f a3, v4f a4, v4f a5, v4f a6, v4f a7, v4f a8, double z, v2f a9, int k, v4i a10) {
  u64 h = 0; H(a0); H(a1); H(a2); H(a3); H(a4); H(a5); H(a6); H(a7); H(a8); H(z); H(a9); H(k); H(a10);
  v2d r = { (double)(h & 0xffffffff), (double)(h >> 32) }; return r; }
P v4h f2(v4h a, v16b b) { u64 h = 0; H(a); H(b); v4h r = { (short)h, (short)(h >> 16), (short)(h >> 32), (short)(h >> 48) }; return r; }
P u64 c0(v4f (*f)(v4f, int, v2d, v4i, double, v16b, v4h, struct hfa, v2f)) {
  v4f a = {1.5f, -2, 3, 4}; v2d b = {5.25, -6}; v4i c = {7, -8, 9, 10}; v16b e = {1, 2, 3, 4, 5, 6, 7, 8, 9, -10, 11, 12, 13, 14, 15, 16};
  v4h q = {17, -18, 19, 20}; struct hfa g = {21, 22}; v2f x = {23, 24};
  v4f r = f(a, 99, b, c, 0.125, e, q, g, x); u64 h = 0; H(r); return h; }
P u64 c1(v2d (*f)(v4f, v4f, v4f, v4f, v4f, v4f, v4f, v4f, v4f, double, v2f, int, v4i)) {
  v4f a[9]; for (int i = 0; i < 9; i++) { v4f t = {i, i + 0.5f, -i, i * 3}; a[i] = t; } v2f a9 = {7, 8}; v4i a10 = {1, 2, 3, 4};
  v2d r = f(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], 2.5, a9, 42, a10); u64 h = 0; H(r); return h; }
P u64 c2(v4h (*f)(v4h, v16b)) { v4h a = {1, 2, 3, 4}; v16b b = {0}; b[3] = 9; v4h r = f(a, b); u64 h = 0; H(r); return h; }
'''


@pytest.mark.skipif(reference() is None or driver.TARGET not in ('arm64', 'arm64_apple', 'arm64_win'),
                    reason='needs the reference compiler, on arm64')
def test_vectors_by_value(tmp_path):
    cmd, ext = reference()
    pre = '#ifdef _MSC_VER\n#define P __declspec(dllexport)\n#else\n#define P\n#endif\n'
    src = tmp_path / 'vref.c'
    src.write_text(pre + VECTORS.replace('f0(', 'r_f0(').replace('f1(', 'r_f1(').replace('f2(', 'r_f2(')
                   .replace('c0(', 'r_c0(').replace('c1(', 'r_c1(').replace('c2(', 'r_c2('))
    lib = str(tmp_path / ('vref' + ext))
    subprocess.check_call(cmd + ['-o', lib, str(src)])
    ref = ctypes.CDLL(lib)
    m = cimo.compile('#define P\n' + VECTORS + ''.join(
        'u64 r_c%d(void *); u64 x%d(void) { return r_c%d((void *)f%d); }\n' % (n, n, n, n) for n in range(3)) + ''.join(
        'extern char r_f%d[]; u64 y%d(void) { return c%d((void *)r_f%d); }\n' % (n, n, n, n) for n in range(3)), link=[lib])
    for n in range(3):
        rc = getattr(ref, 'r_c%d' % n)
        rc.restype = ctypes.c_uint64
        rc.argtypes = [ctypes.c_void_p]
        want = rc(ctypes.cast(getattr(ref, 'r_f%d' % n), ctypes.c_void_p).value)
        assert getattr(m, 'x%d' % n)() == want, ('reference calls cimo', n)  # r_c(f): the reference calls cimo's f
        assert getattr(m, 'y%d' % n)() == want, ('cimo calls the reference', n)
        assert getattr(m, 'c%d' % n)(getattr(m, 'f%d' % n).address) == want, ('cimo calls cimo', n)


HALVES = r'''
typedef unsigned long long u64;
P _Float16 hf(_Float16 a, int k, __bf16 b, double d, _Float16 c) { return a * 2 + (float)b + (float)k + (float)(d * 0.5) + c; }
P __bf16 hb(__bf16 a, _Float16 b) { return (float)a - (float)b; }
P u64 hc(_Float16 (*f)(_Float16, int, __bf16, double, _Float16), __bf16 (*g)(__bf16, _Float16)) {
  _Float16 r = f(1.5f16, 3, 2.25bf16, 7.0, -0.75f16); __bf16 s = g(10.0bf16, 0.5f16);
  unsigned short u, v; __builtin_memcpy(&u, &r, 2); __builtin_memcpy(&v, &s, 2); return (u64)u << 16 | v; }
'''


@pytest.mark.skipif(reference() is None or sys.platform == 'win32', reason='needs gcc or Apple clang (Linux, macOS)')
def test_halves_by_value(tmp_path):
    """_Float16 and __bf16 in FP registers (h0 on arm64, the low word of xmm0 on x86-64), both directions."""
    cmd, ext = reference()
    pre = '#define P\n'
    src = tmp_path / 'href.c'
    src.write_text(pre + HALVES.replace('hf(', 'r_hf(').replace('hb(', 'r_hb(').replace('hc(', 'r_hc('))
    lib = str(tmp_path / ('href' + ext))
    try:
        subprocess.check_call(cmd + ['-o', lib, str(src)], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pytest.skip('the reference compiler has no _Float16 or __bf16')
    ref = ctypes.CDLL(lib)
    m = cimo.compile(pre + HALVES + 'u64 r_hc(void *, void *); u64 x(void) { return r_hc((void *)hf, (void *)hb); }\n'
                     'extern char r_hf[], r_hb[]; u64 y(void) { return hc((void *)r_hf, (void *)r_hb); }\n', link=[lib])
    rc = ref.r_hc
    rc.restype = ctypes.c_uint64
    rc.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    want = rc(ctypes.cast(ref.r_hf, ctypes.c_void_p).value, ctypes.cast(ref.r_hb, ctypes.c_void_p).value)
    assert m.x() == want, 'the reference calls cimo'
    assert m.y() == want, 'cimo calls the reference'
    assert m.hc(m.hf.address, m.hb.address) == want, 'cimo calls cimo'


INT128 = r'''
typedef unsigned long long u64;
P __int128 qf(int a, __int128 b, int c, unsigned __int128 d, int e, int f, int g, __int128 h, __int128 i) {
  return b * 3 + (__int128)d + a + c * 10 + e + f + g + (h >> 5) - i; }
P u64 qc(__int128 (*f)(int, __int128, int, unsigned __int128, int, int, int, __int128, __int128)) {
  __int128 r = f(1, ((__int128)7 << 70) + 5, 2, ((unsigned __int128)9 << 100) | 3, 4, 5, 6, (__int128)-1 << 80, 12345);
  return (u64)r ^ (u64)(r >> 64) * 31; }
'''


@pytest.mark.skipif(reference() is None or sys.platform == 'win32', reason='needs gcc or Apple clang (Linux, macOS)')
def test_int128_by_value(tmp_path):
    """__int128 in a register pair (AAPCS64: an even one), on the stack 16-byte aligned, returned in two registers."""
    cmd, ext = reference()
    src = tmp_path / 'qref.c'
    src.write_text('#define P\n' + INT128.replace('qf(', 'r_qf(').replace('qc(', 'r_qc('))
    lib = str(tmp_path / ('qref' + ext))
    subprocess.check_call(cmd + ['-o', lib, str(src)])
    ref = ctypes.CDLL(lib)
    m = cimo.compile('#define P\n' + INT128 + 'u64 r_qc(void *); u64 x(void) { return r_qc((void *)qf); }\n'
                     'extern char r_qf[]; u64 y(void) { return qc((void *)r_qf); }\n', link=[lib])
    rc = ref.r_qc
    rc.restype = ctypes.c_uint64
    rc.argtypes = [ctypes.c_void_p]
    want = rc(ctypes.cast(ref.r_qf, ctypes.c_void_p).value)
    assert m.x() == want, 'the reference calls cimo'
    assert m.y() == want, 'cimo calls the reference'
    assert m.qc(m.qf.address) == want, 'cimo calls cimo'
