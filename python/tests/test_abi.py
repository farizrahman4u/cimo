"""Calling convention against a reference compiler, with random signatures (abi_gen.py): gcc on
Linux, clang (Visual Studio's, or $CIMO_WIN_CLANG) on Windows, Apple clang (Xcode or the Command
Line Tools) on macOS. For each signature, cimo calling
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
    if sys.platform == 'win32':
        from test_windows import _clang
        c = _clang()
        if not c or driver.TARGET != 'arm64_win':
            return None
        return [c, '--target=aarch64-pc-windows-msvc', '-O1', '-shared', '-nostdlib',
                '-fuse-ld=lld', '-Wl,/noentry', '-w'], '.dll'
    if driver.TARGET == 'arm64' and _have('gcc'):
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


@pytest.mark.skipif(reference() is None, reason='needs gcc (Linux arm64), clang (Windows arm64) or Apple clang (macOS)')
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
]


def check_packed(d):
    """As check(), for PACKED; each struct gets its own translation of `s`."""
    tests = []
    for n, (decl, init, expr) in enumerate(PACKED):
        decl = re.sub(r'\b(s|i|u)\b(?= \{| b;)', lambda m: '%s%d' % (m.group(1), n), decl)
        decl = re.sub(r'\bunion u\b', 'union u%d' % n, decl)
        tests.append(decl + '''
double @f%(n)d(int p, struct s%(n)d v, int q) { return (%(expr)s) + p * 1e6 + q * 1e7; }
double @call%(n)d(double (*f)(int, struct s%(n)d, int)) { struct s%(n)d v = %(init)s; return f(5, v, 6); }
''' % {'n': n, 'expr': expr, 'init': init})
    code = '\n'.join(tests)
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


@pytest.mark.skipif(reference() is None, reason='needs gcc (Linux arm64) or clang (Windows arm64)')
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
