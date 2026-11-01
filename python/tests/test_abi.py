"""Calling convention against a reference compiler, with random signatures (abi_gen.py): gcc on
Linux, clang (Visual Studio's, or $CIMO_WIN_CLANG) on Windows arm64, MinGW-w64 gcc ($CIMO_WIN_GCC)
on Windows x64. For each signature, cimo calling
the reference, the reference calling cimo, and cimo calling itself must give what the reference
gives calling itself.

    python tests/test_abi.py [SEED [COUNT]]      # one more seed, from the command line
"""

import ctypes
import os
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


@pytest.mark.skipif(reference() is None, reason='needs gcc (Linux, Windows x64: MinGW) or clang (Windows arm64)')
@pytest.mark.parametrize('seed', SEEDS)
def test_random_signatures(seed):
    # in a child process: an ABI mismatch often crashes
    p = subprocess.run([sys.executable, os.path.abspath(__file__), str(seed), str(COUNT)],
                       capture_output=True, universal_newlines=True, timeout=600)
    assert p.returncode == 0 and ' 0 mismatches' in p.stdout, (
        'exit status %d (the C source: python tests/abi_gen.py %d %d)\n%s%s'
        % (p.returncode, seed, COUNT, p.stdout, p.stderr[-2000:]))


if __name__ == '__main__':
    import shutil
    import tempfile
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
