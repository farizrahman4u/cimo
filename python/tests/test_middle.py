"""Middle end (cimo.middle): differential tests against gcc.

Every function of tests/middle/loops.c (loop shapes: trip counts 0..100, negative and
symbolic strides, unsigned wrap, INT_MAX bounds, aliasing pointers without restrict, nested
loops, early exits...) and every kernel of scratch/qbe_eval/kernels.c at edge sizes is run
three ways on identical inputs: cimo with the middle end, cimo without it, and gcc -O0
-ffp-contract=off. Return values and every buffer must match bit for bit.
"""

import ctypes
import os
import platform
import shutil
import struct
import subprocess

import pytest

np = pytest.importorskip('numpy')

import cimo  # noqa: E402
from cimo import driver  # noqa: E402


@pytest.fixture(scope='module', autouse=True)
def _no_fma():
    """These tests compare bit for bit against gcc -ffp-contract=off."""
    import cimo.driver
    old = cimo.driver.FMA
    cimo.driver.FMA = False
    yield
    cimo.driver.FMA = old

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LOOPS = os.path.join(HERE, 'middle', 'loops.c')
KERNELS = os.path.join(REPO, 'scratch', 'qbe_eval', 'kernels.c')
CC = os.environ.get('CC', 'cc')

pytestmark = [
    pytest.mark.skipif(platform.machine().lower() not in ('arm64', 'aarch64'),
                       reason='arm64 only'),
    pytest.mark.skipif(not shutil.which(CC), reason='no C compiler for the reference'),
]


def _compile(src, middle, tmp):
    old = os.environ.get('CIMO_CACHE_DIR')
    oldm = driver.MIDDLE
    os.environ['CIMO_CACHE_DIR'] = str(tmp / 'cache')
    driver.MIDDLE = middle
    try:
        return cimo.compile(src)
    finally:
        driver.MIDDLE = oldm
        if old is None:
            del os.environ['CIMO_CACHE_DIR']
        else:
            os.environ['CIMO_CACHE_DIR'] = old


def _gcc(path, tmp, name):
    so = str(tmp / (name + '.so'))
    subprocess.run([CC, '-O0', '-ffp-contract=off', '-fwrapv', '-shared', '-fPIC', path,
                    '-o', so, '-lm'], check=True)
    return ctypes.CDLL(so)


def _names(path):
    import re
    return re.findall(r'^double (\w+)\(', open(path).read(), re.M)


@pytest.fixture(scope='module')
def loops(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('middle')
    src = open(LOOPS).read()
    stats = {}
    from cimo import middle
    old = middle.stats
    middle.stats = stats
    try:
        on = _compile(src, True, tmp)
    finally:
        middle.stats = old
    off = _compile(src, False, tmp)
    return on, off, _gcc(LOOPS, tmp, 'loops'), stats


NS = list(range(0, 22)) + [31, 32, 33, 63, 64, 65, 100]
KS = [0, 1, 3, -1, -5, 5]
OFF = 256
SIZE = 1024


def _buffers(seed):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1, 1, SIZE).astype(np.float32),
            rng.uniform(-1, 1, SIZE).astype(np.float32),
            rng.integers(-1000, 1000, SIZE, dtype=np.int32),
            rng.integers(-1000, 1000, SIZE, dtype=np.int32),
            rng.uniform(-1, 1, SIZE),
            rng.integers(0, 256, SIZE, dtype=np.uint8)]


def _call(f, n, k, bufs, alias):
    fa, fb, ia, ib, da, ua = bufs
    P = lambda a, off=OFF: a.ctypes.data + off * a.itemsize
    pfb = P(fa, OFF + alias) if alias is not None else P(fb)
    pib = P(ia, OFF + alias) if alias is not None else P(ib)
    r = f(n, k, P(fa), pfb, P(ia), pib, P(da), P(ua))
    return struct.pack('<d', r), b''.join(b.tobytes() for b in bufs)


def _cases(name):
    out = [(n, k, None) for n in NS for k in (KS if n < 22 else KS[:3])]
    if name.endswith('_alias'):
        out += [(n, 1, d) for n in (0, 1, 2, 5, 8, 13, 33) for d in (-3, -1, 0, 1, 2, 5)]
    return out


@pytest.mark.parametrize('name', _names(LOOPS))
def test_loop_shapes(loops, name):
    on, off, ref, _ = loops
    fr = getattr(ref, name)
    fr.restype = ctypes.c_double
    vp = ctypes.c_void_p
    fr.argtypes = [ctypes.c_int, ctypes.c_int, vp, vp, vp, vp, vp, vp]
    for seed, (n, k, alias) in enumerate(_cases(name)):
        res = []
        for f in (getattr(on, name), getattr(off, name), fr):
            bufs = _buffers(seed)
            res.append(_call(f, n, k, bufs, alias))
        assert res[1] == res[2], 'middle end off differs from gcc: n=%d k=%d alias=%r' % (
            n, k, alias)
        assert res[0] == res[2], 'middle end on differs from gcc: n=%d k=%d alias=%r' % (
            n, k, alias)


def test_loop_shapes_transformed(loops):
    """The suite actually exercises the transformations."""
    stats = loops[3]
    assert stats.get('versioned', 0) >= 50, stats
    for k in ('full_unroll', 'jam', 'ifconv', 'forward', 'dead_store'):
        assert stats.get(k, 0) >= 1, (k, stats)


# ---- kernels.c at edge sizes


@pytest.fixture(scope='module')
def kernels(tmp_path_factory):
    if not os.path.exists(KERNELS):
        pytest.skip('scratch/qbe_eval missing')
    tmp = tmp_path_factory.mktemp('middle_k')
    src = open(KERNELS).read()
    return _compile(src, True, tmp), _gcc(KERNELS, tmp, 'kernels')


I, F, D = ctypes.c_int, ctypes.c_float, ctypes.c_double


def _kernel_cases(n, rng):
    f32 = lambda m, lo=-1.0, hi=1.0: rng.uniform(lo, hi, max(m, 1)).astype(np.float32)
    f64 = lambda m: rng.uniform(-1, 1, max(m, 1))
    u8 = lambda m, hi=256: rng.integers(0, hi, max(m, 1), dtype=np.uint8)
    i32 = lambda m: rng.integers(-1 << 20, 1 << 20, max(m, 1), dtype=np.int32)
    w = (n * 7) % 23
    return {
        'saxpy': (None, [n, F(0.5), f32(n), f32(n)]),
        'gemm': (None, [n, f32(n * n), f32(n * n), f32(n * n)]),
        'jacobi2d': (None, [n, f32(n * n), f32(n * n)]),
        'conv3x3': (None, [n, w, f32(n * w), f32(9), f32(n * w)]),
        'rgb2gray': (None, [n, u8(3 * n), u8(n)]),
        'dot': (D, [n, f64(n), f64(n)]),
        'isum': (ctypes.c_int64, [n, i32(n)]),
        'prefix_sum': (None, [n, f32(n), f32(n)]),
        'iir': (None, [n, F(0.3), F(-0.5), F(0.2), f32(n), f32(n)]),
        'edit_distance': (I, [n, w, u8(n, 4), u8(w, 4), np.zeros(w + 1, np.int32),
                              np.zeros(w + 1, np.int32)]),
        'histogram': (None, [n, u8(n), np.zeros(256, np.int32)]),
        'mandelbrot': (None, [n, w, 50, np.zeros(max(n * w, 1), np.int32)]),
        'nbody_acc': (None, [n, f64(n), f64(n), f64(n), f64(n), f64(n), f64(n)]),
        'flood_fill': (I, [max(n, 1), max(w, 1), 0, 0, (u8(max(n, 1) * max(w, 1), 3) == 0)
                           .astype(np.uint8), np.zeros(4 * max(n, 1) * max(w, 1) + 4,
                                                       np.int32)]),
        'sieve': (I, [n * 37, np.ones(n * 37 + 1, np.uint8)]),
    }


@pytest.mark.parametrize('n', list(range(0, 22)) + [37, 64])
def test_kernels_edge_sizes(kernels, n):
    mod, ref = kernels
    for name, (restype, args) in _kernel_cases(n, np.random.default_rng(n)).items():
        mine = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
        theirs = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
        r1 = getattr(mod, name)(*[a.value if isinstance(a, F) else a for a in mine])
        fn = getattr(ref, name)
        fn.restype = restype
        fn.argtypes = [ctypes.c_void_p if isinstance(a, np.ndarray) else
                       (F if isinstance(a, F) else I) for a in theirs]
        r2 = fn(*[a.ctypes.data if isinstance(a, np.ndarray) else a for a in theirs])
        pk = lambda v: struct.pack('<d', v) if isinstance(v, float) else v
        assert pk(r1) == pk(r2), (name, n, r1, r2)
        for i, (a, b) in enumerate(zip(mine, theirs)):
            if isinstance(a, np.ndarray):
                assert a.tobytes() == b.tobytes(), '%s n=%d: argument %d differs' % (name, n, i)
