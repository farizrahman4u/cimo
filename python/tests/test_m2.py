"""M2 acceptance: the 15 kernels of scratch/qbe_eval/kernels.c compile through cimo.compile, run
in-process (cimo.asm + cimo.jit, no cc, no dlopen of generated code) and match gcc bit for bit;
scratch/qbe_eval/edge.c passes against the in-process kernels.

The gcc reference is built with `cc -O2 -ffp-contract=off` (QBE never fuses multiply-add, so
results are bit-exact) as a shared library and called through ctypes on the same inputs.
Skips when cc (for the gcc reference) is missing.
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EVAL = os.path.join(REPO, 'scratch', 'qbe_eval')
KERNELS = os.path.join(EVAL, 'kernels.c')
CC = os.environ.get('CC', 'cc')

pytestmark = [
    pytest.mark.skipif(platform.machine().lower() not in ('arm64', 'aarch64'),
                       reason='arm64 only'),
    pytest.mark.skipif(not shutil.which(CC), reason='no C compiler for the reference'),
    pytest.mark.skipif(not os.path.exists(KERNELS), reason='scratch/qbe_eval missing'),
]


@pytest.fixture(scope='module')
def build(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('m2')
    old = os.environ.get('CIMO_CACHE_DIR')
    os.environ['CIMO_CACHE_DIR'] = str(tmp / 'cache')  # exercise the full pipeline
    try:
        with open(KERNELS) as f:
            mod = cimo.compile(f.read())
    finally:
        if old is None:
            del os.environ['CIMO_CACHE_DIR']
        else:
            os.environ['CIMO_CACHE_DIR'] = old
    ref = str(tmp / 'ref.so')
    subprocess.run([CC, '-O2', '-ffp-contract=off', '-shared', '-fPIC', KERNELS, '-o', ref,
                    '-lm'], check=True)
    return mod, ctypes.CDLL(ref), tmp


rng = np.random.default_rng(1)


def f32(n, lo=0.0, hi=1.0):
    return rng.uniform(lo, hi, n).astype(np.float32)


def f64(n):
    return rng.uniform(0, 1, n)


def u8(n, hi=256):
    return rng.integers(0, hi, n, dtype=np.uint8)


def _flood_grid(w, h):
    g = (rng.integers(0, 100, w * h) < 30).astype(np.uint8)
    g[(h // 2) * w + w // 2] = 0
    return g


I, F, D = ctypes.c_int, ctypes.c_float, ctypes.c_double

# name: (restype, [argument factories]); ints and floats are passed by value, arrays by pointer
CASES = {
    'saxpy': (None, [lambda: 1000, lambda: F(0.5), lambda: f32(1000), lambda: f32(1000)]),
    'gemm': (None, [lambda: 37, lambda: f32(37 * 37), lambda: f32(37 * 37),
                    lambda: f32(37 * 37)]),
    'jacobi2d': (None, [lambda: 50, lambda: f32(2500), lambda: f32(2500)]),
    'conv3x3': (None, [lambda: 40, lambda: 33, lambda: f32(40 * 33), lambda: f32(9, -1, 1),
                       lambda: f32(40 * 33)]),
    'rgb2gray': (None, [lambda: 999, lambda: u8(3 * 999), lambda: u8(999)]),
    'dot': (D, [lambda: 1001, lambda: f64(1001), lambda: f64(1001)]),
    'isum': (ctypes.c_long, [lambda: 999, lambda: rng.integers(-1 << 30, 1 << 30, 999,
                                                               dtype=np.int32)]),
    'prefix_sum': (None, [lambda: 1000, lambda: f32(1000), lambda: f32(1000)]),
    'iir': (None, [lambda: 1000, lambda: F(0.3), lambda: F(-0.5), lambda: F(0.2),
                   lambda: f32(1000, -0.5, 0.5), lambda: f32(1000)]),
    'edit_distance': (I, [lambda: 300, lambda: 280, lambda: u8(300, 4) + 97,
                          lambda: u8(280, 4) + 97, lambda: np.zeros(281, np.int32),
                          lambda: np.zeros(281, np.int32)]),
    'histogram': (None, [lambda: 10000, lambda: u8(10000) & u8(10000),
                         lambda: np.full(256, 7, np.int32)]),
    'mandelbrot': (None, [lambda: 80, lambda: 60, lambda: 256,
                          lambda: np.zeros(80 * 60, np.int32)]),
    'nbody_acc': (None, [lambda: 100, lambda: f64(100), lambda: f64(100), lambda: f64(100),
                         lambda: np.zeros(100), lambda: np.zeros(100), lambda: np.zeros(100)]),
    'flood_fill': (I, [lambda: 64, lambda: 48, lambda: 32, lambda: 24,
                       lambda: _flood_grid(64, 48), lambda: np.zeros(64 * 48 * 4, np.int32)]),
    'sieve': (I, [lambda: 100000, lambda: np.ones(100001, np.uint8)]),
}


def _bits(v):
    if isinstance(v, float):
        return struct.pack('<d', v)
    return v


@pytest.mark.parametrize('name', sorted(CASES))
def test_kernel_matches_gcc(build, name):
    mod, ref, _ = build
    restype, factories = CASES[name]
    args = [f() for f in factories]
    mine = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
    theirs = [a.copy() if isinstance(a, np.ndarray) else a for a in args]

    k = getattr(mod, name)
    r1 = k(*[a.value if isinstance(a, F) else a for a in mine])

    fn = getattr(ref, name)
    fn.restype = restype
    fn.argtypes = [ctypes.c_void_p if isinstance(a, np.ndarray) else
                   (F if isinstance(a, F) else I) for a in theirs]
    r2 = fn(*[a.ctypes.data if isinstance(a, np.ndarray) else a for a in theirs])

    assert _bits(r1) == _bits(r2), (r1, r2)
    for i, (a, b) in enumerate(zip(mine, theirs)):
        if isinstance(a, np.ndarray):
            assert a.tobytes() == b.tobytes(), 'argument %d differs' % i
    # and the kernel did something
    changed = any(isinstance(a, np.ndarray) and a.tobytes() != o.tobytes()
                  for a, o in zip(mine, args))
    assert changed or r1 not in (None, 0)


def test_edge_bit_exact(build):
    """edge.c, unchanged, with its four kernels pointing at the in-process cimo code."""
    mod, _, tmp = build
    shim = tmp / 'edge_shim.c'
    shim.write_text('#define main edge_main\n'
                    '#define gemm (*p_gemm)\n#define saxpy (*p_saxpy)\n'
                    '#define jacobi2d (*p_jacobi2d)\n#define conv3x3 (*p_conv3x3)\n'
                    '#include "%s"\n' % os.path.join(EVAL, 'edge.c'))
    so = str(tmp / 'edge.so')
    subprocess.run([CC, '-O0', '-ffp-contract=off', '-shared', '-fPIC', str(shim), '-o', so,
                    '-lm'], check=True)
    lib = ctypes.CDLL(so)
    for k in ('gemm', 'saxpy', 'jacobi2d', 'conv3x3'):
        ctypes.c_void_p.in_dll(lib, 'p_' + k).value = getattr(mod, k).address
    lib.edge_main.restype = ctypes.c_int
    bad = lib.edge_main()
    ctypes.CDLL(None).fflush(None)
    assert bad == 0


def test_no_external_toolchain_for_loading(build):
    """The generated code is loaded by cimo.jit, not dlopen: kernel addresses are inside the
    image's anonymous mapping."""
    mod, _, _ = build
    img = mod._image
    base, size = img._maps[0]
    assert base <= mod.saxpy.address < base + size
