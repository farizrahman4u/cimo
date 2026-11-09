"""cimo.gpu: explicit GPU kernels to PTX, loaded through the NVIDIA driver.

Everywhere: the PTX text (checked by ptxas when one is found: CIMO_PTXAS, PATH, or the nvidia-cuda-nvcc wheel; ptxas is
a test oracle only), the errors, and the device library's accuracy (compiled for the CPU). With libcuda: the kernels run
and match references. With nvcc too: the same sources built by nvcc give the same results (tests/gpu/nvcc/cimo/gpu.h).
Speed: tests/gpu_bench.py and tests/gpu_passes.py."""

import array
import glob
import math
import os
import random
import shutil
import struct
import subprocess
import sys

import pytest

import cimo
import cimo.gpu as G
from cimo import CompileError

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, 'gpu')
KERNELS = sorted(glob.glob(os.path.join(KDIR, '*.c')))
H = '#include <cimo/gpu.h>\n#include <math.h>\n'


def _ptxas():
    p = os.environ.get('CIMO_PTXAS') or shutil.which('ptxas')
    if p:
        return p
    try:
        import nvidia.cuda_nvcc
        cand = os.path.join(list(nvidia.cuda_nvcc.__path__)[0], 'bin', 'ptxas')
        return cand if os.path.exists(cand) else None
    except ImportError:
        return None


PTXAS = _ptxas()


def assemble(text, sm, tmp_path):
    """ptxas accepts the PTX (when there is a ptxas); returns its -v output."""
    if PTXAS is None:
        return ''
    p = tmp_path / 'k.ptx'
    p.write_text(text)
    r = subprocess.run([PTXAS, '-arch=sm_%d' % sm, '-v', str(p), '-o', os.devnull], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stderr


# ---- PTX text (any machine)

@pytest.mark.parametrize('path', KERNELS, ids=[os.path.basename(p) for p in KERNELS])
@pytest.mark.parametrize('sm', [80, 89])
def test_kernels_assemble(path, sm, tmp_path):
    text = G.ptx(open(path).read(), sm=sm)
    assert '.target sm_%d\n' % sm in text and '.address_size 64' in text
    out = assemble(text, sm, tmp_path)
    assert 'stack frame' not in out or ' 0 bytes stack frame' in out or 'gemm_gelu_ref' in out, out


def test_versions():
    src = H + '__gpu_kernel void k(float *p) { p[0] = 1; }'
    assert '.version 7.0\n.target sm_80\n' in G.ptx(src, sm=80)
    assert '.version 7.8\n.target sm_89\n' in G.ptx(src, sm=89)
    assert '.version 6.0\n.target sm_70\n' in G.ptx(src, sm=70)
    assert '.version 8.1\n' in G.ptx(src, sm=80, version=(8, 1))
    wg = H + '__gpu_kernel void k(void) { __gpu_asm("wgmma.fence.sync.aligned;"); }'
    assert '.target sm_90a\n' in G.ptx(wg, sm=90)
    with pytest.raises(ValueError, match='older'):
        G.ptx(src, sm=35)


def test_thread_ids_and_shared(tmp_path):
    src = H + '''
extern __gpu_shared float dyn[];
__gpu_kernel void k(int *out) {
    __gpu_shared int s[64];
    s[gpu_thread_id(0)] = gpu_block_id(1) + gpu_grid_dim(2) + gpu_block_dim(0) + gpu_lane_id();
    gpu_barrier();
    dyn[gpu_thread_id(0)] = s[63 - gpu_thread_id(0)];
    out[gpu_thread_id(1)] = (int)dyn[0];
}'''
    t = G.ptx(src, sm=80)
    for reg in ('%tid.x', '%tid.y', '%ctaid.y', '%nctaid.z', '%ntid.x', '%laneid'):
        assert reg in t
    assert 'bar.sync 0;' in t and '.extern .shared .align 16 .b8 dyn[];' in t and '.shared .align 4 .b8' in t
    assert 'st.shared' in t and 'ld.global' not in t and 'st.global' in t
    assemble(t, 80, tmp_path)


def test_types_and_math(tmp_path):
    src = H + '''
__gpu_kernel void k(const __bf16 *b, _Float16 *h, float *f, double *d, long *l, unsigned *u, short *s) {
    int i = gpu_thread_id(0);
    float x = (float)b[i];
    h[i] = (_Float16)(x * 2);
    f[i] = sqrtf(x) + fmaf(x, f[i], 1.0f) + fmaxf(x, f[i]) + fminf(x, 0) + fabsf(x) + floorf(x) + roundf(x) + expf(x);
    d[i] = sqrt(d[i]) / 3.0 + (double)l[i];
    l[i] = l[i] / 7 + l[i] % 5 + (l[i] >> 3) + __builtin_popcountl(l[i]) + __builtin_clzl(l[i] | 1);
    u[i] = u[i] / 3u + __builtin_ctz(u[i] | 1) + __builtin_bswap32(u[i]);
    s[i] = (short)(s[i] * 3);
}'''
    t = G.ptx(src, sm=80)
    for ins in ('sqrt.rn.f32', 'fma.rn.f32', 'max.f32', 'min.f32', 'abs.f32', 'cvt.rmi.f32.f32', 'copysign.f32',
                'cvt.rn.f16.f32', 'div.rn.f64', 'sqrt.rn.f64', 'popc.b64', 'clz.b64', 'brev.b32', 'prmt.b32',
                'ld.global.u16', 'st.global.u16', 'div.u32'):
        assert ins in t, ins
    assert 'call' not in t  # expf and the rest inline
    assemble(t, 80, tmp_path)


def test_asm_operands(tmp_path):
    src = H + '''
__gpu_kernel void k(float *f, unsigned *u, unsigned short *h, unsigned long *l) {
    float a = f[0], b = f[1];
    unsigned x;
    unsigned short y;
    unsigned long z;
    __gpu_asm("add.f32 %0, %1, %2;" : "=f"(a) : "f"(a), "f"(b));
    __gpu_asm("mov.u32 %0, %%laneid;" : "=r"(x));
    __gpu_asm("add.u32 %0, %0, %1;" : "+r"(x) : "n"(5));
    __gpu_asm("mov.b16 %0, %1;" : "=h"(y) : "h"(h[0]));
    __gpu_asm("add.u64 %0, %1, 1;" : "=l"(z) : "0"(l[0]));
    f[0] = a; u[0] = x; h[1] = y; l[1] = z;
}'''
    t = G.ptx(src, sm=80)
    assert 'mov.u32 %rx' in t and '%laneid;' in t and ', 5;' in t
    assert 'mov.b16 %hx' in t
    assemble(t, 80, tmp_path)


def test_device_functions_and_structs(tmp_path):
    src = H + '''
struct P { float s; int n; double d; };
__attribute__((noinline)) static float dev(float x, struct P p) { return x * p.s + p.n; }
__attribute__((noinline)) static struct P mk(float s) { struct P p = { s, 3, 1.5 }; return p; }
__gpu_kernel void k(struct P p, float *out) {
    struct P q = mk(2.0f);
    out[gpu_thread_id(0)] = dev(out[0], p) + q.s + (float)q.d;
}'''
    t = G.ptx(src, sm=80)
    assert '.func (.param .f32 dev_retval) dev(' in t and 'call.uni (retval0), dev' in t
    assert '.param .align 8 .b8 k_param_0[16]' in t
    assemble(t, 80, tmp_path)


def test_unused_functions_left_out():
    src = H + 'static float unused(float x) { return x; }\nfloat used_nowhere(float x) { return x * 2; }\n' \
        '__gpu_kernel void k(float *p) { p[0] = 1; }'
    t = G.ptx(src, sm=80)
    assert 'used_nowhere' not in t and 'unused' not in t


def test_arrays_in_registers(tmp_path):
    """Small constant-trip loops unroll, also with branches or asm in them, so the arrays they index are registers."""
    for body in ('for (int j = 0; j < 8; j++) a[j] = a[j] > 0 ? expf(a[j]) : a[j];',
                 'for (int j = 0; j < 8; j++) __gpu_asm("add.f32 %0, %0, 1.0;" : "+f"(a[j]));',
                 'for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) a[i * 2 + j % 2] += a[j];'):
        src = H + '__gpu_kernel void k(float *y) { float a[8]; for (int j = 0; j < 8; j++) a[j] = y[j]; ' + body + \
            ' for (int j = 0; j < 8; j++) y[j] = a[j]; }'
        t = G.ptx(src, sm=80)
        assert '__local_depot' not in t, body
        assemble(t, 80, tmp_path)


@pytest.mark.parametrize('src,msg', [
    ('static int r(int x) { return x ? r(x - 1) + 1 : 0; }\n__gpu_kernel void k(int *p) { p[0] = r(p[1]); }',
     'recursion'),
    ('__gpu_kernel void k(int *p) { if (*p) k(p); }', 'cannot be called'),
    ('#include <stdlib.h>\n__gpu_kernel void k(int *p) { p[0] = *(int *)malloc(4); }', 'malloc is not available'),
    ('__gpu_kernel void k(int *p, int (*f)(int)) { p[0] = f(1); }', 'function pointers'),
    ('static int g(int x) { return x; }\n__gpu_kernel void k(long *p) { p[0] = (long)&g; }', 'function pointers'),
    ('__gpu_kernel void k(int *p, int n) { int a[n]; a[0] = 1; p[0] = a[n - 1]; }', 'variable-length'),
    ('__gpu_kernel int k(int *p) { return 1; }', 'must return void'),
    ('static __gpu_kernel void k(int *p) { }', 'must not be static'),
    ('__gpu_kernel void k(int *p) { __gpu_shared int s[4] = {1}; p[0] = s[0]; }', 'cannot have an initializer'),
    ('static void v(int n, ...) { }\n__gpu_kernel void k(int *p) { v(1, 2); }', 'variadic'),
    ('__gpu_kernel void k(int *p) { __gpu_asm("mov.u32 %0, 1;" : "=q"(p[0])); }', "constraint 'q'"),
    ('__gpu_kernel void k(int *p) { p[0] = gpu_thread_id(p[1]); }', 'must be a constant'),
    ('void f(int *p) { p[0] = 1; }', 'no __gpu_kernel'),
])
def test_rejected(src, msg):
    with pytest.raises(CompileError, match=msg):
        G.ptx(H + src, sm=80)


def test_gpu_attributes_need_a_gpu_compile():
    with pytest.raises(CompileError, match='device=gpu'):
        cimo.compile(H + '__gpu_kernel void k(int *p) { p[0] = 1; }', cache=False)


def test_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('CIMO_CACHE_DIR', str(tmp_path))
    src = H + '__gpu_kernel void k(float *p) { p[0] = 2; } /* test_cache */'
    a = G.ptx(src, sm=80)
    assert glob.glob(str(tmp_path / 'gpu-*.json'))
    G._memo.clear()
    assert G.ptx(src, sm=80) == a
    from cimo import linker
    assert 'gpu' not in linker._COMPILER  # gpu/ is in GPU kernels' keys only


def test_onefile_leaves_gpu_out():
    from cimo import onefile
    assert not any(r.startswith('gpu/') for r in onefile._package_files())
    files = onefile._package_files(gpu=True)
    assert 'gpu/__init__.py' in files and 'gpu/include/cimo/gpu.h' in files and 'gpu/host/cuda.c' in files


def test_expf_accuracy():
    """gpu/lib/expf.c, compiled for the CPU: within 1 ulp of the correctly rounded result."""
    from cimo import resources as R
    m = cimo.compile(R.read_text(R.path('gpu/lib/expf.c')).replace('float expf(float x)', 'float gexpf(float x)'),
                     cache=False)
    f32 = lambda v: struct.unpack('<f', struct.pack('<f', v))[0]

    def ulps(a, b):
        ia, ib = (struct.unpack('<i', struct.pack('<f', v))[0] for v in (a, b))
        return abs(ia - ib)
    rnd = random.Random(1)
    xs = [rnd.uniform(-104, 89) for _ in range(20000)] + [rnd.uniform(-1, 1) for _ in range(5000)]
    xs += [0.0, -0.0, 1e-30, -1e-30, 88.72, -87.33, -103.9, 88.7228]
    worst = 0
    for x in xs:
        x = f32(x)
        try:
            want = f32(math.exp(x))
        except OverflowError:
            want = math.inf
        got = m.gexpf(x)
        if want == 0 or math.isinf(want):
            assert got == want, (x, got, want)
            continue
        worst = max(worst, ulps(got, want))
    assert worst <= 1, worst
    assert m.gexpf(math.inf) == math.inf and m.gexpf(-math.inf) == 0 and math.isnan(m.gexpf(math.nan))
    assert m.gexpf(100.0) == math.inf and m.gexpf(-200.0) == 0


# ---- on an NVIDIA GPU

def _gpu():
    try:
        return G.device()
    except G.GpuError:
        return None


gpu = pytest.fixture(scope='module')(lambda: _gpu() or pytest.skip('no NVIDIA driver or GPU'))


def bf(v):
    b = struct.unpack('<I', struct.pack('<f', v))[0]
    return (b + 0x7fff + ((b >> 16) & 1)) >> 16


def fb(b):
    return struct.unpack('<f', struct.pack('<I', b << 16))[0]


def test_device(gpu):
    assert gpu.sm >= 50 and gpu.name and gpu.multiprocessors > 0 and gpu.memory > 0
    assert G.device() is gpu and G.devices() >= 1
    assert G.max_ptx() >= (7, 0)


def test_saxpy(gpu):
    k = cimo.compile(open(os.path.join(KDIR, 'saxpy.c')).read(), device=gpu)
    n = 10000
    x = array.array('f', (random.random() for _ in range(n)))
    y = array.array('f', (random.random() for _ in range(n)))
    dx, dy = gpu.array(x), gpu.array(y)
    k.saxpy(2.0, dx, dy, n, grid=((n + 255) // 256,), block=256)
    out = array.array('f', bytes(4 * n))
    dy.to_host(out)
    assert all(o == struct.unpack('<f', struct.pack('<f', 2.0 * a + b))[0] for o, a, b in zip(out, x, y))
    # a host buffer: copied in and back, and the call waits
    y2 = array.array('f', y)
    k.saxpy(2.0, x, y2, n, grid=((n + 255) // 256,), block=(256,))
    assert y2 == out
    # __cuda_array_interface__, and a device address
    cai = dx.__cuda_array_interface__
    assert cai['data'][0] == dx.ptr and cai['shape'] == (n,) and cai['typestr'] == '<f4'

    class Other:
        __cuda_array_interface__ = cai
    k.saxpy(0.0, Other(), dy.ptr, n, grid=((n + 255) // 256,), block=256)
    gpu.sync()
    assert dy.to_host() == out.tobytes()


def test_blocked_matmul(gpu):
    k = gpu.compile(open(os.path.join(KDIR, 'matmul_tiled.c')).read())
    n = 128
    A = array.array('f', (random.uniform(-1, 1) for _ in range(n * n)))
    B = array.array('f', (random.uniform(-1, 1) for _ in range(n * n)))
    C = array.array('f', bytes(4 * n * n))
    k.matmul(A, B, C, n, grid=(n // 64, n // 64), block=(16, 16))
    for i in range(0, n, 7):
        for j in range(0, n, 5):
            assert abs(C[i * n + j] - sum(A[i * n + t] * B[t * n + j] for t in range(n))) < 1e-4


def test_mma_bf16(gpu):
    if gpu.sm < 80:
        pytest.skip('mma.sync on bf16 needs sm_80')
    k = gpu.compile(open(os.path.join(KDIR, 'gemm_mma.c')).read())
    m, n, kk = 32, 64, 64
    A = array.array('H', (bf(random.uniform(-1, 1)) for _ in range(m * kk)))
    Bt = array.array('H', (bf(random.uniform(-1, 1)) for _ in range(n * kk)))
    C = array.array('f', bytes(4 * m * n))
    k.gemm_mma(A, Bt, C, m, n, kk, grid=(n // 32, m // 16), block=128)
    for i in range(m):
        for j in range(n):
            assert abs(C[i * n + j] - sum(fb(A[i * kk + t]) * fb(Bt[j * kk + t]) for t in range(kk))) < 1e-4


def test_gemm_gelu(gpu):
    if gpu.sm < 80:
        pytest.skip('mma.sync on bf16 needs sm_80')
    k = gpu.compile(open(os.path.join(KDIR, 'gemm_gelu.c')).read())
    m, n, kk = 64, 128, 96
    A = gpu.array(array.array('H', (bf(random.uniform(-1, 1)) for _ in range(m * kk))))
    Bt = gpu.array(array.array('H', (bf(random.uniform(-1, 1)) for _ in range(n * kk))))
    bias = gpu.array(array.array('f', (random.uniform(-1, 1) for _ in range(n))))
    C, R = gpu.zeros(4 * m * n), gpu.zeros(4 * m * n)
    k.gemm_gelu(A, Bt, bias, C, m, n, kk, grid=(n // 64, m // 64), block=128)
    k.gemm_gelu_ref(A, Bt, bias, R, m, n, kk, grid=(1, m), block=n)
    c, r = array.array('f', C.to_host()), array.array('f', R.to_host())
    assert max(abs(p - q) for p, q in zip(c, r)) < 1e-4


def test_softmax_and_expf(gpu):
    k = gpu.compile(open(os.path.join(KDIR, 'softmax.c')).read())
    rows, cols = 4, 1000
    X = array.array('H', (bf(random.uniform(-10, 10)) for _ in range(rows * cols)))
    Y = array.array('f', bytes(4 * rows * cols))
    k.softmax(X, Y, cols, grid=rows, block=256)
    for i in range(rows):
        row = [fb(X[i * cols + j]) for j in range(cols)]
        mx = max(row)
        s = sum(math.exp(v - mx) for v in row)
        for j in range(cols):
            w = math.exp(row[j] - mx) / s
            assert abs(Y[i * cols + j] - w) <= 2e-6 * w + 1e-12


def test_struct_dynamic_shared_device_call(gpu):
    src = H + '''
struct P { float s; int n; double d; };
__attribute__((noinline)) static float dev(float x, struct P p) { return x * p.s + p.n; }
extern __gpu_shared float dyn[];
__gpu_kernel void k(struct P p, float *out) {
    int i = gpu_thread_id(0);
    dyn[i] = (float)i;
    gpu_barrier();
    out[i] = dev(dyn[(i + 1) % gpu_block_dim(0)], p) + (float)p.d;
}'''
    k = gpu.compile(src)
    out = array.array('f', bytes(4 * 64))
    k.k(struct.pack('<fid', 0.5, 3, 0.25), out, grid=1, block=64, shared=64 * 4)
    assert list(out) == [((i + 1) % 64) * 0.5 + 3 + 0.25 for i in range(64)]


def test_half_and_bf16(gpu):
    src = H + '''
__gpu_kernel void k(const __bf16 *b, const _Float16 *h, _Float16 *ho, __bf16 *bo, float *f, int n) {
    int i = gpu_block_id(0) * gpu_block_dim(0) + gpu_thread_id(0);
    if (i < n) {
        f[i] = (float)b[i] + (float)h[i];
        ho[i] = (_Float16)f[i];
        bo[i] = (__bf16)f[i];
    }
}'''
    k = gpu.compile(src)
    n = 3000
    xs = [random.uniform(-100, 100) for _ in range(n)]
    ys = [random.uniform(-100, 100) for _ in range(n)]
    b = array.array('H', (bf(v) for v in xs))
    h = array.array('H', (struct.unpack('<H', struct.pack('<e', v))[0] for v in ys))
    ho, bo, f = array.array('H', bytes(2 * n)), array.array('H', bytes(2 * n)), array.array('f', bytes(4 * n))
    k.k(b, h, ho, bo, f, n, grid=(n + 127) // 128, block=128)
    for i in range(n):
        want = struct.unpack('<f', struct.pack('<f', fb(b[i]) + struct.unpack('<e', struct.pack('<H', h[i]))[0]))[0]
        assert f[i] == want
        assert ho[i] == struct.unpack('<H', struct.pack('<e', want))[0]
        assert bo[i] == bf(want)


def test_async_and_timing(gpu):
    k = gpu.compile(open(os.path.join(KDIR, 'saxpy.c')).read())
    n = 1 << 20
    d = gpu.zeros(4 * n)
    ms = gpu.time(lambda: k.saxpy(1.0, d, d, n, grid=n // 256, block=256), 10)
    assert ms > 0
    for _ in range(100):
        k.saxpy(1.0, d, d, n, grid=n // 256, block=256)
    gpu.sync()


def test_driver_errors(gpu):
    with pytest.raises(CompileError, match='rejected the PTX'):
        gpu.compile(H + '__gpu_kernel void k(int *p) { __gpu_asm("not.an.instruction %0;" : "=r"(p[0])); }')
    k = gpu.compile(open(os.path.join(KDIR, 'saxpy.c')).read())
    with pytest.raises(G.GpuError, match='cuLaunchKernel'):
        k.saxpy(1.0, 0, 0, 0, grid=1, block=4096)
    with pytest.raises(TypeError):
        k.saxpy(1.0, 0, 0, grid=1, block=1)


# ---- nvcc as the oracle

NVCC = os.environ.get('CIMO_NVCC') or shutil.which('nvcc') or (
    '/usr/local/cuda/bin/nvcc' if os.path.exists('/usr/local/cuda/bin/nvcc') else None)


def test_same_as_nvcc(gpu):
    if NVCC is None:
        pytest.skip('no nvcc')
    sys.path.insert(0, HERE)
    import gpu_bench
    for fname, kname, mk, launch, oi, check in gpu_bench.cases(gpu):
        path = os.path.join(KDIR, fname)
        if fname.startswith(('gemm', 'matmul')) and gpu.sm < 80:
            continue
        ours = gpu.compile(open(path).read())
        theirs = gpu.load(gpu_bench.nvcc_build(NVCC, path, gpu.sm, 'ptx'), ours.signatures)
        res = []
        for m in (ours, theirs):
            args = mk()
            getattr(m, kname)(*args, **launch)
            res.append(array.array('f', args[oi].to_host()))
        assert check(res[0]), fname
        assert max(abs(p - q) for p, q in zip(res[0][::101], res[1][::101])) < 1e-3, fname
