"""Execution model (M5): abort/assert/trap/stack overflow unwind to KernelError, and the module
heap (per-thread heaps, cross-thread free, grow, reset, realloc/calloc/aligned_alloc)."""

import ctypes
import os
import platform
import random
import threading

import pytest

import cimo
from cimo import execution

pytestmark = pytest.mark.skipif(platform.machine().lower() not in ('arm64', 'aarch64'),
                                reason='arm64 only')

RUNTIME = os.path.join(os.path.dirname(cimo.__file__), 'runtime')


def _links_runtime():
    m = cimo.compile('#include <stdlib.h>\nlong f(void) { return (long)malloc(1); }')
    return m._image.prefix + '__cimo_heap_reset' in m._image.symbols


_LINKED = None


def build(src):
    """Compile src with the heap and abort() from the runtime: linked by the driver once it
    does that, prepended as source until then."""
    global _LINKED
    if _LINKED is None:
        try:
            _LINKED = _links_runtime()
        except cimo.CompileError:
            _LINKED = False
    if not _LINKED:
        rt = ''.join(open(os.path.join(RUNTIME, f)).read() for f in ('heap.c', 'abort.c'))
        src = rt + '#line 1 "<test>"\n' + src
    return cimo.compile(src)


# ---------------------------------------------------------------------------------------------
# unwinding

ERRORS = r'''
#include <assert.h>
#include <stdlib.h>
long depth_assert(long d, long x) { if (d) return 1 + depth_assert(d - 1, x); assert(x > 0); return x; }
long depth_abort(long d, long x) { if (d) return 1 + depth_abort(d - 1, x); if (x == 3) abort(); return x; }
double fp(double a, long x) { assert(x != 7); return a * 2; }
long trap(long x) { if (x) __builtin_trap(); return 5; }
_Noreturn void never(void) { }
long unreach(long x) { if (x) never(); return 6; }
long rec(long n) { return n == 0 ? 0 : 1 + rec(n - 1); }
long even(long n);
long odd(long n) { return n == 0 ? 0 : even(n - 1); }
long even(long n) { return n == 0 ? 1 : odd(n - 1); }
long callback(long (*cb)(long), long x) { long r = cb(x); assert(r != 99); return r + 1; }
long many(long a, long b, long c, long d, long e, long f, long g, long h, long i, long j)
{ assert(a != 0); return a + b + c + d + e + f + g + h + i + j; }
'''


@pytest.fixture(scope='module')
def em():
    return build(ERRORS)


def test_assert_message_and_recovery(em):
    for d in (0, 1, 5, 50, 1000):
        with pytest.raises(cimo.KernelError) as ei:
            em.depth_assert(d, -1)
        e = ei.value
        assert e.kind == 'assert' and e.expr == 'x > 0' and e.function == 'depth_assert'
        assert e.file == '<string>' and e.line == 4 and e.kernel == 'depth_assert'
        assert 'assertion failed: x > 0, at <string>:4 in depth_assert()' in str(e)
        assert em.depth_assert(d, 3) == d + 3


def test_abort_and_recovery(em):
    for d in (0, 3, 200):
        with pytest.raises(cimo.KernelError) as ei:
            em.depth_abort(d, 3)
        assert ei.value.kind == 'abort' and ei.value.function == 'depth_abort'
        assert em.depth_abort(d, 4) == d + 4


def test_float_return_and_many_args(em):
    assert em.fp(1.5, 0) == 3.0
    with pytest.raises(cimo.KernelError):
        em.fp(1.5, 7)
    assert em.fp(2.5, 0) == 5.0
    assert em.many(*range(1, 11)) == 55  # stack-passed arguments reach the kernel
    with pytest.raises(cimo.KernelError):
        em.many(0, *range(2, 11))
    assert em.many(*range(1, 11)) == 55


def test_trap_and_unreachable(em):
    with pytest.raises(cimo.KernelError) as ei:
        em.trap(1)
    assert ei.value.kind == 'trap' and ei.value.function == 'trap'
    assert em.trap(0) == 5
    with pytest.raises(cimo.KernelError) as ei:
        em.unreach(1)  # falls off the end of a _Noreturn function
    assert ei.value.kind == 'trap'
    assert em.unreach(0) == 6


def test_stack_overflow(em):
    assert em.rec(10000) == 10000
    with pytest.raises(cimo.KernelError) as ei:
        em.rec(1 << 40)
    assert ei.value.kind == 'stack overflow' and ei.value.function == 'rec'
    with pytest.raises(cimo.KernelError) as ei:
        em.odd(1 << 40)  # mutual recursion
    assert ei.value.function in ('odd', 'even')
    assert em.rec(10000) == 10000 and em.even(10) == 1


def test_stack_overflow_in_thread(em):
    out = []

    def run():
        try:
            em.rec(1 << 40)
        except cimo.KernelError as e:
            out.append(e.kind)
        out.append(em.rec(1000))

    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert out == ['stack overflow', 1000]


def test_stack_checked_only_recursive():
    ssa = '''export
function w $a(w %x) {
@s
	%r =w call $b(w %x)
	ret %r
}
function w $b(w %x) {
@s
	%r =w call $c(w %x)
	ret %r
}
function w $c(w %x) {
@s
	%r =w call $b(w %x)
	ret %r
}
function w $d(l %f) {
@s
	%r =w call %f()
	ret %r
}
function w $e() {
@s
	%p =l copy $d
	%r =w call $a(w 1)
	ret %r
}
function w $leaf() {
@s
	ret 0
}
'''
    assert execution.stack_checked(ssa) == {'b', 'c', 'd'}  # d can call itself through $d


CB = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_long)


def test_callback_reentry(em):
    seen = []

    def cb(x):
        try:
            em.depth_abort(10, 3)  # inner call into the same module unwinds only itself
        except cimo.KernelError as e:
            seen.append(e.kind)
        return em.depth_assert(2, x)

    f = CB(cb)
    p = ctypes.cast(f, ctypes.c_void_p).value
    assert em.callback(p, 5) == 8
    assert seen == ['abort']
    with pytest.raises(cimo.KernelError) as ei:  # the outer kernel aborts after the callback
        em.callback(p, 97)
    assert ei.value.function == 'callback'
    assert em.callback(p, 1) == 4
    assert em._state.ctx().depth == 0


def test_nesting_limit(em):
    def cb(x):
        try:
            return em.callback(p, x + 1) if x < 40 else x
        except cimo.KernelError as e:
            errors.append(e.kind)
            return -1000

    errors = []
    f = CB(cb)
    p = ctypes.cast(f, ctypes.c_void_p).value
    em.callback(p, 0)
    assert errors == ['nesting']
    assert em.callback(p, 39) == 42 and em._state.ctx().depth == 0


def test_errors_from_many_threads(em):
    n, iters = 8, 300
    bad = []

    def run(k):
        rng = random.Random(k)
        for _ in range(iters):
            d, x = rng.randrange(30), rng.randrange(-2, 3)
            try:
                r = em.depth_assert(d, x)
                ok = x > 0 and r == d + x
            except cimo.KernelError as e:
                ok = x <= 0 and e.line == 4
            if not ok:
                bad.append((k, d, x))

    ts = [threading.Thread(target=run, args=(k,)) for k in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not bad


def test_raw_address_still_callable(em):
    f = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_long)(em.rec.address)
    assert f(10) == 10


# ---------------------------------------------------------------------------------------------
# heap

HEAP = r'''
#include <stdlib.h>
#include <stdint.h>

static unsigned long next(unsigned long *s) { *s = *s * 6364136223846793005UL + 1442695040888963407UL; return *s >> 33; }

static void fill(unsigned char *p, long n, long tag) { for (long i = 0; i < n; i++) p[i] = (unsigned char)(tag + i * 7); }
static long check(unsigned char *p, long n, long tag) { for (long i = 0; i < n; i++) if (p[i] != (unsigned char)(tag + i * 7)) return 0; return 1; }

static long pick(unsigned long *s)
{
	unsigned long r = next(s) % 1000;
	if (r < 700) return 1 + next(s) % 128;
	if (r < 950) return 129 + next(s) % 4000;
	if (r < 998) return 4096 + next(s) % 100000;
	return 262145 + next(s) % 300000;   /* large: mapped directly */
}

/* slots[i] = pointer, sizes[i] = size, tags[i] = tag; allocate into empty slots */
long produce(long seed, void **slots, long *sizes, long *tags, long n)
{
	unsigned long s = seed;
	for (long i = 0; i < n; i++) {
		long sz = pick(&s);
		unsigned char *p = malloc(sz);
		if (!p) return -1;
		if ((uintptr_t)p % 16) return -2;
		fill(p, sz, seed + i);
		slots[i] = p; sizes[i] = sz; tags[i] = seed + i;
	}
	return 0;
}

/* check and free every slot (blocks may belong to another thread's heap) */
long consume(void **slots, long *sizes, long *tags, long n)
{
	long bad = 0;
	for (long i = 0; i < n; i++) {
		if (!check(slots[i], sizes[i], tags[i])) bad++;
		free(slots[i]);
		slots[i] = 0;
	}
	return bad;
}

/* local churn: random malloc/free/realloc with patterns, all within this call */
long churn(long seed, long iters)
{
	void *p[64] = {0}; long sz[64] = {0}; long tag[64] = {0};
	unsigned long s = seed;
	long bad = 0;
	for (long it = 0; it < iters; it++) {
		long k = next(&s) % 64;
		if (p[k]) {
			if (!check(p[k], sz[k], tag[k])) bad++;
			if (next(&s) % 4 == 0) {
				long nsz = pick(&s);
				unsigned char *q = realloc(p[k], nsz);
				long keep = nsz < sz[k] ? nsz : sz[k];
				if (!check(q, keep, tag[k])) bad++;
				fill(q, nsz, tag[k]);
				p[k] = q; sz[k] = nsz;
				continue;
			}
			free(p[k]);
			p[k] = 0;
		} else {
			sz[k] = pick(&s);
			tag[k] = it;
			p[k] = malloc(sz[k]);
			if (!p[k]) return -1;
			fill(p[k], sz[k], tag[k]);
		}
	}
	for (long k = 0; k < 64; k++) {
		if (p[k] && !check(p[k], sz[k], tag[k])) bad++;
		free(p[k]);
	}
	return bad;
}

long alloc_n(long n, long size) { long ok = 0; for (long i = 0; i < n; i++) ok += malloc(size) != 0; return ok; }

long make_buffer(long n) { double *p = malloc(n * sizeof(double)); for (long i = 0; i < n; i++) p[i] = i * 0.5; return (long)p; }

long realloc_test(void)
{
	char *p = malloc(10);
	for (int i = 0; i < 10; i++) p[i] = i;
	char *q = realloc(p, 5);            /* shrink: same block */
	if (q != p) return 1;
	q = realloc(p, 100000);
	for (int i = 0; i < 10; i++) if (q[i] != i) return 2;
	char *r = realloc(q, 1000000);       /* small -> large */
	for (int i = 0; i < 10; i++) if (r[i] != i) return 3;
	r[999999] = 1;
	char *t = realloc(0, 7);             /* realloc(NULL) is malloc */
	if (!t) return 4;
	free(t);
	free(r);
	free(0);
	return 0;
}

long aligned_test(void)
{
	for (long al = 1; al <= 65536; al *= 2) {
		for (long n = 1; n < 300000; n = n * 3 + 1) {
			unsigned char *p = aligned_alloc(al, n);
			if (!p) return al * 1000 + 1;
			if ((uintptr_t)p % (al < 16 ? 16 : al)) return al * 1000 + 2;
			for (long i = 0; i < n; i++) p[i] = i;
			p = realloc(p, n + 10);
			for (long i = 0; i < n; i++) if (p[i] != (unsigned char)i) return al * 1000 + 3;
			free(p);
		}
	}
	if (aligned_alloc(24, 10)) return 5;   /* not a power of two */
	return 0;
}

long calloc_test(void)
{
	for (long n = 1; n < 2000; n += 37) {
		unsigned char *p = malloc(n);
		for (long i = 0; i < n; i++) p[i] = 0xab;
		free(p);
		unsigned char *q = calloc(n, 1);   /* reuses the block just freed */
		for (long i = 0; i < n; i++) if (q[i]) return n;
		free(q);
	}
	unsigned char *z = calloc(1000, 1000);
	for (long i = 0; i < 1000000; i++) if (z[i]) return -1;
	free(z);
	if (calloc(1UL << 40, 1UL << 40)) return -2;   /* overflow */
	return 0;
}

long large_cycle(long n) { for (long i = 0; i < n; i++) { char *p = malloc(1 << 20); p[0] = 1; p[(1 << 20) - 1] = 2; free(p); } return 0; }
'''


@pytest.fixture(scope='module')
def hm():
    return build(HEAP)


def test_realloc_aligned_calloc(hm):
    assert hm.realloc_test() == 0
    assert hm.aligned_test() == 0
    assert hm.calloc_test() == 0


def test_size_classes():
    """cls_of / cls_size as heap.c computes them: every size maps to the smallest class that
    holds it."""
    def cls_of(n):
        if n <= 128:
            return (n - 1) // 16 if n else 0
        b = (n - 1).bit_length() - 1
        return 8 + (b - 7) * 4 + ((n - 1 - (1 << b)) >> (b - 2))

    def cls_size(c):
        if c < 8:
            return (c + 1) * 16
        b = 7 + (c - 8) // 4
        return (1 << b) + ((c - 8) % 4 + 1) * (1 << (b - 2))

    sizes = [cls_size(c) for c in range(52)]
    assert sizes[-1] == 262144 and sizes == sorted(set(sizes))
    for n in list(range(0, 5000)) + [262143, 262144, 200000, 131073]:
        c = cls_of(n)
        assert sizes[c] >= n and (c == 0 or sizes[c - 1] < n)


def test_buffer_to_python_and_free(hm):
    p = hm.make_buffer(1000)
    a = (ctypes.c_double * 1000).from_address(p)
    assert a[0] == 0 and a[999] == 499.5
    hm.free(p)
    # free from another Python thread (deferred to the owner heap)
    ptrs = [hm.make_buffer(10) for _ in range(100)]
    t = threading.Thread(target=lambda: [hm.free(x) for x in ptrs])
    t.start()
    t.join()
    assert hm.churn(1, 2000) == 0  # the owner drains them on its next malloc


def test_grow_and_large_release(hm):
    before = hm.heap_bytes()
    assert hm.alloc_n(20000, 200) == 20000  # 4.5 MB of small blocks: several chunks
    assert hm.heap_bytes() >= before + 4000000
    hm.reset_heap()
    b = hm.heap_bytes()
    assert hm.large_cycle(50) == 0  # 1 MB blocks are mapped and unmapped one by one
    assert hm.heap_bytes() == b


def test_reset_reuses_chunks(hm):
    hm.reset_heap()
    hm.alloc_n(30000, 300)
    hm.reset_heap()
    b = hm.heap_bytes()
    for _ in range(5):
        assert hm.alloc_n(30000, 300) == 30000
        hm.reset_heap()
    assert hm.heap_bytes() == b


def test_churn_threads(hm):
    out = []

    def run(k):
        out.append(hm.churn(k, 20000))

    ts = [threading.Thread(target=run, args=(k,)) for k in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out == [0] * 8


def test_cross_thread_free(hm):
    """Thread i allocates into its own slots while freeing (and checking) the blocks thread
    i+1 allocated in the previous round: every free is a cross-thread free, racing with the
    owner's mallocs."""
    nthreads, n, rounds = 8, 400, 12
    bufs = [[((ctypes.c_void_p * n)(), (ctypes.c_long * n)(), (ctypes.c_long * n)())
             for _ in range(nthreads)] for _ in range(2)]
    barrier = threading.Barrier(nthreads)
    errors = []

    def run(i):
        for r in range(rounds):
            mine = bufs[r % 2][i]
            rc = hm.produce(1000 * r + i * 100000 + 1, *mine, n)
            if rc:
                errors.append(('produce', i, r, rc))
            barrier.wait()
            other = bufs[r % 2][(i + 1) % nthreads]
            bad = hm.consume(*other, n)
            if bad:
                errors.append(('consume', i, r, bad))
            barrier.wait()

    ts = [threading.Thread(target=run, args=(i,)) for i in range(nthreads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors


def test_concurrent_produce_consume(hm):
    """Frees of thread j's blocks run while thread j is mallocing (double-buffered rounds)."""
    nthreads, n, rounds = 6, 300, 10
    bufs = [[((ctypes.c_void_p * n)(), (ctypes.c_long * n)(), (ctypes.c_long * n)())
             for _ in range(nthreads)] for _ in range(2)]
    barrier = threading.Barrier(nthreads)
    errors = []

    def run(i):
        for r in range(rounds + 1):
            if r < rounds:
                rc = hm.produce(7 * r + i * 1000 + 3, *bufs[r % 2][i], n)
                if rc:
                    errors.append(('produce', i, r, rc))
            if r > 0:
                bad = hm.consume(*bufs[(r - 1) % 2][(i + 1) % nthreads], n)
                if bad:
                    errors.append(('consume', i, r, bad))
            barrier.wait()

    ts = [threading.Thread(target=run, args=(i,)) for i in range(nthreads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors


def test_reset_refuses_while_running(hm):
    m = build(HEAP + 'long cbk(long (*cb)(long)) { return cb(1); }')
    seen = []

    def cb(x):
        hm.reset_heap()  # another module: idle, so allowed
        try:
            m.reset_heap()
        except RuntimeError:
            seen.append('refused')
        return x

    f = CB(cb)
    assert m.cbk(ctypes.cast(f, ctypes.c_void_p).value) == 1
    assert seen == ['refused']
    m.reset_heap()
