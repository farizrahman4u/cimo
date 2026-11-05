"""Threads, as scramjit's parallel=True uses them: many threads calling into one module at once, with the heap, errors,
faults and output crossing between them, modules compiled and dropped while others run, and the same under
free-threaded CPython (CI runs this file under python3.13t). Each case runs in its own process: a failure is a crash."""

import os
import subprocess
import sys
import textwrap

import pytest

import hosts

pytestmark = pytest.mark.skipif(not hosts.RUNS, reason='cimo does not run here')

PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRELUDE = r'''
import cimo, ctypes, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
SRC = r"""
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
long work(long k, long n) { long s = 0; for (long i = 0; i < n; i++) s += (i * k) ^ (s >> 3); return s; }
long *make(long n, long seed) { long *p = malloc(n * sizeof(long)); for (long i = 0; i < n; i++) p[i] = seed + i; return p; }
long sum_free(long *p, long n) { long s = 0; for (long i = 0; i < n; i++) s += p[i]; free(p); return s; }
int checked(int x) { assert(x >= 0); return x + 1; }
int rd(long a) { return *(int *)a; }
void say(int k) { printf("t%d\n", k); }
"""
k = cimo.compile(SRC)
def work(kk, n):
    s = 0
    for i in range(n):
        s = (s + ((i * kk) ^ (s >> 3))) & 0xffffffffffffffff
        s = s - (1 << 64) if s >> 63 else s
    return s
WANT = {kk: work(kk, 50) for kk in range(16)}
def run(threads, fn):
    errors = []
    def wrap(t):
        try:
            fn(t)
        except BaseException as e:
            errors.append((t, repr(e)))
    ts = [threading.Thread(target=wrap, args=(t,)) for t in range(threads)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert not errors, errors[:5]
'''


def check(code, timeout=300):
    r = subprocess.run([sys.executable, '-c', PRELUDE + textwrap.dedent(code)], cwd=PY, capture_output=True,
                       text=True, timeout=timeout, env=dict(os.environ, PYTHONPATH=PY))
    assert r.returncode == 0, (r.returncode, r.stdout[-1000:], r.stderr[-2000:])
    return r.stdout


def test_many_threads_one_module():
    """8 threads x 1,000 calls, mixed: work whose result is known, blocks malloc'd by one thread and freed by another,
    asserts that fail, faults, and printf, all at once."""
    out = check('''
        import queue
        handoff = queue.Queue()
        counts = [0] * 8
        def go(t):
            r = random.Random(t)
            for i in range(1000):
                c = r.randrange(6)
                if c == 0:
                    kk = r.randrange(16)
                    assert k.work(kk, 50) == WANT[kk]
                elif c == 1:
                    n = r.randrange(1, 64)
                    handoff.put((k.make(n, t * 1000 + i), n, t * 1000 + i))
                elif c == 2:
                    try:
                        p, n, seed = handoff.get_nowait()
                    except queue.Empty:
                        continue
                    assert k.sum_free(p, n) == n * seed + n * (n - 1) // 2   # freed on this thread, made on another
                elif c == 3:
                    try:
                        k.checked(-1)
                        raise AssertionError('no KernelError')
                    except cimo.KernelError as e:
                        assert e.kind == 'assert'
                elif c == 4:
                    try:
                        k.rd(8 + 8 * t)
                        raise AssertionError('no KernelError')
                    except cimo.KernelError as e:
                        assert e.kind == 'memory fault' and e.address == 8 + 8 * t
                else:
                    k.say(t)
                    counts[t] += 1
        run(8, go)
        while not handoff.empty():
            p, n, seed = handoff.get()
            assert k.sum_free(p, n) == n * seed + n * (n - 1) // 2
        print('printed', sum(counts))
    ''')
    lines = out.splitlines()
    want = int(lines[-1].split()[1])
    assert sum(1 for line in lines if line.startswith('t')) == want


def test_modules_compiled_and_dropped_while_others_run():
    """Threads compile their own modules, call them, fault in them and drop them, while others keep calling a shared
    one: the fault handler's table, the context tables and alternate stacks change under running kernels."""
    check('''
        import gc
        stop = []
        def shared(t):
            n = 0
            while not stop:
                assert k.work(3, 50) == WANT[3]
                n += 1
            assert n > 0
        def churn(t):
            for i in range(12):
                m = cimo.compile('long f%d_%d(long x) { return x * %d; }\\nint g%d_%d(long a) { return *(int *)a; }'
                                 % (t, i, i, t, i))
                assert getattr(m, 'f%d_%d' % (t, i))(7) == 7 * i
                try:
                    getattr(m, 'g%d_%d' % (t, i))(16)
                    raise AssertionError('no KernelError')
                except cimo.KernelError as e:
                    assert e.kind == 'memory fault'
                del m
                gc.collect()
        sh = [threading.Thread(target=shared, args=(t,)) for t in range(3)]
        for t in sh: t.start()
        run(4, churn)
        stop.append(1)
        for t in sh: t.join()
        try:
            k.rd(24)
        except cimo.KernelError as e:
            assert e.kind == 'memory fault'
    ''')


def test_thread_pool_reused():
    """A pool's threads reused across calls and modules, as parallel=True would: each thread's context and
    alternate stack made once, results right."""
    check('''
        k2 = cimo.compile('long twice(long x) { return 2 * x; }')
        def call(x):
            if x < 0:
                try:
                    k.checked(x)
                    return 'no error'
                except cimo.KernelError as e:
                    return e.kind
            return k.checked(x)
        with ThreadPoolExecutor(8) as ex:
            for _ in range(5):
                assert list(ex.map(lambda kk: k.work(kk, 50), range(16))) == [WANT[kk] for kk in range(16)]
                assert list(ex.map(k2.twice, range(100))) == [2 * x for x in range(100)]
                assert list(ex.map(call, range(-8, 8))) == ['assert'] * 8 + [x + 1 for x in range(8)]
    ''')


def test_gil_state():
    """Under free-threaded CPython the GIL stays off with cimo imported and kernels running (an extension that needs
    it would switch it back on, with a warning)."""
    out = check('''
        assert k.work(3, 50) == WANT[3]
        print(getattr(sys, '_is_gil_enabled', lambda: True)())
    ''')
    if 'free-threading' in sys.version:
        assert out.strip().splitlines()[-1] == 'False'
