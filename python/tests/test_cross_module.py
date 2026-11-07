"""A kernel calling another module's function by address (`Kernel.address`): the callee's malloc gets a context
for the calling thread on first use, and its errors and faults go to the caller's entry frame, the innermost on the
stack, named by the callee's function."""

import threading

import pytest

import cimo

A = r'''
#include <stdlib.h>
#include <assert.h>
long sum(long n) { long *p = malloc(n * sizeof *p), s = 0; if (!p) return -1;
                   for (long i = 0; i < n; i++) { p[i] = i; s += p[i]; } free(p); return s; }
long small(long n) { assert(n < 5); return n; }
long load(long n) { return *(volatile long *)n; }
void stop(void) { abort(); }
void boom(void) { __builtin_trap(); }
'''
B = r'''
long twice(long (*f)(long), long n) { return f(n) + f(2 * n); }
void call(void (*f)(void)) { f(); }
'''


@pytest.fixture(scope='module')
def ab():
    return cimo.compile(A), cimo.compile(B)


def test_malloc_on_a_thread_that_never_called_the_callee(ab):
    a, b = ab
    assert b.twice(a.sum.address, 1000) == 499500 + 1999000
    out = []
    ts = [threading.Thread(target=lambda: out.append(b.twice(a.sum.address, 100))) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out == [4950 + 19900] * 4


@pytest.mark.parametrize('call, match', [
    (lambda a, b: b.twice(a.small.address, 3), r'assertion failed: n < 5, at .*:6 in small\(\) \(calling twice\)'),
    (lambda a, b: b.twice(a.load.address, 0), r'memory fault at 0x0 in load\(\) \(calling twice\)'),
    (lambda a, b: b.call(a.stop.address), r'abort\(\) called in stop\(\) \(calling call\)'),
    (lambda a, b: b.call(a.boom.address), r'trap .* in boom\(\) \(calling call\)'),
])
@pytest.mark.skipif(not cimo.fault.ENABLED, reason="CIMO_FAULTS=0: errors without a frame end the process")
def test_errors_in_the_callee_raise_in_the_caller(ab, call, match):
    a, b = ab
    with pytest.raises(cimo.KernelError, match=match):
        call(a, b)
    # both modules stay usable, and the callee keeps no error of its own
    assert b.twice(a.small.address, 1) == 3
    assert a.small(2) == 2
