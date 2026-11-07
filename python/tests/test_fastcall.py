"""A native kernel's CPython entry (driver._pyentry): the call runs no Python code for floats, ints and writable
buffers, and gives what the ctypes path gives for everything, errors and output included."""

import array
import ctypes
import sys
import threading
import time

import pytest

import cimo
from cimo import driver

pytestmark = pytest.mark.skipif(driver._PYODIDE or sys.implementation.name != 'cpython',
                                reason='CPython on a native target')

SRC = r'''
#include <stdio.h>
#include <string.h>
double add(double a, double b, long n) { return a + b + n; }
float half(float x) { return x / 2; }
int wrap(int k, short s, signed char c) { return k + s + c; }
unsigned big(unsigned k) { return k * 3u; }
unsigned long ubig(unsigned long k) { return k; }
long sum(const int *x, long n) { long s = 0; for (long i = 0; i < n; i++) s += x[i]; return s; }
long first(const char *p) { return p ? p[0] : -1; }
void fill(char *p, long n, int v) { memset(p, v, n); }
void hello(int k) { printf("hi %d\n", k); }
int boom(int k) { if (k) __builtin_trap(); return 1; }
long spin(const volatile int *flag, long most) { for (long i = 0; i < most; i++) if (*flag) return i; return -1; }
int many(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, double x, double y, double z,
         double u, double v, double w, double s, double t, double q, double r) {
    return a + b + c + d + e + f + g + h + i + j + (int)(x + y + z + u + v + w + s + t + q + r); }
'''


@pytest.fixture(scope='module')
def mods():
    fast = cimo.compile(SRC + '// fast', cache=False)
    old = driver._pyapi
    driver._pyapi = lambda image: None
    try:
        slow = cimo.compile(SRC + '// slow', cache=False)
    finally:
        driver._pyapi = old
    return fast, slow


class Idx:
    def __index__(self):
        return 5


class Flt:
    def __float__(self):
        return 2.25


def outcome(f, *args):
    try:
        return 'ok', f(*args)
    except Exception as e:
        return type(e).__name__, str(e)


def test_the_call_is_a_builtin(mods):
    fast, slow = mods
    assert type(type(fast.add).__dict__['__call__']).__name__ == 'builtin_function_or_method'
    assert type(type(slow.add).__dict__['__call__']).__name__ == 'function'


@pytest.mark.parametrize('name, args', [
    ('add', (1.5, 2.5, 3)), ('add', (1, 2, 3)), ('add', (True, 2.5, False)), ('add', (Flt(), 1.0, Idx())),
    ('add', (1.5, 2.5, 2 ** 70)), ('add', (1.5, 2.5, -2 ** 63)), ('add', (1.5, 2.5, 3.0)), ('add', ('x', 1, 2)),
    ('add', (1.5,)), ('add', (1.5, 2.5, 3, 4)), ('add', (float('nan'), -0.0, 0)), ('add', (1e308, 1e308, 0)),
    ('half', (3.0,)), ('half', (1e300,)), ('half', (7,)),
    ('wrap', (2 ** 31 + 5, 70000, 300)), ('wrap', (-1, -1, -1)), ('wrap', (2 ** 40, 0, 0)), ('wrap', (Idx(), 1, 2)),
    ('big', (2 ** 31,)), ('big', (-1,)), ('ubig', (2 ** 64 - 1,)), ('ubig', (2 ** 63,)), ('ubig', (-5,)),
    ('first', (b'abc',)), ('first', (bytearray(b'xyz'),)), ('first', (memoryview(b'q'),)),
    ('first', (memoryview(bytearray(b'w')),)), ('first', (0,)), ('first', (array.array('b', [9]),)),
    ('first', ((ctypes.c_char * 2)(b'k'),)), ('first', (bytearray(),)), ('first', (None,)), ('first', (1.5,)),
    ('boom', (1,)), ('boom', (0,)),
    ('many', tuple(range(10)) + tuple(x / 4 for x in range(10))),
])
def test_same_as_ctypes(mods, name, args):
    fast, slow = mods
    a, b = outcome(getattr(fast, name), *args), outcome(getattr(slow, name), *args)
    assert a == b or (a[0] == b[0] == 'ok' and a[1] != a[1] and b[1] != b[1]), (a, b)


def test_buffers(mods):
    fast, _ = mods
    x = array.array('i', range(100))
    assert fast.sum(x, 100) == 4950
    b = bytearray(8)
    assert fast.fill(b, 8, 7) is None and b == bytearray([7] * 8)
    b.extend(b'more')  # the buffer was released: it can resize
    assert len(b) == 12


def test_output_comes_before_the_next_print(mods, capsys):
    fast, _ = mods
    for k in range(3):
        fast.hello(k)
        print('py', k)
    assert capsys.readouterr().out == 'hi 0\npy 0\nhi 1\npy 1\nhi 2\npy 2\n'


def test_threads_new_and_reused(mods):
    """Every new thread goes through Python first, also one that reuses a finished thread's key."""
    fast, _ = mods
    bad = []

    def run(k):
        for i in range(200):
            if fast.add(k, i, 1) != k + i + 1:
                bad.append(k)
        try:
            fast.boom(1)
        except cimo.KernelError:
            pass
        else:
            bad.append(('no error', k))
    for _ in range(5):
        ts = [threading.Thread(target=run, args=(k,)) for k in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert not bad


def test_the_gil_is_released(mods):
    """Python runs while a kernel does: the main thread sets the flag the kernel waits for."""
    fast, _ = mods
    flag = (ctypes.c_int * 1)()
    out = []
    t = threading.Thread(target=lambda: out.append(fast.spin(flag, 1 << 40)))
    t.start()
    time.sleep(0.05)
    flag[0] = 1
    t.join(30)
    assert out and out[0] > 0
