"""Each pass here changes the code of a program written for it, and what the program computes stays the same. Found by
replacing every pass with one that does nothing and running the tests: nothing else failed for these."""

import ctypes
import importlib
import re

import pytest

import cimo
from cimo import driver

BUF = (ctypes.c_long * 64)(*[(i * 37 + 5) % 23 - 7 for i in range(64)])
A = ctypes.addressof(BUF)
END = A + 8 * 20

CASES = {
    'sccp': ('cimo.backend.scalar', 'sccp', False,
             'int f(int n) { int x = 1; for (int i = 0; i < n; i++) { if (x != 1) x = 2; } return x; }', [(0,), (5,)]),
    'diamonds': ('cimo.backend.scalar', '_diamonds', False,
                 'long f(long c, long n) { long s = 0; for (long i = 0; i < n; i++) { int t = c ? 1 : 2; (void)t; s += i; } return s; }',
                 [(0, 5), (1, 9)]),
    'domcse': ('cimo.backend.cse', 'domcse', False,
               'long f(long *p, long i, long c) { long x = p[i + 1]; if (c) p[i + 1] = x + 1; if (c > 1) x += p[i + 1]; return x + p[i + 1]; }',
               [(A, 1, 0), (A, 2, 1), (A, 3, 2)]),
    'pre': ('cimo.backend.memopt', 'pre', False,
            'long f(long *p, long c, long *q) { long s = 0; if (c) s = *p; else q[1] = 3; return s + *p; }',
            [(A, 0, A + 80), (A, 1, A + 80)]),
    'sinkinc': ('cimo.backend.arm64.isel', 'sinkinc', None,
                'long f(long *p, long *e) { long s = 0; while (p != e) s += *p++; return s; }', [(A, A), (A, END)]),
    'unswitch': ('cimo.middle.unswitch', 'transform', False,
                 'long f(long *a, long n, long c, long *b) { for (long i = 0; i < n; i++) { if (c) a[i] += 1; b[i] = a[i] * 3; } return b[0]; }',
                 [(A, 8, 0, A + 200), (A, 8, 1, A + 200)]),
}


def _asm(src, mark):
    text = driver._build(src + '\nstatic int mark_%s;' % mark)[0]
    names = {}
    return re.sub(r'\b\.?L\d+\b', lambda m: names.setdefault(m[0], 'N%d' % len(names)), re.sub(r'mark_\w+', '', text))


@pytest.mark.parametrize('name', sorted(CASES))
def test_pass_changes_the_code_and_not_the_result(name, monkeypatch):
    mod, fn, off, src, calls = CASES[name]
    on = _asm(src, 'on_' + name)
    m = cimo.compile(src, cache=False)
    results = []
    for args in calls:
        saved = bytes(BUF)
        results.append(m.f(*args))
        ctypes.memmove(BUF, saved, len(saved))
    monkeypatch.setattr(importlib.import_module(mod), fn, lambda *a, **k: off)
    assert _asm(src, 'off_' + name) != on, name + ' changed nothing'
    w = cimo.compile(src + '\nstatic int off_mark;', cache=False)
    for args, want in zip(calls, results):
        saved = bytes(BUF)
        assert w.f(*args) == want, (name, args)
        ctypes.memmove(BUF, saved, len(saved))
