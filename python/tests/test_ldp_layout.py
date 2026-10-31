"""ldp/stp pairing and loop layout (cimo extensions cimo.backend.ext.ldp / .layout), arm64.

Pairing must never reorder accesses that may overlap: loads only move past loads and past
accesses off the same base with disjoint ranges; stores likewise. Checked on IR snippets (the
generated assembly) and by running code where the pointers alias.
"""

import ctypes
import platform
import re

import pytest

import extcompile as X
from cimo.backend import ext

ARM64 = platform.machine().lower() in ('arm64', 'aarch64')
run_only = pytest.mark.skipif(not ARM64, reason='arm64 only')


def body(text, name='f'):
    return text.split('\n%s:' % name)[1].split('/* end function')[0]


def pairs(text, name='f'):
    return [l.strip() for l in body(text, name).splitlines()
            if l.strip().startswith(('ldp', 'stp')) and 'x29' not in l]


def fn(lines, ret='w', args='l %p, l %q'):
    return ('export function %s $f(%s) {\n@start\n' % (ret, args)
            + ''.join('\t%s\n' % l for l in lines) + '}\n')


ADD = ['%p4 =l add %p, 4', '%p8 =l add %p, 8']


@pytest.mark.parametrize('lines,want', [
    # adjacent loads
    (ADD + ['%a =w loadw %p', '%b =w loadw %p4', '%s =w add %a, %b', 'ret %s'], ['ldp']),
    # in either order
    (ADD + ['%b =w loadw %p4', '%a =w loadw %p', '%s =w sub %a, %b', 'ret %s'], ['ldp']),
    # a store to another base in between: q may point at p+4
    (ADD + ['%a =w loadw %p', 'storew 5, %q', '%b =w loadw %p4', '%s =w add %a, %b',
            'ret %s'], []),
    # a store to the same base, not overlapping: the load moves past it
    (ADD + ['%a =w loadw %p', 'storew 5, %p8', '%b =w loadw %p4', '%s =w add %a, %b',
            'ret %s'], ['ldp']),
    # overlapping store in between
    (ADD + ['%a =w loadw %p', 'storew 5, %p4', '%b =w loadw %p4', '%s =w add %a, %b',
            'ret %s'], []),
    (ADD + ['%a =w loadw %p', 'storel 5, %p', '%b =w loadw %p4', '%s =w add %a, %b',
            'ret %s'], []),
    # a call in between
    (ADD + ['%a =w loadw %p', 'call $g()', '%b =w loadw %p4', '%s =w add %a, %b',
            'ret %s'], []),
    # stores: the first sinks to the second past an unrelated computation
    (ADD + ['storew 1, %p', '%x =w add 0, 0', 'storew 2, %p4', 'ret 0'], ['stp']),
    # ... but not past a load that may read it
    (ADD + ['storew 1, %p', '%x =w loadw %q', 'storew 2, %p4', 'ret %x'], []),
    (ADD + ['storew 1, %p', '%x =w loadw %p8', 'storew 2, %p4', 'ret %x'], ['stp']),
    # different sizes or kinds don't pair
    (ADD + ['%a =w loadw %p', '%b =l loadl %p4', '%c =l extsw %a', '%s =l add %c, %b',
            '%t =w copy %s', 'ret %t'], []),
    (ADD + ['%a =w loadsh %p', '%b =w loadsh %p4', '%s =w add %a, %b', 'ret %s'], []),
    # not adjacent
    (ADD + ['%a =w loadw %p', '%b =w loadw %p8', '%s =w add %a, %b', 'ret %s'], []),
    # floats and doubles
    (['%p4 =l add %p, 4', '%a =s loads %p', '%b =s loads %p4', '%s =s add %a, %b',
      '%w =w stosi %s', 'ret %w'], ['ldp']),
    (['%p8 =l add %p, 8', 'stored d_1, %p8', 'stored d_2, %p', 'ret 0'], ['stp']),
])
def test_pairing(lines, want):
    ssa = fn(lines)
    off = X.backend(ssa)
    on = X.backend(ssa, ('ldp',))
    assert not pairs(off)
    assert [p.split('\t')[0] for p in pairs(on)] == want, body(on)


def test_pair_offsets_out_of_range():
    ssa = fn(['%a1 =l add %p, 256', '%a2 =l add %p, 260', '%a =w loadw %a1',
              '%b =w loadw %a2', '%s =w add %a, %b', 'ret %s'])
    assert not pairs(X.backend(ssa, ('ldp',)))  # 256 > 63*4
    ssa = fn(['%a1 =l add %p, 248', '%a2 =l add %p, 252', '%a =w loadw %a1',
              '%b =w loadw %a2', '%s =w add %a, %b', 'ret %s'])
    assert len(pairs(X.backend(ssa, ('ldp',)))) == 1


def test_disjoint_hook():
    ssa = fn(ADD + ['%a =w loadw %p', 'storew 5, %q', '%b =w loadw %p4', '%s =w add %a, %b',
                    'ret %s'])
    old = ext.disjoint
    ext.disjoint = lambda fn, a, b: True
    try:
        assert len(pairs(X.backend(ssa, ('ldp',)))) == 1
    finally:
        ext.disjoint = old


C = r'''
int load2(int *p, int *q) { int a = p[0]; q[0] = 5; int b = p[1]; return a * 100 + b; }
int store2(int *p, int *q) { p[0] = 1; int x = q[0]; p[1] = 2; return x; }
long sum(long *p, int n) { long s = 0; for (int i = 0; i < n; i += 2) s += p[i] - p[i + 1]; return s; }
void copy4(float *restrict d, const float *restrict s, int n)
{ for (int i = 0; i < n; i++) { d[4*i] = s[4*i]; d[4*i+1] = s[4*i+1]; d[4*i+2] = s[4*i+2]; d[4*i+3] = s[4*i+3]; } }
int loop(int n) { int s = 0; for (int i = 0; i < n; i++) s += i; while (n > 3) n -= 3; return s + n; }
int nested(int n) { int s = 0; for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) s += j ^ i; return s; }
'''


@run_only
def test_aliasing_runs():
    for flags in ((), ('ldp',), ('ldp', 'layout')):
        m = X.module(C, flags)
        buf = (ctypes.c_int * 4)(1, 2, 3, 4)
        a = ctypes.addressof(buf)
        assert m.load2(a, a + 4) == 105  # q aliases p+1: the store must be seen
        buf[:] = [1, 2, 3, 4]
        assert m.store2(a, a) == 1 and list(buf) == [1, 2, 3, 4]
        buf[:] = [0, 0, 9, 9]
        assert m.store2(a, a + 4) == 0 and list(buf)[:2] == [1, 2]
        ls = (ctypes.c_int64 * 10)(*range(10, 20))
        assert m.sum(ctypes.addressof(ls), 10) == -5
        src = (ctypes.c_float * 40)(*[i * 0.5 for i in range(40)])
        dst = (ctypes.c_float * 40)()
        m.copy4(ctypes.addressof(dst), ctypes.addressof(src), 10)
        assert list(dst) == list(src)
        for n in (0, 1, 2, 7, 100):
            k = n
            while k > 3:
                k -= 3
            assert m.loop(n) == sum(range(n)) + k
            assert m.nested(n) == sum(j ^ i for i in range(n) for j in range(i))


def test_layout_rotates_loops():
    off = X.asm(C)
    on = X.asm(C, ('layout',))
    for name in ('loop', 'nested'):
        b0, b1 = body(off, name), body(on, name)
        assert b1.count('\tb\t') < b0.count('\tb\t'), b1
    # the hot loop of `loop` ends in its own conditional branch back
    b1 = body(on, 'loop')
    m = re.search(r'\n(\.L\d+):\n(?:\t[^\n]*\n)*?\tb(lt|gt|le|ge)\t\1\n', b1)
    assert m, b1
