"""FMA end to end (cimo extension, cimo.backend.ext.fma + frontend.compile(fma=True)).

- frontend: `a*b + c`, `c + a*b`, `a*b - c`, `c - a*b` become `fma` only within one expression
  (clang's -ffp-contract=on rule); never across statements, casts, conversions or commas;
  without fma=True the output is stock cproc.
- backend: the `fma` IR instruction parses only with the flag, prints back, survives every
  pass, and is selected as fmadd/fmsub/fnmadd/fnmsub; operands are seen by liveness, spilling
  and register allocation (a high-pressure function is checked).
- results: bit-identical to the same C with explicit fma()/fmaf() calls compiled by cc with
  -ffp-contract=off (so the reference contracts exactly where we say C allows it).
"""

import ctypes
import math
import os
import platform
import random
import re
import shutil
import struct
import subprocess

import pytest

import extcompile as X
from cimo.backend import ext
from cimo.backend.main import main as qbe_main

CC = os.environ.get('CC', 'cc')
ARM64 = platform.machine().lower() in ('arm64', 'aarch64')


def fns(ssa):
    """function name -> body text"""
    return {m.group(1): m.group(2)
            for m in re.finditer(r'function [^$]*\$(\w+)\(.*?\n(.*?)\n}', ssa, re.S)}


CONTRACT = r'''
float f1(float a, float b, float c) { return a * b + c; }
float f2(float a, float b, float c) { return c + a * b; }
float f3(float a, float b, float c) { return a * b - c; }
float f4(float a, float b, float c) { return c - a * b; }
double d1(double a, double b, double c) { return (a * b) + c; }
double d2(double a, double b, double c, double d) { return a * b - c * d; }
double d3(double *s, double a, double b) { return *s += a * b; }
double d4(double a, double b, double c, double d) { return a * b + c * d + a; }
'''

NO_CONTRACT = r'''
float n1(float a, float b, float c) { float t = a * b; return t + c; }
float n2(float a, float b, float c) { return (float)(a * b) + c; }
double n3(float a, float b, double c) { return a * b + c; }
float n4(float a, float b, float c) { return (a, a * b) + c; }
float n5(float a, float b, float c) { float t; return (t = a * b) + c; }
int n6(int a, int b, int c) { return a * b + c; }
float n7(float a, float b, float c) { return -(a * b) + c; }
float n8(float a, float b, float c) { return a * b * c + c; }
'''


def test_frontend_contracts():
    got = fns(X.ir(CONTRACT, fma=True))
    for name in ('f1', 'f2', 'f3', 'f4', 'd1', 'd2', 'd3'):
        assert got[name].count(' fma ') == 1, name
    assert ' mul ' not in got['f1']
    assert got['f3'].count(' neg ') == 1 and got['f4'].count(' neg ') == 1
    assert got['d2'].count(' mul ') == 1  # c*d stays a multiply, a*b is fused
    assert got['d4'].count(' fma ') == 1  # (a*b + c*d) + a: only the inner add is fused
    stock = X.ir(CONTRACT)
    assert ' fma ' not in stock
    assert stock == X.ir(CONTRACT, fma=False)


def test_frontend_does_not_contract():
    got = fns(X.ir(NO_CONTRACT, fma=True))
    for name, body in got.items():
        if name == 'n8':
            assert body.count(' fma ') == 1  # (a*b)*c + c: the outer product is fused
        else:
            assert ' fma ' not in body, name


def test_fma_keyword_needs_flag(tmp_path, capsys):
    p = tmp_path / 'f.ssa'
    p.write_text('export function d $f(d %a, d %b, d %c) {\n@start\n'
                 '\t%r =d fma %a, %b, %c\n\tret %r\n}\n')
    assert qbe_main([str(p)]) == 1
    assert 'unknown keyword fma' in capsys.readouterr().err
    with ext.options(fma=True):
        assert qbe_main(['-d', 'P', str(p)]) == 0
    assert '%r =d fma %a, %b, %c' in capsys.readouterr().err


@pytest.mark.parametrize('expr,mn', [
    ('%a, %b, %c', 'fmadd'),
    ('%na, %b, %c', 'fmsub'),
    ('%a, %nb, %c', 'fmsub'),
    ('%a, %b, %nc', 'fnmsub'),
    ('%na, %b, %nc', 'fnmadd'),
    ('%na, %nb, %nc', 'fnmsub'),
])
@pytest.mark.parametrize('k', 'sd')
def test_select(expr, mn, k):
    ssa = ('export function {k} $f({k} %a, {k} %b, {k} %c) {{\n@start\n'
           '\t%na ={k} neg %a\n\t%nb ={k} neg %b\n\t%nc ={k} neg %c\n'
           '\t%r ={k} fma {e}\n\tret %r\n}}\n').format(k=k, e=expr)
    s = X.backend(ssa, ('fma',))
    body = s.split('f:')[1].split('ret')[0]
    assert re.search(r'\b%s\t%s0, ' % (mn, k), body), body
    assert 'fneg' not in body and 'fmul' not in body


def test_constant_operands():
    ssa = ('export function s $f(s %a) {\n@start\n'
           '\t%r =s fma s_3, %a, s_0.5\n\tret %r\n}\n')
    s = X.backend(ssa, ('fma',))
    assert 'fmadd\ts0, ' in s


# ---------------------------------------------------------------------------------------------
# results against cc with explicit fma calls

def _pressure(rnd, k, nvar=40, nout=64):
    """A function with nvar values all live to the end, combined by nout fmas: forces spills
    of fma operands. Returns (contracted C, reference C with explicit fma calls)."""
    t = 'double' if k == 'd' else 'float'
    f = 'fma' if k == 'd' else 'fmaf'
    head = 'void press_%s(const %s *x, %s *y)\n{\n' % (k, t, t)
    loads = ''.join('\t%s v%d = x[%d];\n' % (t, i, i) for i in range(nvar))
    mine, ref = [], []
    for o in range(nout):
        a, b, c = rnd.sample(range(nvar), 3)
        form = rnd.randrange(4)
        va, vb, vc = 'v%d' % a, 'v%d' % b, 'v%d' % c
        mine.append(['%s * %s + %s', '%s + %s * %s', '%s * %s - %s', '%s - %s * %s'][form]
                    % ((va, vb, vc) if form in (0, 2) else (vc, va, vb)))
        ref.append(['%s(%s, %s, %s)', '%s(%s, %s, %s)', '%s(%s, %s, -%s)',
                    '%s(-%s, %s, %s)'][form] % (f, va, vb, vc))
    tail = '\t%s s = 0;\n' % t + ''.join('\ts = s + v%d;\n' % i for i in range(nvar)) \
        + '\ty[%d] = s;\n}\n' % nout
    body = lambda es: ''.join('\ty[%d] = %s;\n' % (i, e) for i, e in enumerate(es))
    return head + loads + body(mine) + tail, head + loads + body(ref) + tail


FUNCS = r'''
float f1(float a, float b, float c) { return a * b + c; }
float f2(float a, float b, float c) { return c + a * b; }
float f3(float a, float b, float c) { return a * b - c; }
float f4(float a, float b, float c) { return c - a * b; }
double g1(double a, double b, double c) { return a * b + c; }
double g2(double a, double b, double c) { return c + a * b; }
double g3(double a, double b, double c) { return a * b - c; }
double g4(double a, double b, double c) { return c - a * b; }
double g5(double a, double b, double c) { return a * b - c * a; }
double dot(int n, const double *x, const double *y)
{ double s = 0; for (int i = 0; i < n; i++) s += x[i] * y[i]; return s; }
float iir(int n, float b0, float a1, float a2, const float *x, float *y)
{
	float y1 = 0, y2 = 0;
	for (int i = 0; i < n; i++) { float v = b0 * x[i] - a1 * y1 - a2 * y2; y[i] = v; y2 = y1; y1 = v; }
	return y1;
}
float nf(float a, float b, float c) { float t = a * b; return t + c; }
'''

REF = r'''
#include <math.h>
float f1(float a, float b, float c) { return fmaf(a, b, c); }
float f2(float a, float b, float c) { return fmaf(a, b, c); }
float f3(float a, float b, float c) { return fmaf(a, b, -c); }
float f4(float a, float b, float c) { return fmaf(-a, b, c); }
double g1(double a, double b, double c) { return fma(a, b, c); }
double g2(double a, double b, double c) { return fma(a, b, c); }
double g3(double a, double b, double c) { return fma(a, b, -c); }
double g4(double a, double b, double c) { return fma(-a, b, c); }
double g5(double a, double b, double c) { return fma(a, b, -(c * a)); }
double dot(int n, const double *x, const double *y)
{ double s = 0; for (int i = 0; i < n; i++) s = fma(x[i], y[i], s); return s; }
float iir(int n, float b0, float a1, float a2, const float *x, float *y)
{
	float y1 = 0, y2 = 0;
	for (int i = 0; i < n; i++) { float v = fmaf(-a2, y2, fmaf(b0, x[i], -(a1 * y1))); y[i] = v; y2 = y1; y1 = v; }
	return y1;
}
float nf(float a, float b, float c) { float t = a * b; return t + c; }
'''



def needs_run(f):
    f = pytest.mark.skipif(not ARM64, reason='arm64 only')(f)
    return pytest.mark.skipif(not shutil.which(CC), reason='no C compiler for the reference')(f)


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('fma')
    rnd = random.Random(7)
    pd, rd = _pressure(rnd, 'd')
    ps, rs = _pressure(rnd, 's')
    mine = X.module(FUNCS + pd + ps, ('fma',))
    src = tmp / 'ref.c'
    src.write_text(REF + rd + rs)
    so = str(tmp / 'ref.so')
    subprocess.run([CC, '-O2', '-ffp-contract=off', '-shared', '-fPIC', str(src), '-o', so,
                    '-lm'], check=True)
    return mine, ctypes.CDLL(so), X.asm(FUNCS + pd + ps, ('fma',))


SPECIAL = [0.0, -0.0, 1.0, -1.0, 0.1, 3.0, 1e300, -1e-300, 5e-324, math.inf, -math.inf,
           math.nan, 1 / 3, 2.0 ** 52 + 1]


def _bits(x, k):
    if x != x:
        return 'nan'  # NaN sign/payload is not specified by C
    return struct.pack('<f' if k == 's' else '<d', x)


def _vals(rnd, k, n):
    out = [rnd.choice(SPECIAL) if rnd.random() < 0.2 else rnd.uniform(-4, 4) for _ in range(n)]
    if k == 's':
        out = [struct.unpack('<f', struct.pack('<f', v if abs(v) < 3e38 or v != v
                                               else math.copysign(math.inf, v)))[0]
               for v in out]
    return out


@pytest.mark.parametrize('name', ['f1', 'f2', 'f3', 'f4', 'g1', 'g2', 'g3', 'g4', 'g5', 'nf'])
@needs_run
def test_scalar_results(built, name):
    mine, ref, _ = built
    k = 's' if name[0] in 'fn' else 'd'
    ct = ctypes.c_float if k == 's' else ctypes.c_double
    fn = getattr(ref, name)
    fn.restype = ct
    fn.argtypes = [ct] * 3
    rnd = random.Random(name)
    fused = 0
    for _ in range(3000):
        a, b, c = _vals(rnd, k, 3)
        r1 = getattr(mine, name)(a, b, c)
        r2 = fn(a, b, c)
        assert _bits(r1, k) == _bits(r2, k), (name, a, b, c, r1, r2)
        p = ct(ct(a).value * ct(b).value).value
        fused += r1 != ct(p + c).value
    if name != 'nf':
        assert fused > 100  # the result really is fused


@needs_run
def test_loops_and_pressure(built):
    mine, ref, text = built
    assert 'fmadd' in text and 'fmsub' in text and 'fnmsub' in text
    rnd = random.Random(3)
    n = 1001
    D = ctypes.c_double * n
    x, y = D(*[rnd.uniform(-1, 1) for _ in range(n)]), D(*[rnd.uniform(-1, 1) for _ in range(n)])
    ref.dot.restype = ctypes.c_double
    assert mine.dot(n, ctypes.addressof(x), ctypes.addressof(y)) == ref.dot(n, x, y)

    F = ctypes.c_float * n
    xs = F(*[rnd.uniform(-0.5, 0.5) for _ in range(n)])
    y1, y2 = F(), F()
    ref.iir.restype = ctypes.c_float
    ref.iir.argtypes = [ctypes.c_int] + [ctypes.c_float] * 3 + [ctypes.c_void_p] * 2
    r1 = mine.iir(n, 0.3, -0.5, 0.2, ctypes.addressof(xs), ctypes.addressof(y1))
    r2 = ref.iir(n, 0.3, -0.5, 0.2, ctypes.addressof(xs), ctypes.addressof(y2))
    assert _bits(r1, 's') == _bits(r2, 's') and bytes(y1) == bytes(y2)

    for k, T in (('d', ctypes.c_double), ('s', ctypes.c_float)):
        for trial in range(20):
            xv = (T * 40)(*_vals(rnd, k, 40))
            o1, o2 = (T * 65)(), (T * 65)()
            getattr(mine, 'press_' + k)(ctypes.addressof(xv), ctypes.addressof(o1))
            getattr(ref, 'press_' + k)(xv, o2)
            assert [_bits(v, k) for v in o1] == [_bits(v, k) for v in o2], (k, trial)
