"""Port of eval.c: constant folding.

Integer division and remainder by zero (and INT64_MIN / -1) follow aarch64, where
C cproc runs: the quotient is 0 (INT64_MIN for the overflow) and the remainder is
the dividend. Float division by zero gives inf/nan as in C.
"""

import math
import struct

from .cc import (M64, EXPRIDENT, EXPRCONST, EXPRCOMPOUND, EXPRUNARY, EXPRSTRING, EXPRCAST,
                 EXPRBINARY, PROPFLOAT, PROPINT, TYPEPOINTER, DECLCONST, SDSTATIC, TSUB, TMUL,
                 TDIV, TMOD, TADD, TSHL, TSHR, TBAND, TBOR, TXOR, TLESS, TGREATER, TLEQ, TGEQ,
                 TEQL, TNEQ, TLOR, TLAND)
from .token import tok, error
from . import type as TY
from . import qbe as Q
from . import decl as D
from . import util as U

F = 1 << 8
S = 2 << 8

_f32 = struct.Struct('<f')


def f32(x):
    """(float)x as a double."""
    try:
        return _f32.unpack(_f32.pack(x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def _s64(u):
    return u - (1 << 64) if u >> 63 else u


def _sdiv(a, b):
    if b == 0:
        return 0
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _srem(a, b):
    if b == 0:
        return a
    return a - _sdiv(a, b) * b


def _fdiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        if a != a or a == 0:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)


def cast(expr):
    t = expr.type
    if t.prop & PROPFLOAT:
        if t.size == 4:
            expr.f = f32(expr.f)
    elif t.prop & PROPINT:
        m = 1 << (t.width - 1)
        expr.u &= m | (m - 1)
        if t.issigned:
            expr.u = ((expr.u ^ m) - m) & M64


def unary(expr, op, l):
    expr.kind = EXPRCONST
    if l.type.prop & PROPFLOAT:
        op |= F
    if op == TSUB:
        expr.u = (-l.u) & M64
    elif op == TSUB | F:
        expr.f = -l.f
    else:
        U.fatal("internal error; unknown unary expression")
    cast(expr)


def binary(expr, op, l, r):
    expr.kind = EXPRCONST
    if l.type.prop & PROPFLOAT:
        op |= F
    elif l.type.prop & PROPINT and l.type.issigned:
        op |= S
    base = op & 0xff
    kind = op & ~0xff
    if kind == F:
        a = l.f
        b = r.f
        if base == TMUL:
            expr.f = a * b
        elif base == TDIV:
            expr.f = _fdiv(a, b)
        elif base == TADD:
            expr.f = a + b
        elif base == TSUB:
            expr.f = a - b
        elif base == TLESS:
            expr.u = int(a < b)
        elif base == TGREATER:
            expr.u = int(a > b)
        elif base == TLEQ:
            expr.u = int(a <= b)
        elif base == TGEQ:
            expr.u = int(a >= b)
        elif base == TEQL:
            expr.u = int(a == b)
        elif base == TNEQ:
            expr.u = int(a != b)
        else:
            U.fatal("internal error; unknown binary expression")
    else:
        a = l.u
        b = r.u
        sgn = kind == S
        if base == TMUL:
            expr.u = (a * b) & M64
        elif base == TDIV:
            if sgn:
                expr.u = _sdiv(_s64(a), _s64(b)) & M64
            else:
                expr.u = a // b if b else 0
        elif base == TMOD:
            if sgn:
                expr.u = _srem(_s64(a), _s64(b)) & M64
            else:
                expr.u = a % b if b else a
        elif base == TADD:
            expr.u = (a + b) & M64
        elif base == TSUB:
            expr.u = (a - b) & M64
        elif base == TSHL:
            expr.u = (a << (b & 63)) & M64
        elif base == TSHR:
            if sgn:
                expr.u = (_s64(a) >> (b & 63)) & M64
            else:
                expr.u = a >> (b & 63)
        elif base == TBAND:
            expr.u = a & b
        elif base == TBOR:
            expr.u = a | b
        elif base == TXOR:
            expr.u = a ^ b
        elif base == TLESS:
            expr.u = int(_s64(a) < _s64(b)) if sgn else int(a < b)
        elif base == TGREATER:
            expr.u = int(_s64(a) > _s64(b)) if sgn else int(a > b)
        elif base == TLEQ:
            expr.u = int(_s64(a) <= _s64(b)) if sgn else int(a <= b)
        elif base == TGEQ:
            expr.u = int(_s64(a) >= _s64(b)) if sgn else int(a >= b)
        elif base == TEQL:
            expr.u = int(a == b)
        elif base == TNEQ:
            expr.u = int(a != b)
        else:
            U.fatal("internal error; unknown binary expression")
    cast(expr)


def eval(expr):
    t = expr.type
    k = expr.kind
    if k == EXPRIDENT:
        d = expr.decl
        if d.kind == DECLCONST:
            expr.kind = EXPRCONST
            expr.u = d.enumconst
    elif k == EXPRCOMPOUND:
        d = expr.decl
        if d.storage == SDSTATIC:
            d.value = Q.mkglobal(d)
            Q.emitdata(d, expr.init)
            expr.kind = EXPRIDENT
            expr.decl = d
    elif k == EXPRUNARY:
        l = eval(expr.base)
        op = expr.op
        if op == TBAND:
            if l.kind == EXPRUNARY:
                if l.op == TMUL:
                    expr = eval(l.base)
            elif l.kind == EXPRSTRING:
                l.decl = D.stringdecl(l)
                l.kind = EXPRIDENT
                expr.base = l
        elif op == TMUL:
            pass
        else:
            if l.kind == EXPRCONST:
                unary(expr, expr.op, l)
    elif k == EXPRCAST:
        l = eval(expr.base)
        if l.kind == EXPRCONST:
            expr.kind = EXPRCONST
            if l.type.prop & PROPINT and t.prop & PROPFLOAT:
                if l.type.issigned:
                    expr.f = float(l.i)
                else:
                    expr.f = float(l.u)
            elif l.type.prop & PROPFLOAT and t.prop & PROPINT:
                lf = l.f
                if t.issigned:
                    if lf < -9223372036854775808.0 or lf >= 9223372036854775808.0:
                        error(tok.loc, "integer part of floating-point constant %s cannot be represented as signed integer", _g(lf))
                    expr.i = int(lf) if lf == lf else 0  # fcvtzs: NaN -> 0
                else:
                    if lf < 0.0 or lf >= 18446744073709551616.0:
                        error(tok.loc, "integer part of floating-point constant %s cannot be represented as unsigned integer", _g(lf))
                    expr.u = int(lf) if lf == lf else 0  # fcvtzu: NaN -> 0
            else:
                expr.u = l.u
            cast(expr)
        elif l.type.kind == TYPEPOINTER:
            # A cast from a pointer to integer is not a valid constant expression, but
            # C11 allows implementations to recognize other forms of constant
            # expressions (6.6p10), and some programs expect this functionality.
            if t.kind == TYPEPOINTER or t.prop & PROPINT and t.size == TY.typelong.size:
                expr = l
    elif k == EXPRBINARY:
        l = eval(expr.l)
        r = eval(expr.r)
        expr.l = l
        expr.r = r
        op = expr.op
        if op == TADD or op == TSUB:
            if op == TADD and r.kind == EXPRBINARY:
                l, r = r, l
            if r.kind == EXPRCONST:
                if l.kind == EXPRCONST:
                    binary(expr, expr.op, l, r)
                elif (l.kind == EXPRBINARY and l.type.kind == TYPEPOINTER and l.op == TADD
                      and l.r.kind == EXPRCONST):
                    # (P + C1) +- C2  ->  P + (C1 +- C2)
                    binary(expr.r, expr.op, l.r, r)
                    expr.op = TADD
                    expr.l = l.l
        elif op == TLOR:
            if l.kind == EXPRCONST:
                return l if l.u else r
        elif op == TLAND:
            if l.kind == EXPRCONST:
                return r if l.u else l
        else:
            if l.kind == EXPRCONST and r.kind == EXPRCONST:
                binary(expr, expr.op, l, r)
    return expr


def _g(x):
    """printf %g"""
    if x != x:
        return '-nan' if math.copysign(1.0, x) < 0 else 'nan'
    return '%g' % x
