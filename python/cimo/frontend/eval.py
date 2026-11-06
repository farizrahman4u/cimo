"""Port of eval.c: constant folding.

Integer division and remainder by zero (and INT64_MIN / -1) follow aarch64, where
C cproc runs: the quotient is 0 (INT64_MIN for the overflow) and the remainder is
the dividend. Float division by zero gives inf/nan as in C.
"""

import math
import struct

from .cc import (M64, EXPRCPLX, EXPRIDENT, EXPRCONST, EXPRCOMPOUND, EXPRUNARY, EXPRSTRING, EXPRCAST,
                 EXPRBINARY, PROPFLOAT, PROPINT, PROPARITH, TYPEPOINTER, TYPEBOOL, DECLCONST, SDSTATIC, TSUB, TMUL,
                 TDIV, TMOD, TADD, TSHL, TSHR, TBAND, TBOR, TXOR, TLESS, TGREATER, TLEQ, TGEQ,
                 TEQL, TNEQ, TLOR, TLAND, Expr)
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
            expr.u = ((expr.u ^ m) - m) & (M64 if t.width <= 64 else m | (m - 1))  # cimo: __int128's 128 bits


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


def _cxconst(e):
    """cimo: (real, imaginary or None) Python floats of a constant arithmetic e, or None."""
    if e.kind == EXPRCPLX:
        if e.l.kind == EXPRCONST and e.r.kind == EXPRCONST:
            return e.l.f, e.r.f
        return None
    if e.kind != EXPRCONST or not (e.type.prop & PROPARITH):
        return None
    if e.type.prop & PROPFLOAT:
        return e.f, None
    return float(e.i if e.type.issigned else e.u), None


def _cxmake(t, re, im):
    rd = f32 if t.base.size == 4 else float
    parts = []
    for v in (re, im):
        c = Expr(EXPRCONST, t.base, None)
        c.f = rd(v)
        parts.append(c)
    e = Expr(EXPRCPLX, t, None)
    e.l, e.r = parts
    return e


def _cxeval(expr):
    """cimo: constant folding with complex operands or results (static initializers: `1.0 + 2.0i`)."""
    t = expr.type
    k = expr.kind
    if k == EXPRCPLX:
        expr.l = eval(expr.l)
        expr.r = eval(expr.r)
        return expr
    if k == EXPRCAST:
        expr.base = eval(expr.base)
        p = _cxconst(expr.base)
        if p is None:
            return expr
        re, im = p
        if t.iscomplex:
            return _cxmake(t, re, im or 0.0)
        if t.kind == TYPEBOOL:
            return _intconst(int(re != 0 or (im or 0.0) != 0))
        c = Expr(EXPRCAST, t, Expr(EXPRCONST, expr.base.type.base, None))
        c.base.f = re
        return eval(c)
    if k == EXPRUNARY:
        expr.base = eval(expr.base)
        p = _cxconst(expr.base)
        if p is None:
            return expr
        re, im = p
        return _cxmake(t, -re if expr.op == TSUB else re, -(im or 0.0))
    if k == EXPRBINARY:
        expr.l = eval(expr.l)
        expr.r = eval(expr.r)
        p, q = _cxconst(expr.l), _cxconst(expr.r)
        if p is None or q is None:
            return expr
        (a, b), (c, d) = p, q
        op = expr.op
        if op == TEQL or op == TNEQ:
            same = a == c and (b or 0.0) == (d or 0.0)
            return _intconst(int(same == (op == TEQL)))
        if op == TADD or op == TSUB:
            sg = 1 if op == TADD else -1
            im = b if d is None else (sg * d if b is None else b + sg * d)
            return _cxmake(t, a + sg * c, im or 0.0)
        if op == TMUL:
            if b is None or d is None:
                x, (y, z) = (a, (c, d or 0.0)) if b is None else (c, (a, b))
                return _cxmake(t, x * y, x * z)
            return _cxmake(t, a * c - b * d, a * d + b * c)
        if op == TDIV:
            b = b or 0.0
            if d is None:
                return _cxmake(t, a / c, b / c) if c else expr
            den = c * c + d * d
            if not den:
                return expr
            return _cxmake(t, (a * c + b * d) / den, (b * c - a * d) / den)
    return expr


_M128 = (1 << 128) - 1


def _ival(e):
    """cimo: the value of integer or floating constant e as a Python int or float."""
    t = e.type
    if t.prop & PROPFLOAT:
        return e.f
    if TY.isi128(t):
        u = e.u & _M128
        return u - (1 << 128) if t.issigned and u >> 127 else u
    return e.i if t.issigned else e.u


def _i128const(t, v):
    e = Expr(EXPRCONST, t, None)
    e.u = v & _M128
    return e


def _i128eval(expr):
    """cimo: constant folding with __int128 operands or results (Python ints, wrapping at 128 bits)."""
    t = expr.type
    k = expr.kind
    if k == EXPRCAST:
        expr.base = eval(expr.base)
        b = expr.base
        if b.kind != EXPRCONST or not (b.type.prop & (PROPINT | PROPFLOAT)):
            return expr
        v = _ival(b)
        if TY.isi128(t):
            if isinstance(v, float):
                if v != v or abs(v) >= 2.0 ** 128:
                    return expr
                v = int(v)
            return _i128const(t, v)
        if t.kind == TYPEBOOL:
            return _intconst(int(v != 0))
        c = Expr(EXPRCONST, t, None)
        if t.prop & PROPFLOAT:
            m = abs(v)
            if t.size == 4 and m >> 53:  # rounded once, as in the cast below
                e = m.bit_length() - 26
                m = (m >> e | (1 if m & ((1 << e) - 1) else 0)) << e
            c.f = float(-m if v < 0 else m)
        elif t.prop & PROPINT:
            c.u = v & M64
        else:
            return expr
        cast(c)
        return c
    if k == EXPRUNARY:
        expr.base = eval(expr.base)
        if expr.base.kind != EXPRCONST or expr.op != TSUB:
            return expr
        return _i128const(t, -_ival(expr.base))
    if k == EXPRBINARY:
        expr.l = eval(expr.l)
        expr.r = eval(expr.r)
        l, r = expr.l, expr.r
        if l.kind != EXPRCONST or r.kind != EXPRCONST:
            return expr
        a, b = _ival(l), _ival(r)
        op = expr.op
        if op in (TLESS, TGREATER, TLEQ, TGEQ, TEQL, TNEQ):
            res = {TLESS: a < b, TGREATER: a > b, TLEQ: a <= b, TGEQ: a >= b, TEQL: a == b, TNEQ: a != b}[op]
            return _intconst(int(res))
        if op == TADD:
            v = a + b
        elif op == TSUB:
            v = a - b
        elif op == TMUL:
            v = a * b
        elif op in (TDIV, TMOD):
            if b == 0:
                return expr
            q = abs(a) // abs(b)
            q = q if (a < 0) == (b < 0) else -q
            v = q if op == TDIV else a - q * b
        elif op == TBAND:
            v = a & b
        elif op == TBOR:
            v = a | b
        elif op == TXOR:
            v = a ^ b
        elif op == TSHL:
            if not 0 <= b < 128:
                return expr
            v = a << b
        elif op == TSHR:
            if not 0 <= b < 128:
                return expr
            v = a >> b
        else:
            return expr
        return _i128const(t, v)
    return expr


def _isi128expr(expr):
    t = expr.type
    k = expr.kind
    if t is not None and TY.isi128(t):
        return k in (EXPRCAST, EXPRUNARY, EXPRBINARY)
    if k == EXPRCAST:
        return TY.isi128(expr.base.type)
    if k == EXPRBINARY:
        return TY.isi128(expr.l.type) or TY.isi128(expr.r.type)
    return False


def eval(expr):
    t = expr.type
    k = expr.kind
    if t is not None and TY.isvec(t) or k == EXPRCAST and TY.isvec(expr.base.type):
        return expr  # cimo: vectors are never constants here
    if _isi128expr(expr):
        return _i128eval(expr)
    if (t is not None and t.iscomplex) or k == EXPRCPLX or (
            k in (EXPRCAST, EXPRBINARY) and (getattr(expr.base, 'type', None) is not None and expr.base.type.iscomplex
                                             or k == EXPRBINARY and (expr.l.type.iscomplex or expr.r.type.iscomplex))):
        return _cxeval(expr)
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
            if t.kind == TYPEBOOL:
                # cimo: compares with 0 (C11 6.3.1.2); cproc kept the low bit of an integer
                # ((_Bool)2 was 0) and truncated a float ((_Bool)0.5 was 0, -0.5 an error)
                expr.u = int(_truth(l))
            elif l.type.prop & PROPINT and t.prop & PROPFLOAT:
                n = l.i if l.type.issigned else l.u
                if t.size == 4 and abs(n) >> 53:
                    # cimo: rounded once, to float: the bits below float's 24 (and a guard bit) become one
                    # sticky bit, so the double below is exact and rounds to float as n does (via double,
                    # 2^62 + 2^38 + 1 was a tie, rounded to even: 2^62, not 2^62 + 2^39)
                    m = abs(n)
                    e = m.bit_length() - 26
                    m = (m >> e | (1 if m & ((1 << e) - 1) else 0)) << e
                    n = -m if n < 0 else m
                expr.f = float(n)
            elif l.type.prop & PROPFLOAT and t.prop & PROPINT:
                lf = l.f
                if t.issigned:
                    if lf < -9223372036854775808.0 or lf >= 9223372036854775808.0:
                        error(tok.loc, "integer part of floating-point constant %s cannot be represented as signed integer", _g(lf))
                    expr.i = int(lf) if lf == lf else 0  # fcvtzs: NaN -> 0
                else:
                    # cimo: -1 < lf < 0 truncates to 0, representable (cproc: lf < 0.0)
                    if lf <= -1.0 or lf >= 18446744073709551616.0:
                        error(tok.loc, "integer part of floating-point constant %s cannot be represented as unsigned integer", _g(lf))
                    expr.u = int(lf) if lf == lf else 0  # fcvtzu: NaN -> 0
            else:
                expr.u = l.u
            if t.kind != TYPEBOOL:
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
        elif op == TLOR or op == TLAND:
            # cimo: the result is an int 0 or 1 (C11 6.5.13, 6.5.14); cproc returned the
            # operand itself (`5L || x` was the long 5, `1 && 7` was 7)
            if l.kind == EXPRCONST:
                if _truth(l) == (op == TLOR):
                    return _intconst(1 if op == TLOR else 0)
                if r.kind == EXPRCONST:
                    return _intconst(int(_truth(r)))
        else:
            if l.kind == EXPRCONST and r.kind == EXPRCONST:
                binary(expr, expr.op, l, r)
    return expr


def _truth(e):
    """Whether constant e compares unequal to 0."""
    return e.f != 0 if e.type.prop & PROPFLOAT else e.u != 0


def _intconst(n):
    e = Expr(EXPRCONST, TY.typeint, None)
    e.u = n
    return e


def _g(x):
    """printf %g"""
    if x != x:
        return '-nan' if math.copysign(1.0, x) < 0 else 'nan'
    return '%g' % x
