"""Port of QBE fold.c: boring folding code.

C integer semantics are emulated explicitly: values are held as Python ints and
wrapped with M64 / U32; Con.i holds the signed int64 view. Float to integer
conversions follow aarch64 (fcvtzs/fcvtzu: saturating, NaN gives 0), the host
C QBE runs on.
"""

import math

from . import defs as D
from .defs import (R, RTmp, RCon, CON_Z, CBits, CAddr, Con, Sym, KWIDE, Kw, Kl, Ks, Kd,
                   M64, U32, s64, s32, f32, bitsd, bitsf, Oadd, Osub, Oneg, Odiv, Orem,
                   Oudiv, Ourem, Omul, Oand, Oor, Oxor, Osar, Oshr, Oshl, Oextsb, Oextub,
                   Oextsh, Oextuh, Oextsw, Oextuw, Ostosi, Ostoui, Odtosi, Odtoui, Ocast,
                   Oswtof, Ouwtof, Osltof, Oultof, Oexts, Otruncd, Ocmpw, Ocmpw1, Ocmpl,
                   Ocmpl1, Ocmps, Ocmps1, Ocmpd, Ocmpd1, Ciule, Ciult, Cisle, Cislt, Cisgt,
                   Cisge, Ciugt, Ciuge, Cieq, Cine, Cfle, Cflt, Cfgt, Cfge, Cfne, Cfeq, Cfo,
                   Cfuo, optab, rtype, rval)
from .util import die, err, symeq, newcon

INT64_MIN = -(1 << 63)
INT32_MIN = -(1 << 31)


def iscon(c, w, k):
    if c.type != CBits:
        return 0
    if w:
        return int((c.i & M64) == (k & M64))
    else:
        return int((c.i & U32) == (k & U32))


def _sx(x, n):
    """(intN_t)x, as a Python int"""
    x &= (1 << n) - 1
    return x - (1 << n) if x >> (n - 1) else x


def _cdiv(a, b):
    """C signed division (truncates toward zero)"""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _crem(a, b):
    return a - b * _cdiv(a, b)


def _ftoi(f, lo, hi):
    """aarch64 fcvtz[su]: truncate, saturate to [lo, hi], NaN gives 0"""
    if f != f:
        return 0
    if f <= lo:
        return lo
    if f >= hi:
        return hi
    return max(lo, min(hi, int(f)))


def _i2f32(n):
    """(float)n for an integer n, rounded once (to nearest even)"""
    m = abs(n)
    if m <= 1 << 53:
        return f32(float(n))
    k = m.bit_length() - 53
    q = m >> k
    if m & ((1 << k) - 1):
        q |= 1  # sticky
    x = f32(math.ldexp(float(q), k))
    return -x if n < 0 else x


def foldint(res, op, w, cl, cr):
    """Writes res (a Con); returns 1 on failure."""
    sym = Sym()
    typ = CBits
    ls = cl.i
    rs = cr.i
    lu = ls & M64
    ru = rs & M64
    if op == Oadd:
        if cl.type == CAddr:
            if cr.type == CAddr:
                return 1
            typ = CAddr
            sym = cl.sym.copy()
        elif cr.type == CAddr:
            typ = CAddr
            sym = cr.sym.copy()
    elif op == Osub:
        if cl.type == CAddr:
            if cr.type != CAddr:
                typ = CAddr
                sym = cl.sym.copy()
            elif not symeq(cl.sym, cr.sym):
                return 1
        elif cr.type == CAddr:
            return 1
    elif cl.type == CAddr or cr.type == CAddr:
        return 1
    if op == Odiv or op == Orem or op == Oudiv or op == Ourem:
        if iscon(cr, w, 0):
            return 1
        if op == Odiv or op == Orem:
            x = INT64_MIN if w else INT32_MIN
            if iscon(cr, w, -1):
                if iscon(cl, w, x):
                    return 1
    if op == Oadd:
        x = lu + ru
    elif op == Osub:
        x = lu - ru
    elif op == Oneg:
        x = -lu
    elif op == Odiv:
        x = _cdiv(ls, rs) if w else _cdiv(s32(ls), s32(rs))
    elif op == Orem:
        x = _crem(ls, rs) if w else _crem(s32(ls), s32(rs))
    elif op == Oudiv:
        x = lu // ru if w else (lu & U32) // (ru & U32)
    elif op == Ourem:
        x = lu % ru if w else (lu & U32) % (ru & U32)
    elif op == Omul:
        x = lu * ru
    elif op == Oand:
        x = lu & ru
    elif op == Oor:
        x = lu | ru
    elif op == Oxor:
        x = lu ^ ru
    elif op == Osar:
        x = (ls if w else s32(ls)) >> (ru & (31 | w << 5))
    elif op == Oshr:
        x = (lu if w else lu & U32) >> (ru & (31 | w << 5))
    elif op == Oshl:
        x = lu << (ru & (31 | w << 5))
    elif op == Oextsb:
        x = _sx(lu, 8)
    elif op == Oextub:
        x = lu & 0xff
    elif op == Oextsh:
        x = _sx(lu, 16)
    elif op == Oextuh:
        x = lu & 0xffff
    elif op == Oextsw:
        x = _sx(lu, 32)
    elif op == Oextuw:
        x = lu & U32
    elif op == Ostosi:
        f = cl.s
        x = _ftoi(f, INT64_MIN, (1 << 63) - 1) if w else _ftoi(f, INT32_MIN, (1 << 31) - 1)
    elif op == Ostoui:
        f = cl.s
        x = _ftoi(f, 0, M64) if w else _ftoi(f, 0, U32)
    elif op == Odtosi:
        f = cl.d
        x = _ftoi(f, INT64_MIN, (1 << 63) - 1) if w else _ftoi(f, INT32_MIN, (1 << 31) - 1)
    elif op == Odtoui:
        f = cl.d
        x = _ftoi(f, 0, M64) if w else _ftoi(f, 0, U32)
    elif op == Ocast:
        x = lu
        if cl.type == CAddr:
            typ = CAddr
            sym = cl.sym.copy()
    else:
        if Ocmpw <= op <= Ocmpl1:
            if op <= Ocmpw1:
                ls = s32(lu)
                rs = s32(ru)
                lu = ls & M64
                ru = rs & M64
            else:
                op -= Ocmpl - Ocmpw
            c = op - Ocmpw
            if c == Ciule:
                x = lu <= ru
            elif c == Ciult:
                x = lu < ru
            elif c == Cisle:
                x = ls <= rs
            elif c == Cislt:
                x = ls < rs
            elif c == Cisgt:
                x = ls > rs
            elif c == Cisge:
                x = ls >= rs
            elif c == Ciugt:
                x = lu > ru
            elif c == Ciuge:
                x = lu >= ru
            elif c == Cieq:
                x = lu == ru
            elif c == Cine:
                x = lu != ru
            else:
                die("unreachable")
        elif Ocmps <= op <= Ocmps1 or Ocmpd <= op <= Ocmpd1:
            if op <= Ocmps1:
                c = op - Ocmps
                lf = bitsf(lu)
                rf = bitsf(ru)
            else:
                c = op - Ocmpd
                lf = bitsd(lu)
                rf = bitsd(ru)
            if c == Cfle:
                x = lf <= rf
            elif c == Cflt:
                x = lf < rf
            elif c == Cfgt:
                x = lf > rf
            elif c == Cfge:
                x = lf >= rf
            elif c == Cfne:
                x = lf != rf
            elif c == Cfeq:
                x = lf == rf
            elif c == Cfo:
                x = lf < rf or lf >= rf
            elif c == Cfuo:
                x = not (lf < rf or lf >= rf)
            else:
                die("unreachable")
        else:
            die("unreachable")
        x = int(x)
    res.type = typ
    res.sym = sym
    res.i = s64(x)
    res.flt = 0
    return 0


def foldflt(res, op, w, cl, cr):
    if cl.type != CBits or cr.type != CBits:
        err("invalid address operand for '%s'", optab[op].name)
    res.type = CBits
    res.sym = Sym()
    res.i = 0
    res.flt = 0
    if w:
        ld = cl.d
        rd = cr.d
        if op == Oadd:
            xd = ld + rd
        elif op == Osub:
            xd = ld - rd
        elif op == Oneg:
            xd = -ld
        elif op == Odiv:
            xd = _fdiv(ld, rd)
        elif op == Omul:
            xd = ld * rd
        elif op == Oswtof:
            xd = float(s32(cl.i))
        elif op == Ouwtof:
            xd = float(cl.i & U32)
        elif op == Osltof:
            xd = float(cl.i)
        elif op == Oultof:
            xd = float(cl.i & M64)
        elif op == Oexts:
            xd = cl.s
        elif op == Ocast:
            xd = ld
        else:
            die("unreachable")
        res.d = xd
        res.flt = 2
    else:
        ls = cl.s
        rs = cr.s
        if op == Oadd:
            xs = f32(ls + rs)
        elif op == Osub:
            xs = f32(ls - rs)
        elif op == Oneg:
            xs = -ls
        elif op == Odiv:
            xs = f32(_fdiv(ls, rs))
        elif op == Omul:
            xs = f32(ls * rs)
        elif op == Oswtof:
            xs = _i2f32(s32(cl.i))
        elif op == Ouwtof:
            xs = _i2f32(cl.i & U32)
        elif op == Osltof:
            xs = _i2f32(cl.i)
        elif op == Oultof:
            xs = _i2f32(cl.i & M64)
        elif op == Otruncd:
            xs = f32(cl.d)
        elif op == Ocast:
            xs = ls
        else:
            die("unreachable")
        res.s = xs
        res.flt = 1


def _fdiv(a, b):
    """IEEE division (Python raises on division by zero)"""
    if b == 0.0:
        if a != a:
            return a
        if a == 0.0:
            return math.nan
        neg = (math.copysign(1.0, a) < 0) != (math.copysign(1.0, b) < 0)
        return -math.inf if neg else math.inf
    return a / b


def opfold(op, cls, cl, cr, fn):
    c = Con()
    if cls == Kw or cls == Kl:
        if foldint(c, op, cls == Kl, cl, cr):
            return R
    else:
        foldflt(c, op, cls == Kd, cl, cr)
    if not KWIDE(cls):
        c.i &= 0xffffffff
    r = newcon(c, fn)
    assert not (cls == Ks or cls == Kd) or c.flt
    return r


def foldref(fn, i):
    """used by GVN"""
    if rtype(i.to) != RTmp:
        return R
    if optab[i.op].canfold:
        if rtype(i.arg[0]) != RCon:
            return R
        cl = fn.con[rval(i.arg[0])]
        rr = i.arg[1]
        if rr == R:
            rr = CON_Z
        if rtype(rr) != RCon:
            return R
        cr = fn.con[rval(rr)]

        return opfold(i.op, i.cls, cl, cr, fn)
    return R
