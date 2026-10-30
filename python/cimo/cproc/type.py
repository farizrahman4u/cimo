"""Port of type.c."""

from .cc import (Type, M64, M32, TYPEVOID, TYPEBOOL, TYPECHAR, TYPESHORT, TYPEINT,
                 TYPEENUM, TYPELONG, TYPELLONG, TYPEFLOAT, TYPEDOUBLE, TYPELDOUBLE, TYPEPOINTER,
                 TYPEARRAY, TYPEFUNC, TYPESTRUCT, TYPEUNION, TYPENULLPTR, TYPEBITINT, PROPCHAR,
                 PROPINT, PROPREAL, PROPARITH, PROPSCALAR, PROPFLOAT, PROPVM, QUALNONE)
from . import util as U
from . import token as TK


def INTTYPE(k, n, s, p):
    return Type(kind=k, size=n, align=n, issigned=s, width=n * 8,
                prop=PROPSCALAR | PROPARITH | PROPREAL | PROPINT | p)


def FLTTYPE(k, n):
    return Type(kind=k, size=n, align=n, prop=PROPSCALAR | PROPARITH | PROPREAL | PROPFLOAT)


typevoid = Type(kind=TYPEVOID, incomplete=True)

typebool = INTTYPE(TYPEBOOL, 1, False, 0)

typechar = INTTYPE(TYPECHAR, 1, True, PROPCHAR)
typeschar = INTTYPE(TYPECHAR, 1, True, PROPCHAR)
typeuchar = INTTYPE(TYPECHAR, 1, False, PROPCHAR)

typeshort = INTTYPE(TYPESHORT, 2, True, 0)
typeushort = INTTYPE(TYPESHORT, 2, False, 0)

typeint = INTTYPE(TYPEINT, 4, True, 0)
typeuint = INTTYPE(TYPEINT, 4, False, 0)

typelong = INTTYPE(TYPELONG, 8, True, 0)
typeulong = INTTYPE(TYPELONG, 8, False, 0)

typellong = INTTYPE(TYPELLONG, 8, True, 0)
typeullong = INTTYPE(TYPELLONG, 8, False, 0)

typefloat = FLTTYPE(TYPEFLOAT, 4)
typedouble = FLTTYPE(TYPEDOUBLE, 8)
typeldouble = FLTTYPE(TYPELDOUBLE, 16)

typenullptr = Type(kind=TYPENULLPTR, size=8, align=8, prop=PROPSCALAR)

typeadjvalist = None


def mktype(kind, prop):
    return Type(kind, prop)


def mkpointertype(base, qual):
    t = mktype(TYPEPOINTER, PROPSCALAR)
    t.base = base
    t.qual = qual
    t.size = 8
    t.align = 8
    if base:
        t.prop |= base.prop & PROPVM
    return t


def mkarraytype(base, qual, ln):
    t = mktype(TYPEARRAY, 0)
    t.base = base
    t.qual = qual
    t.length = None
    t.ptrqual = QUALNONE
    t.incomplete = not ln
    if t.base:
        t.align = t.base.align
        t.size = (t.base.size * ln) & M64  # XXX: overflow?
    return t


def mkbitinttype(width, sign):
    t = mktype(TYPEBITINT, PROPSCALAR | PROPARITH | PROPREAL | PROPINT)
    t.issigned = sign
    t.width = width
    if width < 1 + sign or width > 64:
        TK.error(TK.tok.loc, "invalid %s _BitInt width %d", "signed" if sign else "unsigned", width)
    # calculate byte size
    t.size = 1
    while t.size * 8 < width:
        t.size <<= 1
    t.align = t.size
    return t


def typerank(t):
    """We define type rank using the number of bits in the type shifted left by 4.
    The least significant 4 bits are used to establish a ranking between integer
    types with the same width (such as long and long long on 64-bit platforms)."""
    if t.kind == TYPEENUM:
        t = t.base
    assert t.prop & PROPINT
    k = t.kind
    if k == TYPEBOOL:
        return 0
    if k == TYPECHAR:
        return 0x081
    if k == TYPESHORT:
        return 0x101
    if k == TYPEINT:
        return 0x201
    if k == TYPELONG:
        return t.size << 3 + 4 | 0x1
    if k == TYPELLONG:
        return 0x402
    if k == TYPEBITINT:
        return t.width << 4
    U.fatal("internal error; unhandled integer type")
    return -1


def typecompatible(t1, t2):
    if t1 is t2:
        return True
    if t1.kind != t2.kind:
        # enum types are compatible with their underlying type, but not with each
        # other (unless they are the same type)
        return (t1.kind == TYPEENUM and t2 is t1.base or
                t2.kind == TYPEENUM and t1 is t2.base)
    k = t1.kind
    if k == TYPEBITINT:
        return t1.width == t2.width and t1.issigned == t2.issigned
    if k == TYPEPOINTER:
        pass
    elif k == TYPEARRAY:
        if not (t1.incomplete or t2.incomplete):
            e1 = t1.length
            e2 = t2.length
            if e1 and e2 and e1.kind == EXPRCONST and e2.kind == EXPRCONST and e1.u != e2.u:
                return False
    elif k == TYPEFUNC:
        if t1.isvararg != t2.isvararg:
            return False
        p1 = t1.params
        p2 = t2.params
        while p1 and p2:
            if not typecompatible(p1.type, p2.type):
                return False
            p1 = p1.next
            p2 = p2.next
        if p1 or p2:
            return False
    else:
        return False
    # derived:
    return t1.qual == t2.qual and typecompatible(t1.base, t2.base)


def typesame(t1, t2):
    # XXX: implement
    return typecompatible(t1, t2)


def typecomposite(t1, t2):
    # XXX: implement 6.2.7
    # XXX: merge with typecompatible?
    return t1


def typepromote(t, width):
    """width is unsigned: -1 is M32."""
    if t is typefloat:
        return typedouble
    if t.kind == TYPEBITINT:
        return t
    if t.prop & PROPINT and (typerank(t) <= typerank(typeint) or width <= typeint.size * 8):
        if width == M32:
            width = t.size * 8
        return typeint if ((width - t.issigned) & M32) < typeint.size * 8 else typeuint
    return t


def typecommonreal(t1, w1, t2, w2):
    assert t1.prop & PROPREAL and t2.prop & PROPREAL
    if t1 is typeldouble or t2 is typeldouble:
        return typeldouble
    if t1 is typedouble or t2 is typedouble:
        return typedouble
    if t1 is typefloat or t2 is typefloat:
        return typefloat
    t1 = typepromote(t1, w1)
    t2 = typepromote(t2, w2)
    if t1 is t2:
        return t1
    if t1.issigned == t2.issigned:
        return t1 if typerank(t1) > typerank(t2) else t2
    if t1.issigned:
        t1, t2 = t2, t1
    # t1 is unsigned and t2 is signed
    if typerank(t1) >= typerank(t2):
        return t1
    # t1 has a lower rank than t2
    if t1.width < t2.width:
        return t2
    k = t2.kind
    if k == TYPEINT:
        return typeuint
    if k == TYPELONG:
        return typeulong
    if k == TYPELLONG:
        return typellong
    U.fatal("internal error; could not find common real type")
    return None


def typeadjust(t, tq):
    """function parameter type adjustment (C11 6.7.6.3p7). tq is a Ref."""
    if t.kind == TYPEARRAY:
        ptrqual = t.ptrqual
        t = mkpointertype(t.base, tq.v | t.qual)
        tq.v = ptrqual
    elif t.kind == TYPEFUNC:
        assert tq.v == QUALNONE
        t = mkpointertype(t, QUALNONE)
    return t


def typemember(t, name, offset):
    """offset is a Ref (unsigned long long *)."""
    assert t.kind == TYPESTRUCT or t.kind == TYPEUNION
    m = t.members
    while m:
        if m.name is not None:
            if m.name == name:
                offset.v = (offset.v + m.offset) & M64
                return m
        else:
            sub = typemember(m.type, name, offset)
            if sub:
                offset.v = (offset.v + m.offset) & M64
                return sub
        m = m.next
    return None


def typehasint(t, i, sign):
    assert t.prop & PROPINT
    if sign and i >= 0x8000000000000000:
        return t.issigned and i >= ((M64 << ((t.size << 3) - 1)) & M64)
    return i <= M64 >> (((8 - t.size) << 3) + t.issigned)


from .cc import EXPRCONST  # noqa: E402
