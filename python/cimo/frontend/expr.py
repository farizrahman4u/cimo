"""Port of expr.c."""

import re

from .cc import (Expr, Ref, M64, M32, EXPRIDENT, EXPRCONST, EXPRSTRING, EXPRCALL, EXPRBITFIELD,
                 EXPRINCDEC, EXPRCOMPOUND, EXPRUNARY, EXPRCAST, EXPRBINARY, EXPRCOND, EXPRASSIGN,
                 EXPRCOMMA, EXPRBUILTIN, EXPRTEMP, EXPRSIZEOF, TYPEARRAY, TYPEFUNC, TYPESTRUCT,
                 TYPEUNION, TYPEPOINTER, TYPEBOOL, TYPENULLPTR, TYPEVOID, PROPINT, PROPARITH,
                 PROPSCALAR, PROPREAL, PROPVM, QUALNONE, QUALCONST, DECLOBJECT,
                 DECLBUILTIN, SDSTATIC, SDAUTO, LINKNONE, BUILTINALLOCA, BUILTINCONSTANTP,
                 BUILTINEXPECT, BUILTININFF, BUILTINNANF, BUILTINOFFSETOF,
                 BUILTINTYPESCOMPATIBLEP, BUILTINUNREACHABLE, BUILTINVAARG, BUILTINVACOPY,
                 BUILTINVAEND, BUILTINVASTART,
                 TNONE, TIDENT, TSTRINGLIT, TCHARCONST, TNUMBER, TTRUE, TFALSE, TNULLPTR,
                 TLPAREN, TRPAREN, T_GENERIC, TCOMMA, TDEFAULT, TCOLON, TLBRACK, TRBRACK,
                 TPERIOD, TARROW, TINC, TDEC, TBAND, TMUL, TADD, TSUB, TBNOT, TLNOT, TSIZEOF,
                 TALIGNOF, T__ALIGNOF__, TLBRACE, TLOR, TLAND, TBOR, TXOR, TEQL, TNEQ, TLESS,
                 TGREATER, TLEQ, TGEQ, TSHL, TSHR, TDIV, TMOD, TQUESTION, TASSIGN, TMULASSIGN,
                 TDIVASSIGN, TMODASSIGN, TADDASSIGN, TSUBASSIGN, TSHLASSIGN, TSHRASSIGN,
                 TBANDASSIGN, TXORASSIGN, TBORASSIGN)
from .token import tok, tokenstr, error
from .pp import next, expect, consume
from . import type as TY
from .type import (typecompatible, typesame, typecomposite, typepromote, typecommonreal,
                   mkpointertype, mkarraytype, mkbitinttype, typehasint, typemember)
from . import targ as TG
from . import scope as SC
from . import decl as D
from . import eval as EV
from . import init as IN
from . import utf as UTF
from . import util as U


def mkexpr(k, t, b):
    return Expr(k, t, b)


def delexpr(e):
    """free() only; the port leaves the objects to the garbage collector (the C code
    also unlinks argument and comma lists, which nothing reads afterwards)."""
    pass


def mkconstexpr(t, n):
    e = Expr(EXPRCONST, t, None)
    e.u = n & M64
    return e


def decay(e):
    """6.3.2.1 Conversion of arrays and function designators"""
    # XXX: combine with decl.c:adjust in some way?
    t = e.type
    tq = e.qual
    if t.kind == TYPEARRAY:
        # XXX: qualifiers should be applied to the element type of array types, not the
        # array type itself
        e = mkunaryexpr(TBAND, e)
        e.type = mkpointertype(t.base, t.qual | tq)
        e.decayed = True
    elif t.kind == TYPEFUNC:
        e = mkunaryexpr(TBAND, e)
        e.decayed = True
    return e


def mkunaryexpr(op, base):
    if op == TBAND:
        if base.decayed:
            base = base.base
        # Allow struct and union types even if they are not lvalues, since we take their
        # address when compiling member access.
        k = base.type.kind
        if not base.lvalue and k != TYPEFUNC and k != TYPESTRUCT and k != TYPEUNION:
            error(tok.loc, "'&' operand is not an lvalue or function designator")
        if base.kind == EXPRBITFIELD:
            error(tok.loc, "cannot take address of bit-field")
        expr = Expr(EXPRUNARY, mkpointertype(base.type, base.qual), base)
        expr.op = op
        return expr
    if op == TMUL:
        if base.type.kind != TYPEPOINTER:
            error(tok.loc, "cannot dereference non-pointer")
        if base.kind == EXPRUNARY and base.op == TBAND and base.base.kind != EXPRSTRING:
            type = base.type.base
            expr = base.base
            expr.type = type
        else:
            expr = Expr(EXPRUNARY, base.type.base, base)
            expr.qual = base.type.qual
            expr.lvalue = True
            expr.op = op
        return decay(expr)
    # other unary operators get compiled as equivalent binary ones
    U.fatal("internal error: unknown unary operator %d", op)


def bitfieldwidth(e):
    """Returns unsigned: -1 is M32."""
    if e.kind != EXPRBITFIELD:
        return M32
    return (e.type.size * 8 - e.bits[0] - e.bits[1]) & M32


def exprconvert(e, t):
    if typecompatible(e.type, t):
        return e
    return Expr(EXPRCAST, t, e)


def nullpointer(e):
    if e.kind != EXPRCONST:
        return False
    if e.type.kind == TYPENULLPTR:
        return True
    if not (e.type.prop & PROPINT) and (e.type.kind != TYPEPOINTER or e.type.base is not TY.typevoid):
        return False
    return e.u == 0


def exprassign(e, t):
    et = e.type
    k = t.kind
    if k == TYPEBOOL:
        if not (et.prop & PROPARITH) and et.kind != TYPEPOINTER and et.kind != TYPENULLPTR:
            error(tok.loc, "assignment to bool must be from arithmetic, pointer, or nullptr_t type")
    elif k == TYPEPOINTER:
        if not nullpointer(e):
            if et.kind != TYPEPOINTER:
                error(tok.loc, "assignment to pointer must be from pointer or null pointer constant")
            if t.base is not TY.typevoid and et.base is not TY.typevoid and not typecompatible(t.base, et.base):
                error(tok.loc, "base types of pointer assignment must be compatible or void")
            if (et.qual & t.qual) != et.qual:
                error(tok.loc, "assignment to pointer discards qualifiers")
    elif k == TYPENULLPTR:
        if not nullpointer(e):
            error(tok.loc, "assignment to nullptr_t must be from null pointer constant or expression with type nullptr_t")
    elif k == TYPESTRUCT or k == TYPEUNION:
        if not typecompatible(t, et):
            error(tok.loc, "assignment to %s type must be from compatible type", "union" if k == TYPEUNION else "struct")
    else:
        assert t.prop & PROPARITH
        if not (et.prop & PROPARITH):
            error(tok.loc, "assignment to arithmetic type must be from arithmetic type")
    return exprconvert(e, t)


def exprpromote(e):
    t = typepromote(e.type, bitfieldwidth(e))
    return exprconvert(e, t)


def commonreal(e1, e2):
    """Returns (t, e1, e2)."""
    t = typecommonreal(e1.type, bitfieldwidth(e1), e2.type, bitfieldwidth(e2))
    e1 = exprconvert(e1, t)
    e2 = exprconvert(e2, t)
    return t, e1, e2


def mksizeofexpr(t):
    if t.kind == TYPEARRAY and t.size == 0:
        e = Expr(EXPRSIZEOF, TY.typeulong, None)
        e.sztype = t
    else:
        e = mkconstexpr(TY.typeulong, t.size)
    return e


def mkbinaryexpr(loc, op, l, r):
    t = None
    lp = l.type.prop
    rp = r.type.prop
    if op == TLOR or op == TLAND:
        if not (lp & PROPSCALAR):
            error(loc, "left operand of '%s' operator must be scalar", tokenstr(op))
        if not (rp & PROPSCALAR):
            error(loc, "right operand of '%s' operator must be scalar", tokenstr(op))
        t = TY.typeint
    elif op == TEQL or op == TNEQ:
        t = TY.typeint
        if lp & PROPARITH and rp & PROPARITH:
            _, l, r = commonreal(l, r)
        else:
            if l.type.kind != TYPEPOINTER:
                l, r = r, l
            if l.type.kind != TYPEPOINTER:
                error(loc, "invalid operands to '%s' operator", tokenstr(op))
            if nullpointer(EV.eval(r)):
                r = exprconvert(r, l.type)
            elif nullpointer(EV.eval(l)):
                l = exprconvert(l, r.type)
            else:
                if r.type.kind != TYPEPOINTER:
                    error(loc, "invalid operands to '%s' operator", tokenstr(op))
                if l.type.base.kind == TYPEVOID:
                    l, r = r, l
                if r.type.base.kind == TYPEVOID and l.type.base.kind != TYPEFUNC:
                    r = exprconvert(r, l.type)
                elif not typecompatible(l.type.base, r.type.base):
                    error(loc, "pointer operands to '%s' operator are to incompatible types", tokenstr(op))
    elif op == TLESS or op == TGREATER or op == TLEQ or op == TGEQ:
        t = TY.typeint
        if lp & PROPREAL and rp & PROPREAL:
            _, l, r = commonreal(l, r)
        elif l.type.kind == TYPEPOINTER and r.type.kind == TYPEPOINTER:
            if not typecompatible(l.type.base, r.type.base) or l.type.base.kind == TYPEFUNC:
                error(loc, "pointer operands to '%s' operator must be to compatible object types", tokenstr(op))
        else:
            error(loc, "invalid operands to '%s' operator", tokenstr(op))
    elif op == TBOR or op == TXOR or op == TBAND:
        t, l, r = commonreal(l, r)
    elif op == TADD:
        if lp & PROPARITH and rp & PROPARITH:
            t, l, r = commonreal(l, r)
        else:
            if r.type.kind == TYPEPOINTER:
                l, r = r, l
                rp = lp
            if l.type.kind != TYPEPOINTER or not (rp & PROPINT):
                error(loc, "invalid operands to '+' operator")
            t = l.type
            if t.base.incomplete or t.base.kind == TYPEFUNC:
                error(loc, "pointer operand to '+' must be to complete object type")
            r = mkbinaryexpr(loc, TMUL, exprconvert(r, TY.typeulong), mksizeofexpr(t.base))
    elif op == TSUB:
        if lp & PROPARITH and rp & PROPARITH:
            t, l, r = commonreal(l, r)
        else:
            if l.type.kind != TYPEPOINTER or not (rp & PROPINT) and r.type.kind != TYPEPOINTER:
                error(loc, "invalid operands to '-' operator")
            if l.type.base.incomplete or l.type.base.kind == TYPEFUNC:
                error(loc, "pointer operand to '-' must be to complete object type")
            if rp & PROPINT:
                t = l.type
                r = mkbinaryexpr(loc, TMUL, exprconvert(r, TY.typeulong), mksizeofexpr(t.base))
            else:
                if not typecompatible(l.type.base, r.type.base):
                    error(tok.loc, "pointer operands to '-' are to incompatible types")
                op = TDIV
                t = TY.typelong
                e = mkbinaryexpr(loc, TSUB, exprconvert(l, TY.typelong), exprconvert(r, TY.typelong))
                r = mksizeofexpr(l.type.base)
                l = e
    elif op == TMOD:
        if not (lp & PROPINT) or not (rp & PROPINT):
            error(loc, "operands to '%%' operator must be integer")
        t, l, r = commonreal(l, r)
    elif op == TMUL or op == TDIV:
        if not (lp & PROPARITH) or not (rp & PROPARITH):
            error(loc, "operands to '%s' operator must be arithmetic", tokenstr(op))
        t, l, r = commonreal(l, r)
    elif op == TSHL or op == TSHR:
        if not (lp & PROPINT) or not (rp & PROPINT):
            error(loc, "operands to '%s' operator must be integer", tokenstr(op))
        l = exprpromote(l)
        r = exprpromote(r)
        t = l.type
    else:
        U.fatal("internal error: unknown binary operator %d", op)
    e = Expr(EXPRBINARY, t, None)
    e.op = op
    e.l = l
    e.r = r
    return e


def inttype(val, decimal, suffix):
    U_, L, LL, WB = 1, 2, 4, 8
    types = (TY.typeint, TY.typeuint, TY.typelong, TY.typeulong, TY.typellong, TY.typeullong)

    def invalid():
        error(tok.loc, "invalid integer constant suffix '%s'", suffix)

    # parse suffix
    flags = 0
    s = 0
    n = len(suffix)
    while s < n:
        c = suffix[s]
        c1 = suffix[s + 1] if s + 1 < n else ''
        if c == 'u' or c == 'U':
            new = U_
        elif c == 'l' or c == 'L':
            if c1 == c:
                s += 1
                new = LL
            else:
                new = L
        elif c == 'w':
            if c1 != 'b':
                invalid()
            s += 1
            new = WB
        elif c == 'W':
            if c1 != 'B':
                invalid()
            s += 1
            new = WB
        else:
            invalid()
        if flags & new or flags & ~U_ and new & ~U_:
            invalid()
        flags |= new
        s += 1

    # find integer type
    sign = not (flags & U_)
    if flags & WB:
        i = 1
        while True:
            if i > 64 - sign:
                error(tok.loc, "no suitable type for constant '%s'", tok.lit)
            if val <= M64 >> (64 - i):
                break
            i += 1
        return mkbitinttype(i + sign, sign)
    assert flags < len(types)
    step = 2 if not sign or decimal else 1
    i = flags
    while i < len(types):
        t = types[i]
        if typehasint(t, val, False):
            return t
        i += step
    error(tok.loc, "no suitable type for constant '%s'", tok.lit)


def octval(c):
    if '0' <= c <= '7':
        return ord(c) - 48
    return -1


def hexval(c):
    if '0' <= c <= '9':
        return ord(c) - 48
    if 'a' <= c <= 'f':
        return ord(c) - 97 + 10
    if 'A' <= c <= 'F':
        return ord(c) - 65 + 10
    return -1


_SIMPLEESC = {"'": 0x27, '"': 0x22, '?': 0x3f, '\\': 0x5c, 'a': 7, 'b': 8, 'f': 12, 'n': 10,
              'r': 13, 't': 9, 'v': 11}


def decodechar(src, pos, desc, loc):
    """Decodes one (possibly escaped) character of src at pos.
    Returns (chr, hexoct, number of chars consumed)."""
    hexoct = False
    s = pos
    n = len(src)
    if src[s] == '\\':
        s += 1
        c0 = src[s]
        if c0 in _SIMPLEESC:
            c = _SIMPLEESC[c0]
            s += 1
        elif c0 == 'x':
            s += 1
            c = 0
            v = hexval(src[s] if s < n else '')
            assert v >= 0
            while True:
                if c > M32 // 16:
                    error(tok.loc, "character constant escape is out of range")
                c = c * 16 + v
                s += 1
                v = hexval(src[s] if s < n else '')
                if v < 0:
                    break
            hexoct = True
        else:
            c = 0
            v = octval(c0)
            assert v >= 0
            i = 0
            while True:
                if c > M32 // 8:
                    error(tok.loc, "character constant escape is out of range")
                c = c * 8 + v
                s += 1
                v = octval(src[s] if s < n else '')
                if not (v >= 0):
                    break
                i += 1
                if not (i < 3):
                    break
            hexoct = True
    else:
        c, ln = UTF.utf8dec(src, s, 4)
        if ln == -1:
            error(loc, "%s contains invalid UTF-8", desc)
        s += ln
    return c, hexoct, s - pos


def encodechar8(dst, chr, hexoct):
    if not hexoct:
        dst.extend(UTF.utf8enc(chr))
    else:
        dst.append(chr & 0xff)


def encodechar16(dst, chr, hexoct):
    if not hexoct:
        for u in UTF.utf16enc(chr):
            dst.append(u & 0xff)
            dst.append(u >> 8 & 0xff)
    else:
        dst.append(chr & 0xff)
        dst.append(chr >> 8 & 0xff)


def encodechar32(dst, chr, hexoct):
    dst.append(chr & 0xff)
    dst.append(chr >> 8 & 0xff)
    dst.append(chr >> 16 & 0xff)
    dst.append(chr >> 24 & 0xff)


class StringLit(object):
    """struct stringlit: size (in elements) and data (bytearray of raw bytes)."""
    __slots__ = ('size', 'data')

    def __init__(self):
        self.size = 0
        self.data = None


def stringconcat(st, forceutf8):
    """st is an object with .ssize/.sdata (an Expr) or a StringLit (.size/.data).
    Returns the element type."""
    assert tok.kind == TSTRINGLIT
    parts = []
    kind = ''
    while True:
        src = tok.lit
        p = 0
        c = src[0]
        if c == 'u':
            if src[1] == '8':
                p += 1
            newkind = src[p]
            p += 1
        elif c == 'L' or c == 'U':
            newkind = c
            p += 1
        elif c == '"':
            newkind = ''
        else:
            assert 0
        if kind != newkind and kind and newkind:
            error(tok.loc, "adjacent string literals have differing prefixes")
        if newkind:
            kind = newkind
        parts.append((tok.loc, src, p + 1))
        next()
        if tok.kind != TSTRINGLIT:
            break
    if forceutf8:
        kind = '8'
    if kind == '':
        t = TY.typechar
    elif kind == '8':
        t = TY.typeuchar
    elif kind == 'u':
        t = TY.typeushort
    elif kind == 'U':
        t = TY.typeuint
    elif kind == 'L':
        t = TG.targ.typewchar
    else:
        assert 0
    if t.size == 1:
        width = 1
        encodechar = encodechar8
    elif t.size == 2:
        width = 2
        encodechar = encodechar16
    elif t.size == 4:
        width = 4
        encodechar = encodechar32
    else:
        assert 0
    buf = bytearray()
    for loc, src, p in parts:
        while src[p] != '"':
            chr, hexoct, n = decodechar(src, p, "string literal", loc)
            p += n
            encodechar(buf, chr, hexoct)
    encodechar(buf, 0, False)
    size = len(buf) // width
    if isinstance(st, StringLit):
        st.data = buf
        st.size = size
    else:
        st.sdata = buf
        st.ssize = size
    return t


def generic(s):
    match = None
    dflt = None
    next()
    expect(TLPAREN, "after '_Generic'")
    e = assignexpr(s)
    expect(TCOMMA, "after generic selector expression")
    want = e.type
    delexpr(e)
    while True:
        if consume(TDEFAULT):
            if dflt:
                error(tok.loc, "multiple default expressions in generic association list")
            expect(TCOLON, "after 'default'")
            dflt = assignexpr(s)
        else:
            qual = Ref(QUALNONE)
            t = D.typename(s, qual, None)
            if not t:
                error(tok.loc, "expected typename for generic association")
            if t.kind == TYPEFUNC:
                error(tok.loc, "generic association must have object type")
            if t.incomplete:
                error(tok.loc, "generic association must have complete type")
            if t.prop & PROPVM:
                error(tok.loc, "generic association has variably modified type")
            expect(TCOLON, "after type name")
            e = assignexpr(s)
            if typecompatible(t, want) and qual.v == QUALNONE:
                if match:
                    error(tok.loc, "generic selector matches multiple associations")
                match = e
            else:
                delexpr(e)
        if not consume(TCOMMA):
            break
    expect(TRPAREN, "after generic assocation list")
    if not match:
        if not dflt:
            error(tok.loc, "generic selector matches no associations and no default was specified")
        match = dflt
    elif dflt:
        delexpr(dflt)
    return match


_DECFLT = re.compile(r'(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')
_HEXFLT = re.compile(r'0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?')


def strtod(s):
    """glibc strtod for the strings that can reach it (pp-numbers starting with a digit
    or '.'). Returns (value, number of chars consumed); 0 consumed means no conversion."""
    m = _HEXFLT.match(s)
    if m:
        txt = m.group(0)
        if 'p' not in txt and 'P' not in txt:
            v = float.fromhex(txt + 'p0')
        else:
            try:
                v = float.fromhex(txt)
            except OverflowError:
                v = float('inf')
        return v, m.end()
    m = _DECFLT.match(s)
    if m:
        return float(m.group(0)), m.end()
    return 0.0, 0


_DIGITS = '0123456789abcdefghijklmnopqrstuvwxyz'


def strtoull(s, base):
    """glibc strtoull(s, &end, base) for base 2, 8, 10 or 16. Returns (value, consumed)."""
    i = 0
    n = len(s)
    while i < n and s[i] in ' \t\n\v\f\r':
        i += 1
    neg = False
    if i < n and s[i] in '+-':
        neg = s[i] == '-'
        i += 1
    if base == 16 and s[i:i + 2] in ('0x', '0X') and i + 2 < n and hexval(s[i + 2]) >= 0:
        i += 2
    start = i
    v = 0
    while i < n:
        d = _DIGITS.find(s[i].lower())
        if d < 0 or d >= base:
            break
        v = v * base + d
        i += 1
    if i == start:
        return 0, 0
    if v > M64:
        return M64, i
    return (-v if neg else v) & M64, i


def primaryexpr(s):
    k = tok.kind
    if k == TSTRINGLIT:
        e = Expr(EXPRSTRING, None, None)
        t = stringconcat(e, False)
        e.type = mkarraytype(t, QUALNONE, e.ssize)
        e.lvalue = True
        e = decay(e)
    elif k == TCHARCONST:
        src = tok.lit
        p = 0
        ordinary = False
        c = src[0]
        if c == 'L':
            p += 1
            t = TG.targ.typewchar
        elif c == 'u':
            p += 1
            if src[p] == '8':
                p += 1
                t = TY.typeuchar
            else:
                t = TY.typeushort
        elif c == 'U':
            p += 1
            t = TY.typeuint
        else:
            t = TY.typeuchar
            ordinary = True
        assert src[p] == "'"
        p += 1
        chr, hexoct, n = decodechar(src, p, "character constant", tok.loc)
        p += n
        if hexoct and not typehasint(t, chr, False):
            error(tok.loc, "character constant escape is out of range")
        val = chr
        if ordinary:
            if TY.typechar.issigned:
                val = ((val ^ 0x80) - 0x80) & M64
            t = TY.typeint
        e = mkconstexpr(t, val)
        c = src[p] if p < len(src) else '\0'  # C reads the NUL terminator
        if c != "'":
            error(tok.loc, "character constant contains more than one character: %c", c)
        next()
    elif k == TNUMBER:
        e = Expr(EXPRCONST, None, None)
        lit = tok.lit
        if lit[0] == '0':
            c1 = lit[1:2]
            if c1 == 'x' or c1 == 'X':
                base = 16
            elif c1 == 'b' or c1 == 'B':
                base = 2
            else:
                base = 8
        else:
            base = 10
        chars = ".pP" if base == 16 else ".eE"
        if any(c in lit for c in chars):
            # floating constant
            v, n = strtod(lit)
            e.f = v
            if n == 0:
                error(tok.loc, "invalid floating constant '%s'", lit)
            end = lit[n:]
            if not end:
                e.type = TY.typedouble
            elif (end[0] == 'f' or end[0] == 'F') and len(end) == 1:
                e.type = TY.typefloat
            elif (end[0] == 'l' or end[0] == 'L') and len(end) == 1:
                e.type = TY.typeldouble
            else:
                error(tok.loc, "invalid floating constant suffix '%s'", end)
        else:
            src = lit
            if base == 2:
                src = src[2:]
            # integer constant
            v, n = strtoull(src, base)
            e.u = v
            if n == 0:
                error(tok.loc, "invalid integer constant '%s'", lit)
            e.type = inttype(e.u, base == 10, src[n:])
        next()
    elif k == TTRUE or k == TFALSE:
        e = Expr(EXPRCONST, TY.typebool, None)
        e.u = 1 if tok.kind == TTRUE else 0
        next()
    elif k == TNULLPTR:
        e = Expr(EXPRCONST, TY.typenullptr, None)
        e.u = 0
        next()
    elif k == TLPAREN:
        next()
        e = expr(s)
        expect(TRPAREN, "after expression")
    elif k == T_GENERIC:
        e = generic(s)
    else:
        if tok.kind >= TIDENT:
            d = SC.scopegetdecl(s, tokenstr(tok.kind), 1)
            if not d:
                error(tok.loc, "undeclared identifier: %s", tokenstr(tok.kind))
            e = Expr(EXPRIDENT, d.type, None)
            e.qual = d.qual
            e.lvalue = d.kind == DECLOBJECT
            e.decl = d
            if d.kind != DECLBUILTIN:
                e = decay(e)
            next()
        else:
            error(tok.loc, "expected primary expression")
    return e


def designator(s, t, offset):
    """TODO: merge with init.c:designator(). offset is a Ref."""
    while True:
        if tok.kind == TLBRACK:
            if t.kind != TYPEARRAY:
                error(tok.loc, "index designator is only valid for array types")
            next()
            i = intconstexpr(s, False)
            expect(TRBRACK, "for index designator")
            t = t.base
            offset.v = (offset.v + i * t.size) & M64
        elif tok.kind == TPERIOD:
            if t.kind != TYPESTRUCT and t.kind != TYPEUNION:
                error(tok.loc, "member designator only valid for struct/union types")
            next()
            name = expect(TIDENT, "for member designator")
            m = typemember(t, name, offset)
            if not m:
                error(tok.loc, "%s has no member named '%s'", "union" if t.kind == TYPEUNION else "struct", name)
            t = m.type
        else:
            return


def builtinfunc(s, kind):
    if kind == BUILTINALLOCA:
        e = exprassign(assignexpr(s), TY.typeulong)
        e = Expr(EXPRBUILTIN, mkpointertype(TY.typevoid, QUALNONE), e)
        e.bkind = BUILTINALLOCA
    elif kind == BUILTINCONSTANTP:
        e = mkconstexpr(TY.typeint, 1 if EV.eval(condexpr(s)).kind == EXPRCONST else 0)
    elif kind == BUILTINEXPECT:
        # just a no-op for now
        # TODO: check that the expression and the expected value have type 'long'
        e = assignexpr(s)
        expect(TCOMMA, "after expression")
        delexpr(assignexpr(s))
    elif kind == BUILTININFF:
        e = Expr(EXPRCONST, TY.typefloat, None)
        # TODO: use INFINITY here when we can handle musl's math.h
        e.f = float('inf')
    elif kind == BUILTINNANF:
        e = assignexpr(s)
        if not e.decayed or e.base.kind != EXPRSTRING or e.base.ssize > 1:
            error(tok.loc, "__builtin_nanf currently only supports empty string literals")
        e = Expr(EXPRCONST, TY.typefloat, None)
        # TODO: use NAN here when we can handle musl's math.h
        e.f = float('nan')
    elif kind == BUILTINOFFSETOF:
        t = D.typename(s, None, None)
        expect(TCOMMA, "after type name")
        name = expect(TIDENT, "after ','")
        if t.kind != TYPESTRUCT and t.kind != TYPEUNION:
            error(tok.loc, "type is not a struct/union type")
        offset = Ref(0)
        m = typemember(t, name, offset)
        if not m:
            error(tok.loc, "struct/union has no member named '%s'", name)
        designator(s, m.type, offset)
        e = mkconstexpr(TY.typeulong, offset.v)
    elif kind == BUILTINTYPESCOMPATIBLEP:
        t = D.typename(s, None, None)
        expect(TCOMMA, "after type name")
        e = mkconstexpr(TY.typeint, 1 if typecompatible(t, D.typename(s, None, None)) else 0)
    elif kind == BUILTINUNREACHABLE:
        e = Expr(EXPRBUILTIN, TY.typevoid, None)
        e.bkind = BUILTINUNREACHABLE
    elif kind == BUILTINVAARG:
        e = Expr(EXPRBUILTIN, None, assignexpr(s))
        e.bkind = BUILTINVAARG
        if not typesame(e.base.type, TY.typeadjvalist):
            error(tok.loc, "va_arg argument must have type va_list")
        if TY.typeadjvalist is TG.targ.typevalist:
            e.base = mkunaryexpr(TBAND, e.base)
        expect(TCOMMA, "after va_list")
        qual = Ref(e.qual)
        toeval = Ref(None)
        e.type = D.typename(s, qual, toeval)
        e.qual = qual.v
        e.toeval = toeval.v
    elif kind == BUILTINVACOPY:
        e = Expr(EXPRASSIGN, TY.typevoid, None)
        e.l = assignexpr(s)
        if not typesame(e.l.type, TY.typeadjvalist):
            error(tok.loc, "va_copy destination must have type va_list")
        if TY.typeadjvalist is not TG.targ.typevalist:
            e.l = mkunaryexpr(TMUL, e.l)
        expect(TCOMMA, "after target va_list")
        e.r = assignexpr(s)
        if not typesame(e.r.type, TY.typeadjvalist):
            error(tok.loc, "va_copy source must have type va_list")
        if TY.typeadjvalist is not TG.targ.typevalist:
            e.r = mkunaryexpr(TMUL, e.r)
    elif kind == BUILTINVAEND:
        e = assignexpr(s)
        if not typesame(e.type, TY.typeadjvalist):
            error(tok.loc, "va_end argument must have type va_list")
        e = Expr(EXPRCAST, TY.typevoid, e)
    elif kind == BUILTINVASTART:
        e = Expr(EXPRBUILTIN, TY.typevoid, assignexpr(s))
        e.bkind = BUILTINVASTART
        if not typesame(e.base.type, TY.typeadjvalist):
            error(tok.loc, "va_start argument must have type va_list")
        if TY.typeadjvalist is TG.targ.typevalist:
            e.base = mkunaryexpr(TBAND, e.base)
        if consume(TCOMMA):
            delexpr(assignexpr(s))
    else:
        U.fatal("internal error; unknown builtin")
    return e


def mkincdecexpr(op, base, post):
    if not base.lvalue:
        error(tok.loc, "operand of '%s' operator must be an lvalue", tokenstr(op))
    if base.qual & QUALCONST:
        error(tok.loc, "operand of '%s' operator is const qualified", tokenstr(op))
    e = Expr(EXPRINCDEC, base.type, base)
    e.op = op
    e.post = post
    return e


def postfixexpr(s, r):
    if not r:
        r = primaryexpr(s)
    while True:
        k = tok.kind
        if k == TLBRACK:  # subscript
            next()
            arr = r
            idx = expr(s)
            if arr.type.kind != TYPEPOINTER:
                if idx.type.kind != TYPEPOINTER:
                    error(tok.loc, "either array or index must be pointer type")
                arr, idx = idx, arr
            if arr.type.base.incomplete:
                error(tok.loc, "array is pointer to incomplete type")
            if not (idx.type.prop & PROPINT):
                error(tok.loc, "index is not an integer type")
            e = mkunaryexpr(TMUL, mkbinaryexpr(tok.loc, TADD, arr, idx))
            expect(TRBRACK, "after array index")
        elif k == TLPAREN:  # function call
            next()
            if r.kind == EXPRIDENT and r.decl.kind == DECLBUILTIN:
                e = builtinfunc(s, r.decl.builtin)
                expect(TRPAREN, "after builtin parameters")
                r = e
                continue
            if r.type.kind != TYPEPOINTER or r.type.base.kind != TYPEFUNC:
                error(tok.loc, "called object is not a function")
            t = r.type.base
            e = Expr(EXPRCALL, t.base, r)
            e.args = None
            e.nargs = 0
            p = t.params
            last = None
            while tok.kind != TRPAREN:
                if e.args:
                    expect(TCOMMA, "or ')' after function call argument")
                if not p and not t.isvararg:
                    error(tok.loc, "too many arguments for function call")
                a = assignexpr(s)
                if t.isvararg and not p:
                    a = exprpromote(a)
                else:
                    a = exprassign(a, p.type)
                if last is None:
                    e.args = a
                else:
                    last.next = a
                last = a
                e.nargs += 1
                if p:
                    p = p.next
            if p and not t.isvararg:
                error(tok.loc, "not enough arguments for function call")
            e = decay(e)
            next()
        elif k == TPERIOD or k == TARROW:
            if k == TPERIOD:
                r = mkunaryexpr(TBAND, r)
            op = tok.kind
            if r.type.kind != TYPEPOINTER:
                error(tok.loc, "'%s' operator must be applied to pointer to struct/union", tokenstr(op))
            t = r.type.base
            tq = r.type.qual
            if t.kind != TYPESTRUCT and t.kind != TYPEUNION:
                error(tok.loc, "'%s' operator must be applied to pointer to struct/union", tokenstr(op))
            next()
            if tok.kind < TIDENT:
                error(tok.loc, "expected identifier after '%s' operator", tokenstr(op))
            lvalue = op == TARROW or r.base.lvalue
            offset = Ref(0)
            m = typemember(t, tokenstr(tok.kind), offset)
            if not m:
                error(tok.loc, "struct/union has no member named '%s'", tok.lit)
            r = mkbinaryexpr(tok.loc, TADD, exprconvert(r, TY.typeulong), mkconstexpr(TY.typeulong, offset.v))
            r.type = mkpointertype(m.type, tq | m.qual)
            r = mkunaryexpr(TMUL, r)
            r.lvalue = lvalue
            if m.bits[0] or m.bits[1]:
                e = Expr(EXPRBITFIELD, r.type, r)
                e.lvalue = lvalue
                e.bits = m.bits
            else:
                e = r
            next()
        elif k == TINC or k == TDEC:
            e = mkincdecexpr(tok.kind, r, True)
            next()
        else:
            return r
        r = e


def unaryexpr(s):
    op = tok.kind
    if op == TINC or op == TDEC:
        next()
        l = unaryexpr(s)
        e = mkincdecexpr(op, l, False)
    elif op == TBAND or op == TMUL:
        next()
        return mkunaryexpr(op, castexpr(s))
    elif op == TADD:
        next()
        e = castexpr(s)
        if not (e.type.prop & PROPARITH):
            error(tok.loc, "operand of unary '+' operator must have arithmetic type")
        if e.type.prop & PROPINT:
            e = exprpromote(e)
    elif op == TSUB:
        next()
        e = castexpr(s)
        if not (e.type.prop & PROPARITH):
            error(tok.loc, "operand of unary '-' operator must have arithmetic type")
        if e.type.prop & PROPINT:
            e = exprpromote(e)
        e = Expr(EXPRUNARY, e.type, e)
        e.op = op
    elif op == TBNOT:
        next()
        e = castexpr(s)
        if not (e.type.prop & PROPINT):
            error(tok.loc, "operand of '~' operator must have integer type")
        e = exprpromote(e)
        e = mkbinaryexpr(tok.loc, TXOR, e, mkconstexpr(e.type, M64))
    elif op == TLNOT:
        next()
        e = castexpr(s)
        if not (e.type.prop & PROPSCALAR):
            error(tok.loc, "operator '!' must have scalar operand")
        e = mkbinaryexpr(tok.loc, TEQL, e, mkconstexpr(TY.typeint, 0))
    elif op == TSIZEOF or op == TALIGNOF or op == T__ALIGNOF__:
        next()
        if consume(TLPAREN):
            t = D.typename(s, None, None)
            if t:
                expect(TRPAREN, "after type name")
                # might be part of a compound literal
                if op == TSIZEOF and tok.kind == TLBRACE:
                    IN.parseinit(s, t)
                e = None
            else:
                e = expr(s)
                expect(TRPAREN, "after expression")
                if op == TSIZEOF:
                    e = postfixexpr(s, e)
        elif op == TSIZEOF:
            t = None
            e = unaryexpr(s)
        else:
            error(tok.loc, "expected ')' after 'alignof'")
        if not t:
            if e.decayed:
                e = e.base
            if e.kind == EXPRBITFIELD:
                error(tok.loc, "%s operator applied to bitfield expression", tokenstr(op))
            t = e.type
        if t.incomplete:
            error(tok.loc, "%s operator applied to incomplete type", tokenstr(op))
        if t.kind == TYPEFUNC:
            error(tok.loc, "%s operator applied to function type", tokenstr(op))
        if t.kind == TYPEARRAY and t.size == 0 and op == TSIZEOF:
            e = Expr(EXPRSIZEOF, TY.typeulong, e)
            e.sztype = t
        else:
            # XXX: __alignof__ is not the same as alignof on 32-bit archs
            # this needs to be considered if we gain support for those archs
            e = mkconstexpr(TY.typeulong, t.size if op == TSIZEOF else t.align)
    else:
        e = postfixexpr(s, None)
    return e


def castexpr(s):
    ct = None
    r = None
    last = None  # the cast whose base is `*end`; None means end == &r
    while consume(TLPAREN):
        tq = Ref(QUALNONE)
        toeval = Ref(None)
        t = D.typename(s, tq, toeval)
        if not t:
            e = expr(s)
            expect(TRPAREN, "after expression to match '('")
            e = postfixexpr(s, e)
            break
        expect(TRPAREN, "after type name")
        if tok.kind == TLBRACE:
            e = Expr(EXPRCOMPOUND, t, None)
            e.toeval = toeval.v
            e.qual = tq.v
            e.lvalue = True
            d = D.mkdecl(None, DECLOBJECT, t, tq.v, LINKNONE)
            d.storage = SDSTATIC if s is SC.filescope else SDAUTO
            e.decl = d
            e.init = IN.parseinit(s, t)
            e = postfixexpr(s, decay(e))
            break
        if t is not TY.typevoid and not (t.prop & PROPSCALAR):
            error(tok.loc, "cast type must be scalar")
        e = Expr(EXPRCAST, t, None)
        e.toeval = toeval.v
        if last is None:
            r = e
        else:
            last.base = e
        last = e
        ct = t
    else:
        e = unaryexpr(s)

    # done:
    if ct and ct is not TY.typevoid and not (e.type.prop & PROPSCALAR):
        error(tok.loc, "cast operand must have scalar type")
    if last is None:
        r = e
    else:
        last.base = e
    return r


_PREC = {TLOR: 0, TLAND: 1, TBOR: 2, TXOR: 3, TBAND: 4, TEQL: 5, TNEQ: 5, TLESS: 6,
         TGREATER: 6, TLEQ: 6, TGEQ: 6, TSHL: 7, TSHR: 7, TADD: 8, TSUB: 8, TMUL: 9, TDIV: 9,
         TMOD: 9}


def precedence(t):
    return _PREC.get(t, -1)


def binaryexpr(s, l, i):
    if not l:
        l = castexpr(s)
    while True:
        j = _PREC.get(tok.kind, -1)
        if j < i:
            break
        op = tok.kind
        loc = tok.loc
        next()
        r = castexpr(s)
        while True:
            k = _PREC.get(tok.kind, -1)
            if k <= j:
                break
            r = binaryexpr(s, r, k)
        l = mkbinaryexpr(loc, op, l, r)
    return l


def condexpr(s):
    e = binaryexpr(s, None, 0)
    if not consume(TQUESTION):
        return e
    l = e if tok.kind == TCOLON else expr(s)
    expect(TCOLON, "in conditional expression")
    r = condexpr(s)

    lt = l.type
    rt = r.type
    if lt is rt:
        t = lt
    elif lt.prop & PROPARITH and rt.prop & PROPARITH:
        t = typecommonreal(lt, bitfieldwidth(l), rt, bitfieldwidth(r))
    elif lt is TY.typevoid and rt is TY.typevoid:
        t = TY.typevoid
    else:
        l = EV.eval(l)
        r = EV.eval(r)
        if nullpointer(l) and rt.kind == TYPEPOINTER:
            t = rt
        elif nullpointer(r) and lt.kind == TYPEPOINTER:
            t = lt
        elif lt.kind == TYPEPOINTER and rt.kind == TYPEPOINTER:
            tq = lt.qual | rt.qual
            lt = lt.base
            rt = rt.base
            if lt is TY.typevoid or rt is TY.typevoid:
                t = TY.typevoid
            elif typecompatible(lt, rt):
                t = typecomposite(lt, rt)
            else:
                error(tok.loc, "operands of conditional operator must have compatible types")
            t = mkpointertype(t, tq)
        else:
            error(tok.loc, "invalid operands to conditional operator")
    e = EV.eval(e)
    if e.kind == EXPRCONST and e.type.prop & PROPINT:
        return exprconvert(l if e.u else r, t)
    e = Expr(EXPRCOND, t, e)
    e.ct = l
    e.cf = r
    return e


def intconstexpr(s, allowneg):
    e = EV.eval(condexpr(s))
    if e.kind != EXPRCONST or not (e.type.prop & PROPINT):
        error(tok.loc, "not an integer constant expression")
    if not allowneg and e.type.issigned and e.u >> 63:
        error(tok.loc, "integer constant expression cannot be negative")
    return e.u


def mkassignexpr(l, r):
    e = Expr(EXPRASSIGN, l.type, None)
    e.l = l
    e.r = exprassign(r, l.type)
    return e


_ASSIGNOPS = {TASSIGN: TNONE, TMULASSIGN: TMUL, TDIVASSIGN: TDIV, TMODASSIGN: TMOD,
              TADDASSIGN: TADD, TSUBASSIGN: TSUB, TSHLASSIGN: TSHL, TSHRASSIGN: TSHR,
              TBANDASSIGN: TBAND, TXORASSIGN: TXOR, TBORASSIGN: TBOR}


def assignexpr(s):
    l = condexpr(s)
    if l.kind == EXPRBINARY or l.kind == EXPRCOMMA or l.kind == EXPRCAST:
        return l
    op = _ASSIGNOPS.get(tok.kind)
    if op is None:
        return l
    if not l.lvalue:
        error(tok.loc, "left side of assignment expression is not an lvalue")
    next()
    r = assignexpr(s)
    if not op:
        return mkassignexpr(l, r)
    # rewrite `E1 OP= E2` as `T = &E1, *T = *T OP E2`, where T is a temporary slot
    if l.kind == EXPRBITFIELD:
        bit = l
        l = l.base
    else:
        bit = None
    tmp = Expr(EXPRTEMP, mkpointertype(l.type, l.qual), None)
    tmp.lvalue = True
    tmp.temp = None
    e = mkassignexpr(tmp, mkunaryexpr(TBAND, l))
    l = mkunaryexpr(TMUL, tmp)
    if bit:
        bit.base = l
        l = bit
    r = mkbinaryexpr(tok.loc, op, l, r)
    e.next = mkassignexpr(l, r)
    return Expr(EXPRCOMMA, l.type, e)


def expr(s):
    r = None
    last = None
    while True:
        e = assignexpr(s)
        if last is None:
            r = e
        else:
            last.next = e
        last = e
        if tok.kind != TCOMMA:
            break
        next()
    if not r.next:
        return r
    return Expr(EXPRCOMMA, e.type, r)
