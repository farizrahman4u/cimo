"""Port of decl.c."""

from .cc import (Decl, Member, ListNode, Attr, Ref, Bitfield, listinsert, M64, ALIGNUP,
                 ALIGNDOWN, TYPESTRUCT, TYPEUNION, TYPEENUM, TYPEARRAY, TYPEFUNC, TYPEPOINTER,
                 TYPEVOID, PROPSCALAR, PROPARITH, PROPREAL, PROPINT, PROPVM, PROPFLEX, QUALNONE,
                 QUALCONST, QUALVOLATILE, QUALRESTRICT, DECLTYPE, DECLOBJECT, DECLFUNC,
                 DECLCONST, LINKNONE, LINKINTERN, LINKEXTERN, SDSTATIC, SDTHREAD, SDAUTO,
                 EXPRCONST, EXPRSTRING, ATTRPACKED, ATTRALIGNED, ATTRNORETURN, BITS0,
                 TTYPEDEF, TEXTERN, TSTATIC, TTHREAD_LOCAL, TAUTO, TREGISTER, TCONST, TVOLATILE,
                 TRESTRICT, T_ATOMIC, TINLINE, T_NORETURN, TSTRUCT, TUNION, TENUM, TIDENT,
                 TCOLON, TLBRACE, TRBRACE, TSEMICOLON, TASSIGN, TCOMMA, TVOID, TCHAR, TSHORT,
                 TINT, TLONG, TFLOAT, TDOUBLE, TSIGNED, TUNSIGNED, T_BITINT, TLPAREN, TRPAREN,
                 TBOOL, T_COMPLEX, TTYPEOF, TTYPEOF_UNQUAL, TALIGNAS, T__ATTRIBUTE__, TMUL,
                 TLBRACK, TRBRACK, TELLIPSIS, TSTATIC_ASSERT, TSTRINGLIT, T__ASM__)
from .token import tok, tokenstr, tokencheck, error
from .pp import next, consume, expect, peek
from . import type as TY
from .type import (mktype, mkpointertype, mkarraytype, mkbitinttype, typehasint, typeadjust,
                   typecompatible, typesame, typecomposite)
from . import scope as SC
from .scope import mkscope, delscope, scopegetdecl, scopeputdecl, scopegettag, scopeputtag
from . import attr as A
from . import expr as E
from . import eval as EV
from . import init as IN
from . import qbe as Q
from . import stmt as ST
from . import util as U

ULLONG_MAX = M64
INT_MAX = 0x7fffffff

tentativedefns = None
tentativedefnsend = None  # last decl in the list (None: &tentativedefns)
strings = {}  # static map in stringdecl()


def _reset():
    global tentativedefns, tentativedefnsend
    tentativedefns = None
    tentativedefnsend = None
    strings.clear()


class QualType(object):
    __slots__ = ('type', 'qual', 'expr')

    def __init__(self, type, qual, expr):
        self.type = type
        self.qual = qual
        self.expr = expr


# enum storageclass
SCNONE = 0
SCTYPEDEF = 1 << 1
SCEXTERN = 1 << 2
SCSTATIC = 1 << 3
SCAUTO = 1 << 4
SCREGISTER = 1 << 5
SCTHREADLOCAL = 1 << 6

# enum typespec
SPECNONE = 0
SPECVOID = 1 << 1
SPECCHAR = 1 << 2
SPECBOOL = 1 << 3
SPECINT = 1 << 4
SPECFLOAT = 1 << 5
SPECDOUBLE = 1 << 6
SPECSHORT = 1 << 7
SPECLONG = 1 << 8
SPECLONG2 = 1 << 9
SPECSIGNED = 1 << 10
SPECUNSIGNED = 1 << 11
SPECCOMPLEX = 1 << 12
SPECBITINT = 1 << 13
SPECLONGLONG = SPECLONG | SPECLONG2

# enum funcspec
FUNCNONE = 0
FUNCINLINE = 1 << 1
FUNCNORETURN = 1 << 2


class StructBuilder(object):
    """last: the member whose `next` is the end of the list (None: t.members)."""
    __slots__ = ('type', 'last', 'bits', 'pack')


def mkdecl(name, k, t, tq, linkage):
    d = Decl(name, k, linkage, t, tq)
    if k == DECLOBJECT and t:
        d.align = t.align
    return d


_SC = {TTYPEDEF: SCTYPEDEF, TEXTERN: SCEXTERN, TSTATIC: SCSTATIC, TTHREAD_LOCAL: SCTHREADLOCAL,
       TAUTO: SCAUTO, TREGISTER: SCREGISTER}


def storageclass(sc):
    """6.7.1 Storage-class specifiers"""
    new = _SC.get(tok.kind)
    if new is None:
        return 0
    if sc is None:
        error(tok.loc, "storage class not allowed in this declaration")
    v = sc.v
    if v == SCNONE:
        allowed = ~SCNONE
    elif v == SCTHREADLOCAL:
        allowed = SCSTATIC | SCEXTERN
    elif v == SCSTATIC or v == SCEXTERN:
        allowed = SCTHREADLOCAL
    else:
        allowed = SCNONE
    if new & ~allowed:
        error(tok.loc, "invalid combination of storage class specifiers")
    sc.v |= new
    next()
    return 1


def typequal(tq):
    """6.7.3 Type qualifiers"""
    k = tok.kind
    if k == TCONST:
        tq.v |= QUALCONST
    elif k == TVOLATILE:
        tq.v |= QUALVOLATILE
    elif k == TRESTRICT:
        tq.v |= QUALRESTRICT
    elif k == T_ATOMIC:
        error(tok.loc, "_Atomic type qualifier is not yet supported")
    else:
        return 0
    next()
    return 1


def funcspec(fs):
    """6.7.4 Function specifiers"""
    k = tok.kind
    if k == TINLINE:
        new = FUNCINLINE
    elif k == T_NORETURN:
        new = FUNCNORETURN
    else:
        return 0
    if fs is None:
        error(tok.loc, "function specifier not allowed in this declaration")
    fs.v |= new
    next()
    return 1


def tagspec(s):
    inttypes = ((TY.typeuint, TY.typeint), (TY.typeulong, TY.typelong),
                (TY.typeullong, TY.typellong))
    allowedattr = 0
    k = tok.kind
    if k == TSTRUCT:
        kind = TYPESTRUCT
        allowedattr |= ATTRPACKED
    elif k == TUNION:
        kind = TYPEUNION
        allowedattr |= ATTRPACKED
    elif k == TENUM:
        kind = TYPEENUM
    else:
        U.fatal("internal error: unknown tag kind")
    next()
    a = Attr()
    a.kind = 0
    A.attr(a, allowedattr)
    A.gnuattr(a, allowedattr)
    tag = None
    t = None
    et = None
    if tok.kind >= TIDENT:
        tag = tokenstr(tok.kind)
        next()
    if kind == TYPEENUM and consume(TCOLON):
        et = declspecs(s, None, None, None).type
        if not et:
            error(tok.loc, "no type in enum type specifier")
    if tag:
        t = scopegettag(s, tag, tok.kind != TLBRACE and tok.kind != TSEMICOLON)
    if t:
        if t.kind != kind:
            error(tok.loc, "redeclaration of tag '%s' with different kind", tag)
    else:
        if kind == TYPEENUM:
            t = mktype(kind, PROPSCALAR | PROPARITH | PROPREAL | PROPINT)
            t.base = et
        else:
            t = mktype(kind, 0)
            t.size = 0
            t.align = 0
            t.tag = tag
            t.members = None
        t.incomplete = True
        if tag:
            scopeputtag(s, tag, t)
    if tok.kind != TLBRACE:
        return t
    if not t.incomplete:
        error(tok.loc, "redefinition of tag '%s'", tag)
    next()
    if t.kind == TYPESTRUCT or t.kind == TYPEUNION:
        b = StructBuilder()
        b.type = t
        b.last = None
        b.bits = 0
        b.pack = bool(a.kind & ATTRPACKED)
        while True:
            structdecl(s, b)
            if tok.kind == TRBRACE:
                break
        if not t.members:
            error(tok.loc, "struct/union has no members")
        next()
        if not b.pack:
            t.size = ALIGNUP(t.size, t.align)
    elif t.kind == TYPEENUM:
        enumconsts = None
        if et:
            t.size = t.base.size
            t.align = t.base.align
            t.issigned = t.base.issigned
            t.width = t.base.width
            t.incomplete = False
            et = t
        else:
            et = TY.typeint
        mx = 0
        mn = 0
        value = 0
        while tok.kind >= TIDENT:
            name = tokenstr(tok.kind)
            next()
            A.attr(None, 0)
            invalid = False
            if consume(TASSIGN):
                e = EV.eval(E.condexpr(s))
                if e.kind != EXPRCONST or not (e.type.prop & PROPINT):
                    error(tok.loc, "expected integer constant expression")
                value = e.u
                if not t.base:
                    et = TY.typeint if typehasint(TY.typeint, value, e.type.issigned) else e.type
                elif not typehasint(et, value, e.type.issigned):
                    invalid = True
            elif value == 0 and not et.issigned or value == 1 << 63 and et.issigned:
                error(tok.loc, "no %ssigned integer type can represent enumerator value", "" if et.issigned else "un")
            elif not typehasint(et, value, et.issigned):
                if t.base:
                    invalid = True
                else:
                    sign = et.issigned
                    i = 0
                    while i < len(inttypes):
                        et = inttypes[i][1 if sign else 0]
                        if typehasint(et, value, sign):
                            break
                        i += 1
                    assert i < len(inttypes)
            if invalid:
                # fixed underlying type
                error(tok.loc, "enumerator '%s' value cannot be represented in underlying type", name)
            d = mkdecl(name, DECLCONST, et, QUALNONE, LINKNONE)
            d.enumconst = value
            d.value = Q.mkintconst(value)
            d.next = enumconsts
            enumconsts = d
            if et.issigned and value >= 1 << 63:
                if ((-value) & M64) > mn:
                    mn = (-value) & M64
            elif value > mx:
                mx = value
            scopeputdecl(s, d)
            if not consume(TCOMMA):
                break
            value = (value + 1) & M64
        expect(TRBRACE, "to close enum specifier")
        if not t.base:
            if mn <= 0x80000000 and mx <= 0x7fffffff:
                t.base = TY.typeint if mn else TY.typeuint
            else:
                sign = mn > 0
                i = 0
                while i < len(inttypes):
                    et = inttypes[i][1 if sign else 0]
                    if typehasint(et, mx, False) and typehasint(et, (-mn) & M64, True):
                        break
                    i += 1
                if i == len(inttypes):
                    error(tok.loc, "no integer type can represent all enumerator values")
                t.base = et
                d = enumconsts
                while d:
                    d.type = t
                    d = d.next
            t.size = t.base.size
            t.align = t.base.align
            t.issigned = t.base.issigned
            t.width = t.base.width
    t.incomplete = False
    return t


_TS = {
    SPECNONE: None,
    SPECCHAR: 'char', SPECSIGNED | SPECCHAR: 'schar', SPECUNSIGNED | SPECCHAR: 'uchar',
    SPECSHORT: 'short', SPECSHORT | SPECINT: 'short', SPECSIGNED | SPECSHORT: 'short',
    SPECSIGNED | SPECSHORT | SPECINT: 'short',
    SPECUNSIGNED | SPECSHORT: 'ushort', SPECUNSIGNED | SPECSHORT | SPECINT: 'ushort',
    SPECINT: 'int', SPECSIGNED: 'int', SPECSIGNED | SPECINT: 'int',
    SPECUNSIGNED: 'uint', SPECUNSIGNED | SPECINT: 'uint',
    SPECLONG: 'long', SPECLONG | SPECINT: 'long', SPECSIGNED | SPECLONG: 'long',
    SPECSIGNED | SPECLONG | SPECINT: 'long',
    SPECUNSIGNED | SPECLONG: 'ulong', SPECUNSIGNED | SPECLONG | SPECINT: 'ulong',
    SPECLONGLONG: 'llong', SPECLONGLONG | SPECINT: 'llong', SPECSIGNED | SPECLONGLONG: 'llong',
    SPECSIGNED | SPECLONGLONG | SPECINT: 'llong',
    SPECUNSIGNED | SPECLONGLONG: 'ullong', SPECUNSIGNED | SPECLONGLONG | SPECINT: 'ullong',
    SPECBITINT: 'sbitint', SPECBITINT | SPECSIGNED: 'sbitint', SPECBITINT | SPECUNSIGNED: 'ubitint',
    SPECFLOAT: 'float', SPECDOUBLE: 'double', SPECLONG | SPECDOUBLE: 'ldouble',
}


def declspecs(s, sc, fs, align):
    """6.7 Declarations. sc, fs, align are Refs or None."""
    ts = SPECNONE
    tq = Ref(QUALNONE)
    ntypes = 0
    typeofexpr = None
    bits = 0

    t = None
    if sc:
        sc.v = SCNONE
    if fs:
        fs.v = FUNCNONE
    if align:
        align.v = 0
    while True:
        if typequal(tq) or storageclass(sc) or funcspec(fs):
            continue
        op = tok.kind
        # 6.7.2 Type specifiers
        if op == TVOID:
            t = TY.typevoid
            ntypes += 1
            next()
        elif op == TCHAR:
            ts |= SPECCHAR
            ntypes += 1
            next()
        elif op == TSHORT:
            if ts & SPECSHORT:
                error(tok.loc, "duplicate 'short'")
            ts |= SPECSHORT
            next()
        elif op == TINT:
            ts |= SPECINT
            ntypes += 1
            next()
        elif op == TLONG:
            if ts & SPECLONG2:
                error(tok.loc, "too many 'long'")
            if ts & SPECLONG:
                ts |= SPECLONG2
            ts |= SPECLONG
            next()
        elif op == TFLOAT:
            ts |= SPECFLOAT
            ntypes += 1
            next()
        elif op == TDOUBLE:
            ts |= SPECDOUBLE
            ntypes += 1
            next()
        elif op == TSIGNED:
            if ts & SPECSIGNED:
                error(tok.loc, "duplicate 'signed'")
            ts |= SPECSIGNED
            next()
        elif op == TUNSIGNED:
            if ts & SPECUNSIGNED:
                error(tok.loc, "duplicate 'unsigned'")
            ts |= SPECUNSIGNED
            next()
        elif op == T_BITINT:
            ts |= SPECBITINT
            ntypes += 1
            next()
            expect(TLPAREN, "after _BitInt")
            bits = E.intconstexpr(s, False)
            expect(TRPAREN, "after _BitInt width")
        elif op == TBOOL:
            t = TY.typebool
            ntypes += 1
            next()
        elif op == T_COMPLEX:
            error(tok.loc, "_Complex is not yet supported")
        elif op == T_ATOMIC:
            error(tok.loc, "_Atomic is not yet supported")
        elif op == TSTRUCT or op == TUNION or op == TENUM:
            t = tagspec(s)
            ntypes += 1
        elif op == TTYPEOF or op == TTYPEOF_UNQUAL:
            next()
            expect(TLPAREN, "after 'typeof'")
            toeval = Ref(typeofexpr)
            t = typename(s, tq, toeval)
            typeofexpr = toeval.v
            if not t:
                e = E.expr(s)
                if e.decayed:
                    e = e.base
                t = e.type
                if op == TTYPEOF:
                    tq.v |= e.qual
                if t.prop & PROPVM:
                    typeofexpr = e
            ntypes += 1
            expect(TRPAREN, "to close 'typeof'")
        # 6.7.5 Alignment specifier
        elif op == TALIGNAS:
            if not align:
                error(tok.loc, "alignment specifier not allowed in this declaration")
            next()
            expect(TLPAREN, "after 'alignas'")
            other = typename(s, None, None)
            i = other.align if other else E.intconstexpr(s, False)
            if i & (i - 1) or i > INT_MAX:
                error(tok.loc, "invalid alignment: %d", i)
            if i > align.v:
                align.v = i
            expect(TRPAREN, "to close 'alignas' specifier")
        elif op == T__ATTRIBUTE__:
            A.gnuattr(None, 0)
        else:
            if op < TIDENT or t or ts:
                break
            d = scopegetdecl(s, tokenstr(tok.kind), 1)
            if not d or d.kind != DECLTYPE:
                break
            t = d.type
            tq.v |= d.qual
            ntypes += 1
            next()
        if ntypes > 1 or (t and ts):
            error(tok.loc, "multiple types in declaration specifiers")
    # done:
    if ts not in _TS:
        error(tok.loc, "invalid combination of type specifiers")
    n = _TS[ts]
    if n is not None:
        if n == 'sbitint':
            t = mkbitinttype(bits, True)
        elif n == 'ubitint':
            t = mkbitinttype(bits, False)
        else:
            t = getattr(TY, 'type' + n)
    if tq.v & QUALRESTRICT:
        # C23 6.7.4.1p2
        other = t
        while other.kind == TYPEARRAY:
            other = other.base
        if not other or other.kind != TYPEPOINTER:
            error(tok.loc, "'restrict' applied to non-pointer type")
        if other.base.kind == TYPEFUNC:
            error(tok.loc, "'restrict' applied to function pointer")
    if not t and (tq.v or sc and sc.v or fs and fs.v):
        error(tok.loc, "declaration has no type specifier")
    # TODO: consider delaying attribute parsing to declarator(), so we can tell the
    # difference between the start of an attribute and an array declarator.
    A.attr(None, 0)
    return QualType(t, tq.v, typeofexpr)


def istypename(s, name):
    """6.7.6 Declarators"""
    d = scopegetdecl(s, name, 1)
    return d and d.kind == DECLTYPE


def _link(t):
    if t.link is None:
        t.link = ListNode(t)
    return t.link


def declaratortypes(s, result, name, align, funcscope, allowabstract):
    """When parsing a declarator, qualifiers for derived types are temporarily stored
    in the `qual` field of the type itself (elsewhere this field is used for the
    qualifiers of the base type). This is corrected in declarator().
    name, align, funcscope are Refs or None."""
    while consume(TMUL):
        A.attr(None, 0)
        tq = Ref(QUALNONE)
        while typequal(tq):
            pass
        t = mkpointertype(None, tq.v)
        listinsert(result, _link(t))
    if name:
        name.v = None
    a = Attr()
    a.kind = 0
    ptr = result.next
    gofunc = False
    if tok.kind == TLPAREN:
        next()
        recurse = True
        if allowabstract:
            k = tok.kind
            if k == TMUL or k == TLPAREN:
                pass
            elif tok.kind >= TIDENT and not istypename(s, tokenstr(tok.kind)):
                pass
            else:
                recurse = False
                gofunc = True
        if recurse:
            declaratortypes(s, result, name, align, funcscope, allowabstract)
            expect(TRPAREN, "after parenthesized declarator")
            allowedattr = -1
    else:
        if tok.kind >= TIDENT:
            if not name:
                error(tok.loc, "identifier not allowed in abstract declarator")
            name.v = tokenstr(tok.kind)
            next()
        elif not allowabstract:
            error(tok.loc, "expected '(' or identifier")
        # the aligned attribute is used in the definition of max_align_t from
        # gcc/clang (used with glibc), dragonflybsd, freebsd, and openbsd
        allowedattr = ATTRALIGNED if align else 0
    while True:
        k = tok.kind
        if gofunc or k == TLPAREN:  # function declarator
            if not gofunc:
                next()
            gofunc = False
            # func:
            t = mktype(TYPEFUNC, 0)
            t.qual = QUALNONE
            t.isvararg = False
            t.params = None
            t.nparam = 0
            paramlast = None
            s = mkscope(s)
            d = None
            while True:
                if consume(TELLIPSIS):
                    t.isvararg = True
                    break
                if tok.kind == TRPAREN:
                    break
                d = parameter(s)
                if d.name:
                    scopeputdecl(s, d)
                if paramlast is None:
                    t.params = d
                else:
                    paramlast.next = d
                paramlast = d
                t.nparam += 1
                if not consume(TCOMMA):
                    break
            expect(TRPAREN, "to close function declarator")
            if funcscope and ptr.prev is result:
                # we may need to re-open the scope later if this is a function definition
                funcscope.v = s
                s = s.parent
            else:
                s = delscope(s)
            if t.nparam == 1 and not t.isvararg and d.type.kind == TYPEVOID and not d.name:
                t.params = None
                t.nparam = 0
            listinsert(ptr.prev, _link(t))
            allowedattr = 0
        elif k == TLBRACK:  # array declarator
            if allowedattr != -1 and A.attr(a, allowedattr):
                pass  # goto attr
            else:
                next()
                t = mkarraytype(None, QUALNONE, 0)
                pq = Ref(t.ptrqual)
                while consume(TSTATIC) or typequal(pq):
                    pass
                t.ptrqual = pq.v
                if tok.kind == TMUL and peek(TRBRACK):
                    t.prop |= PROPVM
                    t.incomplete = False
                elif not consume(TRBRACK):
                    e = E.assignexpr(s)
                    if not (e.type.prop & PROPINT):
                        error(tok.loc, "array length expression must have integer type")
                    t.length = e
                    t.incomplete = False
                    expect(TRBRACK, "after array length")
                listinsert(ptr.prev, _link(t))
                allowedattr = 0
                continue
            # attr:
            if ptr.prev is result:
                if a.kind & ATTRALIGNED and a.align > align.v:
                    align.v = a.align
        elif k == T__ATTRIBUTE__:
            if allowedattr == -1:
                error(tok.loc, "attribute not allowed after parenthesized declarator")
            A.gnuattr(a, allowedattr)
            # attr:
            # attribute applies to identifier if ptr->prev == result, otherwise type ptr->prev
            if ptr.prev is result:
                if a.kind & ATTRALIGNED and a.align > align.v:
                    align.v = a.align
        else:
            return


def declarator(s, base, name, align, funcscope, allowabstract):
    if funcscope:
        funcscope.v = None
    result = ListNode()
    result.prev = result.next = result
    declaratortypes(s, result, name, align, funcscope, allowabstract)
    btype = base.type
    bqual = base.qual
    l = result.prev
    while l is not result:
        prev = l.prev
        t = l.owner
        tq = t.qual
        t.base = btype
        t.qual = bqual
        t.prop |= btype.prop & PROPVM
        if t.kind == TYPEFUNC:
            if btype.kind == TYPEFUNC:
                error(tok.loc, "function declarator specifies function return type")
            if btype.kind == TYPEARRAY:
                error(tok.loc, "function declarator specifies array return type")
        elif t.kind == TYPEARRAY:
            if btype.incomplete:
                error(tok.loc, "array element has incomplete type")
            if btype.kind == TYPEFUNC:
                error(tok.loc, "array element has function type")
            if t.ptrqual:
                # TODO: check if we are in a function prototype
                if prev is not result:
                    error(tok.loc, "nested array type has qualifiers")
            t.align = btype.align
            t.size = 0
            if t.length:
                e = EV.eval(t.length)
                if e.kind == EXPRCONST and btype.size:
                    if e.type.issigned and e.u >> 63:
                        error(tok.loc, "array length must be non-negative")
                    if e.u > ULLONG_MAX // btype.size:
                        error(tok.loc, "array length is too large")
                    t.size = btype.size * e.u
                else:
                    t.prop |= PROPVM
                    t.length = e
                    t.asize = None
        btype = t
        bqual = tq
        l = prev
    return QualType(btype, bqual, base.expr)


def parameter(s):
    A.attr(None, 0)
    sc = Ref()
    t = declspecs(s, sc, None, None)
    if not t.type:
        error(tok.loc, "no type in parameter declaration")
    if sc.v and sc.v != SCREGISTER:
        error(tok.loc, "parameter declaration has invalid storage-class specifier")
    name = Ref()
    t = declarator(s, t, name, None, None, True)
    q = Ref(t.qual)
    t.type = typeadjust(t.type, q)
    t.qual = q.v
    d = mkdecl(name.v, DECLOBJECT, t.type, t.qual, LINKNONE)
    d.storage = SDAUTO
    return d


def addmember(b, mt, name, align, width):
    """width is unsigned long long: -1 is M64."""
    t = b.type
    if t.kind == TYPESTRUCT and t.prop & PROPFLEX:
        error(tok.loc, "struct has member '%s' after flexible array member", name)
    if mt.type.incomplete:
        if mt.type.kind != TYPEARRAY:
            error(tok.loc, "struct member '%s' has incomplete type", name)
        t.prop |= PROPFLEX
    if mt.type.prop & PROPFLEX:
        # ISO C doesn't allow structures with members containing FAMs, but allow it as
        # an extension
        t.prop |= PROPFLEX
    if mt.type.kind == TYPEFUNC:
        error(tok.loc, "struct member '%s' has function type", name)
    if mt.type.prop & PROPVM:
        error(tok.loc, "struct member '%s' has variably modified type", name)
    assert mt.type.align > 0
    if name or width == M64:
        m = Member()
        m.type = mt.type
        m.qual = mt.qual
        m.name = name
        m.next = None
        if b.last is None:
            t.members = m
        else:
            b.last.next = m
        b.last = m
    else:
        m = None
    if width == M64:
        before = 0
        after = 0
        m.bits = BITS0
        if align < mt.type.align:
            if align:
                error(tok.loc, "specified alignment of struct member '%s' is less strict than is required by type", name)
            align = 1 if b.pack else mt.type.align
        if t.kind == TYPESTRUCT:
            m.offset = ALIGNUP(t.size, align)
            t.size = m.offset + mt.type.size
        else:
            m.offset = 0
            if t.size < mt.type.size:
                t.size = mt.type.size
        b.bits = 0
    else:  # bit-field
        if not (mt.type.prop & PROPINT):
            error(tok.loc, "bit-field '%s' has invalid type", name)
        if align:
            error(tok.loc, "alignment specified for bit-field '%s'", name)
        if b.pack:
            error(tok.loc, "bit-field '%s' in packed struct is not supported", name)
        if not width and name:
            error(tok.loc, "bit-field '%s' with zero width must not have declarator", name)
        if width > mt.type.width:
            error(tok.loc, "bit-field '%s' exceeds width of underlying type", name)
        align = mt.type.align
        if t.kind == TYPESTRUCT:
            # calculate end of the storage-unit for this bit-field
            end = ALIGNUP(t.size, mt.type.size)
            if not width or width > (end - t.size) * 8 + b.bits:
                # no room, allocate a new storage-unit
                t.size = end
                b.bits = 0
            if m:
                m.offset = ALIGNDOWN(t.size - (1 if b.bits else 0), mt.type.size)
                before = _short((t.size - m.offset) * 8 - b.bits)
                after = _short(mt.type.size * 8 - width - before)
                m.bits = Bitfield(before, after)
            t.size = (t.size + (((width - b.bits + 7) & M64) // 8)) & M64
            b.bits = ((b.bits - width) % 8) & 0xffffffff
        elif m:
            m.offset = 0
            m.bits = Bitfield(0, _short(mt.type.size * 8 - width))
            if t.size < mt.type.size:
                t.size = mt.type.size
    if m and t.align < align:
        t.align = align


def _short(v):
    v &= 0xffff
    return v - 0x10000 if v & 0x8000 else v


def staticassert(s):
    if not consume(TSTATIC_ASSERT):
        return False
    expect(TLPAREN, "after static_assert")
    c = E.intconstexpr(s, True)
    if consume(TCOMMA):
        tokencheck(tok, TSTRINGLIT, "after static assertion expression")
        msg = E.StringLit()
        E.stringconcat(msg, True)
        if not c:
            txt = bytes(msg.data[:msg.size - 1]).split(b'\0', 1)[0].decode('latin-1')
            error(tok.loc, "static assertion failed: %s", txt)
    elif not c:
        error(tok.loc, "static assertion failed")
    expect(TRPAREN, "after static assertion")
    expect(TSEMICOLON, "after static assertion")
    return True


def structdecl(s, b):
    if staticassert(s):
        return
    A.attr(None, 0)
    align = Ref(0)
    base = declspecs(s, None, None, align)
    if not base.type:
        error(tok.loc, "no type in struct member declaration")
    if tok.kind == TSEMICOLON:
        if (base.type.kind != TYPESTRUCT and base.type.kind != TYPEUNION) or base.type.tag:
            error(tok.loc, "struct declaration must declare at least one member")
        next()
        addmember(b, base, None, align.v, M64)
        return
    while True:
        if consume(TCOLON):
            width = E.intconstexpr(s, False)
            addmember(b, base, None, 0, width)
        else:
            name = Ref()
            mt = declarator(s, base, name, align, None, False)
            width = E.intconstexpr(s, False) if consume(TCOLON) else M64
            addmember(b, mt, name.v, align.v, width)
        if tok.kind == TSEMICOLON:
            break
        expect(TCOMMA, "or ';' after declarator")
    next()


def typename(s, tq, toeval):
    """6.7.7 Type names. tq and toeval are Refs or None."""
    t = declspecs(s, None, None, None)
    if t.type:
        t = declarator(s, t, None, None, None, True)
        if tq:
            tq.v |= t.qual
        if toeval:
            toeval.v = t.expr
    return t.type


def getlinkage(kind, sc, prior, filescope):
    if sc & SCSTATIC:
        return LINKINTERN if filescope else LINKNONE
    if sc & SCEXTERN or kind == DECLFUNC:
        return prior.linkage if prior else LINKEXTERN
    return LINKEXTERN if filescope else LINKNONE


def declcommon(s, kind, name, asmname, t, tq, sc, prior):
    kindstr = "function" if kind == DECLFUNC else "object"
    if prior:
        if prior.linkage == LINKNONE:
            error(tok.loc, "%s '%s' with no linkage redeclared", kindstr, name)
        linkage = getlinkage(kind, sc, prior, s is SC.filescope)
        if prior.linkage != linkage:
            error(tok.loc, "%s '%s' redeclared with different linkage", kindstr, name)
        if not typecompatible(t, prior.type) or tq != prior.qual:
            error(tok.loc, "%s '%s' redeclared with incompatible type", kindstr, name)
        if asmname and (not prior.asmname or prior.asmname != asmname):
            error(tok.loc, "%s '%s' redeclared with different assembler name", kindstr, name)
        prior.type = typecomposite(t, prior.type)
        return prior
    if s.parent:
        prior = scopegetdecl(s.parent, name, True)
    linkage = getlinkage(kind, sc, prior, s is SC.filescope)
    if linkage != LINKNONE and s.parent:
        # XXX: should maintain map of identifiers with linkage to their declaration, and use that
        if s.parent is not SC.filescope:
            prior = scopegetdecl(SC.filescope, name, False)
        if prior and prior.linkage != LINKNONE:
            if prior.kind != kind:
                error(tok.loc, "'%s' redeclared with different kind", name)
            if prior.linkage != linkage:
                error(tok.loc, "%s '%s' redeclared with different linkage", kindstr, name)
            if not typecompatible(t, prior.type) or tq != prior.qual:
                error(tok.loc, "%s '%s' redeclared with incompatible type", kindstr, name)
            if not asmname:
                asmname = prior.asmname
            elif not prior.asmname or prior.asmname != asmname:
                error(tok.loc, "%s '%s' redeclared with different assembler name", kindstr, name)
            t = typecomposite(t, prior.type)
    d = mkdecl(name, kind, t, tq, linkage)
    d.asmname = asmname
    scopeputdecl(s, d)
    return d


def defineobj(d, init, hasinit, f):
    if d.type.incomplete:
        error(tok.loc, "object '%s' has incomplete type", d.name)
    if d.align < d.type.align:
        d.align = d.type.align
    if d.storage == SDAUTO:
        Q.funcinit(f, d, init, hasinit)
    else:
        Q.emitdata(d, init)
    d.defined = True


def decl(s, f):
    global tentativedefnsend, tentativedefns
    allowfunc = not f
    if staticassert(s):
        return True
    a = Attr()
    a.kind = 0
    if A.attr(a, ATTRNORETURN) and consume(TSEMICOLON):
        return True
    sc = Ref()
    fs = Ref()
    alignr = Ref()
    base = declspecs(s, sc, fs, alignr)
    sc = sc.v
    fs = fs.v
    align = alignr.v
    if not base.type:
        return False
    if f:
        if sc == SCTHREADLOCAL:
            error(tok.loc, "block scope declaration containing 'thread_local' must contain 'static' or 'extern'")
    else:
        # 6.9p2
        if sc & SCAUTO:
            error(tok.loc, "external declaration must not contain 'auto'")
        if sc & SCREGISTER:
            error(tok.loc, "external declaration must not contain 'register'")
    if consume(TSEMICOLON):
        # XXX 6.7p2 error unless in function parameter/struct/union, or tag/enum members are declared
        return True
    while True:
        namer = Ref()
        funcscoper = Ref()
        qt = declarator(s, base, namer, None, funcscoper, False)
        name = namer.v
        funcscope = funcscoper.v
        t = qt.type
        tq = qt.qual
        if consume(T__ASM__):
            lit = E.StringLit()
            expect(TLPAREN, "after __asm__")
            tokencheck(tok, TSTRINGLIT, "after __asm__")
            E.stringconcat(lit, True)
            asmname = bytes(lit.data).split(b'\0', 1)[0].decode('latin-1')
            expect(TRPAREN, "after assembler name")
            allowfunc = 0
        else:
            asmname = None
        A.gnuattr(a, 0)  # appertains to identifier
        kind = DECLTYPE if sc & SCTYPEDEF else DECLFUNC if t.kind == TYPEFUNC else DECLOBJECT
        prior = scopegetdecl(s, name, False)
        if prior and prior.kind != kind:
            error(tok.loc, "'%s' redeclared with different kind", name)
        if kind == DECLTYPE:
            if align:
                error(tok.loc, "typedef '%s' declared with alignment specifier", name)
            if asmname:
                error(tok.loc, "typedef '%s' declared with assembler label", name)
            if not prior:
                scopeputdecl(s, mkdecl(name, DECLTYPE, t, tq, LINKNONE))
            elif not typesame(prior.type, t) or prior.qual != tq:
                error(tok.loc, "typedef '%s' redefined with different type", name)
        elif kind == DECLOBJECT:
            if align and align < t.align:
                error(tok.loc, "object '%s' requires alignment %d, which is stricter than specified alignment %d", name, t.align, align)
            d = declcommon(s, kind, name, asmname, t, tq, sc, prior)
            if d.align < align:
                d.align = align
            if d.linkage == LINKNONE and not (sc & SCSTATIC):
                d.storage = SDAUTO
            else:
                d.storage = SDTHREAD if sc & SCTHREADLOCAL else SDSTATIC
                if t.prop & PROPVM:
                    error(tok.loc, "object '%s' with %s storage duration cannot have variably modified type", name, "static" if d.storage == SDSTATIC else "thread")
                d.value = Q.mkglobal(d)

            if base.expr:
                Q.funcexpr(f, base.expr)
            init = None
            hasinit = False
            if consume(TASSIGN):
                if f and d.linkage != LINKNONE:
                    error(tok.loc, "object '%s' with block scope and %s linkage cannot have initializer", name, "external" if d.linkage == LINKEXTERN else "internal")
                if d.defined:
                    error(tok.loc, "object '%s' redefined", name)
                init = IN.parseinit(s, d.type)
                hasinit = True
            elif sc & SCEXTERN:
                pass
            elif d.linkage != LINKNONE and d.storage == SDSTATIC:
                if not d.defined and not d.tentative:
                    d.tentative = True
                    if tentativedefnsend is None:
                        tentativedefns = d
                    else:
                        tentativedefnsend.next = d
                    tentativedefnsend = d
            else:
                defineobj(d, init, hasinit, f)
            if hasinit:
                defineobj(d, init, hasinit, f)
        elif kind == DECLFUNC:
            if align:
                error(tok.loc, "function '%s' declared with alignment specifier", name)
            if f and sc and sc != SCEXTERN:  # 6.7.1p7
                error(tok.loc, "function '%s' with block scope may only have storage class 'extern'", name)
            d = declcommon(s, kind, name, asmname, t, tq, sc, prior)
            d.value = Q.mkglobal(d)
            d.inlinedefn = bool(d.linkage == LINKEXTERN and fs & FUNCINLINE and not (sc & SCEXTERN)
                                and (not prior or prior.inlinedefn))
            d.isnoreturn = bool(fs & FUNCNORETURN or a.kind & ATTRNORETURN)
            if tok.kind == TLBRACE:
                if not allowfunc:
                    error(tok.loc, "function definition not allowed")
                if d.defined:
                    error(tok.loc, "function '%s' redefined", name)
                # re-open scope from function declarator
                assert funcscope
                s = funcscope
                f = Q.mkfunc(d, name, t, s)
                ST.stmt(f, s)
                if d.isnoreturn:
                    Q.funchlt(f)
                # XXX: need to keep track of function in case a later declaration specifies extern
                if not d.inlinedefn:
                    Q.emitfunc(f, d.linkage == LINKEXTERN)
                s = delscope(s)
                Q.delfunc(f)
                d.defined = True
                return True
            elif funcscope:
                delscope(funcscope)
        if consume(TSEMICOLON):
            return True
        expect(TCOMMA, "or ';' after declarator")
        allowfunc = 0


def stringdecl(expr):
    assert expr.kind == EXPRSTRING
    key = bytes(expr.sdata[:expr.ssize])
    d = strings.get(key)
    if d is None:
        d = mkdecl("string", DECLOBJECT, expr.type, QUALNONE, LINKNONE)
        d.value = Q.mkglobal(d)
        Q.emitdata(d, IN.mkinit(0, expr.type.size, BITS0, expr))
        strings[key] = d
    return d


def emittentativedefns():
    d = tentativedefns
    while d:
        if not d.defined:
            defineobj(d, None, False, None)
        d = d.next
