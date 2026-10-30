"""Port of qbe.c: the QBE IR back end. Output goes to util.out."""

import math

from .cc import (M64, BITS0, TYPEVOID, TYPEPOINTER, TYPEBOOL, TYPEBITINT, TYPEARRAY,
                 TYPEFUNC, TYPESTRUCT, TYPEUNION, PROPSCALAR, PROPFLOAT, PROPINT, PROPREAL,
                 PROPVM, QUALNONE, QUALCONST, QUALVOLATILE, DECLOBJECT, DECLFUNC, DECLCONST,
                 LINKNONE, LINKEXTERN, SDSTATIC, SDTHREAD, EXPRIDENT, EXPRCONST, EXPRSTRING,
                 EXPRCALL, EXPRBITFIELD, EXPRINCDEC, EXPRCOMPOUND, EXPRUNARY, EXPRCAST,
                 EXPRBINARY, EXPRCOND, EXPRASSIGN, EXPRCOMMA, EXPRBUILTIN, EXPRTEMP, EXPRSIZEOF,
                 BUILTINVASTART, BUILTINVAARG, BUILTINALLOCA, BUILTINUNREACHABLE, TBAND, TMUL,
                 TSUB, TLOR, TLAND, TDIV, TMOD, TADD, TSHL, TSHR, TBOR, TXOR, TLESS, TGREATER,
                 TLEQ, TGEQ, TEQL, TNEQ, TINC)
from .token import tok, error
from . import type as TY
from . import targ as TG
from . import scope as SC
from . import decl as D
from . import eval as EV
from . import tree as TR
from . import util as U

out = U.out
w = out.append

VALUE_NONE = 0
VALUE_GLOBAL = 1
VALUE_INTCONST = 2
VALUE_FLTCONST = 3
VALUE_DBLCONST = 4
VALUE_TEMP = 5
VALUE_TYPE = 6
VALUE_LABEL = 7
VALUE_EXTERN = 1 << 4
VALUE_THREAD = 1 << 5
VALUE_QUOTE = 1 << 6


class Value(object):
    """struct value; union u is name / i / f."""
    __slots__ = ('kind', 'id', 'name', 'i', 'f')

    def __init__(self, kind=VALUE_NONE):
        self.kind = kind
        self.id = 0
        self.name = None
        self.i = 0
        self.f = 0.0


class LValue(object):
    __slots__ = ('addr', 'bits')

    def __init__(self, addr=None, bits=BITS0):
        self.addr = addr
        self.bits = bits


_OPS = [
    ('IADD', 'add'), ('ISUB', 'sub'), ('INEG', 'neg'), ('IDIV', 'div'),
    ('IMUL', 'mul'), ('IUDIV', 'udiv'), ('IREM', 'rem'), ('IUREM', 'urem'),
    ('IOR', 'or'), ('IXOR', 'xor'), ('IAND', 'and'), ('ISAR', 'sar'),
    ('ISHR', 'shr'), ('ISHL', 'shl'), ('ISTORED', 'stored'), ('ISTORES', 'stores'),
    ('ISTOREL', 'storel'), ('ISTOREW', 'storew'), ('ISTOREH', 'storeh'), ('ISTOREB', 'storeb'),
    ('ILOADD', 'loadd'), ('ILOADS', 'loads'), ('ILOADL', 'loadl'), ('ILOADW', 'loadw'),
    ('ILOADSH', 'loadsh'), ('ILOADUH', 'loaduh'), ('ILOADSB', 'loadsb'), ('ILOADUB', 'loadub'),
    ('IALLOC4', 'alloc4'), ('IALLOC8', 'alloc8'), ('IALLOC16', 'alloc16'), ('ICEQW', 'ceqw'),
    ('ICNEW', 'cnew'), ('ICSLEW', 'cslew'), ('ICSLTW', 'csltw'), ('ICSGEW', 'csgew'),
    ('ICSGTW', 'csgtw'), ('ICULEW', 'culew'), ('ICULTW', 'cultw'), ('ICUGEW', 'cugew'),
    ('ICUGTW', 'cugtw'), ('ICEQL', 'ceql'), ('ICNEL', 'cnel'), ('ICSLEL', 'cslel'),
    ('ICSLTL', 'csltl'), ('ICSGEL', 'csgel'), ('ICSGTL', 'csgtl'), ('ICULEL', 'culel'),
    ('ICULTL', 'cultl'), ('ICUGEL', 'cugel'), ('ICUGTL', 'cugtl'), ('ICEQS', 'ceqs'),
    ('ICNES', 'cnes'), ('ICLES', 'cles'), ('ICLTS', 'clts'), ('ICGES', 'cges'),
    ('ICGTS', 'cgts'), ('ICOS', 'cos'), ('ICUOS', 'cuos'), ('ICEQD', 'ceqd'),
    ('ICNED', 'cned'), ('ICLED', 'cled'), ('ICLTD', 'cltd'), ('ICGED', 'cged'),
    ('ICGTD', 'cgtd'), ('ICOD', 'cod'), ('ICUOD', 'cuod'), ('IEXTSW', 'extsw'),
    ('IEXTUW', 'extuw'), ('IEXTSH', 'extsh'), ('IEXTUH', 'extuh'), ('IEXTSB', 'extsb'),
    ('IEXTUB', 'extub'), ('IEXTS', 'exts'), ('ITRUNCD', 'truncd'), ('ISTOSI', 'stosi'),
    ('ISTOUI', 'stoui'), ('IDTOSI', 'dtosi'), ('IDTOUI', 'dtoui'), ('ISWTOF', 'swtof'),
    ('IUWTOF', 'uwtof'), ('ISLTOF', 'sltof'), ('IULTOF', 'ultof'), ('ICAST', 'cast'),
    ('ICOPY', 'copy'), ('ICALL', 'call'), ('IVASTART', 'vastart'), ('IVAARG', 'vaarg'),
]
INONE = 0
IADD = 1
ISUB = 2
INEG = 3
IDIV = 4
IMUL = 5
IUDIV = 6
IREM = 7
IUREM = 8
IOR = 9
IXOR = 10
IAND = 11
ISAR = 12
ISHR = 13
ISHL = 14
ISTORED = 15
ISTORES = 16
ISTOREL = 17
ISTOREW = 18
ISTOREH = 19
ISTOREB = 20
ILOADD = 21
ILOADS = 22
ILOADL = 23
ILOADW = 24
ILOADSH = 25
ILOADUH = 26
ILOADSB = 27
ILOADUB = 28
IALLOC4 = 29
IALLOC8 = 30
IALLOC16 = 31
ICEQW = 32
ICNEW = 33
ICSLEW = 34
ICSLTW = 35
ICSGEW = 36
ICSGTW = 37
ICULEW = 38
ICULTW = 39
ICUGEW = 40
ICUGTW = 41
ICEQL = 42
ICNEL = 43
ICSLEL = 44
ICSLTL = 45
ICSGEL = 46
ICSGTL = 47
ICULEL = 48
ICULTL = 49
ICUGEL = 50
ICUGTL = 51
ICEQS = 52
ICNES = 53
ICLES = 54
ICLTS = 55
ICGES = 56
ICGTS = 57
ICOS = 58
ICUOS = 59
ICEQD = 60
ICNED = 61
ICLED = 62
ICLTD = 63
ICGED = 64
ICGTD = 65
ICOD = 66
ICUOD = 67
IEXTSW = 68
IEXTUW = 69
IEXTSH = 70
IEXTUH = 71
IEXTSB = 72
IEXTUB = 73
IEXTS = 74
ITRUNCD = 75
ISTOSI = 76
ISTOUI = 77
IDTOSI = 78
IDTOUI = 79
ISWTOF = 80
IUWTOF = 81
ISLTOF = 82
IULTOF = 83
ICAST = 84
ICOPY = 85
ICALL = 86
IVASTART = 87
IVAARG = 88
IARG = 89
IVARARG = 90
IFMA = 91  # cimo extension: "fma a, b, c" = a*b + c rounded once (arg2 is c)
# cimo extension: the math instructions (cimo.backend.ext.math), see funcbuiltin
_MATHINST = ('fsqrt', 'fabs', 'fmin', 'fmax', 'ffloor', 'fceil', 'ftrunc', 'fround')
IFSQRT, IFABS, IFMIN, IFMAX, IFFLOOR, IFCEIL, IFTRUNC, IFROUND = range(92, 100)

# cimo extension: contract a*b+c into fma within one expression, like clang's
# -ffp-contract=on; off by default (stock cproc output), set by compile(fma=True)
contract = False

instname = [None] + [_s for _n, _s in _OPS]
# IARG, IVARARG and IFMA are printed by emitinst itself
instname += [None] * (IFSQRT - len(instname)) + list(_MATHINST)

# cimo extension: calls to these C library functions become one IR instruction (and
# fma/fmaf the fma instruction), when the callee is the external function of that name with
# the standard prototype; off by default (stock cproc output), set by compile(builtins=True)
builtins = False
_BUILTINS = {}
for _i, _n in ((IFSQRT, 'sqrt'), (IFABS, 'fabs'), (IFMIN, 'fmin'), (IFMAX, 'fmax'),
               (IFFLOOR, 'floor'), (IFCEIL, 'ceil'), (IFTRUNC, 'trunc'), (IFROUND, 'round'),
               (IFMA, 'fma')):
    _nargs = 3 if _i == IFMA else 2 if _i in (IFMIN, IFMAX) else 1
    _BUILTINS[_n] = (_i, _nargs, 8)
    _BUILTINS[_n + 'f'] = (_i, _nargs, 4)
del _i, _n, _nargs


class QbeType(object):
    __slots__ = ('base', 'data', 'load', 'store')

    def __init__(self, base, data, load, store):
        self.base = base
        self.data = data
        self.load = load
        self.store = store


class Inst(object):
    __slots__ = ('kind', 'cls', 'res', 'arg0', 'arg1', 'arg2')


JUMP_NONE, JUMP_JMP, JUMP_JNZ, JUMP_RET, JUMP_HLT = range(5)


class Block(object):
    __slots__ = ('label', 'insts', 'phi_class', 'phi_blk', 'phi_val', 'phi_res', 'jump_kind',
                 'jump_arg', 'jump_blk', 'next')


class SwitchCase(object):
    """struct switchcase: treenode + body."""
    __slots__ = ('key', 'child', 'height', 'new', 'body')


class Func(object):
    __slots__ = ('decl', 'namedecl', 'name', 'paramtemps', 'type', 'start', 'end', 'gotos',
                 'lastid')


ptrclass = 'l'

_blockid = 0
_globalid = 0
_typeid = 0


def _reset():
    global _blockid, _globalid, _typeid
    _blockid = 0
    _globalid = 0
    _typeid = 0


def switchcase(cases, i, b):
    c = TR.treeinsert(cases, i, SwitchCase)
    if not c.new:
        error(tok.loc, "multiple 'case' labels with same value")
    c.body = b


# values

def mkblock(name):
    global _blockid
    b = Block()
    b.label = Value(VALUE_LABEL)
    b.label.name = name
    _blockid = (_blockid + 1) & 0xffffffff
    b.label.id = _blockid
    b.insts = []
    b.jump_kind = JUMP_NONE
    b.jump_arg = None
    b.jump_blk = [None, None]
    b.phi_class = 0
    b.phi_blk = [None, None]
    b.phi_val = [None, None]
    b.phi_res = Value(VALUE_NONE)
    b.next = None
    return b


def mkglobal(d):
    global _globalid
    v = Value(VALUE_GLOBAL)
    if d.linkage == LINKEXTERN:
        v.kind |= VALUE_EXTERN
    if d.kind == DECLOBJECT and d.storage == SDTHREAD:
        v.kind |= VALUE_THREAD
    if d.asmname is not None:
        v.kind |= VALUE_QUOTE
        v.name = d.asmname
        v.id = 0
    else:
        v.name = d.name
        if d.linkage == LINKNONE:
            _globalid = (_globalid + 1) & 0xffffffff
            v.id = _globalid
        else:
            v.id = 0
    return v


def mkintconst(n):
    v = Value(VALUE_INTCONST)
    v.i = n & M64
    return v


def mkfltconst(kind, n):
    v = Value(kind)
    v.f = n
    return v


_ub = QbeType('w', 'b', ILOADUB, ISTOREB)
_sb = QbeType('w', 'b', ILOADSB, ISTOREB)
_uh = QbeType('w', 'h', ILOADUH, ISTOREH)
_sh = QbeType('w', 'h', ILOADSH, ISTOREH)
_w = QbeType('w', 'w', ILOADW, ISTOREW)
_l = QbeType('l', 'l', ILOADL, ISTOREL)
_s = QbeType('s', 's', ILOADS, ISTORES)
_d = QbeType('d', 'd', ILOADD, ISTORED)
_v = QbeType(0, 0, 0, 0)


def qbetype(t):
    if t is TY.typevoid:
        return _v
    if not (t.prop & PROPSCALAR):
        return _l
    sz = t.size
    if sz == 1:
        return _sb if t.issigned else _ub
    if sz == 2:
        return _sh if t.issigned else _uh
    if sz == 4:
        return _s if t.prop & PROPFLOAT else _w
    if sz == 8:
        return _d if t.prop & PROPFLOAT else _l
    if sz == 16:
        U.fatal("long double is not yet supported")
    assert 0


# functions

def functemp(f, v):
    v.kind = VALUE_TEMP
    v.name = None
    f.lastid = (f.lastid + 1) & 0xffffffff
    v.id = f.lastid


def mkinst(f, op, cls, arg0, arg1):
    inst = Inst()
    inst.kind = op
    inst.cls = cls
    inst.arg0 = arg0
    inst.arg1 = arg1
    inst.res = Value(VALUE_NONE)
    if cls and op != IARG:
        functemp(f, inst.res)
    return inst


def funcinst(f, op, cls, arg0, arg1):
    if f.end.jump_kind:
        b = mkblock("dead")
        funclabel(f, b)
    inst = mkinst(f, op, cls, arg0, arg1)
    f.end.insts.append(inst)
    return inst.res


def funcbits(f, t, v, b):
    cls = 'w' if t.size <= 4 else 'l'
    bits = b[1]
    if not bits and t.kind == TYPEBITINT:
        bits = t.size * 8 - t.width
    if bits:
        bits += (((t.size + 3) & -4) - t.size) << 3
        v = funcinst(f, ISHL, cls, v, mkintconst(bits))
    bits += b[0]
    if bits:
        v = funcinst(f, ISAR if t.issigned else ISHR, cls, v, mkintconst(bits))
    return v


def convert(f, dst, src, l):
    r = None
    if src is dst:
        return l
    if src.kind == TYPEPOINTER:
        src = TY.typeulong
    if dst.kind == TYPEPOINTER:
        dst = TY.typeulong
    if dst.kind == TYPEVOID:
        return None
    if not (src.prop & PROPREAL) or not (dst.prop & PROPREAL):
        U.fatal("internal error; unsupported conversion")
    if dst.kind == TYPEBOOL:
        cls = 'w'
        if src.prop & PROPINT:
            r = mkintconst(0)
            sz = src.size
            if sz == 1:
                op = ICNEW
                if src.kind != TYPEBITINT:
                    l = funcinst(f, IEXTUB, 'w', l, None)
            elif sz == 2:
                op = ICNEW
                if src.kind != TYPEBITINT:
                    l = funcinst(f, IEXTUH, 'w', l, None)
            elif sz == 4:
                op = ICNEW
            elif sz == 8:
                op = ICNEL
            else:
                U.fatal("internal error; unknown integer conversion")
        else:
            assert src.prop & PROPFLOAT
            if src.size == 4:
                op = ICNES
                r = mkfltconst(VALUE_FLTCONST, 0.0)
            elif src.size == 8:
                op = ICNED
                r = mkfltconst(VALUE_DBLCONST, 0.0)
            else:
                U.fatal("internal error; unknown floating point conversion")
    elif dst.prop & PROPINT:
        cls = 'l' if dst.size == 8 else 'w'
        if src.prop & PROPINT:
            if dst.width <= src.width:
                if dst.kind == TYPEBITINT:
                    return funcbits(f, dst, l, BITS0)
                return l
            if src.kind == TYPEBITINT and (src.size == 8 or dst.size == 4):
                return l
            sz = src.size
            if sz == 4:
                op = IEXTSW if src.issigned else IEXTUW
            elif sz == 2:
                op = IEXTSH if src.issigned else IEXTUH
            elif sz == 1:
                op = IEXTSB if src.issigned else IEXTUB
            else:
                U.fatal("internal error; unknown integer conversion")
        else:
            if dst.issigned:
                op = IDTOSI if src.size == 8 else ISTOSI
            else:
                op = IDTOUI if src.size == 8 else ISTOUI
    else:
        cls = 'd' if dst.size == 8 else 's'
        if src.prop & PROPINT:
            if src.size < 4:
                l = convert(f, TY.typeint if src.issigned else TY.typeuint, src, l)
            if src.issigned:
                op = ISLTOF if src.size == 8 else ISWTOF
            else:
                op = IULTOF if src.size == 8 else IUWTOF
        else:
            assert src.prop & PROPFLOAT
            if src.size == dst.size:
                return l
            op = IEXTS if src.size < dst.size else ITRUNCD
    return funcinst(f, op, cls, l, r)


def calcvla(f, t):
    if not (t.prop & PROPVM):
        return
    if t.base:
        calcvla(f, t.base)
    if t.kind == TYPEFUNC or t.size:
        return
    assert t.kind == TYPEARRAY
    if not t.asize:
        assert t.base.size or t.base.kind == TYPEARRAY
        assert t.length
        length = convert(f, TY.typeulong, t.length.type, funcexpr(f, t.length))
        basesize = mkintconst(t.base.size) if t.base.size else t.base.asize
        t.asize = funcinst(f, IMUL, 'l', length, basesize)


def funcalloc(f, d):
    assert not d.type.incomplete
    calcvla(f, d.type)
    end = f.end
    if d.type.size:
        f.end = f.start
        v = mkintconst(d.type.size)
    else:
        assert d.type.kind == TYPEARRAY
        assert d.type.asize
        v = d.type.asize
    align = d.align
    if align in (1, 2, 4):
        op = IALLOC4
    elif align == 8:
        op = IALLOC8
    elif align == 16:
        op = IALLOC16
    else:
        v = funcinst(f, IADD, ptrclass, v, mkintconst(align - 16))
        op = IALLOC16
    v = funcinst(f, op, ptrclass, v, None)
    if align > 16:
        # TODO: implement alloc32 in QBE and use that instead
        v = funcinst(f, IADD, ptrclass, v, mkintconst(align - 16))
        v = funcinst(f, IAND, ptrclass, v, mkintconst(-align))
    d.value = v
    f.end = end


def funccopy(f, dst, src, size, align):
    assert (align & (align - 1)) == 0
    cls = 'w'
    if align == 1:
        load, store = ILOADUB, ISTOREB
    elif align == 2:
        load, store = ILOADUH, ISTOREH
    elif align == 4:
        load, store = ILOADW, ISTOREW
    else:
        load, store, align, cls = ILOADL, ISTOREL, 8, 'l'
    inc = mkintconst(align)
    off = 0
    while True:
        tmp = funcinst(f, load, cls, src, None)
        funcinst(f, store, 0, tmp, dst)
        off += align
        if off >= size:
            break
        src = funcinst(f, IADD, ptrclass, src, inc)
        dst = funcinst(f, IADD, ptrclass, dst, inc)


def funcstore(f, t, tq, lval, v):
    if tq & QUALVOLATILE:
        error(tok.loc, "volatile store is not yet supported")
    if tq & QUALCONST:
        error(tok.loc, "cannot store to 'const' object")
    tp = t.prop
    assert not lval.bits[0] and not lval.bits[1] or tp & PROPINT
    r = v
    k = t.kind
    if k == TYPESTRUCT or k == TYPEUNION or k == TYPEARRAY:
        funccopy(f, lval.addr, v, t.size, t.align)
    else:
        if k == TYPEPOINTER:
            t = TY.typeulong
        assert tp & PROPSCALAR
        qt = qbetype(t)
        bits = lval.bits[0] + lval.bits[1]
        if bits:
            mask = ((M64 >> (64 - t.size * 8 + bits)) << lval.bits[0]) & M64
            v = funcinst(f, ISHL, qt.base, v, mkintconst(lval.bits[0]))
            r = funcbits(f, t, v, lval.bits)
            v = funcinst(f, IAND, qt.base, v, mkintconst(mask))
            v = funcinst(f, IOR, qt.base, v,
                         funcinst(f, IAND, qt.base,
                                  funcinst(f, qt.load, qt.base, lval.addr, None),
                                  mkintconst(~mask & M64)))
        funcinst(f, qt.store, 0, v, lval.addr)
    return r


def funcload(f, t, lval):
    k = t.kind
    if k == TYPESTRUCT or k == TYPEUNION or k == TYPEARRAY:
        return lval.addr
    qt = qbetype(t)
    v = funcinst(f, qt.load, qt.base, lval.addr, None)
    return funcbits(f, t, v, lval.bits)


def mkfunc(decl, name, t, s):
    f = Func()
    f.decl = decl
    f.name = name
    f.type = t
    f.start = f.end = mkblock("start")
    f.lastid = 0
    f.gotos = {}
    emittype(t.base)

    # allocate space for parameters
    f.paramtemps = []
    d = t.params
    while d:
        v = Value()
        f.paramtemps.append(v)
        emittype(d.type)
        functemp(f, v)
        if d.name:
            if d.type.value:
                d.value = v
            else:
                funcalloc(f, d)
                funcstore(f, d.type, QUALNONE, LValue(d.value), v)
        d = d.next

    t = TY.mkarraytype(TY.typechar, QUALCONST, len(name) + 1)
    d = D.mkdecl("__func__", DECLOBJECT, t, QUALNONE, LINKNONE)
    d.storage = SDSTATIC
    d.value = mkglobal(d)
    SC.scopeputdecl(s, d)
    f.namedecl = d

    funclabel(f, mkblock("body"))

    return f


def delfunc(f):
    pass


def functype(f):
    return f.type


def funclabel(f, b):
    f.end.next = b
    f.end = b


def funcjmp(f, l):
    b = f.end
    if not b.jump_kind:
        b.jump_kind = JUMP_JMP
        b.jump_blk[0] = l


def funcjnz(f, v, t, l1, l2):
    b = f.end
    if b.jump_kind:
        return
    if t:
        assert t.prop & PROPSCALAR
        # Ideally we would just do this conversion unconditionally, but QBE is not
        # currently able to optimize the conversion away for int.
        if t.prop & PROPINT and t.size < 4:
            v = convert(f, TY.typeint, t, v)
        elif t.prop & PROPFLOAT or t.size > 4:
            v = convert(f, TY.typebool, t, v)
    b.jump_kind = JUMP_JNZ
    b.jump_arg = v
    b.jump_blk[0] = l1
    b.jump_blk[1] = l2


def funcret(f, v):
    b = f.end
    if not b.jump_kind:
        b.jump_kind = JUMP_RET
        b.jump_arg = v


def funchlt(f):
    b = f.end
    if not b.jump_kind:
        b.jump_kind = JUMP_HLT


class GotoLabel(object):
    __slots__ = ('label', 'defined')


def funcgoto(f, name):
    g = f.gotos.get(name)
    if g is None:
        g = GotoLabel()
        g.defined = False
        g.label = mkblock(name)
        f.gotos[name] = g
    return g


def funclval(f, e):
    lval = LValue()
    if e.kind == EXPRBITFIELD:
        lval.bits = e.bits
        e = e.base
    k = e.kind
    if k == EXPRIDENT:
        d = e.decl
        if d.kind != DECLOBJECT and d.kind != DECLFUNC:
            error(tok.loc, "identifier '%s' is not an object or function", d.name)
        if d is f.namedecl:
            w("data ")
            emitname(d.value)
            w(" = { b \"%s\", b 0 }\n" % f.name)
            f.namedecl = None
        lval.addr = d.value
    elif k == EXPRSTRING:
        d = D.stringdecl(e)
        lval.addr = d.value
    elif k == EXPRCOMPOUND:
        if e.toeval:
            funcexpr(f, e.toeval)
        d = e.decl
        funcinit(f, d, e.init, True)
        lval.addr = d.value
    elif k == EXPRUNARY:
        if e.op != TMUL:
            error(tok.loc, "expression is not an object")
        lval.addr = funcexpr(f, e.base)
    else:
        if e.type.kind != TYPESTRUCT and e.type.kind != TYPEUNION:
            error(tok.loc, "expression is not an object")
        lval.addr = funcexpr(f, e)
    return lval


_one = Value(VALUE_INTCONST)
_one.i = 1


def funcbranch(f, e, bt, bf):
    """returns the expression value in the true branch"""
    # Maybe we we could do something for EXPRCOND as well.
    k = e.kind
    if k == EXPRBINARY:
        l = e.l
        r = e.r
        op = e.op
        if op == TEQL or op == TNEQ:
            r = EV.eval(r)
            if r.kind == EXPRCONST and r.type.prop & PROPINT and r.u == 0:
                if e.op == TEQL:
                    bt, bf = bf, bt
                funcbranch(f, l, bt, bf)
                return _one
        elif op == TLOR or op == TLAND:
            if e.op == TLOR:
                b = mkblock("logic_or")
                funcbranch(f, l, bt, b)
            else:
                b = mkblock("logic_and")
                funcbranch(f, l, b, bf)
            funclabel(f, b)
            funcbranch(f, r, bt, bf)
            return _one
    elif k == EXPRCOMMA:
        e = e.base
        while e.next:
            funcexpr(f, e)
            e = e.next
        return funcbranch(f, e, bt, bf)
    v = funcexpr(f, e)
    funcjnz(f, v, e.type, bt, bf)
    return v


def _binop(op, t):
    if op == TMUL:
        return IMUL
    if op == TDIV:
        return IDIV if not (t.prop & PROPINT) or t.issigned else IUDIV
    if op == TMOD:
        return IREM if t.issigned else IUREM
    if op == TADD:
        return IADD
    if op == TSUB:
        return ISUB
    if op == TSHL:
        return ISHL
    if op == TSHR:
        return ISAR if t.issigned else ISHR
    if op == TBOR:
        return IOR
    if op == TBAND:
        return IAND
    if op == TXOR:
        return IXOR
    fl = t.prop & PROPFLOAT
    small = t.size <= 4
    if op == TLESS:
        if small:
            return ICLTS if fl else ICSLTW if t.issigned else ICULTW
        return ICLTD if fl else ICSLTL if t.issigned else ICULTL
    if op == TGREATER:
        if small:
            return ICGTS if fl else ICSGTW if t.issigned else ICUGTW
        return ICGTD if fl else ICSGTL if t.issigned else ICUGTL
    if op == TLEQ:
        if small:
            return ICLES if fl else ICSLEW if t.issigned else ICULEW
        return ICLED if fl else ICSLEL if t.issigned else ICULEL
    if op == TGEQ:
        if small:
            return ICGES if fl else ICSGEW if t.issigned else ICUGEW
        return ICGED if fl else ICSGEL if t.issigned else ICUGEL
    if op == TEQL:
        if small:
            return ICEQS if fl else ICEQW
        return ICEQD if fl else ICEQL
    if op == TNEQ:
        if small:
            return ICNES if fl else ICNEW
        return ICNED if fl else ICNEL
    return INONE


def _ismul(e, t):
    return e.kind == EXPRBINARY and e.op == TMUL and e.type is t


def funcfma(f, e):
    """cimo extension: a*b + c, c + a*b, a*b - c or c - a*b, where the product is an
    operand of the add/sub itself (no cast, assignment or comma in between), as one fma.
    Same choice as clang: the left operand is fused if it is a product, else the right.
    Negating an fma operand is exact, so a*b - c = fma(a, b, -c), c - a*b = fma(-a, b, c)."""
    t = e.type
    if not (t.prop & PROPFLOAT) or t.size not in (4, 8) or e.op not in (TADD, TSUB):
        return None
    cls = qbetype(t).base
    if _ismul(e.l, t):
        a = funcexpr(f, e.l.l)
        b = funcexpr(f, e.l.r)
        c = funcexpr(f, e.r)
        if e.op == TSUB:
            c = funcinst(f, INEG, cls, c, None)
    elif _ismul(e.r, t):
        c = funcexpr(f, e.l)
        a = funcexpr(f, e.r.l)
        b = funcexpr(f, e.r.r)
        if e.op == TSUB:
            a = funcinst(f, INEG, cls, a, None)
    else:
        return None
    if f.end.jump_kind:
        funclabel(f, mkblock("dead"))
    inst = mkinst(f, IFMA, cls, a, b)
    inst.arg2 = c
    f.end.insts.append(inst)
    return inst.res


def funcbuiltin(f, e):
    """cimo extension: a call of sqrt, fabs, fmin, fmax, floor, ceil, trunc, round, fma (or
    the float versions) as one instruction, if the callee is the declared external function
    with the standard prototype (the names are reserved, C11 7.1.3). The results are the
    C functions' exactly: fminnm/fmaxnm return the number when one operand is a quiet NaN
    (and treat -0 < +0), frinta rounds halfway cases away from zero."""
    b = e.base
    if not (b.kind == EXPRUNARY and b.op == TBAND and b.base.kind == EXPRIDENT):
        return None
    d = b.base.decl
    spec = _BUILTINS.get(d.name)
    if spec is None or d.kind != DECLFUNC or d.linkage != LINKEXTERN or d.asmname:
        return None
    op, nargs, size = spec
    t = TY.typedouble if size == 8 else TY.typefloat
    ft = d.type
    if ft.base is not t or ft.isvararg or e.type is not t:
        return None
    p = ft.params
    args = []
    arg = e.args
    for _ in range(nargs):
        if not p or p.type is not t or not arg or arg.type is not t:
            return None
        args.append(arg)
        p = p.next
        arg = arg.next
    if p or arg:
        return None
    vals = [funcexpr(f, a) for a in args]
    cls = qbetype(t).base
    if op == IFMA:
        if f.end.jump_kind:
            funclabel(f, mkblock("dead"))
        inst = mkinst(f, IFMA, cls, vals[0], vals[1])
        inst.arg2 = vals[2]
        f.end.insts.append(inst)
        return inst.res
    return funcinst(f, op, cls, vals[0], vals[1] if nargs == 2 else None)


def funcexpr(f, e):
    calcvla(f, e.type)
    k = e.kind
    if k == EXPRIDENT:
        d = e.decl
        if d.kind == DECLOBJECT:
            return funcload(f, e.type, LValue(d.value))
        if d.kind == DECLCONST:
            return d.value
        U.fatal("unimplemented declaration kind %d", d.kind)
    elif k == EXPRCONST:
        t = e.type
        if t.prop & PROPINT or t.kind == TYPEPOINTER:
            return mkintconst(e.u)
        assert t.prop & PROPFLOAT
        return mkfltconst(VALUE_FLTCONST if t.size == 4 else VALUE_DBLCONST, e.f)
    elif k == EXPRBITFIELD or k == EXPRCOMPOUND:
        lval = funclval(f, e)
        return funcload(f, e.type, lval)
    elif k == EXPRINCDEC:
        lval = funclval(f, e.base)
        l = funcload(f, e.base.type, lval)
        t = e.type
        if t.kind == TYPEPOINTER:
            if t.base.kind == TYPEARRAY and t.base.size == 0:
                r = t.base.asize
            else:
                r = mkintconst(t.base.size)
        elif t.prop & PROPINT:
            r = mkintconst(1)
        elif t.prop & PROPFLOAT:
            r = mkfltconst(VALUE_FLTCONST if t.size == 4 else VALUE_DBLCONST, 1.0)
        else:
            U.fatal("not a scalar")
        v = funcinst(f, IADD if e.op == TINC else ISUB, qbetype(t).base, l, r)
        v = funcstore(f, e.type, e.qual, lval, v)
        return l if e.post else v
    elif k == EXPRCALL:
        if builtins:
            v = funcbuiltin(f, e)
            if v is not None:
                return v
        argvals = []
        arg = e.args
        while arg:
            emittype(arg.type)
            argvals.append(funcexpr(f, arg))
            arg = arg.next
        t = e.type
        emittype(t)
        v = funcinst(f, ICALL, qbetype(t).base, funcexpr(f, e.base), t.value)
        ftype = e.base.type.base
        arg = e.args
        i = 0
        while arg:
            if ftype.isvararg and i == ftype.nparam:
                funcinst(f, IVARARG, 0, None, None)
            t = arg.type
            funcinst(f, IARG, qbetype(t).base, argvals[i], t.value)
            arg = arg.next
            i += 1
        e = e.base
        if e.kind == EXPRUNARY and e.op == TBAND:
            e = e.base
            if e.kind == EXPRIDENT and e.decl.isnoreturn:
                funchlt(f)
        return v
    elif k == EXPRUNARY:
        op = e.op
        if op == TBAND:
            lval = funclval(f, e.base)
            return lval.addr
        if op == TMUL:
            r = funcexpr(f, e.base)
            return funcload(f, e.type, LValue(r))
        if op == TSUB:
            r = funcexpr(f, e.base)
            return funcinst(f, INEG, qbetype(e.type).base, r, None)
        U.fatal("internal error; unknown unary expression")
    elif k == EXPRCAST:
        if e.toeval:
            funcexpr(f, e.toeval)
        l = funcexpr(f, e.base)
        return convert(f, e.type, e.base.type, l)
    elif k == EXPRBINARY:
        if e.op == TLOR or e.op == TLAND:
            b0 = mkblock("logic_true")
            b1 = mkblock("logic_false")
            b2 = mkblock("logic_join")

            funcbranch(f, e, b0, b1)
            funclabel(f, b0)
            funcjmp(f, b2)
            funclabel(f, b1)

            b2.phi_class = 'w'
            b2.phi_blk[0] = b0
            b2.phi_val[0] = mkintconst(1)
            b2.phi_blk[1] = b1
            b2.phi_val[1] = mkintconst(0)
            functemp(f, b2.phi_res)
            funclabel(f, b2)

            return b2.phi_res
        if contract:
            v = funcfma(f, e)
            if v is not None:
                return v
        l = funcexpr(f, e.l)
        r = funcexpr(f, e.r)
        t = e.l.type
        if t.kind == TYPEPOINTER:
            t = TY.typeulong
        op = _binop(e.op, t)
        if op == INONE:
            U.fatal("internal error; unimplemented binary expression")
        v = funcinst(f, op, qbetype(e.type).base, l, r)
        if t.kind == TYPEBITINT:
            v = funcbits(f, t, v, BITS0)
        return v
    elif k == EXPRCOND:
        b0 = mkblock("cond_true")
        b1 = mkblock("cond_false")
        b2 = mkblock("cond_join")

        v = funcbranch(f, e.base, b0, b1)

        funclabel(f, b0)
        if e.ct is not e.base:
            v = funcexpr(f, e.ct)
        b2.phi_val[0] = convert(f, e.type, e.ct.type, v)
        b2.phi_blk[0] = f.end
        funcjmp(f, b2)

        funclabel(f, b1)
        v = funcexpr(f, e.cf)
        b2.phi_val[1] = convert(f, e.type, e.cf.type, v)
        b2.phi_blk[1] = f.end

        funclabel(f, b2)
        if e.type is TY.typevoid:
            return None
        functemp(f, b2.phi_res)
        b2.phi_class = qbetype(e.type).base
        return b2.phi_res
    elif k == EXPRASSIGN:
        r = funcexpr(f, e.r)
        if e.l.kind == EXPRTEMP:
            e.l.temp = r
        else:
            lval = funclval(f, e.l)
            r = funcstore(f, e.l.type, e.l.qual, lval, r)
        return r
    elif k == EXPRCOMMA:
        e = e.base
        while e.next:
            funcexpr(f, e)
            e = e.next
        return funcexpr(f, e)
    elif k == EXPRBUILTIN:
        bk = e.bkind
        if bk == BUILTINVASTART:
            l = funcexpr(f, e.base)
            funcinst(f, IVASTART, 0, l, None)
        elif bk == BUILTINVAARG:
            if e.toeval:
                funcexpr(f, e.toeval)
            # https://todo.sr.ht/~mcf/cproc/52
            if not (e.type.prop & PROPSCALAR):
                error(tok.loc, "va_arg with non-scalar type is not yet supported")
            l = funcexpr(f, e.base)
            return funcinst(f, IVAARG, qbetype(e.type).base, l, None)
        elif bk == BUILTINALLOCA:
            l = funcexpr(f, e.base)
            return funcinst(f, IALLOC16, ptrclass, l, None)
        elif bk == BUILTINUNREACHABLE:
            funchlt(f)
            return None
        else:
            U.fatal("internal error: unimplemented builtin")
        return None
    elif k == EXPRTEMP:
        assert e.temp
        return e.temp
    elif k == EXPRSIZEOF:
        t = e.sztype
        assert t.kind == TYPEARRAY
        calcvla(f, t)
        # if the sizeof operand has VLA type, we must evaluate it
        if e.base:
            funcexpr(f, e.base)
        return t.asize
    U.fatal("unimplemented expression %d", e.kind)


_zstore = {1: ISTOREB, 2: ISTOREH, 4: ISTOREW, 8: ISTOREL}
_z = Value(VALUE_INTCONST)


def zero(func, addr, align, offset, end):
    a = 1
    while offset < end:
        if ((align - (offset & (align - 1))) & M64) & a:
            tmp = funcinst(func, IADD, ptrclass, addr, mkintconst(offset)) if offset else addr
            funcinst(func, _zstore[a], 0, _z, tmp)
            offset += a
        if a < align:
            a <<= 1


def _strelem(data, i, w):
    if w == 1:
        return data[i]
    return int.from_bytes(data[i * w:i * w + w], 'little')


def funcinit(func, d, init, hasinit):
    offset = 0
    mx = 0
    funcalloc(func, d)
    if not hasinit:
        return
    dst = LValue()
    while init:
        zero(func, d.value, d.type.align, offset, init.start)
        dst.bits = init.bits
        ex = init.expr
        if ex.kind == EXPRSTRING:
            ew = ex.type.base.size
            i = 0
            while i < ex.ssize and i * ew < init.end - init.start:
                v = mkintconst(init.start + i * ew)
                dst.addr = funcinst(func, IADD, ptrclass, d.value, v)
                v = mkintconst(_strelem(ex.sdata, i, ew))
                funcstore(func, ex.type.base, QUALNONE, dst, v)
                i += 1
            offset = init.start + i * ew
        else:
            if offset < init.end and (dst.bits[0] or dst.bits[1]):
                zero(func, d.value, d.type.align, offset, init.end)
            dst.addr = d.value
            # QBE's memopt does not eliminate the store for ptr + 0, so only emit the
            # add if the offset is non-zero
            if init.start > 0:
                dst.addr = funcinst(func, IADD, ptrclass, dst.addr, mkintconst(init.start))
            src = funcexpr(func, ex)
            funcstore(func, ex.type, QUALNONE, dst, src)
            offset = init.end
        if mx < offset:
            mx = offset
        init = init.next
    zero(func, d.value, d.type.align, mx, d.type.size)


def casesearch(f, cls, v, c, defaultlabel, mn, mx):
    if not c:
        funcjmp(f, defaultlabel)
        return
    if mn == mx:
        funcjmp(f, c.body)
        return
    label0 = mkblock("switch_ne")
    label1 = mkblock("switch_lt")
    label2 = mkblock("switch_gt")

    key = mkintconst(c.key)
    res = funcinst(f, ICEQW if cls == 'w' else ICEQL, 'w', v, key)
    funcjnz(f, res, None, c.body, label0)
    funclabel(f, label0)
    res = funcinst(f, ICULTW if cls == 'w' else ICULTL, 'w', v, key)
    funcjnz(f, res, None, label1, label2)
    funclabel(f, label1)
    casesearch(f, cls, v, c.child[0], defaultlabel, mn, (c.key - 1) & M64)
    funclabel(f, label2)
    casesearch(f, cls, v, c.child[1], defaultlabel, (c.key + 1) & M64, mx)


def funcswitch(f, v, c, defaultlabel):
    casesearch(f, qbetype(c.type).base, v, c.root, defaultlabel, 0, M64)


# emit

_sigil = {VALUE_TEMP: '%', VALUE_GLOBAL: '$', VALUE_TYPE: ':', VALUE_LABEL: '@'}


def emitname(v):
    kind = v.kind & 0xf
    sg = _sigil.get(kind)
    if not sg:
        U.fatal("invalid value")
    w(sg)
    if kind == VALUE_GLOBAL and v.id:
        w(".L")
    if v.kind & VALUE_QUOTE:
        w('"')
    if v.name:
        w(v.name)
    if v.kind & VALUE_QUOTE:
        w('"')
    if v.id:
        w(".%d" % v.id)


def fmtg17(x):
    """printf("%.17g", x)"""
    if x != x:
        return '-nan' if math.copysign(1.0, x) < 0 else 'nan'
    return '%.17g' % x


def emitvalue(v):
    kind = v.kind & 0xf
    if kind == VALUE_INTCONST:
        w(str(v.i))
    elif kind == VALUE_FLTCONST:
        w("s_" + fmtg17(v.f))
    elif kind == VALUE_DBLCONST:
        w("d_" + fmtg17(v.f))
    else:
        if kind == VALUE_GLOBAL:
            if v.kind & VALUE_EXTERN:
                w("extern ")
            if v.kind & VALUE_THREAD:
                w("thread ")
        emitname(v)


def emitclass(cls, v):
    if v and v.kind == VALUE_TYPE:
        emitname(v)
    elif cls:
        w(cls)
    else:
        U.fatal("type has no QBE representation")


def emittype(t):
    """XXX: need to consider _Alignas on struct members"""
    global _typeid
    if t.value or t.kind != TYPESTRUCT and t.kind != TYPEUNION:
        return
    t.value = Value(VALUE_TYPE)
    t.value.name = t.tag
    _typeid = (_typeid + 1) & 0xffffffff
    t.value.id = _typeid
    m = t.members
    while m:
        sub = m.type
        while sub.kind == TYPEARRAY:
            sub = sub.base
        emittype(sub)
        m = m.next
    w("type ")
    emitname(t.value)
    if t is TG.targ.typevalist:
        w(" = align %d { %d }\n" % (t.align, t.size))
        return
    w(" = { ")
    m = t.members
    off = 0
    while m:
        if t.kind == TYPESTRUCT:
            # look for a subsequent member with a larger storage unit
            other = m.next
            while other:
                if other.offset >= ((m.offset + 1 + 7) & -8):
                    break
                if other.offset <= m.offset:
                    m = other
                other = other.next
            off = m.offset + m.type.size
        else:
            w("{ ")
        sub = m.type
        while sub.kind == TYPEARRAY:
            sub = sub.base
        emitclass(qbetype(sub).data, sub.value)
        if m.type.size > sub.size:
            w(" %d" % (m.type.size // sub.size))
        if t.kind == TYPESTRUCT:
            w(", ")
            # skip subsequent members contained within the same storage unit
            while True:
                m = m.next
                if not (m and m.offset < off):
                    break
        else:
            w(" } ")
            m = m.next
    w("}\n")


def emitinst(insts, i, n):
    inst = insts[i]
    w('\t')
    if inst.kind == IFMA:
        emitvalue(inst.res)
        w(" =")
        emitclass(inst.cls, None)
        w(" fma ")
        emitvalue(inst.arg0)
        w(", ")
        emitvalue(inst.arg1)
        w(", ")
        emitvalue(inst.arg2)
        w('\n')
        return i + 1
    assert inst.kind < len(instname)
    if inst.res.kind:
        emitvalue(inst.res)
        w(" =")
        emitclass(inst.cls, inst.arg1)
        w(' ')
    w(instname[inst.kind])
    w(' ')
    emitvalue(inst.arg0)
    i += 1
    op = inst.kind
    if op == ICALL:
        w('(')
        first = 1
        while i != n:
            inst = insts[i]
            if inst.kind == IVARARG:
                w(", ...")
                i += 1
                continue
            if inst.kind != IARG:
                break
            if first:
                first = 0
            else:
                w(", ")
            emitclass(inst.cls, inst.arg1)
            w(' ')
            emitvalue(inst.arg0)
            i += 1
        w(')')
    else:
        if inst.arg1:
            w(", ")
            emitvalue(inst.arg1)
    w('\n')
    return i


def emitjump(b):
    k = b.jump_kind
    if k == JUMP_NONE:
        pass
    elif k == JUMP_RET:
        w("\tret")
        if b.jump_arg:
            w(' ')
            emitvalue(b.jump_arg)
        w('\n')
    elif k == JUMP_JMP:
        w("\tjmp ")
        emitname(b.jump_blk[0].label)
        w('\n')
    elif k == JUMP_JNZ:
        w("\tjnz ")
        emitvalue(b.jump_arg)
        w(", ")
        emitname(b.jump_blk[0].label)
        w(", ")
        emitname(b.jump_blk[1].label)
        w('\n')
    elif k == JUMP_HLT:
        w("\thlt\n")
    else:
        assert 0


# cimo extension, off by default (None): when set to a dict, emitfunc records for each
# function {name: [(param temporary, is_restrict), ...]} for the middle end's alias analysis
paraminfo = None


def _recordparams(f):
    from .cc import QUALRESTRICT
    out = []
    p = f.type.params
    vi = 0
    while p:
        v = f.paramtemps[vi]
        name = '%' + (v.name or '') + ('.%d' % v.id if v.id else '')
        out.append((name, bool(p.qual & QUALRESTRICT)))
        p = p.next
        vi += 1
    paraminfo[f.decl.value.name] = out


def emitfunc(f, glob):
    if f.end.jump_kind == JUMP_NONE:
        v = None
        # implicitly return 0 from main if we reach the end of the function
        if f.name == "main" and f.type.base is TY.typeint:
            v = mkintconst(0)
        funcret(f, v)
    if glob:
        w("export\n")
    w("function ")
    if f.type.base is not TY.typevoid:
        emitclass(qbetype(f.type.base).base, f.type.base.value)
        w(' ')
    emitname(f.decl.value)
    w('(')
    p = f.type.params
    vi = 0
    while p:
        if p is not f.type.params:
            w(", ")
        emitclass(qbetype(p.type).base, p.type.value)
        w(' ')
        emitname(f.paramtemps[vi])
        p = p.next
        vi += 1
    if f.type.isvararg:
        if f.type.params:
            w(", ")
        w("...")
    w(") {\n")
    if paraminfo is not None:
        _recordparams(f)
    b = f.start
    while b:
        emitname(b.label)
        w('\n')
        if b.phi_res.kind:
            w('\t')
            emitvalue(b.phi_res)
            w(" =%s phi " % b.phi_class)
            emitname(b.phi_blk[0].label)
            w(' ')
            emitvalue(b.phi_val[0])
            w(", ")
            emitname(b.phi_blk[1].label)
            w(' ')
            emitvalue(b.phi_val[1])
            w('\n')
        insts = b.insts
        n = len(insts)
        i = 0
        while i != n:
            i = emitinst(insts, i, n)
        emitjump(b)
        b = b.next
    w("}\n")


def _isprint(c):
    return 0x20 <= c < 0x7f


def dataitem(expr, size):
    k = expr.kind
    if k == EXPRUNARY:
        if expr.op != TBAND:
            U.fatal("not a address expr")
        expr = expr.base
        if expr.kind != EXPRIDENT:
            error(tok.loc, "initializer is not a constant expression")
        decl = expr.decl
        if decl.kind == DECLOBJECT and decl.storage != SDSTATIC:
            error(tok.loc, "initializer is not a constant expression")
        emitname(decl.value)
    elif k == EXPRBINARY:
        if expr.op != TADD or expr.l.kind != EXPRUNARY or expr.r.kind != EXPRCONST:
            error(tok.loc, "initializer is not a constant expression")
        dataitem(expr.l, 0)
        w(" + ")
        dataitem(expr.r, 0)
    elif k == EXPRCONST:
        if expr.type.prop & PROPFLOAT:
            w("%s_%s" % ('s' if expr.type.size == 4 else 'd', fmtg17(expr.f)))
        else:
            w(str(expr.u))
    elif k == EXPRSTRING:
        ew = expr.type.base.size
        data = expr.sdata
        i = 0
        if ew == 1:
            w('"')
            while i < expr.ssize and i < size:
                c = data[i]
                if _isprint(c) and c != 0x22 and c != 0x5c:
                    w(chr(c))
                else:
                    w("\\%03o" % c)
                i += 1
            w('"')
        else:
            while i < expr.ssize and i * ew < size:
                if ew == 2 or ew == 4:
                    w("%d " % _strelem(data, i, ew))
                else:
                    assert 0
                i += 1
        if i * ew < size:
            w(", z %d" % (size - i * ew))
    else:
        error(tok.loc, "initializer is not a constant expression")


def emitdata(d, init):
    offset = 0
    bits = 0
    align = d.align
    cur = init
    while cur:
        cur.expr = EV.eval(cur.expr)
        cur = cur.next
    if d.storage == SDTHREAD:
        w("thread ")
    if d.linkage == LINKEXTERN:
        w("export ")
    w("data ")
    emitname(d.value)
    w(" = align %d { " % align)

    while init:
        cur = init
        while True:
            init = init.next
            if not (init and init.start * 8 + init.bits[0] < cur.end * 8 - cur.bits[1]):
                break
            # XXX: Currently, if multiple union members are initialized, these
            # assertions may not hold. (https://todo.sr.ht/~mcf/cproc/38)
            assert cur.expr.kind == EXPRSTRING
            assert init.expr.kind == EXPRCONST
            bs = cur.expr.type.base.size
            i = (init.start - cur.start) // bs
            val = init.expr.u
            cur.expr.sdata[i * bs:i * bs + bs] = (val & ((1 << (8 * bs)) - 1)).to_bytes(bs, 'little')
        start = cur.start + cur.bits[0] // 8
        end = cur.end - (cur.bits[1] + 7) // 8
        if offset < start and bits:
            w("b %d, " % (bits & 0xffffffff))  # unfinished byte from previous bit-field
            offset += 1
            bits = 0
        if offset < start:
            w("z %d, " % (start - offset))
        if cur.bits[0] or cur.bits[1]:
            # little-endian target specific
            assert cur.expr.type.prop & PROPINT
            assert cur.expr.kind == EXPRCONST
            bits = (bits | (cur.expr.u << (cur.bits[0] % 8))) & M64
            offset = start
            while offset < end:
                w("b %d, " % (bits & 0xff))
                offset += 1
                bits >>= 8
            # clear the upper `after` bits in the last byte, or all bits when `after`
            # is 0 (we ended on a byte boundary).
            bits &= 0x7f >> (cur.bits[1] + 7) % 8
        else:
            t = cur.expr.type
            if t.kind == TYPEARRAY:
                t = t.base
            w("%s " % qbetype(t).data)
            dataitem(cur.expr, cur.end - cur.start)
            w(", ")
        offset = end
    if bits:
        w("b %d, " % (bits & 0xffffffff))
        offset += 1
    assert offset <= d.type.size
    if offset < d.type.size:
        w("z %d " % (d.type.size - offset))
    w("}\n")
