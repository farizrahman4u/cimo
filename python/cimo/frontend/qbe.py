"""Port of qbe.c: the QBE IR back end. Output goes to util.out."""

import math

from .cc import (Expr, Init, M64, BITS0, TYPEVOID, TYPEPOINTER, TYPENULLPTR, TYPEBOOL, TYPEBITINT, TYPEARRAY,
                 TYPEFUNC, TYPESTRUCT, TYPEUNION, PROPSCALAR, PROPFLOAT, PROPINT, PROPREAL,
                 PROPVM, QUALNONE, QUALCONST, QUALVOLATILE, DECLOBJECT, DECLFUNC, DECLCONST,
                 LINKNONE, LINKEXTERN, SDSTATIC, SDTHREAD, EXPRIDENT, EXPRCONST, EXPRSTRING,
                 EXPRCALL, EXPRBITFIELD, EXPRINCDEC, EXPRCOMPOUND, EXPRUNARY, EXPRCAST,
                 EXPRBINARY, EXPRCOND, EXPRASSIGN, EXPRCOMMA, EXPRBUILTIN, EXPRTEMP, EXPRSIZEOF, EXPRSTMT, EXPRCPLX, EXPRVEC,
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
from . import diag as DG

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
    __slots__ = ('kind', 'id', 'name', 'i', 'f', 'qalign')

    def __init__(self, kind=VALUE_NONE):
        self.kind = kind
        self.id = 0
        self.qalign = 1  # VALUE_TYPE: the alignment QBE gives the type
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
# cimo extension: volatile loads and stores (cimo.backend.ext.volatile), IVOL + (op - ISTORED)
# for ISTORED..ILOADUB
IVOL = 100
volatile = False
# cimo extension: free variable-length arrays when their scope ends (cimo.backend.ext.vla):
# "%s =l getsp" before each VLA allocation, "setsp %s" when control leaves its scope
IGETSP, ISETSP = 114, 115
vlafree = False
# cimo extension: va_arg of struct and union types, lowered here with loads and stores of the
# va_list: 'aapcs' (Linux: register save areas, then the stack), 'slots' (Apple, Windows,
# wasm: a pointer to 8-byte slots); None: an error, as in cproc
vastruct = None
# cimo extension: the GNU bit builtins as instructions (cimo.backend.ext.math): name ->
# (instruction, operand class); set by compile(bitops=True), off for wasm (the runtime's C)
IBCLZ, IBCTZ, IBPOPCNT, IBSWAP, IBSWAP16 = range(116, 121)
bitops = False
# cimo extension: an aggregate copy of at least this many bytes calls memcpy (None: always
# unrolled, as cproc does; the backend's time grows faster than linearly with the unrolled code)
bigcopy = None
# cimo extension: rint and nearbyint as instructions (cimo.backend.ext.math), see funcbuiltin
IFRINT, IFNEARBYINT = 121, 122
# cimo: the high 64 bits of an unsigned 64-bit product (the backend's arm64 umulh), for __int128 multiplication
IUMULH = 123
_BITBUILTINS = {}
for _n, _i in (('clz', IBCLZ), ('ctz', IBCTZ), ('popcount', IBPOPCNT)):
    _BITBUILTINS['__builtin_' + _n] = (_i, 'w')
    _BITBUILTINS['__builtin_' + _n + 'l'] = (_i, 'l')
    _BITBUILTINS['__builtin_' + _n + 'll'] = (_i, 'l')
_BITBUILTINS.update(__builtin_bswap16=(IBSWAP16, 'w'), __builtin_bswap32=(IBSWAP, 'w'),
                    __builtin_bswap64=(IBSWAP, 'l'))
del _n, _i

# cimo extension: contract a*b+c into fma within one expression, like clang's
# -ffp-contract=on; off by default (stock cproc output), set by compile(fma=True)
contract = False
# cimo extension: tell the backend that signed +, - and * don't overflow (C 6.5p5); off by default
# (stock cproc output), set unless compile(wrapv=True)
nowrap = False

instname = [None] + [_s for _n, _s in _OPS]
# IARG, IVARARG and IFMA are printed by emitinst itself
instname += [None] * (IFSQRT - len(instname)) + list(_MATHINST)
instname += ['v' + instname[_o] for _o in range(ISTORED, ILOADUB + 1)]
instname += ['getsp', 'setsp']
instname += ['bclz', 'bctz', 'bpopcnt', 'bswap', 'bswap16']
instname += ['frint', 'fnearbyint']
instname += ['umulh']


def _vol(op, tq):
    """The volatile version of a load or store op for an object with qualifiers tq."""
    return IVOL + op - ISTORED if volatile and tq & QUALVOLATILE else op

# cimo extension: calls to these C library functions become one IR instruction (and
# fma/fmaf the fma instruction), when the callee is the external function of that name with
# the standard prototype; off by default (stock cproc output), set by compile(builtins=True)
builtins = False
_BUILTINS = {}
for _i, _n in ((IFSQRT, 'sqrt'), (IFABS, 'fabs'), (IFMIN, 'fmin'), (IFMAX, 'fmax'),
               (IFFLOOR, 'floor'), (IFCEIL, 'ceil'), (IFTRUNC, 'trunc'), (IFROUND, 'round'),
               (IFRINT, 'rint'), (IFNEARBYINT, 'nearbyint'), (IFMA, 'fma')):
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
    __slots__ = ('kind', 'cls', 'res', 'arg0', 'arg1', 'arg2', 'nsw')


JUMP_NONE, JUMP_JMP, JUMP_JNZ, JUMP_RET, JUMP_HLT = range(5)


class Block(object):
    __slots__ = ('label', 'insts', 'phi_class', 'phi_blk', 'phi_val', 'phi_res', 'jump_kind',
                 'jump_arg', 'jump_blk', 'next')


class SwitchCase(object):
    """struct switchcase: treenode + body."""
    __slots__ = ('key', 'child', 'height', 'new', 'body')


class Func(object):
    __slots__ = ('decl', 'namedecl', 'name', 'paramtemps', 'type', 'start', 'end', 'gotos',
                 'lastid', 'vlas', 'vlagotos', 'gotochecks')


ptrclass = 'l'

_blockid = 0
_globalid = 0
_typeid = 0


def _reset():
    global _blockid, _globalid, _typeid, curfunc
    curfunc = None
    for c in list(TY._complex.values()) + list(TY._vectors.values()) + [TY.typeint128, TY.typeuint128]:  # cimo: these types outlive a
        c.value = None                                                      # compile; their QBE names don't
    _blockid = 0
    _globalid = 0
    _typeid = 0


def switchcase(cases, i, b):
    k = _casevalue(cases, i)
    if any(lo <= k <= hi for lo, hi, _ in cases.ranges):
        error(tok.loc, "multiple 'case' labels with same value")
    cases.keys.append(k)
    c = TR.treeinsert(cases, i, SwitchCase)
    if not c.new:
        error(tok.loc, "multiple 'case' labels with same value")
    c.body = b


def _casevalue(cases, i):
    """cimo: a case constant's value in the switch's (promoted) type."""
    t = cases.type
    bits = 64 if t is None or t.size >= 8 else 32
    v = i & ((1 << bits) - 1)
    if (t is None or t.issigned) and v >> (bits - 1):
        v -= 1 << bits
    return v


def switchrange(cases, i, j, b):
    """cimo: `case i ... j:` (GNU): every value from i to j; empty when j < i."""
    lo, hi = _casevalue(cases, i), _casevalue(cases, j)
    if lo > hi:
        return
    if any(lo <= k <= hi for k in cases.keys) or any(lo <= h and l <= hi for l, h, _ in cases.ranges):
        error(tok.loc, "multiple 'case' labels with same value")
    cases.ranges.append((lo, hi, b))


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
        return _uh if t.prop & PROPFLOAT else _sh if t.issigned else _uh  # cimo: a half's bits
    if sz == 4:
        return _s if t.prop & PROPFLOAT else _w
    if sz == 8:
        return _d if t.prop & PROPFLOAT else _l
    if sz == 16:
        U.fatal("long double is not yet supported")
    assert 0


def abiclass(t):
    """cimo: the QBE class of a parameter or named argument of type t. Apple arm64 packs
    stack arguments at their own size (a char takes 1 byte, not an 8-byte slot), so there a
    char or short goes as QBE's sub-word ABI class (sb, ub, sh, uh), which tells the backend
    its size; elsewhere, and for everything else, as cproc: the base class."""
    qt = qbetype(t)
    if qt.data in ('b', 'h') and TG.targ.name == 'aarch64-apple':
        return ('s' if t.issigned else 'u') + qt.data
    return qt.base


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
    inst.nsw = False
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


def _isnullptr(t):
    # cimo: code for nullptr_t values (C23), which C cproc d1c53dd parses but can't generate:
    # a pointer-sized zero
    return U.diagnostics and t.kind == TYPENULLPTR


def _halfcall(f, name, cls, v, vcls):
    """cimo: a call of a runtime conversion between float and _Float16 or __bf16 bits."""
    fn = Value(VALUE_GLOBAL | VALUE_EXTERN)
    fn.name = name
    fn.id = 0
    r = funcinst(f, ICALL, cls, fn, None)
    funcinst(f, IARG, vcls, v, None)
    return r


def convert(f, dst, src, l):
    r = None
    if src is dst:
        return l
    if dst is not TY.typevoid and (TY.isi128(dst) or TY.isi128(src) or dst.iscomplex or src.iscomplex):
        return _convertvalue(f, dst, src, l)  # cimo: ?: arms and other values converted here
    if TY.ishalf(src) and dst is not TY.typevoid:  # cimo: through float (a runtime call)
        l = _halfcall(f, '__cimo_bf2f' if src.kind == TY.TYPEBF16 else '__cimo_h2f', 's', l, 'w')
        return convert(f, dst, TY.typefloat, l)
    if TY.ishalf(dst):
        l = convert(f, TY.typefloat, src, l)
        return _halfcall(f, '__cimo_f2bf' if dst.kind == TY.TYPEBF16 else '__cimo_f2h', 'w', l, 's')
    if src.kind == TYPEPOINTER or _isnullptr(src):
        src = TY.typeulong
    if dst.kind == TYPEPOINTER or _isnullptr(dst):
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
            if not (src.kind == TYPEBITINT and (src.size == 8 or dst.size == 4)):
                sz = src.size
                if sz == 4:
                    op = IEXTSW if src.issigned else IEXTUW
                elif sz == 2:
                    op = IEXTSH if src.issigned else IEXTUH
                elif sz == 1:
                    op = IEXTSB if src.issigned else IEXTUB
                else:
                    U.fatal("internal error; unknown integer conversion")
                l = funcinst(f, op, cls, l, None)
            if U.diagnostics and dst.kind == TYPEBITINT and src.issigned and not dst.issigned:
                # cimo: a negative value to a wider unsigned _BitInt wraps at its width (-25 as
                # unsigned _BitInt(9) is 487, not the sign-extended -25)
                l = funcbits(f, dst, l, BITS0)
            return l
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
    if d.type.size or not (d.type.prop & PROPVM):  # size 0: a GNU zero-length array
        f.end = f.start
        v = mkintconst(d.type.size)
    else:
        assert d.type.kind == TYPEARRAY
        assert d.type.asize
        v = d.type.asize
        if vlafree:
            f.vlas.append(funcinst(f, IGETSP, ptrclass, None, None))
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


def funccopy(f, dst, src, size, align, tq=QUALNONE):
    assert (align & (align - 1)) == 0
    if bigcopy is not None and size >= bigcopy and not tq & QUALVOLATILE:
        fn = Value(VALUE_GLOBAL | VALUE_EXTERN)
        fn.name = 'memcpy'
        fn.id = 0
        funcinst(f, ICALL, ptrclass, fn, None)
        for v in (dst, src, mkintconst(size)):
            funcinst(f, IARG, ptrclass, v, None)
        return
    cls = 'w'
    if align == 1:
        load, store = ILOADUB, ISTOREB
    elif align == 2:
        load, store = ILOADUH, ISTOREH
    elif align == 4:
        load, store = ILOADW, ISTOREW
    else:
        load, store, align, cls = ILOADL, ISTOREL, 8, 'l'
    load, store = _vol(load, tq), _vol(store, tq)
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


def funcstore(f, t, tq, lval, v, srcq=QUALNONE):
    """srcq: qualifiers of the object an aggregate v is copied from."""
    if tq & QUALVOLATILE and not volatile:
        error(tok.loc, "volatile store is not yet supported")
    if tq & QUALCONST:
        error(tok.loc, "cannot store to 'const' object")
    tp = t.prop
    assert not lval.bits[0] and not lval.bits[1] or tp & PROPINT
    r = v
    k = t.kind
    if k == TYPESTRUCT or k == TYPEUNION or k == TYPEARRAY:
        funccopy(f, lval.addr, v, t.size, t.align, tq | srcq)
    else:
        if k == TYPEPOINTER or _isnullptr(t):
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
                                  funcinst(f, _vol(qt.load, tq), qt.base, lval.addr, None),
                                  mkintconst(~mask & M64)))
        funcinst(f, _vol(qt.store, tq), 0, v, lval.addr)
    return r


def funcload(f, t, lval, tq=QUALNONE):
    k = t.kind
    if k == TYPESTRUCT or k == TYPEUNION or k == TYPEARRAY:
        return lval.addr
    qt = qbetype(t)
    v = funcinst(f, _vol(qt.load, tq), qt.base, lval.addr, None)
    return funcbits(f, t, v, lval.bits)


curfunc = None  # cimo: the function being parsed (statement expressions emit into it)


def mkfunc(decl, name, t, s):
    global curfunc
    f = Func()
    curfunc = f
    f.decl = decl
    f.name = name
    f.type = t
    f.start = f.end = mkblock("start")
    f.lastid = 0
    f.gotos = {}
    f.vlas = []  # cimo (vlafree): stack pointers saved before the VLAs now in scope
    f.vlagotos = []  # (block, VLAs in scope at the goto, GotoLabel) of forward gotos
    f.gotochecks = []  # (VLAs in scope, GotoLabel, location) of forward gotos, for emitfunc
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
    d = f.type.params
    while d:
        if d.lengthexpr is not None:
            funcexpr(f, d.lengthexpr)
        d = d.next

    return f


def delfunc(f):
    global curfunc
    curfunc = None


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
    __slots__ = ('label', 'defined', 'vlas')


def _hfa(t):
    """(member size, count) if t is a homogeneous floating-point aggregate of 1 to 4 floats or
    doubles (AAPCS64), else None. Unions are not treated as HFAs."""
    if t.prop & PROPFLOAT:
        return (t.size, 1) if t.size in (4, 8) else None
    if t.kind == TYPEARRAY:
        r = _hfa(t.base) if t.base.size else None
        return (r[0], r[1] * (t.size // t.base.size)) if r else None
    if t.kind != TYPESTRUCT:
        return None
    fs, n = 0, 0
    m = t.members
    while m:
        if m.bits[0] or m.bits[1]:
            return None
        r = _hfa(m.type)
        if not r or (fs and r[0] != fs):
            return None
        fs, n = r[0], n + r[1]
        m = m.next
    return (fs, n) if 1 <= n <= 4 and fs * n == t.size else None


def _tempalloc(f, size, align):
    """A stack slot in the start block, as for a local variable."""
    end = f.end
    f.end = f.start
    v = funcinst(f, IALLOC16 if align > 8 else IALLOC8, ptrclass, mkintconst(size), None)
    f.end = end
    return v


def _sysv_unaligned(ty, off):
    """x86-64 SysV: whether a member of ty at offset off (a packed struct's) is not at a
    multiple of its own alignment: such a struct is passed in memory (the psABI's MEMORY class
    for unaligned fields; gcc's classify_argument), whatever its size. Bit-fields don't count."""
    if ty.kind == TYPEARRAY:
        n = ty.size // ty.base.size if ty.base.size else 0
        return any(_sysv_unaligned(ty.base, off + k * ty.base.size) for k in range(min(n, 2)))
    if ty.kind in (TYPESTRUCT, TYPEUNION):
        m = ty.members
        while m:
            if not (m.bits[0] or m.bits[1]) and _sysv_unaligned(m.type, off + m.offset):
                return True
            m = m.next
        return False
    return bool(ty.size) and off % min(ty.align, 8) != 0


def _sysv_classes(t):
    """x86-64 SysV: the class of each eightbyte of struct/union t ('i' INTEGER, 's' SSE, 'n' all
    padding: no register, as gcc), or None when it is passed in memory. The answer is the
    backend's own (sysv.typclass, which lowers calls and parameters): va_arg has to read the
    registers the caller wrote. It gets each scalar of t as a union member at its offset."""
    from ..backend import defs as BD
    from ..backend.amd64 import sysv
    q = BD.Typ()
    q.size = t.size
    q.align = t.align.bit_length() - 1
    q.isdark = int(_sysv_unaligned(t, 0))

    def walk(ty, off):
        if ty.kind == TYPEARRAY:
            if ty.base.size:
                for k in range(ty.size // ty.base.size):
                    walk(ty.base, off + k * ty.base.size)
        elif ty.kind in (TYPESTRUCT, TYPEUNION):
            m = ty.members
            while m:
                if m.bits[0] or m.bits[1]:
                    leaf(off + m.offset, BD.Fw)
                else:
                    walk(m.type, off + m.offset)
                m = m.next
        elif ty.size:
            leaf(off, BD.Fs if ty.prop & PROPFLOAT else BD.Fl)

    def leaf(off, kind):
        if off < 16:
            q.fields.append([BD.Field(BD.FPad, off), BD.Field(kind, 1), BD.Field(BD.FEnd)])
            q.nunion += 1

    if t.size <= 16:  # over that it is memory, whatever an array of any length holds
        walk(t, 0)
    a = sysv.AClass()
    sysv.typclass(a, q)
    if a.inmem:
        return None
    return [{BD.Kl: 'i', BD.Kd: 's', BD.Kx: 'n'}[c] for c in a.cls[:a.size // 8]]


def _vaargsysv(f, t, ap, add, load, align16):
    """va_arg of a struct or union with the x86-64 SysV va_list {gp_offset, fp_offset,
    overflow_arg_area, reg_save_area}, as gcc: all of its eightbytes from the register save
    area (gathered into a temporary) if enough registers of each class are left, else the
    whole value from the stack (memory-class values always)."""
    size, align = t.size, t.align
    n8 = (size + 7) & -8
    cls = _sysv_classes(t)
    if cls is None:
        p = load(ILOADL, 'l', add(ap, 8))
        if align > 8:
            p = align16(p)
        funcinst(f, ISTOREL, 0, add(p, n8), add(ap, 8))
        return p
    ni, ns = cls.count('i'), cls.count('s')
    slot = _tempalloc(f, 8, 8)
    buf = _tempalloc(f, 8 * len(cls), max(align, 8))
    breg, bstk, bjoin = mkblock("vaarg_reg"), mkblock("vaarg_stack"), mkblock("vaarg_join")
    gp = load(ILOADW, 'w', ap)
    fpp = add(ap, 4)
    fp = load(ILOADW, 'w', fpp)
    ok = None
    if ni:
        ok = funcinst(f, ICULEW, 'w', gp, mkintconst(48 - 8 * ni))
    if ns:
        c = funcinst(f, ICULEW, 'w', fp, mkintconst(176 - 16 * ns))
        ok = c if ok is None else funcinst(f, IAND, 'w', ok, c)
    funcjnz(f, ok, None, breg, bstk)
    funclabel(f, breg)
    save = load(ILOADL, 'l', add(ap, 16))
    for k, c in enumerate(cls):
        if c == 'n':  # padding only: in no register
            continue
        if c == 'i':
            src = funcinst(f, IADD, ptrclass, save, funcinst(f, IEXTUW, 'l', gp, None))
            gp = funcinst(f, IADD, 'w', gp, mkintconst(8))
        else:
            src = funcinst(f, IADD, ptrclass, save, funcinst(f, IEXTUW, 'l', fp, None))
            fp = funcinst(f, IADD, 'w', fp, mkintconst(16))
        funcinst(f, ISTOREL, 0, load(ILOADL, 'l', src), add(buf, 8 * k))
    if ni:
        funcinst(f, ISTOREW, 0, gp, ap)
    if ns:
        funcinst(f, ISTOREW, 0, fp, fpp)
    funcinst(f, ISTOREL, 0, buf, slot)
    funcjmp(f, bjoin)
    funclabel(f, bstk)
    p = load(ILOADL, 'l', add(ap, 8))
    if align > 8:
        p = align16(p)
    funcinst(f, ISTOREL, 0, add(p, n8), add(ap, 8))
    funcinst(f, ISTOREL, 0, p, slot)
    funclabel(f, bjoin)
    return load(ILOADL, 'l', slot)


def funcvaargstruct(f, t, ap):
    """cimo (vastruct): va_arg(ap, t) for a struct or union t: its address. ap is the
    address of the va_list. As gcc's va_arg: an HFA in floating-point registers, one per
    member; else over 16 bytes, a pointer to a copy; else in general registers; never split
    between registers and the stack. vastruct 'sysv' and 'win64' follow those ABIs instead."""
    size, align = t.size, t.align

    def add(p, n):
        return funcinst(f, IADD, ptrclass, p, mkintconst(n))

    def load(op, cls, p):
        return funcinst(f, op, cls, p, None)

    def align16(p):
        return funcinst(f, IAND, ptrclass, add(p, 15), mkintconst(-16))

    if vastruct == 'sysv':
        return _vaargsysv(f, t, ap, add, load, align16)
    if vastruct == 'win64':
        # an 8-byte slot holds the value if it is 1, 2, 4 or 8 bytes, else a pointer to a copy
        p = load(ILOADL, 'l', ap)
        funcinst(f, ISTOREL, 0, add(p, 8), ap)
        return p if size in (1, 2, 4, 8) else load(ILOADL, 'l', p)

    # an HFA is never passed by reference (AAPCS64 B.3 before B.4), except to a Windows
    # variadic function, which passes it as any other composite
    hfa = _hfa(t) if TG.targ.name != 'aarch64-windows' else None
    if size > 16 and not hfa:
        if vastruct == 'slots':
            p = load(ILOADL, 'l', ap)
            funcinst(f, ISTOREL, 0, add(p, 8), ap)
            return load(ILOADL, 'l', p)
        return funcinst(f, IVAARG, 'l', ap, None)
    n8 = (size + 7) & -8
    if vastruct == 'slots':
        p = load(ILOADL, 'l', ap)
        if align > 8:
            p = align16(p)
        funcinst(f, ISTOREL, 0, add(p, n8), ap)
        return p
    if hfa:
        offat, topat, step, nreg = 28, 16, 16, hfa[1]  # __vr_offs, __vr_top
    else:
        offat, topat, step, nreg = 24, 8, 8, n8 // 8  # __gr_offs, __gr_top
    slot = _tempalloc(f, 8, 8)
    bcheck, breg, bstk, bjoin = (mkblock("vaarg_check"), mkblock("vaarg_reg"),
                                 mkblock("vaarg_stack"), mkblock("vaarg_join"))
    offp = add(ap, offat)
    offs = load(ILOADW, 'w', offp)
    funcjnz(f, funcinst(f, ICSLTW, 'w', offs, mkintconst(0)), None, bcheck, bstk)
    funclabel(f, bcheck)
    if not hfa and align > 8:  # an even register pair
        offs = funcinst(f, IAND, 'w', funcinst(f, IADD, 'w', offs, mkintconst(15)),
                        mkintconst(-16))
    end = funcinst(f, IADD, 'w', offs, mkintconst(step * nreg))
    funcinst(f, ISTOREW, 0, end, offp)
    funcjnz(f, funcinst(f, ICSGTW, 'w', end, mkintconst(0)), None, bstk, breg)
    funclabel(f, breg)
    p = funcinst(f, IADD, ptrclass, load(ILOADL, 'l', add(ap, topat)),
                 funcinst(f, IEXTSW, 'l', offs, None))
    if hfa:  # members are 16 bytes apart in the save area: gather them
        fs = hfa[0]
        buf = _tempalloc(f, size, align)
        lop, sop, cls = (ILOADS, ISTORES, 's') if fs == 4 else (ILOADD, ISTORED, 'd')
        for i in range(nreg):
            funcinst(f, sop, 0, load(lop, cls, add(p, 16 * i)), add(buf, fs * i))
        p = buf
    funcinst(f, ISTOREL, 0, p, slot)
    funcjmp(f, bjoin)
    funclabel(f, bstk)
    p = load(ILOADL, 'l', ap)  # __stack
    if align > 8:
        p = align16(p)
    funcinst(f, ISTOREL, 0, add(p, n8), ap)
    funcinst(f, ISTOREL, 0, p, slot)
    funclabel(f, bjoin)
    return load(ILOADL, 'l', slot)


def funcvlaleave(f, n):
    """Free the VLAs allocated after the first n now in scope (cimo, vlafree)."""
    if len(f.vlas) > n:
        funcinst(f, ISETSP, 0, f.vlas[n], None)


def funcvlagoto(f, g):
    """Jump to a goto label, freeing the VLAs whose scope the jump leaves (cimo, vlafree)."""
    if g.defined:  # backward: the label's VLAs are a prefix of ours
        if tuple(f.vlas[:len(g.vlas)]) != g.vlas:  # C11 6.8.6.1p1
            error(tok.loc, "jump into the scope of a variable length array")
        funcvlaleave(f, len(g.vlas))
        funcjmp(f, g.label)
        return
    f.gotochecks.append((list(f.vlas), g, tok.loc))
    if f.vlas:  # forward: the label's VLAs are not known yet
        t = mkblock("goto_vla")
        funcjmp(f, t)
        f.vlagotos.append((t, list(f.vlas), g))
    else:
        funcjmp(f, g.label)


def funcgoto(f, name):
    g = f.gotos.get(name)
    if g is None:
        g = GotoLabel()
        g.defined = False
        g.vlas = ()
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
            if r.kind == EXPRCONST and r.type.prop & PROPINT and r.u == 0 and l.type.prop & PROPSCALAR:
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
    v = _memcmpzero(f, e) if k == EXPRCALL else None  # only zero or not matters here
    if v is not None:
        funcjnz(f, v, TY.typeint, bt, bf)
        return v
    v = funcexpr(f, e)
    funcjnz(f, v, e.type, bt, bf)
    return v


def _nowrapinst(f, t):
    """The instruction just made is a signed + or - or * of type t: mark it. Sizes below int are the store
    of a value computed in int, and a _BitInt wraps at its own width."""
    if nowrap and t.prop & PROPINT and t.issigned and t.size >= 4 and t.kind != TYPEBITINT:
        f.end.insts[-1].nsw = True


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


# LLVM's LibCallSimplifier, and its inline memcpy and memset: sizes it expands fit in 64 bytes
_INLINEMEM = 64
_CHUNKS = ((8, ILOADL, ISTOREL, 'l'), (4, ILOADW, ISTOREW, 'w'), (2, ILOADUH, ISTOREH, 'w'),
           (1, ILOADUB, ISTOREB, 'w'))


def _stripcast(x):
    while x.kind == EXPRCAST:
        x = x.base
    return x


def _argcount(e):
    n = 0
    a = e.args
    while a:
        n += 1
        a = a.next
    return n


def _chunks(n):
    off = 0
    for size, load, store, cls in _CHUNKS:
        while n - off >= size:
            yield off, size, load, store, cls
            off += size


def _memcmpzero(f, e):
    """cimo extension, LLVM's ExpandMemCmp: memcmp(a, b, n) of a small constant n where only zero or
    not matters, as a word 0 or 1 for "differs" from loads and xors; or None. e is the call."""
    if not builtins or e.kind != EXPRCALL:
        return None
    b = e.base
    if not (b.kind == EXPRUNARY and b.op == TBAND and b.base.kind == EXPRIDENT):
        return None
    d = b.base.decl
    name = d.asmname.lstrip('_') if d.asmname else d.name
    if name != 'memcmp' or d.kind != DECLFUNC or d.linkage != LINKEXTERN or d.defined or _argcount(e) != 3:
        return None
    c = _stripcast(e.args.next.next)
    if c.kind != EXPRCONST or not c.type.prop & PROPINT or not 0 <= c.u <= 32:
        return None
    x = funcexpr(f, e.args)
    y = funcexpr(f, e.args.next)
    diff = None
    for off, size, load, store, cls in _chunks(c.u):
        p = funcinst(f, IADD, 'l', x, mkintconst(off)) if off else x
        q = funcinst(f, IADD, 'l', y, mkintconst(off)) if off else y
        v = funcinst(f, IXOR, cls, funcinst(f, load, cls, p, None), funcinst(f, load, cls, q, None))
        if cls == 'w':
            v = funcinst(f, IEXTUW, 'l', v, None)
        diff = v if diff is None else funcinst(f, IOR, 'l', diff, v)
    if diff is None:
        return mkintconst(0)
    return funcinst(f, ICNEL, 'w', diff, mkintconst(0))


def _libcall(f, e, name, d):
    """A call of a C library function with constant arguments, as simple instructions; or None. The
    callee is the external function of that name (funcbuiltin), and these names are reserved."""
    ft = d.type
    nargs = _argcount(e)
    if name in ('pow', 'powf') and nargs == 2:
        t = TY.typedouble if name == 'pow' else TY.typefloat
        y = _stripcast(e.args.next)
        sign = 1.0
        if y.kind == EXPRUNARY and y.op == TSUB:
            y = _stripcast(y.base)
            sign = -1.0
        if ft.base is not t or ft.isvararg or y.kind != EXPRCONST or e.type is not t \
                or e.args.type is not t:
            return None
        cls = qbetype(t).base
        p = sign * (y.f if y.type.prop & PROPFLOAT else float(y.i))
        if p == 2.0:
            x = funcexpr(f, e.args)
            return funcinst(f, IMUL, cls, x, x)
        if p == 1.0:
            return funcexpr(f, e.args)
        if p == -1.0:
            one = mkfltconst(VALUE_DBLCONST if cls == 'd' else VALUE_FLTCONST, 1.0)
            return funcinst(f, IDIV, cls, one, funcexpr(f, e.args))
        return None
    if name == 'strlen' and nargs == 1:
        s = _stripcast(e.args)
        if s.kind == EXPRUNARY and s.op == TBAND:
            s = s.base
        if s.kind != EXPRSTRING or s.type.base.size != 1 or e.type.size != 8:
            return None
        data = bytes(s.sdata)
        n = data.find(0)
        return mkintconst(len(data) if n < 0 else n)
    if name == 'memcpy' and nargs == 3 and e.type.kind == TYPEPOINTER:
        c = _stripcast(e.args.next.next)
        if c.kind != EXPRCONST or not c.type.prop & PROPINT or not 0 <= c.u <= _INLINEMEM:
            return None
        dst = funcexpr(f, e.args)
        src = funcexpr(f, e.args.next)
        for off, size, load, store, cls in _chunks(c.u):
            a = funcinst(f, IADD, 'l', src, mkintconst(off)) if off else src
            v = funcinst(f, load, cls, a, None)
            a = funcinst(f, IADD, 'l', dst, mkintconst(off)) if off else dst
            funcinst(f, store, 0, v, a)
        return dst
    if name == 'memset' and nargs == 3 and e.type.kind == TYPEPOINTER:
        c = _stripcast(e.args.next.next)
        v = _stripcast(e.args.next)
        if c.kind != EXPRCONST or v.kind != EXPRCONST or not c.type.prop & PROPINT \
                or not v.type.prop & PROPINT or not 0 <= c.u <= _INLINEMEM:
            return None
        dst = funcexpr(f, e.args)
        byte = v.u & 0xff
        for off, size, load, store, cls in _chunks(c.u):
            a = funcinst(f, IADD, 'l', dst, mkintconst(off)) if off else dst
            pat = int.from_bytes(bytes([byte]) * size, 'little')
            funcinst(f, store, 0, mkintconst(pat), a)
        return dst
    return None


def funcbuiltin(f, e):
    """cimo extension: a call of sqrt, fabs, fmin, fmax, floor, ceil, trunc, round, rint,
    nearbyint, fma (or
    the float versions) as one instruction, if the callee is the declared external function
    with the standard prototype (the names are reserved, C11 7.1.3). The results are the
    C functions' exactly: fminnm/fmaxnm return the number when one operand is a quiet NaN
    (and treat -0 < +0), frinta rounds halfway cases away from zero, frintx/frinti round in the current mode
    (frintx raising inexact, as rint does)."""
    b = e.base
    if not (b.kind == EXPRUNARY and b.op == TBAND and b.base.kind == EXPRIDENT):
        return None
    d = b.base.decl
    bit = bitops and not d.asmname and _BITBUILTINS.get(d.name)
    if bit and bitops is not True and instname[bit[0]] not in bitops:
        bit = None  # cimo: not one the target's backend selects (a runtime call instead)
    if bit and d.kind == DECLFUNC and d.linkage == LINKEXTERN and d.type.nparam == 1 \
            and e.args and not e.args.next and e.args.type.prop & PROPINT:
        # the prelude's declaration (a user cannot redeclare these reserved names)
        op, cls = bit
        return funcinst(f, op, cls, convert(f, d.type.params.type, e.args.type,
                                            funcexpr(f, e.args)), None)
    # by its symbol name: __builtin_sqrt is declared as sqrt's alias (the driver's prelude)
    name = d.asmname.lstrip('_') if d.asmname else d.name
    spec = _BUILTINS.get(name)
    if d.kind != DECLFUNC or d.linkage != LINKEXTERN:
        return None
    if d.defined:  # the program's own, which beats the runtime's (one defined after this call
        return None  # can't be seen here, as with gcc's builtins)
    if spec is None:
        return _libcall(f, e, name, d)
    if builtins is not True and (spec[0] == IFMA and 'fma' not in builtins
                                 or spec[0] != IFMA and instname[spec[0]] not in builtins):
        return None  # cimo: not one the target's backend selects (a library call instead)
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


# cimo: complex values (float/double _Complex). A complex value is the address of its two parts, as a struct's
# is: loads, stores, copies, calls and returns are a struct's. Arithmetic loads the parts, computes them and
# stores the result in a slot of its own (C11 Annex G's rules: a real operand has no imaginary part).

def _cxslot(f, t):
    end = f.end
    f.end = f.start
    v = funcinst(f, IALLOC8 if t.align == 8 else IALLOC4 if t.align <= 4 else IALLOC16, ptrclass,
                 mkintconst(t.size), None)
    f.end = end
    return v


def _cxmake(f, t, re, im):
    qt = qbetype(t.base)
    a = _cxslot(f, t)
    funcinst(f, qt.store, 0, re, a)
    if im is None:
        im = mkfltconst(VALUE_FLTCONST if t.base.size == 4 else VALUE_DBLCONST, 0.0)
    funcinst(f, qt.store, 0, im, funcinst(f, IADD, ptrclass, a, mkintconst(t.base.size)))
    return a


def _cxparts(f, e):
    """(real part, imaginary part or None for a real value, the real type's QBE class) of arithmetic e."""
    t = e.type
    if not t.iscomplex:
        return funcexpr(f, e), None
    qt = qbetype(t.base)
    a = funcexpr(f, e)
    re = funcinst(f, qt.load, qt.base, a, None)
    im = funcinst(f, qt.load, qt.base, funcinst(f, IADD, ptrclass, a, mkintconst(t.base.size)), None)
    return re, im


def _cxcall(f, name, cls, args):
    fn = Value(VALUE_GLOBAL | VALUE_EXTERN)
    fn.name = name
    fn.id = 0
    funcinst(f, ICALL, 'w', fn, None)
    for v, c in args:
        funcinst(f, IARG, c, v, None)


def _cxbinary(f, e):
    t = e.type
    k = qbetype(t.base).base
    a, b = _cxparts(f, e.l)
    c, d = _cxparts(f, e.r)
    op = e.op
    if op == TADD or op == TSUB:
        o = IADD if op == TADD else ISUB
        re = funcinst(f, o, k, a, c)
        if b is None:
            im = d if op == TADD or d is None else funcinst(f, INEG, k, d, None)
        elif d is None:
            im = b
        else:
            im = funcinst(f, o, k, b, d)
        return _cxmake(f, t, re, im)
    if op == TMUL:
        if b is None or d is None:  # a real factor scales both parts
            x, (y, z) = (a, (c, d)) if b is None else (c, (a, b))
            return _cxmake(f, t, funcinst(f, IMUL, k, x, y), funcinst(f, IMUL, k, x, z))
        re = funcinst(f, ISUB, k, funcinst(f, IMUL, k, a, c), funcinst(f, IMUL, k, b, d))
        im = funcinst(f, IADD, k, funcinst(f, IMUL, k, a, d), funcinst(f, IMUL, k, b, c))
        r = _cxmake(f, t, re, im)
        # both parts NaN: Annex G's recovery of infinities (G.5.1), in the runtime, as gcc's __muldc3
        uo = ICUOS if k == 's' else ICUOD
        nan = funcinst(f, IAND, 'w', funcinst(f, uo, 'w', re, re), funcinst(f, uo, 'w', im, im))
        slow, done = mkblock("cmul_nan"), mkblock("cmul_done")
        funcjnz(f, nan, None, slow, done)
        funclabel(f, slow)
        _cxcall(f, '__cimo_cmul' + ('f' if k == 's' else ''), k, [(a, k), (b, k), (c, k), (d, k), (r, ptrclass)])
        funcjmp(f, done)
        funclabel(f, done)
        return r
    if op == TDIV:
        if d is None:  # by a real: each part
            return _cxmake(f, t, funcinst(f, IDIV, k, a, c), funcinst(f, IDIV, k, b, c))
        if b is None:
            b = mkfltconst(VALUE_FLTCONST if k == 's' else VALUE_DBLCONST, 0.0)
        r = _cxslot(f, t)
        _cxcall(f, '__cimo_cdiv' + ('f' if k == 's' else ''), k, [(a, k), (b, k), (c, k), (d, k), (r, ptrclass)])
        return r
    U.fatal("internal error: complex operator %d", op)


def _cxcompare(f, e):
    """== and != with a complex operand: both parts."""
    t = e.l.type if e.l.type.iscomplex else e.r.type
    k = qbetype(t.base).base
    a, b = _cxparts(f, e.l)
    c, d = _cxparts(f, e.r)
    zero = mkfltconst(VALUE_FLTCONST if k == 's' else VALUE_DBLCONST, 0.0)
    b = zero if b is None else b
    d = zero if d is None else d
    if e.op == TEQL:
        o = ICEQS if k == 's' else ICEQD
        return funcinst(f, IAND, 'w', funcinst(f, o, 'w', a, c), funcinst(f, o, 'w', b, d))
    o = ICNES if k == 's' else ICNED
    return funcinst(f, IOR, 'w', funcinst(f, o, 'w', a, c), funcinst(f, o, 'w', b, d))


def _cxconvert(f, dst, src, e):
    """A conversion to or from a complex type (e: the operand)."""
    re, im = _cxparts(f, e)
    if dst.iscomplex:
        rs = src.base if src.iscomplex else src
        re = convert(f, dst.base, rs, re)
        if im is not None:
            im = convert(f, dst.base, rs, im)
        return _cxmake(f, dst, re, im)
    if dst.kind == TYPEBOOL:  # either part nonzero
        k = qbetype(src.base).base
        o = ICNES if k == 's' else ICNED
        zero = mkfltconst(VALUE_FLTCONST if k == 's' else VALUE_DBLCONST, 0.0)
        return funcinst(f, IOR, 'w', funcinst(f, o, 'w', re, zero), funcinst(f, o, 'w', im, zero))
    return convert(f, dst, src.base, re)  # to a real type: the real part


# cimo: __int128 values. As a complex value, the address of its two halves (low, then high: little-endian), so
# memory, copies, calls and returns are a struct's; arithmetic is lowered to 64-bit operations on the halves,
# division and the float conversions are runtime calls (runtime/builtins.c).

def _i128parts(f, e):
    """(low, high) halves of integer e (an __int128, or another integer type extended to 128 bits)."""
    t = e.type
    if TY.isi128(t) and e.kind == EXPRCONST:
        return mkintconst(e.u & M64), mkintconst((e.u >> 64) & M64)
    if TY.isi128(t):
        a = funcexpr(f, e)
        return (funcinst(f, ILOADL, 'l', a, None),
                funcinst(f, ILOADL, 'l', funcinst(f, IADD, ptrclass, a, mkintconst(8)), None))
    return _i128extend(f, t, funcexpr(f, e))


def _i128extend(f, t, v):
    if t.kind == TYPEPOINTER:
        t = TY.typeulong
    lo = convert(f, TY.typelong if t.issigned else TY.typeulong, t, v)
    hi = funcinst(f, ISAR, 'l', lo, mkintconst(63)) if t.issigned else mkintconst(0)
    return lo, hi


def _i128make(f, lo, hi):
    a = _cxslot(f, TY.typeint128)
    funcinst(f, ISTOREL, 0, lo, a)
    funcinst(f, ISTOREL, 0, hi, funcinst(f, IADD, ptrclass, a, mkintconst(8)))
    return a


def _umulh(f, a, b):
    """The high 64 bits of the unsigned product a * b: umulh on arm64, else from 32-bit halves."""
    if TG.targ.name.startswith('aarch64'):
        return funcinst(f, IUMULH, 'l', a, b)
    m32 = mkintconst(0xffffffff)
    s32 = mkintconst(32)
    a0, a1 = funcinst(f, IAND, 'l', a, m32), funcinst(f, ISHR, 'l', a, s32)
    b0, b1 = funcinst(f, IAND, 'l', b, m32), funcinst(f, ISHR, 'l', b, s32)
    p00, p01 = funcinst(f, IMUL, 'l', a0, b0), funcinst(f, IMUL, 'l', a0, b1)
    p10, p11 = funcinst(f, IMUL, 'l', a1, b0), funcinst(f, IMUL, 'l', a1, b1)
    mid = funcinst(f, IADD, 'l', funcinst(f, IADD, 'l', funcinst(f, ISHR, 'l', p00, s32),
                                          funcinst(f, IAND, 'l', p01, m32)), funcinst(f, IAND, 'l', p10, m32))
    hi = funcinst(f, IADD, 'l', p11, funcinst(f, ISHR, 'l', p01, s32))
    hi = funcinst(f, IADD, 'l', hi, funcinst(f, ISHR, 'l', p10, s32))
    return funcinst(f, IADD, 'l', hi, funcinst(f, ISHR, 'l', mid, s32))


def _i128shift(f, op, signed, lo, hi, n):
    """lo:hi shifted by n (0 to 127; a constant or a w value)."""
    if n.kind == VALUE_INTCONST:
        k = n.i & 127
        if k == 0:
            return lo, hi
        if op == ISHL:
            if k >= 64:
                return mkintconst(0), funcinst(f, ISHL, 'l', lo, mkintconst(k - 64))
            return (funcinst(f, ISHL, 'l', lo, mkintconst(k)),
                    funcinst(f, IOR, 'l', funcinst(f, ISHL, 'l', hi, mkintconst(k)),
                             funcinst(f, ISHR, 'l', lo, mkintconst(64 - k))))
        sh = ISAR if signed else ISHR
        if k >= 64:
            return (funcinst(f, sh, 'l', hi, mkintconst(k - 64)),
                    funcinst(f, ISAR, 'l', hi, mkintconst(63)) if signed else mkintconst(0))
        return (funcinst(f, IOR, 'l', funcinst(f, ISHR, 'l', lo, mkintconst(k)),
                         funcinst(f, ISHL, 'l', hi, mkintconst(64 - k))),
                funcinst(f, sh, 'l', hi, mkintconst(k)))
    r = _cxslot(f, TY.typeint128)
    kind = 0 if op == ISHL else 2 if signed else 1
    _cxcall(f, '__cimo_i128shift', 'l', [(mkintconst(kind), 'w'), (lo, 'l'), (hi, 'l'), (n, 'w'), (r, ptrclass)])
    return (funcinst(f, ILOADL, 'l', r, None),
            funcinst(f, ILOADL, 'l', funcinst(f, IADD, ptrclass, r, mkintconst(8)), None))


def _i128binary(f, e):
    t = e.type
    op = e.op
    signed = t.issigned
    a0, a1 = _i128parts(f, e.l)
    if op == TSHL or op == TSHR:
        n = convert(f, TY.typeint, e.r.type, funcexpr(f, e.r)) if not TY.isi128(e.r.type) else _i128parts(f, e.r)[0]
        if not TY.isi128(e.r.type) and n.kind != VALUE_INTCONST:
            pass
        lo, hi = _i128shift(f, ISHL if op == TSHL else ISHR, signed, a0, a1, n)
        return _i128make(f, lo, hi)
    b0, b1 = _i128parts(f, e.r)
    if op == TADD:
        lo = funcinst(f, IADD, 'l', a0, b0)
        carry = funcinst(f, IEXTUW, 'l', funcinst(f, ICULTL, 'w', lo, a0), None)
        hi = funcinst(f, IADD, 'l', funcinst(f, IADD, 'l', a1, b1), carry)
    elif op == TSUB:
        lo = funcinst(f, ISUB, 'l', a0, b0)
        borrow = funcinst(f, IEXTUW, 'l', funcinst(f, ICULTL, 'w', a0, b0), None)
        hi = funcinst(f, ISUB, 'l', funcinst(f, ISUB, 'l', a1, b1), borrow)
    elif op == TMUL:
        lo = funcinst(f, IMUL, 'l', a0, b0)
        hi = _umulh(f, a0, b0)
        hi = funcinst(f, IADD, 'l', hi, funcinst(f, IMUL, 'l', a0, b1))
        hi = funcinst(f, IADD, 'l', hi, funcinst(f, IMUL, 'l', a1, b0))
    elif op in (TBAND, TBOR, TXOR):
        o = {TBAND: IAND, TBOR: IOR, TXOR: IXOR}[op]
        lo, hi = funcinst(f, o, 'l', a0, b0), funcinst(f, o, 'l', a1, b1)
    elif op in (TDIV, TMOD):
        r = _cxslot(f, TY.typeint128)
        kind = (1 if op == TMOD else 0) | (2 if signed else 0)
        _cxcall(f, '__cimo_i128div', 'l', [(mkintconst(kind), 'w'), (a0, 'l'), (a1, 'l'), (b0, 'l'), (b1, 'l'),
                                           (r, ptrclass)])
        return r
    else:
        U.fatal("internal error: __int128 operator %d", op)
    return _i128make(f, lo, hi)


def _i128compare(f, e):
    t = e.l.type if TY.isi128(e.l.type) else e.r.type
    a0, a1 = _i128parts(f, e.l)
    b0, b1 = _i128parts(f, e.r)
    op = e.op
    if op == TEQL or op == TNEQ:
        o, j = (ICEQL, IAND) if op == TEQL else (ICNEL, IOR)
        return funcinst(f, j, 'w', funcinst(f, o, 'w', a0, b0), funcinst(f, o, 'w', a1, b1))
    # the high halves decide (signed or not), the low halves (unsigned) when those are equal
    s = t.issigned
    if op == TLESS or op == TGREATER:
        hi = {TLESS: ICSLTL if s else ICULTL, TGREATER: ICSGTL if s else ICUGTL}[op]
        lo = ICULTL if op == TLESS else ICUGTL
    else:
        hi = {TLEQ: ICSLTL if s else ICULTL, TGEQ: ICSGTL if s else ICUGTL}[op]
        lo = ICULEL if op == TLEQ else ICUGEL
    eqh = funcinst(f, ICEQL, 'w', a1, b1)
    return funcinst(f, IOR, 'w', funcinst(f, hi, 'w', a1, b1),
                    funcinst(f, IAND, 'w', eqh, funcinst(f, lo, 'w', a0, b0)))


def _i128convert(f, dst, src, e):
    """A conversion to or from __int128 (e: the operand)."""
    if TY.isi128(dst):
        if src.prop & PROPFLOAT:
            r = _cxslot(f, dst)
            v = convert(f, TY.typedouble, src, funcexpr(f, e))
            _cxcall(f, '__cimo_i128fromd', 'l', [(mkintconst(1 if dst.issigned else 0), 'w'), (v, 'd'), (r, ptrclass)])
            return r
        lo, hi = _i128parts(f, e)
        return _i128make(f, lo, hi)
    lo, hi = _i128parts(f, e)
    if dst.kind == TYPEBOOL:
        return funcinst(f, ICNEL, 'w', funcinst(f, IOR, 'l', lo, hi), mkintconst(0))
    if TY.ishalf(dst):  # through float
        return convert(f, dst, TY.typefloat, _i128convert(f, TY.typefloat, src, e))
    if dst.prop & PROPFLOAT:
        r = _cxslot(f, TY.typedouble)
        name = '__cimo_i128tof' if dst.size == 4 else '__cimo_i128tod'
        _cxcall(f, name, 'l', [(mkintconst(1 if src.issigned else 0), 'w'), (lo, 'l'), (hi, 'l'), (r, ptrclass)])
        return funcinst(f, ILOADS if dst.size == 4 else ILOADD, 's' if dst.size == 4 else 'd', r, None)
    if dst.kind == TYPEPOINTER:
        return lo
    return convert(f, dst, TY.typeulong, lo)  # the low bits


def _convertvalue(f, dst, src, l):
    """cimo: convert() of an __int128 or complex value l (an address) or to one: through a stand-in expression
    (EXPRTEMP) whose value is l."""
    if TY.isi128(dst) and TY.isi128(src):
        return l
    e = Expr(EXPRTEMP, src, None)
    e.temp = l
    if dst.iscomplex or src.iscomplex:
        return _cxconvert(f, dst, src, e)
    return _i128convert(f, dst, src, e)

# cimo: GNU vector types. A vector value is the address of its lanes (a struct's, for memory, copies, calls and
# returns); operators are lowered lane by lane into a slot of their own. Lanes narrower than int are computed in
# w (loaded extended by their signedness, stored truncated), as C computes them.

def _vlane(f, a, el, i):
    qt = qbetype(el)
    return funcinst(f, qt.load, qt.base, funcinst(f, IADD, ptrclass, a, mkintconst(i * el.size)), None)


def _vput(f, a, el, i, v):
    funcinst(f, qbetype(el).store, 0, v, funcinst(f, IADD, ptrclass, a, mkintconst(i * el.size)))


def _voperand(f, e):
    """(address, None) of a vector operand, or (None, value) of a scalar one (computed once)."""
    if TY.isvec(e.type):
        return funcexpr(f, e), None
    return None, funcexpr(f, e)


def _vbinary(f, e):
    vt = e.l.type if TY.isvec(e.l.type) else e.r.type
    el = vt.base
    rt = e.type
    op = e.op
    la, ls = _voperand(f, e.l)
    ra, rs = _voperand(f, e.r)
    rel = e.r.type.base if TY.isvec(e.r.type) else el
    res = _cxslot(f, rt)
    cls = qbetype(el).base
    cmp = op in (TEQL, TNEQ, TLESS, TGREATER, TLEQ, TGEQ)
    o = _binop(op, el)
    for i in range(vt.vlanes):
        a = _vlane(f, la, el, i) if la is not None else ls
        b = _vlane(f, ra, rel, i) if ra is not None else rs
        if op in (TSHL, TSHR) and cls == 'l' and qbetype(rel).base == 'w':
            b = funcinst(f, IEXTUW, 'l', b, None)
        if cmp:
            v = funcinst(f, INEG, 'w', funcinst(f, o, 'w', a, b), None)  # -1 or 0
            if rt.base.size == 8:
                v = funcinst(f, IEXTSW, 'l', v, None)
        else:
            v = funcinst(f, o, cls, a, b)
        _vput(f, res, rt.base, i, v)
    return res


def _vunary(f, e):
    t = e.type
    el = t.base
    cls = qbetype(el).base
    a = funcexpr(f, e.base)
    res = _cxslot(f, t)
    for i in range(t.vlanes):
        x = _vlane(f, a, el, i)
        if e.op == TSUB:
            x = funcinst(f, INEG, cls, x, None)
        else:
            x = funcinst(f, IXOR, cls, x, mkintconst(-1))
        _vput(f, res, el, i, x)
    return res


def _vcast(f, dst, src, e):
    """A vector's bits as another type of the same size, or a scalar's as a vector."""
    v = funcexpr(f, e)
    if TY.isvec(dst):
        res = _cxslot(f, dst)
        if TY.isvec(src):
            funccopy(f, res, v, dst.size, min(dst.align, src.align))
        else:
            if src.kind == TYPEPOINTER:
                src = TY.typeulong
            funcinst(f, qbetype(src).store, 0, v, res)
        return res
    if dst is TY.typevoid:
        return None
    t = TY.typeulong if dst.kind == TYPEPOINTER else dst
    qt = qbetype(t)
    return funcinst(f, qt.load, qt.base, v, None)


def _vbuiltin(f, e):
    t = e.type
    el = t.base
    a = funcexpr(f, e.l)
    st = e.l.type
    res = _cxslot(f, t)
    if e.op == 'convert':
        for i in range(t.vlanes):
            _vput(f, res, el, i, convert(f, el, st.base, _vlane(f, a, st.base, i)))
        return res
    n = st.vlanes
    if e.op == 'shufflevector':
        b = funcexpr(f, e.r)
        for i, k in enumerate(e.args):
            v = mkintconst(0) if k < 0 else _vlane(f, a if k < n else b, el, k % n)
            if k < 0 and el.prop & PROPFLOAT:
                v = mkfltconst(VALUE_FLTCONST if el.size == 4 else VALUE_DBLCONST, 0.0)
            _vput(f, res, el, i, v)
        return res
    m = funcexpr(f, e.ct)
    mel = e.ct.type.base
    base, lanes = a, n
    if e.r is not None:  # a and b side by side, indexed together
        b = funcexpr(f, e.r)
        both = _cxslot(f, TY.mkvector(el, 2 * st.size))
        funccopy(f, both, a, st.size, st.align)
        funccopy(f, funcinst(f, IADD, ptrclass, both, mkintconst(st.size)), b, st.size, st.align)
        base, lanes = both, 2 * n
    for i in range(n):
        k = _vlane(f, m, mel, i)
        if qbetype(mel).base == 'w':
            k = funcinst(f, IEXTUW, 'l', k, None)
        k = funcinst(f, IAND, 'l', k, mkintconst(lanes - 1))
        p = funcinst(f, IADD, ptrclass, base, funcinst(f, IMUL, 'l', k, mkintconst(el.size)))
        qt = qbetype(el)
        _vput(f, res, el, i, funcinst(f, qt.load, qt.base, p, None))
    return res


def funcexpr(f, e):
    if U.diagnostics:
        DG.checktype(e.type)
    calcvla(f, e.type)
    k = e.kind
    if k == EXPRIDENT:
        d = e.decl
        if d.kind == DECLOBJECT:
            return funcload(f, e.type, LValue(d.value), e.qual)
        if d.kind == DECLCONST:
            return d.value
        U.fatal("unimplemented declaration kind %d", d.kind)
    elif k == EXPRCONST:
        t = e.type
        if TY.isi128(t):  # cimo
            return _i128make(f, *_i128parts(f, e))
        if TY.ishalf(t):  # cimo: its 16 bits
            return mkintconst(EV.halfbits(t, e.f))
        if t.prop & PROPINT or t.kind == TYPEPOINTER or _isnullptr(t):
            return mkintconst(e.u)
        assert t.prop & PROPFLOAT
        return mkfltconst(VALUE_FLTCONST if t.size == 4 else VALUE_DBLCONST, e.f)
    elif k == EXPRBITFIELD or k == EXPRCOMPOUND:
        lval = funclval(f, e)
        return funcload(f, e.type, lval, e.qual)
    elif k == EXPRSTMT:
        # cimo: a statement expression's body, emitted when it was parsed into blocks of its own (ct to cf),
        # runs here: the block chain spliced in, its last expression's value (temp) the result
        if e.post:
            U.fatal("internal error: statement expression emitted twice")
        e.post = True
        funcjmp(f, e.ct)
        f.end.next = e.ct
        f.end = e.cf
        return e.temp
    elif k == EXPRINCDEC and TY.ishalf(e.type):  # cimo: through float
        lval = funclval(f, e.base)
        l = funcload(f, e.type, lval, e.base.qual)
        x = convert(f, TY.typefloat, e.type, l)
        x = funcinst(f, IADD if e.op == TINC else ISUB, 's', x, mkfltconst(VALUE_FLTCONST, 1.0))
        v = funcstore(f, e.type, e.qual, lval, convert(f, e.type, TY.typefloat, x))
        return l if e.post else v
    elif k == EXPRINCDEC and TY.isi128(e.type):  # cimo
        lval = funclval(f, e.base)
        a = lval.addr
        a0 = funcinst(f, ILOADL, 'l', a, None)
        a1 = funcinst(f, ILOADL, 'l', funcinst(f, IADD, ptrclass, a, mkintconst(8)), None)
        old = _i128make(f, a0, a1) if e.post else None
        if e.op == TINC:
            lo = funcinst(f, IADD, 'l', a0, mkintconst(1))
            hi = funcinst(f, IADD, 'l', a1, funcinst(f, IEXTUW, 'l', funcinst(f, ICEQL, 'w', lo, mkintconst(0)), None))
        else:
            lo = funcinst(f, ISUB, 'l', a0, mkintconst(1))
            hi = funcinst(f, ISUB, 'l', a1, funcinst(f, IEXTUW, 'l', funcinst(f, ICEQL, 'w', a0, mkintconst(0)), None))
        funcinst(f, ISTOREL, 0, lo, a)
        funcinst(f, ISTOREL, 0, hi, funcinst(f, IADD, ptrclass, a, mkintconst(8)))
        return old if e.post else a
    elif k == EXPRINCDEC:
        lval = funclval(f, e.base)
        l = funcload(f, e.base.type, lval, e.base.qual)
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
        if t.kind == TYPEBOOL:
            # cimo: x + 1 and x - 1 convert back to _Bool (C11 6.5.2.4): 1, and !x; cproc
            # stored 2 and 255
            v = mkintconst(1) if e.op == TINC else funcinst(f, ICEQW, 'w', l, mkintconst(0))
        else:
            v = funcinst(f, IADD if e.op == TINC else ISUB, qbetype(t).base, l, r)
            _nowrapinst(f, t)
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
            funcinst(f, IARG, abiclass(t) if i < ftype.nparam else qbetype(t).base,
                     argvals[i], t.value)
            arg = arg.next
            i += 1
        if ftype.isvararg and i <= ftype.nparam and TG.targ.name == 'aarch64-windows':
            # cimo: Windows passes every argument of a variadic call its own way (floats in
            # x registers), so the backend needs the `...` even when nothing follows it
            funcinst(f, IVARARG, 0, None, None)
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
            return funcload(f, e.type, LValue(r), e.qual)
        if TY.isvec(e.type):  # cimo
            return _vunary(f, e)
        if TY.isi128(e.type) and op == TSUB:  # cimo: 0 - x
            a0, a1 = _i128parts(f, e.base)
            lo = funcinst(f, INEG, 'l', a0, None)
            borrow = funcinst(f, IEXTUW, 'l', funcinst(f, ICNEL, 'w', a0, mkintconst(0)), None)
            return _i128make(f, lo, funcinst(f, ISUB, 'l', funcinst(f, INEG, 'l', a1, None), borrow))
        if e.type.iscomplex:  # cimo: -z negates both parts, ~z (GNU) the imaginary one
            re, im = _cxparts(f, e.base)
            k = qbetype(e.type.base).base
            if op == TSUB:
                re = funcinst(f, INEG, k, re, None)
            return _cxmake(f, e.type, re, funcinst(f, INEG, k, im, None))
        if op == TSUB:
            r = funcexpr(f, e.base)
            v = funcinst(f, INEG, qbetype(e.type).base, r, None)
            if U.diagnostics and e.type.kind == TYPEBITINT:  # cimo: -1uwb is 1 (an unsigned _BitInt(1)), as binary ops wrap
                v = funcbits(f, e.type, v, BITS0)
            return v
        U.fatal("internal error; unknown unary expression")
    elif k == EXPRCAST:
        if e.toeval:
            funcexpr(f, e.toeval)
        if TY.isvec(e.type) or TY.isvec(e.base.type):
            return _vcast(f, e.type, e.base.type, e.base)
        if (e.type.iscomplex or e.base.type.iscomplex) and e.type is not TY.typevoid:
            return _cxconvert(f, e.type, e.base.type, e.base)
        if (TY.isi128(e.type) or TY.isi128(e.base.type)) and e.type is not TY.typevoid:
            if TY.isi128(e.type) and TY.isi128(e.base.type):
                return funcexpr(f, e.base)  # the same bits
            return _i128convert(f, e.type, e.base.type, e.base)
        l = funcexpr(f, e.base)
        return convert(f, e.type, e.base.type, l)
    elif k == EXPRCPLX:
        return _cxmake(f, e.type, funcexpr(f, e.l), funcexpr(f, e.r))
    elif k == EXPRVEC:
        return _vbuiltin(f, e)
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
        if TY.isvec(e.type) or TY.isvec(e.l.type) or TY.isvec(e.r.type):
            return _vbinary(f, e)
        if e.op in (TEQL, TNEQ, TLESS, TGREATER, TLEQ, TGEQ) and (TY.isi128(e.l.type) or TY.isi128(e.r.type)):
            return _i128compare(f, e)
        if TY.isi128(e.type):
            return _i128binary(f, e)
        if e.type.iscomplex:
            return _cxbinary(f, e)
        if (e.op == TEQL or e.op == TNEQ) and (e.l.type.iscomplex or e.r.type.iscomplex):
            return _cxcompare(f, e)
        if contract:
            v = funcfma(f, e)
            if v is not None:
                return v
        if (e.op == TEQL or e.op == TNEQ) and e.l.kind == EXPRCALL:
            r = _stripcast(e.r)
            if r.kind == EXPRCONST and r.type.prop & PROPINT and r.u == 0:
                v = _memcmpzero(f, e.l)
                if v is not None:
                    return v if e.op == TNEQ else funcinst(f, ICEQW, 'w', v, mkintconst(0))
        l = funcexpr(f, e.l)
        r = funcexpr(f, e.r)
        t = e.l.type
        if t.kind == TYPEPOINTER or _isnullptr(t):
            t = TY.typeulong
        op = _binop(e.op, t)
        if op == INONE:
            U.fatal("internal error; unimplemented binary expression")
        v = funcinst(f, op, qbetype(e.type).base, l, r)
        if op in (IADD, ISUB, IMUL) and t is e.l.type:  # not of pointers (t became unsigned long)
            _nowrapinst(f, e.type)
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
            r = funcstore(f, e.l.type, e.l.qual, lval, r, e.r.qual)
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
                if vastruct and e.type.kind in (TYPESTRUCT, TYPEUNION):
                    return funcvaargstruct(f, e.type, funcexpr(f, e.base))
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


def zero(func, addr, align, offset, end, tq=QUALNONE):
    if bigcopy is not None and end - offset >= bigcopy and not tq & QUALVOLATILE:
        # cimo: memset(addr + offset, 0, end - offset) instead of a store per 8 bytes
        fn = Value(VALUE_GLOBAL | VALUE_EXTERN)
        fn.name = 'memset'
        fn.id = 0
        p = funcinst(func, IADD, ptrclass, addr, mkintconst(offset)) if offset else addr
        funcinst(func, ICALL, ptrclass, fn, None)
        for c, v in ((ptrclass, p), ('w', mkintconst(0)), (ptrclass, mkintconst(end - offset))):
            funcinst(func, IARG, c, v, None)
        return
    align = min(align, 8)  # cimo: cproc indexed its store table with 16 for _Alignas(16)
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
        zero(func, d.value, d.type.align, offset, init.start, d.qual)
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
                zero(func, d.value, d.type.align, offset, init.end, d.qual)
            dst.addr = d.value
            # QBE's memopt does not eliminate the store for ptr + 0, so only emit the
            # add if the offset is non-zero
            if init.start > 0:
                dst.addr = funcinst(func, IADD, ptrclass, dst.addr, mkintconst(init.start))
            src = funcexpr(func, ex)
            # cimo: volatile objects are initialized with volatile stores
            funcstore(func, ex.type, d.qual & QUALVOLATILE, dst, src, ex.qual)
            offset = init.end
        if mx < offset:
            mx = offset
        init = init.next
    zero(func, d.value, d.type.align, mx, d.type.size, d.qual)


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
    cls = qbetype(c.type).base
    for lo, hi, b in c.ranges:  # cimo: a case range is one unsigned compare, (v - lo) <= (hi - lo)
        nxt = mkblock("switch_range")
        d = funcinst(f, ISUB, cls, v, mkintconst(lo & M64))
        res = funcinst(f, ICULEW if cls == 'w' else ICULEL, 'w', d, mkintconst((hi - lo) & M64))
        funcjnz(f, res, None, b, nxt)
        funclabel(f, nxt)
    casesearch(f, cls, v, c.root, defaultlabel, 0, M64)


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


def _classstr(cls, v):
    """emitclass's output as a string."""
    n = len(out)
    emitclass(cls, v)
    r = ''.join(out[n:])
    del out[n:]
    return r


def emittype(t):
    """cimo: padding and `align` are emitted where QBE's layout of the fields would differ from
    C's (_Alignas, or a struct aligned beyond its members): cproc's XXX, which left such a
    struct too small and under-aligned for QBE, so calls passed it wrong. A packed member at an
    offset QBE would not place its type at goes as bytes: QBE aligns every field, so such a
    struct came out bigger than in C and took the wrong registers. AAPCS64 (gcc, clang) picks
    registers by the members' alignment, not an `aligned(N)` on the struct itself, so on Linux
    QBE isn't told about that one; Apple and Windows do go by the type's alignment."""
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
    if TG.targ.name == 'x86_64-sysv' and _sysv_unaligned(t, 0):
        # cimo: a struct with an unaligned (packed) member is passed in memory on x86-64
        # SysV, which QBE's classify does for an opaque type (its fields don't matter then)
        w(" = align %d { %d }\n" % (t.align, t.size))
        t.value.qalign = t.align
        return
    if t is TG.targ.typevalist:
        w(" = align %d { %d }\n" % (t.align, t.size))
        t.value.qalign = t.align
        return
    out = []
    m = t.members
    off = 0
    qoff = 0  # where QBE's layout of the fields so far ends
    qalign = 1
    usize = 0
    while m:
        if t.kind == TYPESTRUCT and m.type.size == 0 and not (m.bits[0] or m.bits[1]):
            m = m.next  # cimo: a flexible or zero-length array has no QBE field
            continue
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
            out.append("{ ")
        sub = m.type
        while sub.kind == TYPEARRAY:
            sub = sub.base
        # sub.align is C's; a packed struct has align 1 but QBE aligns it by its fields
        al = sub.value.qalign if sub.value and sub.value.kind == VALUE_TYPE else sub.align
        if t.kind == TYPESTRUCT and m.offset & (al - 1):
            out.append("b %d" % (off - qoff))
            qoff = off
        else:
            qalign = max(qalign, al)
            if t.kind == TYPESTRUCT:
                if m.offset > ((qoff + al - 1) & -al):
                    out.append("b %d, " % (m.offset - qoff))
                qoff = off
            else:
                usize = max(usize, m.type.size)
            out.append(_classstr(qbetype(sub).data, sub.value))
            if m.type.size > sub.size:
                out.append(" %d" % (m.type.size // sub.size))
        if t.kind == TYPESTRUCT:
            out.append(", ")
            # skip subsequent members contained within the same storage unit
            while True:
                m = m.next
                if not (m and m.offset < off):
                    break
        else:
            out.append(" } ")
            m = m.next
    if not out:
        out.append("b 0, ")  # an empty struct (GNU C): size 0
    end = qoff if t.kind == TYPESTRUCT else usize
    # QBE rounds the size up to the type's alignment, the `align` written below if larger.
    # cimo: on x86-64 that alignment counts, since padding written as `b` fields would make an
    # eightbyte of floats INTEGER for QBE's amd64 classify (`struct { _Alignas(16) float
    # m[3]; }` in rdi instead of xmm1, unlike gcc); arm64 keeps the layout tested against gcc
    qround = max(qalign, t.align) if TG.targ.name.startswith('x86_64') else qalign
    if t.size > ((end + qround - 1) & -qround):  # trailing padding QBE would not add
        out.append("b %d, " % (t.size - end) if t.kind == TYPESTRUCT
                   else "{ b %d } " % t.size)
    align = t.align
    if t.fieldalign is not None and TG.targ.name == 'aarch64':
        align = t.fieldalign
    t.value.qalign = max(qalign, align)
    qa = t.value.qalign
    if TG.targ.name == 'x86_64-windows' and t.size != ((max(end, t.size) + qa - 1) & -qa):
        # cimo: QBE rounds the size up to its alignment of the fields (a 7-byte packed struct
        # would be 8 and go in a register); Windows x64 passes what is not 1, 2, 4 or 8 bytes
        # by reference, which winabi does for an opaque type of C's size
        t.value.qalign = t.align
        w(" = align %d { %d }\n" % (t.align, t.size))
        return
    w(" = align %d { " % align if align > qalign else " = { ")
    w("".join(out))
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
    if inst.nsw:
        w('nsw')
    if inst.kind == IGETSP:
        w('\n')
        return i + 1
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


def _ctypename(t):
    """A C type by what it is, for Module.signature (cimo.introspect)."""
    from .cc import QUALCONST, TYPEARRAY, TYPEBOOL, TYPEDOUBLE, TYPEFLOAT, TYPELDOUBLE, TYPEPOINTER, TYPEUNION, TYPESTRUCT, TYPEVOID
    if t.kind == TYPEPOINTER:
        const, base = t.qual & QUALCONST, t.base
        while base.kind == TYPEARRAY:
            const |= base.qual & QUALCONST
            base = base.base
        return ('const ' if const else '') + _ctypename(base) + ' *'
    names = {TYPEVOID: 'void', TYPEBOOL: '_Bool', TYPEFLOAT: 'float', TYPEDOUBLE: 'double', TYPELDOUBLE: 'long double', TYPESTRUCT: 'struct', TYPEUNION: 'union'}
    if t.kind in names:
        return names[t.kind]
    if t.prop & PROPINT:
        return '%sint%d_t' % ('' if t.issigned else 'u', 8 * t.size)
    return 'function'


def _isconst(d):
    from .cc import QUALCONST, TYPEARRAY
    t, const = d.type, d.qual & QUALCONST
    while t.kind == TYPEARRAY:
        const |= t.qual & QUALCONST
        t = t.base
    return const


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
    p, params = f.type.params, []
    while p:
        params.append(_ctypename(p.type))
        p = p.next
    if f.type.isvararg:
        params.append('...')
    paraminfo.setdefault('#signatures', {})[f.decl.value.name] = (_ctypename(f.type.base), tuple(params))
    # for the Python bindings: QBE's classes don't say that a w or l result is unsigned. A narrower
    # unsigned type is zero extended into its w, so it reads the same either way
    r = f.type.base
    if r.prop & PROPINT and not r.issigned and r.size >= 4:
        paraminfo.setdefault('#unsigned', set()).add(f.decl.value.name)


def emitfunc(f, glob):
    if f.end.jump_kind == JUMP_NONE:
        v = None
        # implicitly return 0 from main if we reach the end of the function
        if f.name == "main" and f.type.base is TY.typeint:
            v = mkintconst(0)
        funcret(f, v)
    for vlas, g, loc in f.gotochecks:  # cimo (vlafree): forward gotos, C11 6.8.6.1p1
        if not g.defined and U.diagnostics:
            error(loc, "label '%s' used but not defined", g.label.label.name)
        if g.defined and tuple(vlas[:len(g.vlas)]) != g.vlas:
            error(loc, "jump into the scope of a variable length array")
    if U.diagnostics:
        for name, g in f.gotos.items():
            if not g.defined:
                error(tok.loc, "label '%s' used but not defined", name)
    for t, vlas, g in f.vlagotos:  # forward gotos that leave VLA scopes (vlafree)
        funclabel(f, t)
        if len(vlas) > len(g.vlas):
            funcinst(f, ISETSP, 0, vlas[len(g.vlas)], None)
        funcjmp(f, g.label)
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
        emitclass(abiclass(p.type), p.type.value)
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
        if TY.ishalf(expr.type):  # cimo: its 16 bits
            w(str(EV.halfbits(expr.type, expr.f)))
        elif expr.type.prop & PROPFLOAT:
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


def emitdata(d, init, literal=False):
    if paraminfo is not None and not literal and not _isconst(d) and d.value.name:
        paraminfo.setdefault('#mutable', []).append(d.value.name)
    offset = 0
    bits = 0
    align = d.align
    cur = init
    while cur:
        cur.expr = EV.eval(cur.expr)
        if cur.expr.kind == EXPRCONST and TY.isi128(cur.expr.type):  # cimo: an __int128 constant is two longs
            c = cur.expr
            lo, hi = Expr(EXPRCONST, TY.typeulong, None), Expr(EXPRCONST, TY.typeulong, None)
            lo.u, hi.u = c.u & M64, (c.u >> 64) & M64
            nxt = Init(cur.start + 8, cur.end, BITS0, hi)
            nxt.next = cur.next
            cur.end, cur.expr, cur.next = cur.start + 8, lo, nxt
        if cur.expr.kind == EXPRCPLX:  # cimo: a complex constant is its two parts
            n = cur.expr.type.base.size
            im = Init(cur.start + n, cur.end, BITS0, cur.expr.r)
            im.next = cur.next
            cur.end, cur.expr, cur.next = cur.start + n, cur.expr.l, im
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
