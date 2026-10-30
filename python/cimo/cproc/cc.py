"""Definitions from cc.h, tokens.h and util.h (see PORTING.md).

Token kinds, type/decl/expr kinds and the front end's data structures.
Struct fields keep their C names; C unions are flattened into separate
attributes, except where the port relies on the aliasing (noted below).
"""

import struct as _struct

M64 = 0xffffffffffffffff
M32 = 0xffffffff

# tokens.h, in order: (name, spelling)
TOKENS = [
    ('TNONE', None),
    ('TEOF', None),
    ('TNEWLINE', None),
    ('TOTHER', None),
    ('TNUMBER', None),
    ('TCHARCONST', None),
    ('TSTRINGLIT', None),
    ('TLBRACK', "["),
    ('TRBRACK', "]"),
    ('TLPAREN', "("),
    ('TRPAREN', ")"),
    ('TLBRACE', "{"),
    ('TRBRACE', "}"),
    ('TPERIOD', "."),
    ('TARROW', "->"),
    ('TINC', "++"),
    ('TDEC', "--"),
    ('TBAND', "&"),
    ('TMUL', "*"),
    ('TADD', "+"),
    ('TSUB', "-"),
    ('TBNOT', "~"),
    ('TLNOT', "!"),
    ('TDIV', "/"),
    ('TMOD', "%"),
    ('TSHL', "<<"),
    ('TSHR', ">>"),
    ('TLESS', "<"),
    ('TGREATER', ">"),
    ('TLEQ', "<="),
    ('TGEQ', ">="),
    ('TEQL', "=="),
    ('TNEQ', "!="),
    ('TXOR', "^"),
    ('TBOR', "|"),
    ('TLAND', "&&"),
    ('TLOR', "||"),
    ('TQUESTION', "?"),
    ('TCOLON', ":"),
    ('TCOLONCOLON', "::"),
    ('TSEMICOLON', ";"),
    ('TELLIPSIS', "..."),
    ('TASSIGN', "="),
    ('TMULASSIGN', "*="),
    ('TDIVASSIGN', "/="),
    ('TMODASSIGN', "%="),
    ('TADDASSIGN', "+="),
    ('TSUBASSIGN', "-="),
    ('TSHLASSIGN', "<<="),
    ('TSHRASSIGN', ">>="),
    ('TBANDASSIGN', "&="),
    ('TXORASSIGN', "^="),
    ('TBORASSIGN', "|="),
    ('TCOMMA', ","),
    ('THASH', "#"),
    ('THASHHASH', "##"),
    ('TALIGNAS', "alignas"),
    ('TALIGNOF', "alignof"),
    ('TAUTO', "auto"),
    ('TBOOL', "bool"),
    ('TBREAK', "break"),
    ('TCASE', "case"),
    ('TCHAR', "char"),
    ('TCONST', "const"),
    ('TCONSTEXPR', "constexpr"),
    ('TCONTINUE', "continue"),
    ('TDEFAULT', "default"),
    ('TDO', "do"),
    ('TDOUBLE', "double"),
    ('TELSE', "else"),
    ('TENUM', "enum"),
    ('TEXTERN', "extern"),
    ('TFALSE', "false"),
    ('TFLOAT', "float"),
    ('TFOR', "for"),
    ('TGOTO', "goto"),
    ('TIF', "if"),
    ('TINLINE', "inline"),
    ('TINT', "int"),
    ('TLONG', "long"),
    ('TNULLPTR', "nullptr"),
    ('TREGISTER', "register"),
    ('TRESTRICT', "restrict"),
    ('TRETURN', "return"),
    ('TSHORT', "short"),
    ('TSIGNED', "signed"),
    ('TSIZEOF', "sizeof"),
    ('TSTATIC', "static"),
    ('TSTATIC_ASSERT', "static_assert"),
    ('TSTRUCT', "struct"),
    ('TSWITCH', "switch"),
    ('TTHREAD_LOCAL', "thread_local"),
    ('TTRUE', "true"),
    ('TTYPEDEF', "typedef"),
    ('TTYPEOF', "typeof"),
    ('TTYPEOF_UNQUAL', "typeof_unqual"),
    ('TUNION', "union"),
    ('TUNSIGNED', "unsigned"),
    ('TVOID', "void"),
    ('TVOLATILE', "volatile"),
    ('TWHILE', "while"),
    ('T_ATOMIC', "_Atomic"),
    ('T_BITINT', "_BitInt"),
    ('T_COMPLEX', "_Complex"),
    ('T_DECIMAL128', "_Decimal128"),
    ('T_DECIMAL32', "_Decimal32"),
    ('T_DECIMAL64', "_Decimal64"),
    ('T_GENERIC', "_Generic"),
    ('T_IMAGINARY', "_Imaginary"),
    ('T_NORETURN', "_Noreturn"),
    ('T__ALIGNOF__', "__alignof__"),
    ('T__ASM__', "__asm__"),
    ('T__ATTRIBUTE__', "__attribute__"),
    ('TDEFINE', "define"),
    ('TELIF', "elif"),
    ('TENDIF', "endif"),
    ('TERROR', "error"),
    ('TIFDEF', "ifdef"),
    ('TIFNDEF', "ifndef"),
    ('TINCLUDE', "include"),
    ('TLINE', "line"),
    ('TPRAGMA', "pragma"),
    ('TUNDEF', "undef"),
    ('TWARNING', "warning"),
    ('T__VA_ARGS__', "__VA_ARGS__"),
    ('ALIAS_ALIGNAS', "_Alignas"),
    ('ALIAS_ALIGNOF', "_Alignof"),
    ('ALIAS_BOOL', "_Bool"),
    ('ALIAS_STATIC_ASSERT', "_Static_assert"),
    ('ALIAS_THREAD_LOCAL', "_Thread_local"),
    ('ALIAS__INLINE', "__inline"),
    ('ALIAS__INLINE__', "__inline__"),
    ('ALIAS__SIGNED', "__signed"),
    ('ALIAS__SIGNED__', "__signed__"),
    ('ALIAS__THREAD', "__thread"),
    ('ALIAS__TYPEOF', "__typeof"),
    ('ALIAS__TYPEOF__', "__typeof__"),
    ('ALIAS__VOLATILE__', "__volatile__"),
    ('ALIAS__ASM', "__asm"),
]

TNONE = 0
TEOF = 1
TNEWLINE = 2
TOTHER = 3
TNUMBER = 4
TCHARCONST = 5
TSTRINGLIT = 6
TLBRACK = 7
TRBRACK = 8
TLPAREN = 9
TRPAREN = 10
TLBRACE = 11
TRBRACE = 12
TPERIOD = 13
TARROW = 14
TINC = 15
TDEC = 16
TBAND = 17
TMUL = 18
TADD = 19
TSUB = 20
TBNOT = 21
TLNOT = 22
TDIV = 23
TMOD = 24
TSHL = 25
TSHR = 26
TLESS = 27
TGREATER = 28
TLEQ = 29
TGEQ = 30
TEQL = 31
TNEQ = 32
TXOR = 33
TBOR = 34
TLAND = 35
TLOR = 36
TQUESTION = 37
TCOLON = 38
TCOLONCOLON = 39
TSEMICOLON = 40
TELLIPSIS = 41
TASSIGN = 42
TMULASSIGN = 43
TDIVASSIGN = 44
TMODASSIGN = 45
TADDASSIGN = 46
TSUBASSIGN = 47
TSHLASSIGN = 48
TSHRASSIGN = 49
TBANDASSIGN = 50
TXORASSIGN = 51
TBORASSIGN = 52
TCOMMA = 53
THASH = 54
THASHHASH = 55
TALIGNAS = 56
TALIGNOF = 57
TAUTO = 58
TBOOL = 59
TBREAK = 60
TCASE = 61
TCHAR = 62
TCONST = 63
TCONSTEXPR = 64
TCONTINUE = 65
TDEFAULT = 66
TDO = 67
TDOUBLE = 68
TELSE = 69
TENUM = 70
TEXTERN = 71
TFALSE = 72
TFLOAT = 73
TFOR = 74
TGOTO = 75
TIF = 76
TINLINE = 77
TINT = 78
TLONG = 79
TNULLPTR = 80
TREGISTER = 81
TRESTRICT = 82
TRETURN = 83
TSHORT = 84
TSIGNED = 85
TSIZEOF = 86
TSTATIC = 87
TSTATIC_ASSERT = 88
TSTRUCT = 89
TSWITCH = 90
TTHREAD_LOCAL = 91
TTRUE = 92
TTYPEDEF = 93
TTYPEOF = 94
TTYPEOF_UNQUAL = 95
TUNION = 96
TUNSIGNED = 97
TVOID = 98
TVOLATILE = 99
TWHILE = 100
T_ATOMIC = 101
T_BITINT = 102
T_COMPLEX = 103
T_DECIMAL128 = 104
T_DECIMAL32 = 105
T_DECIMAL64 = 106
T_GENERIC = 107
T_IMAGINARY = 108
T_NORETURN = 109
T__ALIGNOF__ = 110
T__ASM__ = 111
T__ATTRIBUTE__ = 112
TDEFINE = 113
TELIF = 114
TENDIF = 115
TERROR = 116
TIFDEF = 117
TIFNDEF = 118
TINCLUDE = 119
TLINE = 120
TPRAGMA = 121
TUNDEF = 122
TWARNING = 123
T__VA_ARGS__ = 124
ALIAS_ALIGNAS = 125
ALIAS_ALIGNOF = 126
ALIAS_BOOL = 127
ALIAS_STATIC_ASSERT = 128
ALIAS_THREAD_LOCAL = 129
ALIAS__INLINE = 130
ALIAS__INLINE__ = 131
ALIAS__SIGNED = 132
ALIAS__SIGNED__ = 133
ALIAS__THREAD = 134
ALIAS__TYPEOF = 135
ALIAS__TYPEOF__ = 136
ALIAS__VOLATILE__ = 137
ALIAS__ASM = 138
NTOKENS = 139

TPPIDENT = TALIGNAS
TIDENT = TDEFINE


class Location(tuple):
    """struct location: an immutable (file, line, col) tuple, so struct copies are free."""
    __slots__ = ()

    def __new__(cls, file, line, col):
        return tuple.__new__(cls, (file, line, col))

    file = property(lambda self: self[0])
    line = property(lambda self: self[1])
    col = property(lambda self: self[2])


NOLOC = Location(None, 0, 0)


class Token(object):
    __slots__ = ('kind', 'hide', 'space', 'loc', 'lit')

    def __init__(self, kind=0, hide=False, space=False, loc=NOLOC, lit=None):
        self.kind = kind
        self.hide = hide
        self.space = space
        self.loc = loc
        self.lit = lit

    def set(self, o):
        """C struct assignment: *self = *o."""
        self.kind = o.kind
        self.hide = o.hide
        self.space = o.space
        self.loc = o.loc
        self.lit = o.lit

    def copy(self):
        return Token(self.kind, self.hide, self.space, self.loc, self.lit)


# enum typequal
QUALNONE = 0
QUALCONST = 1 << 1
QUALRESTRICT = 1 << 2
QUALVOLATILE = 1 << 3
QUALATOMIC = 1 << 4

# enum typekind
(TYPENONE, TYPEVOID, TYPEBOOL, TYPECHAR, TYPESHORT, TYPEINT, TYPEENUM, TYPELONG, TYPELLONG,
 TYPEFLOAT, TYPEDOUBLE, TYPELDOUBLE, TYPEPOINTER, TYPEARRAY, TYPEFUNC, TYPESTRUCT, TYPEUNION,
 TYPENULLPTR, TYPEBITINT) = range(19)

# enum typeprop
PROPNONE = 0
PROPCHAR = 1 << 0
PROPINT = 1 << 1
PROPREAL = 1 << 2
PROPARITH = 1 << 3
PROPSCALAR = 1 << 4
PROPFLOAT = 1 << 5
PROPVM = 1 << 6
PROPFLEX = 1 << 7


class Bitfield(tuple):
    """struct bitfield: immutable (before, after); assign a new one instead of mutating."""
    __slots__ = ()

    def __new__(cls, before=0, after=0):
        return tuple.__new__(cls, (before, after))

    before = property(lambda self: self[0])
    after = property(lambda self: self[1])


BITS0 = Bitfield(0, 0)


class Member(object):
    __slots__ = ('name', 'type', 'qual', 'offset', 'bits', 'next')

    def __init__(self):
        self.name = None
        self.type = None
        self.qual = 0
        self.offset = 0
        self.bits = BITS0
        self.next = None


class ListNode(object):
    """struct list. `owner` replaces listelement()."""
    __slots__ = ('prev', 'next', 'owner')

    def __init__(self, owner=None):
        self.prev = self.next = None
        self.owner = owner


def listinsert(lst, new):
    new.next = lst.next
    new.prev = lst
    lst.next.prev = new
    lst.next = new


def listremove(lst):
    lst.next.prev = lst.prev
    lst.prev.next = lst.next
    lst.next = lst.prev = None


class Type(object):
    """struct type. Union members: arith (issigned, iscomplex, width), array (length,
    ptrqual, asize = u.array.size), func (isvararg, params, nparam), structunion (tag,
    members). Fields that C leaves uninitialized default to zero/None."""
    __slots__ = ('kind', 'prop', 'align', 'size', 'value', 'base', 'link', 'qual', 'incomplete',
                 'issigned', 'iscomplex', 'width', 'length', 'ptrqual', 'asize',
                 'isvararg', 'params', 'nparam', 'tag', 'members')

    def __init__(self, kind=TYPENONE, prop=0, size=0, align=0, issigned=False, width=0,
                 base=None, incomplete=False, tag=None):
        self.kind = kind
        self.prop = prop
        self.align = align
        self.size = size
        self.value = None
        self.base = base
        self.link = None
        self.qual = 0
        self.incomplete = incomplete
        self.issigned = issigned
        self.iscomplex = False
        self.width = width
        self.length = None
        self.ptrqual = 0
        self.asize = None
        self.isvararg = False
        self.params = None
        self.nparam = 0
        self.tag = tag
        self.members = None


# enum declkind
DECLTYPE, DECLOBJECT, DECLFUNC, DECLCONST, DECLBUILTIN = range(5)
# enum linkage
LINKNONE, LINKINTERN, LINKEXTERN = range(3)
# enum storageduration
SDSTATIC, SDTHREAD, SDAUTO = range(3)
# enum builtinkind
(BUILTINALLOCA, BUILTINCONSTANTP, BUILTINEXPECT, BUILTININFF, BUILTINNANF, BUILTINOFFSETOF,
 BUILTINTYPESCOMPATIBLEP, BUILTINUNREACHABLE, BUILTINVAARG, BUILTINVACOPY, BUILTINVAEND,
 BUILTINVALIST, BUILTINVASTART) = range(13)


class Decl(object):
    """struct decl. Union u: obj (align, storage), func (inlinedefn, isnoreturn),
    enumconst, builtin."""
    __slots__ = ('name', 'kind', 'linkage', 'type', 'qual', 'value', 'asmname', 'defined',
                 'tentative', 'next', 'align', 'storage', 'inlinedefn', 'isnoreturn',
                 'enumconst', 'builtin')

    def __init__(self, name=None, kind=0, linkage=0, type=None, qual=0, builtin=0):
        self.name = name
        self.kind = kind
        self.linkage = linkage
        self.type = type
        self.qual = qual
        self.value = None
        self.asmname = None
        self.defined = False
        self.tentative = False
        self.next = None
        self.align = 0
        self.storage = 0
        self.inlinedefn = False
        self.isnoreturn = False
        self.enumconst = 0
        self.builtin = builtin


class Scope(object):
    """struct scope. The maps are dicts (empty dict == len 0)."""
    __slots__ = ('tags', 'decls', 'breaklabel', 'continuelabel', 'switchcases', 'parent')

    def __init__(self):
        self.tags = {}
        self.decls = {}
        self.breaklabel = None
        self.continuelabel = None
        self.switchcases = None
        self.parent = None


# enum exprkind
(EXPRIDENT, EXPRCONST, EXPRSTRING, EXPRCALL, EXPRBITFIELD, EXPRINCDEC, EXPRCOMPOUND,
 EXPRUNARY, EXPRCAST, EXPRBINARY, EXPRCOND, EXPRASSIGN, EXPRCOMMA, EXPRBUILTIN, EXPRTEMP,
 EXPRSIZEOF) = range(16)

_pack_d = _struct.Struct('<d').pack
_unpack_d = _struct.Struct('<d').unpack
_pack_q = _struct.Struct('<Q').pack
_unpack_q = _struct.Struct('<Q').unpack


def dbits(f):
    """The bits of a double as an unsigned 64-bit int."""
    return _unpack_q(_pack_d(f))[0]


def bitsd(u):
    return _unpack_d(_pack_q(u))[0]


class Expr(object):
    """struct expr. Union u is flattened:
      ident.decl / compound.decl -> decl, compound.init -> init,
      constant.{u,i,f} -> u (the 64 bits, unsigned) with i and f as views,
      string.{size,data} -> ssize, sdata (bytearray of the raw element bytes),
      call.{args,nargs} -> args, nargs, bitfield.bits -> bits, incdec.post -> post,
      binary/assign.{l,r} -> l, r, cond.{t,f} -> ct, cf, builtin.kind -> bkind,
      szof.type -> sztype, temp -> temp."""
    __slots__ = ('kind', 'lvalue', 'decayed', 'type', 'qual', 'op', 'base', 'next', 'toeval',
                 'decl', 'init', 'u', 'ssize', 'sdata', 'args', 'nargs', 'bits', 'post',
                 'l', 'r', 'ct', 'cf', 'bkind', 'sztype', 'temp')

    def __init__(self, kind, type, base):
        self.qual = QUALNONE
        self.type = type
        self.lvalue = False
        self.decayed = False
        self.kind = kind
        self.base = base
        self.next = None
        self.toeval = None
        self.op = 0
        self.u = 0

    @property
    def i(self):
        u = self.u
        return u - (1 << 64) if u >> 63 else u

    @i.setter
    def i(self, v):
        self.u = v & M64

    @property
    def f(self):
        return _unpack_d(_pack_q(self.u))[0]

    @f.setter
    def f(self, v):
        self.u = _unpack_q(_pack_d(v))[0]


class Init(object):
    __slots__ = ('start', 'end', 'expr', 'bits', 'next')

    def __init__(self, start, end, bits, expr):
        self.start = start
        self.end = end
        self.expr = expr
        self.bits = bits
        self.next = None


# enum attrkind
ATTRNORETURN = 1 << 0
ATTRALIGNED = 1 << 2
ATTRCONSTRUCTOR = 1 << 3
ATTRDESTRUCTOR = 1 << 4
ATTRPACKED = 1 << 5
ATTRSECTION = 1 << 6


class Attr(object):
    __slots__ = ('kind', 'align', 'section')

    def __init__(self):
        self.kind = 0
        self.align = 0
        self.section = None


# enum ppflags
PPNEWLINE = 1 << 0


class GotoLabel(object):
    __slots__ = ('label', 'defined')

    def __init__(self, label):
        self.label = label
        self.defined = False


class SwitchCases(object):
    __slots__ = ('root', 'type', 'defaultlabel')

    def __init__(self):
        self.root = None
        self.type = None
        self.defaultlabel = None


class Ref(object):
    """A pointer to a scalar (int *, char **, ...) passed as an out-parameter."""
    __slots__ = ('v',)

    def __init__(self, v=None):
        self.v = v


def ALIGNDOWN(x, n):
    return x & -n


def ALIGNUP(x, n):
    return (x + n - 1) & -n
