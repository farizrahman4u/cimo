"""cimo extension (not in cproc): clear errors for C that cimo does not support.

Everything here is used only when `util.diagnostics` is set (compile(diagnostics=True), which
cimo's driver passes); the command line keeps cproc's exact behavior. The ported modules call
in here from small `if U.diagnostics:` hooks, so their control flow stays cproc's.

Policy: an unsupported feature is an error naming it, with a location, never a silent
miscompile or a misleading parse error. Declaring something cimo cannot generate code for
(`long double`, `__int128`, thread-local objects) is fine, as system headers do; using it or
defining an object of it is an error.
"""

from .cc import (TIDENT, TMUL, TLBRACE, T__ASM__, T__ALIGNOF__, TCONST, TRESTRICT, TVOLATILE,
                 TYPELDOUBLE, TYPESTRUCT, TYPEUNION, TYPEARRAY, TYPELONG, TYPEPOINTER, TYPEFUNC,
                 TYPEENUM, TYPEBOOL, TYPEBITINT, PROPINT, PROPFLOAT, PROPSCALAR, DECLFUNC, DECLOBJECT, LINKEXTERN, SDTHREAD,
                 QUALNONE, Type, Token)
from .token import tok, tokenget, tokenstr, error
from .type import mkpointertype, typevoid, typecompatible

# GNU and standard attributes that do not change the meaning of a program compiled by cimo
# (hints, warnings, linkage/visibility details of a single module); they are parsed and
# ignored. Attributes cproc implements (aligned, packed, section, constructor, destructor,
# noreturn) are handled by attr.py; any other attribute is an error.
IGNOREDATTRS = frozenset('''
    noinline always_inline gnu_inline flatten noclone noipa no_icf optimize target
    hot cold unused used maybe_unused retain const pure malloc leaf nothrow noreturn
    returns_nonnull nonnull nonnull_if_nonzero null_terminated_string_arg sentinel
    format format_arg alloc_size alloc_align assume_aligned warn_unused_result nodiscard
    deprecated unavailable error warning visibility externally_visible weak artificial
    may_alias no_instrument_function no_profile_instrument_function no_sanitize
    no_sanitize_address no_sanitize_thread no_sanitize_undefined no_address_safety_analysis
    no_stack_protector stack_protect no_split_stack no_reorder noplt nocommon common
    access nonstring fd_arg fd_arg_read fd_arg_write designated_init counted_by
    strict_flex_array warn_if_not_aligned fallthrough assume musttail expected_throw
    tainted_args zero_call_used_regs uninitialized unsequenced reproducible
    returns_twice nomips16 mips16 nomicromips micromips long_call short_call far near
'''.split())

_ATTRMSG = {
    'ext_vector_type': "vector types (ext_vector_type) are not supported yet",
    'cleanup': "attribute 'cleanup' is not supported",
    'transparent_union': "attribute 'transparent_union' is not supported",
}

# type names cimo rejects outright
_BADTYPES = frozenset('''_Float128 _Float128x __float128 __float80 __ibm128 _Decimal32 _Decimal64 _Decimal128
    _Accum _Fract _Sat'''.split())
# the ISO/IEC TS 18661-3 (C23) interchange types cimo has, and GNU's half-precision ones
_FLOATN = {'_Float16': 'typehalf', '__fp16': 'typehalf', '__bf16': 'typebf16', '_Float32': 'typefloat',
           '_Float64': 'typedouble', '_Float32x': 'typedouble', '_Float64x': 'typeldouble'}

# identifiers of GNU extensions, reported where they would be "undeclared"
_GNUIDENTS = {
    '__label__': "local labels ('__label__') are not supported",
    '__auto_type': "'__auto_type' is not supported",
    '__builtin_choose_expr': "'__builtin_choose_expr' is not supported",
}

INLINEASM = "inline assembly is not supported"

# __int128 and its typedefs: the frontend lowers their operations to 64-bit halves (qbe._i128*)
from .type import typeint128, typeuint128  # noqa: E402


aliases = {}  # extra keyword aliases, token kind -> token kind (see setup)
inlinedefns = []  # [(decl, func)]: C99 inline definitions, held back (see emitinlinedefns)
referenced = set()  # function declarations named in an expression
declalign = 0  # `aligned` attribute after the declarator being parsed by decl()
declmode = None  # `mode` attribute of the declaration being parsed by decl()
memberpacked = False  # `packed` attribute after a struct member's declarator
declvector = None  # `vector_size` attribute of the declaration being parsed (bytes)
declnoinline = False  # `noinline` or `noipa` in the declaration being parsed by decl()
noinline = set()  # the functions declared so (the backend's inliner leaves their calls: qbe.paraminfo['#noinline'])
_cache = {}


def setup():
    """After tokeninit(): GNU spellings of keywords that cproc lacks."""
    aliases.clear()
    for name, kind in (('asm', T__ASM__), ('__restrict', TRESTRICT),
                       ('__restrict__', TRESTRICT), ('__const', TCONST),
                       ('__const__', TCONST), ('__volatile', TVOLATILE),
                       ('__alignof', T__ALIGNOF__)):
        aliases[tokenget(name)] = kind
    func = tokenget('__func__')
    aliases[tokenget('__PRETTY_FUNCTION__')] = func
    aliases[tokenget('__FUNCTION__')] = func
    global declalign
    del inlinedefns[:]
    referenced.clear()
    noinline.clear()
    declalign = 0
    _cache.clear()


def peekkind():
    """The kind of the token after the current one, without consuming anything."""
    from . import pp as PP
    old = tok.copy()
    PP.next()
    k = tok.kind
    pending = Token()
    pending.set(tok)
    tok.set(old)
    PP.ctxpush([pending], 0, 1, None, pending.space)
    return k


def attribute(name, prefix, loc):
    """An attribute that attr.py does not implement: ignore it or reject it."""
    if prefix in ('gnu', '') and name in IGNOREDATTRS:
        return
    msg = _ATTRMSG.get(name)
    if msg is None:
        msg = "attribute '%s%s' is not supported" % (prefix + '::' if prefix not in ('gnu', '') else '', name)
    error(loc, "%s", msg)


def typespec(name, typed):
    """An identifier in declaration specifiers that is not a typedef name. Returns the type
    for __int128 spellings, errors for other unsupported types, None otherwise. typed: a
    type specifier was already seen (then only `__int128` can follow, as in `unsigned
    __int128`)."""
    if name == '__int128':
        return typeint128
    if typed:
        return None
    if name == '__int128_t':
        return typeint128
    if name == '__uint128_t':
        return typeuint128
    if name in _FLOATN:
        from . import type as TY
        return getattr(TY, _FLOATN[name])
    if name in _BADTYPES:
        error(tok.loc, "type '%s' is not supported", name)
    msg = _GNUIDENTS.get(name)
    if msg:
        error(tok.loc, "%s", msg)
    return None


def undeclared(name):
    """Before cproc's "undeclared identifier" error: a clearer one where possible."""
    msg = _GNUIDENTS.get(name)
    if msg:
        error(tok.loc, "%s", msg)
    if name in _BADTYPES or name in ('__int128', '__int128_t', '__uint128_t'):
        error(tok.loc, "type '%s' is not supported", name)
    if name.startswith('__builtin_') or name.startswith('__sync_') or name.startswith('__atomic_'):
        error(tok.loc, "built-in function '%s' is not supported", name)
    if peekkind() >= TIDENT:
        error(tok.loc, "unknown type name '%s'", name)


def unknowntype():
    """No type at the start of a declaration (file scope, parameter, struct member): an
    undeclared identifier followed by a declarator is an unknown type name, as gcc says."""
    if tok.kind >= TIDENT:
        name = tokenstr(tok.kind)
        typespec(name, False)
        k = peekkind()
        if k >= TIDENT or k == TMUL:
            error(tok.loc, "unknown type name '%s'", name)


def unsupported(t):
    """The message if type t is (or contains) a type cimo cannot generate code for."""
    k = t.kind
    if k == TYPELDOUBLE:
        return "long double is not supported" if t.size == 16 else None
    if k == TYPEARRAY:
        return unsupported(t.base)
    if k == TYPESTRUCT or k == TYPEUNION:
        if t.incomplete:
            return None
        r = _cache.get(id(t), 0)
        if r == 0:
            r = None
            m = t.members
            while m and not r:
                r = unsupported(m.type)
                m = m.next
            _cache[id(t)] = r
        return r
    return None


def checktype(t):
    msg = unsupported(t)
    if msg:
        error(tok.loc, "%s", msg)


def checkfunc(t):
    """A function definition of type t."""
    checktype(t.base)
    p = t.params
    while p:
        checktype(p.type)
        p = p.next


def emitinlinedefns():
    """At the end of the translation unit: a C99 inline definition (cproc emits nothing for
    it, so calls fail to link) becomes a local function when it is referenced, or the
    external definition when a later declaration made it one (`extern` or no `inline`)."""
    from . import qbe as Q
    for d, f in inlinedefns:
        if not d.inlinedefn:
            Q.emitfunc(f, True)
        elif d in referenced:
            Q.emitfunc(f, False)
    del inlinedefns[:]


def identifier(d):
    """An identifier in an expression."""
    if d.kind == DECLFUNC:
        referenced.add(d)
    elif d.kind == DECLOBJECT and d.storage == SDTHREAD:
        error(tok.loc, "thread-local storage is not supported")


def stmtexpr():
    """After '(' in an expression."""
    if tok.kind == TLBRACE:
        error(tok.loc, "statement expressions ('({ ... })', a GNU extension) are not supported")


def imaginary(suffix):
    """Before cproc's "invalid ... constant suffix" error."""
    if suffix.strip('fFlLuU') in ('i', 'j', 'I', 'J'):
        error(tok.loc, "imaginary constants (a GNU extension) are not supported")


def alignedtype(t, align):
    """`typedef struct {...} T __attribute__((aligned(n)))` (glibc's pthread.h): as in gcc,
    T is a variant of the struct with alignment n and the same size."""
    if t.kind in (TYPEARRAY, TYPEFUNC) or t is typeint128 or t is typeuint128:
        error(tok.loc, "GNU attribute 'aligned' on a typedef is only supported for scalar, struct and union types")
    v = Type()
    for f in Type.__slots__:
        setattr(v, f, getattr(t, f))
    v.align = align
    v.value = None
    v.link = None
    if t.kind != TYPESTRUCT and t.kind != TYPEUNION:  # cimo: a scalar variant, compatible with t
        v.origin = t.origin or t
    return v


def mentionslong(t, seen=None):
    """Whether type t is or contains `long` (through pointers, arrays, function types and
    struct members)."""
    k = t.kind
    if k == TYPELONG:
        return True
    if k in (TYPEPOINTER, TYPEARRAY, TYPEENUM):
        return t.base is not None and mentionslong(t.base, seen)
    if k == TYPEFUNC:
        p = t.params
        while p:
            if mentionslong(p.type, seen):
                return True
            p = p.next
        return mentionslong(t.base, seen)
    if k == TYPESTRUCT or k == TYPEUNION:
        seen = seen if seen is not None else set()
        if id(t) in seen:
            return False
        seen.add(id(t))
        m = t.members
        while m:
            if mentionslong(m.type, seen):
                return True
            m = m.next
    return False


def longexterns():
    """After compile(): the names of the file-scope declarations with external linkage and no
    definition whose type mentions `long`. On Windows cimo's long is 64-bit but a DLL's is
    32-bit, so the driver warns when one of these resolves to a `link=` library."""
    from . import scope as SC
    return sorted(d.asmname or d.name for d in SC.filescope.decls.values()
                  if d.kind in (DECLFUNC, DECLOBJECT) and d.linkage == LINKEXTERN
                  and not d.defined and not d.tentative and mentionslong(d.type))


def ptrsimilar(a, b):
    """Pointer targets a and b that cproc calls incompatible but gcc accepts without a
    warning, as cimo does: integer types that differ only in signedness (`int *` and
    `unsigned *`, `char *` and `unsigned char *`), and qualifiers below the top level
    (`int **` and `const int *const *` compared). Code generation does not depend on either."""
    if a.kind == TYPEPOINTER and b.kind == TYPEPOINTER:
        return typecompatible(a.base, b.base) or ptrsimilar(a.base, b.base)
    return (a.prop & PROPINT and b.prop & PROPINT and a.kind == b.kind
            and a.kind not in (TYPEBOOL, TYPEENUM, TYPEBITINT))


_MODES = {'QI': 1, 'byte': 1, 'HI': 2, 'SI': 4, 'DI': 8, 'word': 8, 'pointer': 8, 'TI': 16, 'SF': 4, 'DF': 8}


def modetype(t, mode):
    """`__attribute__((mode(M)))` (GNU): the integer (or floating) type of mode M's size, signed if t is."""
    from . import type as TY
    size = _MODES.get(mode)
    if size is None:
        error(tok.loc, "unknown machine mode '%s'", mode)
    if mode in ('SF', 'DF'):
        if not (t.prop & PROPFLOAT) and not (t.prop & PROPINT):
            error(tok.loc, "mode '%s' applied to a non-arithmetic type", mode)
        return TY.typefloat if size == 4 else TY.typedouble
    if not (t.prop & PROPINT) or t.kind == TYPEBOOL:
        error(tok.loc, "mode '%s' applied to a non-integer type", mode)
    if size == 16:
        return typeint128 if t.issigned else typeuint128
    return {1: (TY.typeuchar, TY.typeschar), 2: (TY.typeushort, TY.typeshort), 4: (TY.typeuint, TY.typeint),
            8: (TY.typeulong, TY.typelong)}[size][bool(t.issigned)]


def vectortype(t, size):
    """`__attribute__((vector_size(N)))` on a declaration of type t: a vector of t (or of an array's, pointer's or
    function's innermost element: gcc applies it there)."""
    from . import type as TY
    if t.kind == TYPEARRAY:
        a = TY.mkarraytype(vectortype(t.base, size), t.qual, t.size // t.base.size if t.base.size else 0)
        a.length, a.incomplete = t.length, t.incomplete
        return a
    if t.kind == TYPEFUNC or t.kind == TYPEPOINTER:  # `vector int f(...)`, `vector int *p`
        v = Type()
        for name in Type.__slots__:
            setattr(v, name, getattr(t, name))
        v.base = vectortype(t.base, size)
        v.value = v.link = None
        return v
    if not (t.prop & (PROPINT | PROPFLOAT)) or t.kind == TYPEBOOL or not (t.prop & PROPSCALAR) or t.kind == TYPEENUM:
        error(tok.loc, "vector_size applied to a type that is not an integer or floating type")
    if size <= 0 or size % t.size or (size // t.size) & (size // t.size - 1):
        error(tok.loc, "vector size %d is not a power-of-two multiple of the element size", size)
    return TY.mkvector(t, size)
