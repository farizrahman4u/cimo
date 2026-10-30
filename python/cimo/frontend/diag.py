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
                 TYPELDOUBLE, TYPESTRUCT, TYPEUNION, TYPEARRAY, DECLFUNC, DECLOBJECT, SDTHREAD,
                 QUALNONE, Type, Token)
from .token import tok, tokenget, tokenstr, error
from .type import mkpointertype, typevoid

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
'''.split())

_ATTRMSG = {
    'vector_size': "vector types (vector_size) are not supported yet",
    'ext_vector_type': "vector types (ext_vector_type) are not supported yet",
    'cleanup': "attribute 'cleanup' is not supported",
    'transparent_union': "attribute 'transparent_union' is not supported",
}

# type names cimo rejects outright
_BADTYPES = frozenset('''_Float16 __fp16 __bf16 _Float32 _Float32x _Float64 _Float64x
    _Float128 _Float128x __float128 __float80 __ibm128 _Decimal32 _Decimal64 _Decimal128
    _Accum _Fract _Sat'''.split())

# identifiers of GNU extensions, reported where they would be "undeclared"
_GNUIDENTS = {
    '__label__': "local labels ('__label__') are not supported",
    '__auto_type': "'__auto_type' is not supported",
    '__builtin_choose_expr': "'__builtin_choose_expr' is not supported",
    '__real__': "'__real__' is not supported (complex numbers are not supported)",
    '__imag__': "'__imag__' is not supported (complex numbers are not supported)",
}

INLINEASM = "inline assembly is not supported"

# __int128 and its typedefs: 16-byte opaque types, so that system headers that merely
# declare them (glibc's signal.h: struct fpsimd_context) still compile
typeint128 = Type(kind=TYPESTRUCT, size=16, align=16, tag='__int128')
typeuint128 = Type(kind=TYPESTRUCT, size=16, align=16, tag='unsigned __int128')

typevoidptr = mkpointertype(typevoid, QUALNONE)  # nullptr is ((void *)0)

aliases = {}  # extra keyword aliases, token kind -> token kind (see setup)
inlinedefns = []  # [(decl, func)]: C99 inline definitions, held back (see emitinlinedefns)
referenced = set()  # function declarations named in an expression
declalign = 0  # `aligned` attribute after the declarator being parsed by decl()
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
        if t is typeint128 or t is typeuint128:
            return "type '%s' is not supported" % t.tag
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
    if t.kind != TYPESTRUCT and t.kind != TYPEUNION or t is typeint128 or t is typeuint128:
        error(tok.loc, "GNU attribute 'aligned' on a typedef is only supported for struct and union types")
    v = Type()
    for f in Type.__slots__:
        setattr(v, f, getattr(t, f))
    v.align = align
    v.value = None
    v.link = None
    return v
