"""Port of pp.c: cproc's minimal preprocessor (#define, #undef, line markers,
#pragma, #error/#warning; macro expansion without ##).

Token arrays are lists of Token objects; a C `struct token *` into an array is
(list, index). A context frame is [toks, index, ntoken, macro].
"""

import sys

from .cc import (Token, TNEWLINE, TEOF, THASH, THASHHASH, TLPAREN, TRPAREN, TCOMMA, TELLIPSIS,
                 TPPIDENT, TNUMBER, TSTRINGLIT, TCHARCONST, TIF, TIFDEF, TIFNDEF, TELIF, TENDIF,
                 TINCLUDE, TDEFINE, TUNDEF, TLINE, TERROR, TWARNING, TPRAGMA, T__VA_ARGS__,
                 PPNEWLINE, Location,
                 ALIAS_ALIGNAS, ALIAS_ALIGNOF, ALIAS_BOOL, ALIAS__INLINE, ALIAS__INLINE__,
                 ALIAS__SIGNED, ALIAS__SIGNED__, ALIAS_STATIC_ASSERT, ALIAS_THREAD_LOCAL,
                 ALIAS__THREAD, ALIAS__TYPEOF, ALIAS__TYPEOF__, ALIAS__VOLATILE__, ALIAS__ASM,
                 TALIGNAS, TALIGNOF, TBOOL, TINLINE, TSIGNED, TSTATIC_ASSERT, TTHREAD_LOCAL,
                 TTYPEOF, TVOLATILE, T__ASM__)
from . import token as TK
from .token import tok, tokencheck, tokenstr, error
from . import scan as S
from . import util as U

PARAMTOK = 1 << 0  # the parameter is used normally
PARAMSTR = 1 << 1  # the parameter is used with the '#' operator
PARAMVAR = 1 << 2  # the parameter is __VA_ARGS__

MACROOBJ = 0
MACROFUNC = 1


class MacroParam(object):
    __slots__ = ('name', 'flags')

    def __init__(self):
        self.name = None
        self.flags = 0


class MacroArg(object):
    """token -> (toks, start); str is the stringized argument."""
    __slots__ = ('toks', 'start', 'ntoken', 'str')

    def __init__(self):
        self.toks = None
        self.start = 0
        self.ntoken = 0
        self.str = None


class Macro(object):
    __slots__ = ('kind', 'name', 'hide', 'param', 'nparam', 'arg', 'token', 'ntoken')

    def __init__(self):
        self.kind = MACROOBJ
        self.name = None
        self.hide = False
        self.param = []
        self.nparam = 0
        self.arg = None
        self.token = None
        self.ntoken = 0


ppflags = 0
ctx = []  # frames: [toks, index, ntoken, macro]
macros = {}  # token kind - TPPIDENT -> macro
macrodepth = 0
_newline = True  # static in nextinto()
_peekpending = Token()  # static in peek()

_ALIASES = {
    ALIAS_ALIGNAS: TALIGNAS, ALIAS_ALIGNOF: TALIGNOF, ALIAS_BOOL: TBOOL,
    ALIAS__INLINE: TINLINE, ALIAS__INLINE__: TINLINE,
    ALIAS__SIGNED: TSIGNED, ALIAS__SIGNED__: TSIGNED,
    ALIAS_STATIC_ASSERT: TSTATIC_ASSERT,
    ALIAS_THREAD_LOCAL: TTHREAD_LOCAL, ALIAS__THREAD: TTHREAD_LOCAL,
    ALIAS__TYPEOF: TTYPEOF, ALIAS__TYPEOF__: TTYPEOF,
    ALIAS__VOLATILE__: TVOLATILE, ALIAS__ASM: T__ASM__,
}


def _reset():
    global ppflags, macrodepth, _newline, _peekpending
    ppflags = 0
    del ctx[:]
    macros.clear()
    macrodepth = 0
    _newline = True
    _peekpending = Token()


def ppinit():
    next()


def macrodel(m):
    pass


def macroequal(m1, m2):
    """check if two macro definitions are equal, as in C11 6.10.3p2"""
    if m1.kind != m2.kind:
        return False
    if m1.kind == MACROFUNC:
        if m1.nparam != m2.nparam:
            return False
        for i in range(m1.nparam):
            p1 = m1.param[i]
            p2 = m2.param[i]
            if p1.name != p2.name or p1.flags != p2.flags:
                return False
    if m1.ntoken != m2.ntoken:
        return False
    for i in range(m1.ntoken):
        t1 = m1.token[i]
        t2 = m2.token[i]
        if t1.kind != t2.kind:
            return False
        if t1.lit is not None and t1.lit != t2.lit:
            return False
    return True


def macroparam(m, t):
    """find the index of a macro parameter with the given name"""
    if t.kind >= TPPIDENT:
        name = tokenstr(t.kind)
        for i in range(m.nparam):
            if m.param[i].name == name:
                return i
    return -1


def macroget(ident):
    return macros.get(ident - TPPIDENT)


def macrodone(m):
    global macrodepth
    m.hide = False
    macrodepth -= 1


def macrovarargs(m):
    return m.kind == MACROFUNC and m.nparam > 0 and m.param[m.nparam - 1].flags & PARAMVAR


def framenext(f):
    f[2] -= 1
    t = f[0][f[1]]
    f[1] += 1
    return t


def ctxpush(toks, start, n, m, space):
    """push a new context frame"""
    f = [toks, start, n, m]
    ctx.append(f)
    if n > 0:
        toks[start].space = space
    return f


def ctxnext():
    """get the next token from the context"""
    while True:
        while ctx:
            f = ctx[-1]
            if f[2]:
                break
            if f[3]:
                macrodone(f[3])
            ctx.pop()
        if not ctx:
            return None
        m = f[3]
        if m and m.kind == MACROFUNC:
            # try to expand macro parameter
            ft = f[0][f[1]]
            space = ft.space
            if ft.kind == THASH:
                framenext(f)
                t = framenext(f)
                assert t
                i = macroparam(m, t)
                assert i != -1
                f = ctxpush([m.arg[i].str], 0, 1, None, space)
            elif ft.kind >= TPPIDENT:
                i = macroparam(m, ft)
                if i != -1:
                    framenext(f)
                    if m.arg[i].ntoken == 0:
                        continue  # goto again
                    a = m.arg[i]
                    f = ctxpush(a.toks, a.start, a.ntoken, None, space)
            # XXX: token concatenation
        return framenext(f)


def define():
    m = Macro()
    ident = tok.kind
    m.name = tokencheck(tok, TPPIDENT, "after #define")
    m.hide = False
    repl = []
    t = Token()
    repl.append(t)
    S.scan(t)
    params = []
    if t.kind == TLPAREN and not t.space:
        m.kind = MACROFUNC
        # read macro parameter names
        p = None
        while True:
            S.scan(tok)
            if tok.kind == TRPAREN:
                break
            if p:
                if p.flags & PARAMVAR:
                    tokencheck(tok, TRPAREN, "after '...'")
                tokencheck(tok, TCOMMA, "or ')' after macro parameter")
                S.scan(tok)
            p = MacroParam()
            params.append(p)
            p.flags = 0
            if tok.kind == TELLIPSIS:
                p.name = "__VA_ARGS__"
                p.flags |= PARAMVAR
            else:
                p.name = tokencheck(tok, TPPIDENT, "of macro parameter name or '...'")
        S.scan(t)  # first token in replacement list
    else:
        m.kind = MACROOBJ
    m.param = params
    m.nparam = len(params)

    # read macro body
    i = macroparam(m, t)
    while t.kind != TNEWLINE and t.kind != TEOF:
        if t.kind == THASHHASH:
            error(t.loc, "'##' operator is not yet implemented")
        prev = t.kind
        t = Token()
        repl.append(t)
        S.scan(t)
        if t.kind == T__VA_ARGS__ and not macrovarargs(m):
            error(t.loc, "__VA_ARGS__ can only be used in variadic function-like macros")
        if m.kind != MACROFUNC:
            continue
        if i != -1:
            m.param[i].flags |= PARAMTOK
        i = macroparam(m, t)
        if prev == THASH:
            tokencheck(t, TPPIDENT, "after '#' operator")
            if i == -1:
                error(t.loc, "'%s' is not a macro parameter name", t.lit)
            m.param[i].flags |= PARAMSTR
            i = -1
    m.token = repl
    m.ntoken = len(repl) - 1
    tok.set(t)

    i = ident - TPPIDENT
    other = macros.get(i)
    macros[i] = m
    if other:
        if not macroequal(m, other):
            error(tok.loc, "redefinition of macro '%s'", m.name)
        macrodel(other)


def undef():
    ident = tok.kind
    tokencheck(tok, TPPIDENT, "after #undef")
    m = macroget(ident)
    if m:
        macrodel(m)
        macros[ident - TPPIDENT] = None
    S.scan(tok)


def _strtoull0(s):
    """strtoull(s, NULL, 0)"""
    i = 0
    n = len(s)
    while i < n and s[i] in ' \t\n\v\f\r':
        i += 1
    neg = False
    if i < n and s[i] in '+-':
        neg = s[i] == '-'
        i += 1
    base = 10
    if i < n and s[i] == '0':
        if i + 2 < n + 1 and s[i + 1:i + 2] in ('x', 'X') and i + 2 < n and \
                s[i + 2] in '0123456789abcdefABCDEF':
            base = 16
            i += 2
        else:
            base = 8
    v = 0
    digits = '0123456789abcdefghijklmnopqrstuvwxyz'
    while i < n:
        d = digits.find(s[i].lower())
        if d < 0 or d >= base:
            break
        v = v * base + d
        i += 1
    if v > 0xffffffffffffffff:
        return 0xffffffffffffffff
    return (-v if neg else v) & 0xffffffffffffffff


def directive():
    global ppflags
    S.scan(tok)
    if tok.kind == TNEWLINE:
        return  # empty directive
    oldflags = ppflags
    ppflags |= PPNEWLINE
    kind = tok.kind
    if kind == TNUMBER or kind == TLINE:
        if kind == TLINE:
            S.scan(tok)
            tokencheck(tok, TNUMBER, "after #line")
        # line:
        nline = _strtoull0(tok.lit)
        ncol = 1
        S.scan(tok)
        nfile = tok.loc.file
        if tok.kind == TSTRINGLIT:
            # XXX: handle escape sequences (reuse string decoding from expr.c)
            q = tok.lit.find('"')
            assert q >= 0
            q2 = tok.lit.find('"', q + 1)
            assert q2 >= 0
            nfile = tok.lit[q + 1:q2]
            S.scan(tok)
        while tok.kind == TNUMBER:
            S.scan(tok)
        S.scansetloc(Location(nfile, nline, ncol))
    elif kind == TIF:
        error(tok.loc, "#if directive is not implemented")
    elif kind == TIFDEF:
        error(tok.loc, "#ifdef directive is not implemented")
    elif kind == TIFNDEF:
        error(tok.loc, "#ifndef directive is not implemented")
    elif kind == TELIF:
        error(tok.loc, "#elif directive is not implemented")
    elif kind == TENDIF:
        error(tok.loc, "#endif directive is not implemented")
    elif kind == TINCLUDE:
        error(tok.loc, "#include directive is not implemented")
    elif kind == TDEFINE:
        S.scan(tok)
        define()
    elif kind == TUNDEF:
        S.scan(tok)
        undef()
    elif kind == TERROR or kind == TWARNING:
        msg = [TK.diagloc(tok.loc), "%s directive:" % ("#error" if kind == TERROR else "#warning")]
        S.scan(tok)
        while tok.kind != TNEWLINE and tok.kind != TEOF:
            TK.tokenprint(tok, msg)
            S.scan(tok)
        if kind == TERROR:
            raise U.FrontendError(''.join(msg))
        sys.stderr.write(''.join(msg) + '\n')
    elif kind == TPRAGMA:
        while tok.kind != TNEWLINE and tok.kind != TEOF:
            next()
    else:
        error(tok.loc, "invalid preprocessor directive #%s", tokenstr(tok.kind))
    tokencheck(tok, TNEWLINE, "after preprocessing directive")
    ppflags = oldflags


def nextinto(t):
    """get the next token without expanding it"""
    global _newline
    while True:
        S.scan(t)
        if _newline and t.kind == THASH:
            directive()
        else:
            _newline = tok.kind == TNEWLINE
            break


def rawnext():
    t = ctxnext()
    if t is None:
        t = tok
        nextinto(t)
    return t


def peekparen():
    t = ctxnext()
    if t is not None:
        if t.kind == TLPAREN:
            return True
        f = ctx[-1]
        f[1] -= 1
        f[2] += 1
        return False
    pending = []
    while True:
        t = Token()
        pending.append(t)
        nextinto(t)
        if t.kind != TNEWLINE:
            break
    if t.kind == TLPAREN:
        return True
    ctxpush(pending, 0, len(pending), None, pending[0].space)
    return False


def stringize(buf, t):
    """buf is a list of chars."""
    if (t.space or t.kind == TNEWLINE) and len(buf) > 1 and buf[-1] != ' ':
        buf.append(' ')
    lit = t.lit if t.lit is not None else tokenstr(t.kind)
    if t.kind == TSTRINGLIT or t.kind == TCHARCONST:
        for c in lit:
            if c == '\\' or c == '"':
                buf.append('\\')
            buf.append(c)
    elif lit is not None:
        buf.extend(lit)


def expand(t):
    global macrodepth
    if t.kind < TPPIDENT:
        return False
    m = macroget(t.kind)
    if not m or m.hide:
        t.hide = True
    if t.hide:
        return False
    space = t.space
    if m.kind == MACROFUNC:
        if not peekparen():
            return False
        expandfunc(m)
    ctxpush(m.token, 0, m.ntoken, m, space)
    m.hide = True
    macrodepth += 1
    return True


def expandfunc(m):
    # read macro arguments
    paren = 0
    depth = macrodepth
    toks = []
    arg = [MacroArg() for _ in range(m.nparam)]
    t = rawnext()
    i = 0
    while i < m.nparam:
        p = m.param[i]
        if p.flags & PARAMSTR:
            sbuf = ['"']
        arg[i].ntoken = 0
        while True:
            if t.kind == TEOF:
                error(t.loc, "EOF when reading macro parameters")
            if macrodepth <= depth:
                # adjust current macro depth, in case it got shallower
                depth = macrodepth
                if paren == 0 and (t.kind == TRPAREN or t.kind == TCOMMA and not (p.flags & PARAMVAR)):
                    break
                if t.kind == TLPAREN:
                    paren += 1
                elif t.kind == TRPAREN:
                    paren -= 1
                if p.flags & PARAMSTR:
                    stringize(sbuf, t)
            if p.flags & PARAMTOK and not expand(t):
                toks.append(t.copy())
                arg[i].ntoken += 1
            t = rawnext()
        if p.flags & PARAMSTR:
            sbuf.append('"')
            arg[i].str = Token(kind=TSTRINGLIT, lit=''.join(sbuf))
        if t.kind == TRPAREN:
            break
        t = rawnext()
        i += 1
    if i + 1 < m.nparam:
        error(t.loc, "not enough arguments for macro '%s'", m.name)
    if t.kind != TRPAREN:
        error(t.loc, "too many arguments for macro '%s'", m.name)
    pos = 0
    for i in range(m.nparam):
        arg[i].toks = toks
        arg[i].start = pos
        pos += arg[i].ntoken
    m.arg = arg


def next():
    while True:
        t = rawnext()
        if expand(t) or t.kind == TNEWLINE and not (ppflags & PPNEWLINE):
            continue
        break
    tok.set(t)

    # keyword aliases
    k = _ALIASES.get(tok.kind)
    if k is not None:
        tok.kind = k


def peek(kind):
    old = tok.copy()
    next()
    if tok.kind == kind:
        next()
        return True
    _peekpending.set(tok)
    tok.set(old)
    ctxpush([_peekpending], 0, 1, None, _peekpending.space)
    return False


def expect(kind, msg):
    lit = tokencheck(tok, kind, msg)
    next()
    return lit


def consume(kind):
    if tok.kind != kind:
        return False
    next()
    return True
