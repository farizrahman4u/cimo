"""Port of token.c."""

from .cc import (TOKENS, TEOF, TNEWLINE, TOTHER, TNUMBER, TCHARCONST, TSTRINGLIT, TIDENT,
                 TPPIDENT, Token)
from . import util as U

# the current token; a single object updated in place (C: struct token tok)
tok = Token()
tokmap = {}  # maps string to token
tokstr = []  # maps token to string (None for tokens without a spelling)


def tokeninit():
    tokmap.clear()
    del tokstr[:]
    for name, s in TOKENS:
        if s:
            tokmap[s] = len(tokstr)
            tokstr.append(s)
        else:
            tokstr.append(None)


def tokenget(s):
    """Returns the token kind for the identifier/keyword spelled s."""
    k = tokmap.get(s)
    if k is None:
        if len(s) > 8192:
            U.fatal("token is too long")
        k = len(tokstr)
        tokmap[s] = k
        tokstr.append(s)
    return k


def tokenstr(kind):
    assert kind < len(tokstr)
    return tokstr[kind]


def tokenprint(t, f):
    """f is a list of str pieces."""
    if t.space:
        f.append(' ')
    k = t.kind
    if k == TNUMBER or k == TCHARCONST or k == TSTRINGLIT:
        s = t.lit
    elif k == TNEWLINE:
        s = "\n"
    elif k == TEOF:
        return
    else:
        s = tokenstr(k)
    if s is None:
        U.fatal("cannot print token %d", k)
    f.append(s)


def _isprint(c):
    return 0x20 <= c < 0x7f


def tokendesc(kind, lit):
    """Returns the description (C: written to a 64-byte buffer, so at most 63 chars)."""
    quote = True
    assert kind < len(tokstr)
    if kind == TEOF:
        cls = "EOF"
    elif kind == TNUMBER:
        cls = "number"
        quote = True
    elif kind == TCHARCONST:
        cls = "character"
        quote = False
    elif kind == TSTRINGLIT:
        cls = "string"
        quote = False
    elif kind == TNEWLINE:
        cls = "newline"
    elif kind == TOTHER:
        cls = None
    else:
        if kind >= TIDENT:
            cls = "identifier"
            quote = True
        else:
            cls = None
        lit = tokstr[kind]
    if cls and lit is not None:
        buf = ("%s '%s'" if quote else "%s %s") % (cls, lit)
    elif cls:
        buf = cls
    elif kind == TOTHER and not _isprint(ord(lit[0]) if lit else 0):
        buf = "<U+%04x>" % (ord(lit[0]) if lit else 0)
    elif lit is not None:
        buf = "'%s'" % lit
    else:
        buf = "<unknown>"
    return buf[:63]


def tokencheck(t, kind, msg):
    if kind == TPPIDENT or kind == TIDENT:
        if t.kind < kind:
            want = "identifier"
            got = tokendesc(t.kind, t.lit)
            error(t.loc, "expected %s %s, saw %s", want, msg, got)
        return tokenstr(t.kind)
    if t.kind != kind and (kind != TIDENT or t.kind < TIDENT):
        want = tokendesc(kind, None)
        got = tokendesc(t.kind, t.lit)
        error(t.loc, "expected %s %s, saw %s", want, msg, got)
    return t.lit


def diagloc(loc):
    return "%s:%d:%d: " % ("(null)" if loc[0] is None else loc[0], loc[1], loc[2])


def error(loc, fmt, *args):
    msg = fmt % tuple("(null)" if a is None else a for a in args)
    raise U.CprocError(diagloc(loc) + "error: " + msg)
