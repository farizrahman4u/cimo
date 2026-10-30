"""Port of attr.c."""

from .cc import (TIDENT, TCOLONCOLON, TLPAREN, TRPAREN, TLBRACK, TRBRACK, TCOMMA, TSTRINGLIT,
                 T__ATTRIBUTE__, ATTRNORETURN, ATTRALIGNED, ATTRCONSTRUCTOR, ATTRDESTRUCTOR,
                 ATTRPACKED, ATTRSECTION)
from .token import tok, tokenstr, error
from .pp import next, consume, expect, peek
from . import expr as E
from . import scope as SC

PREFIXNONE = 1  # standard attribute
PREFIXGNU = 2
INT_MAX = 0x7fffffff


def strip(name):
    ln = len(name)
    if ln >= 4 and name[0] == '_' and name[1] == '_' and name[ln - 2] == '_' and name[ln - 1] == '_':
        if ln - 4 >= 32:
            return ""
        return name[2:ln - 2]
    return name


def parseattr(a, allowed, prefix):
    prefixname = ""
    if tok.kind < TIDENT:
        return False
    name = strip(tokenstr(tok.kind))
    next()
    if not prefix:
        if consume(TCOLONCOLON):
            if name == "gnu":
                prefix = PREFIXGNU
            name = strip(expect(TIDENT, "after attribute prefix"))
        else:
            prefix = PREFIXNONE
    kind = 0
    if prefix == PREFIXNONE:
        if name == "noreturn" or name == "_Noreturn":
            kind = ATTRNORETURN
    elif prefix == PREFIXGNU:
        prefixname = "GNU "
        if name == "aligned":
            kind = ATTRALIGNED
            if consume(TLPAREN):
                i = E.intconstexpr(SC.filescope, False)
                if not i or i & (i - 1) or i > INT_MAX:
                    error(tok.loc, "invalid alignment %d", i)
                if a:
                    a.align = i
                expect(TRPAREN, "after alignment")
            else:
                if a:
                    a.align = 16
        elif name == "constructor":
            kind = ATTRCONSTRUCTOR
        elif name == "destructor":
            kind = ATTRDESTRUCTOR
        elif name == "packed":
            kind = ATTRPACKED
        elif name == "section":
            kind = ATTRSECTION
            expect(TLPAREN, "after 'section'")
            section = expect(TSTRINGLIT, "for section name")
            if a:
                a.section = section
            expect(TRPAREN, "after section name")
    if kind:
        if not (kind & allowed):
            error(tok.loc, "%sattribute '%s' is not supported here", prefixname, name)
        if a:
            a.kind |= kind
    elif consume(TLPAREN):
        # skip arguments
        paren = 1
        while paren > 0:
            if tok.kind == TLPAREN:
                paren += 1
            elif tok.kind == TRPAREN:
                paren -= 1
            next()
    return True


def attrspec(a, allowed):
    if tok.kind != TLBRACK or not peek(TLBRACK):
        return False
    while parseattr(a, allowed, 0) or consume(TCOMMA):
        pass
    expect(TRBRACK, "to end attribute specifier")
    expect(TRBRACK, "to end attribute specifier")
    return True


def attr(a, allowed):
    if not attrspec(a, allowed):
        return False
    while attrspec(a, allowed):
        pass
    return True


def gnuattrspec(a, allowed):
    if not consume(T__ATTRIBUTE__):
        return False
    while parseattr(a, allowed, PREFIXGNU) or consume(TCOMMA):
        pass
    expect(TLPAREN, "after '__attribute__' to begin attribute specifier")
    expect(TLPAREN, "after '__attribute__' to begin attribute specifier")
    while parseattr(a, allowed, PREFIXGNU) or consume(TCOMMA):
        pass
    expect(TRPAREN, "to end attribute specifier")
    expect(TRPAREN, "to end attribute specifier")
    return True


def gnuattr(a, allowed):
    if not gnuattrspec(a, allowed):
        return False
    while gnuattrspec(a, allowed):
        pass
    return True
