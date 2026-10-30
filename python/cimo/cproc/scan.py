"""Port of scan.c.

The input is a str with one char per byte (bytes decoded as latin-1). A char
is a 1-char str; EOF is ''. getc/ungetc are an index into the text.
"""

from .cc import (Location, TLNOT, TNEQ, THASH, THASHHASH, TMOD, TMODASSIGN, TBAND, TBANDASSIGN,
                 TLAND, TMUL, TMULASSIGN, TADD, TADDASSIGN, TINC, TSUB, TSUBASSIGN, TDEC, TARROW,
                 TDIV, TDIVASSIGN, TLESS, TLEQ, TSHL, TSHLASSIGN, TASSIGN, TEQL, TGREATER, TGEQ,
                 TSHR, TSHRASSIGN, TXOR, TXORASSIGN, TBOR, TBORASSIGN, TLOR, TNEWLINE, TLBRACK,
                 TRBRACK, TLPAREN, TRPAREN, TLBRACE, TRBRACE, TPERIOD, TELLIPSIS, TBNOT,
                 TQUESTION, TCOLON, TCOLONCOLON, TSEMICOLON, TCOMMA, TEOF, TOTHER, TNUMBER,
                 TCHARCONST, TSTRINGLIT)
from . import token as TK
from . import util as U

DIGIT = frozenset('0123456789')
ALPHA = frozenset('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
ALNUM = DIGIT | ALPHA
IDENT = ALNUM | frozenset('_')
XDIGIT = DIGIT | frozenset('abcdefABCDEF')
ODIGIT = frozenset('01234567')
SIMPLEESC = frozenset("'\"?\\abfnrtv\0")  # strchr also matches the terminating NUL


class Scanner(object):
    __slots__ = ('chr', 'usebuf', 'sawspace', 'text', 'pos', 'file', 'line', 'col', 'buf',
                 'next', 'opened')

    def __init__(self):
        self.chr = ''
        self.usebuf = False
        self.sawspace = False
        self.text = None
        self.pos = 0
        self.file = None
        self.line = 1
        self.col = 0
        self.buf = []
        self.next = None


scanner = None


def _reset():
    global scanner
    scanner = None


def nextchar(s):
    if s.usebuf:
        s.buf.append(s.chr if s.chr else '\xff')
    text = s.text
    while True:
        pos = s.pos
        if pos < len(text):
            c = text[pos]
            s.pos = pos + 1
        else:
            c = ''
        s.chr = c
        if c == '\n':
            s.line += 1
            s.col = 0
            break
        s.col += 1
        if c != '\\':
            break
        pos = s.pos
        if pos < len(text):
            c = text[pos]
            s.pos = pos + 1
        else:
            c = ''
        if c != '\n':
            if c:
                s.pos -= 1  # ungetc
            break
        s.line += 1
        s.col = 0


def op2(s, t1, t2):
    nextchar(s)
    if s.chr != '=':
        return t1
    nextchar(s)
    return t2


def op3(s, t1, t2, t3):
    c = s.chr
    nextchar(s)
    if s.chr == '=':
        nextchar(s)
        return t2
    if s.chr != c:
        return t1
    nextchar(s)
    return t3


def op4(s, t1, t2, t3, t4):
    c = s.chr
    nextchar(s)
    if s.chr == '=':
        nextchar(s)
        return t2
    if s.chr != c:
        return t1
    nextchar(s)
    if s.chr != '=':
        return t3
    nextchar(s)
    return t4


def ident(s):
    s.usebuf = True
    while s.chr in IDENT and s.chr:
        nextchar(s)
    tok = TK.tokenget(''.join(s.buf))
    s.usebuf = False
    del s.buf[:]
    return tok


def number(s):
    allowsign = False
    s.usebuf = True
    while True:
        nextchar(s)
        c = s.chr
        if c in ('e', 'E', 'p', 'P'):
            allowsign = True
        elif c in ('+', '-'):
            if not allowsign:
                break
        elif c == '.':
            allowsign = False
        else:
            if c == "'":
                nextchar(s)
                s.buf.pop()  # ignore separator
                c = s.chr
            if not (c and c in IDENT):
                break
            allowsign = False
    return TNUMBER


def escape(s):
    nextchar(s)
    if s.chr == 'x':
        nextchar(s)
        if not (s.chr and s.chr in XDIGIT):
            TK.error(s_loc(s), "invalid hexadecimal escape sequence")
        while True:
            nextchar(s)
            if not (s.chr and s.chr in XDIGIT):
                break
    elif s.chr and s.chr in ODIGIT:
        nextchar(s)
        if s.chr and s.chr in ODIGIT:
            nextchar(s)
            if s.chr and s.chr in ODIGIT:
                nextchar(s)
    elif s.chr and s.chr in SIMPLEESC:
        nextchar(s)
    else:
        TK.error(s_loc(s), "invalid escape sequence")


def s_loc(s):
    return Location(s.file, s.line, s.col)


def charconst(s):
    s.usebuf = True
    nextchar(s)
    while True:
        c = s.chr
        if c == '\\':
            escape(s)
        elif c == "'":
            nextchar(s)
            return TCHARCONST
        elif c == '\n':
            TK.error(s_loc(s), "newline in character constant")
        elif c == '':
            TK.error(s_loc(s), "EOF in character constant")
        else:
            nextchar(s)


def stringlit(s):
    s.usebuf = True
    nextchar(s)
    while True:
        c = s.chr
        if c == '\\':
            escape(s)
        elif c == '"':
            nextchar(s)
            return TSTRINGLIT
        elif c == '\n':
            TK.error(s_loc(s), "newline in string literal")
        elif c == '':
            TK.error(s_loc(s), "EOF in string literal")
        else:
            nextchar(s)


def comment(s):
    c = s.chr
    if c == '/':  # C++-style comment
        while True:
            nextchar(s)
            if s.chr == '\n' or s.chr == '':
                break
    elif c == '*':  # C-style comment
        nextchar(s)
        while True:
            last = s.chr
            nextchar(s)
            if s.chr == '':
                TK.error(s_loc(s), "EOF in comment")
            if last == '*' and s.chr == '/':
                break
        nextchar(s)
    else:
        return False
    s.sawspace = True
    return True


_SIMPLE = {
    '\n': TNEWLINE, '[': TLBRACK, ']': TRBRACK, '(': TLPAREN, ')': TRPAREN, '{': TLBRACE,
    '}': TRBRACE, '~': TBNOT, '?': TQUESTION, ';': TSEMICOLON, ',': TCOMMA,
}


def scankind(s, t):
    while True:
        t.loc = Location(s.file, s.line, s.col)
        c = s.chr
        if c in (' ', '\t', '\f', '\v'):
            s.sawspace = True
            nextchar(s)
            continue
        if c == '/':
            tok = op2(s, TDIV, TDIVASSIGN)
            if tok == TDIV and comment(s):
                continue
            return tok
        break
    k = _SIMPLE.get(c)
    if k is not None:
        nextchar(s)
        return k
    if c == '!':
        return op2(s, TLNOT, TNEQ)
    if c == '"':
        return stringlit(s)
    if c == '#':
        nextchar(s)
        if s.chr != '#':
            return THASH
        nextchar(s)
        return THASHHASH
    if c == '%':
        return op2(s, TMOD, TMODASSIGN)
    if c == '&':
        return op3(s, TBAND, TBANDASSIGN, TLAND)
    if c == "'":
        return charconst(s)
    if c == '*':
        return op2(s, TMUL, TMULASSIGN)
    if c == '+':
        return op3(s, TADD, TADDASSIGN, TINC)
    if c == '-':
        tok = op3(s, TSUB, TSUBASSIGN, TDEC)
        if tok != TSUB or s.chr != '>':
            return tok
        nextchar(s)
        return TARROW
    if c == '<':
        return op4(s, TLESS, TLEQ, TSHL, TSHLASSIGN)
    if c == '=':
        return op2(s, TASSIGN, TEQL)
    if c == '>':
        return op4(s, TGREATER, TGEQ, TSHR, TSHRASSIGN)
    if c == '^':
        return op2(s, TXOR, TXORASSIGN)
    if c == '|':
        return op3(s, TBOR, TBORASSIGN, TLOR)
    if c == '.':
        nextchar(s)
        if s.chr and s.chr in DIGIT:
            s.buf.append('.')
            return number(s)
        if s.chr != '.':
            return TPERIOD
        oldline, oldcol = s.line, s.col
        nextchar(s)
        if s.chr != '.':
            if s.chr:
                s.pos -= 1  # ungetc
            s.line, s.col = oldline, oldcol
            s.chr = '.'
            return TPERIOD
        nextchar(s)
        return TELLIPSIS
    if c == ':':
        nextchar(s)
        if s.chr != ':':
            return TCOLON
        nextchar(s)
        return TCOLONCOLON
    if c in ('L', 'U', 'u'):
        s.usebuf = True
        nextchar(s)
        if s.buf[0] == 'u' and s.chr == '8':
            nextchar(s)
        if s.chr == "'":
            return charconst(s)
        if s.chr == '"':
            return stringlit(s)
        return ident(s)
    if c == '':
        return TEOF
    if c in DIGIT:
        return number(s)
    if c in ALPHA or c == '_':
        return ident(s)
    s.usebuf = True
    nextchar(s)
    return TOTHER


def scanfrom(name, text):
    """Push a scanner for `name`. text is the input (a latin-1 str), or None to read
    the file named `name` in scanopen()."""
    global scanner
    s = Scanner()
    s.text = text
    s.file = name
    s.line = 1
    s.col = 0
    s.next = scanner
    if text is not None:
        nextchar(s)
    scanner = s


def scanopen():
    s = scanner
    if s.text is None:
        try:
            with open(s.file, 'rb') as fp:
                s.text = fp.read().decode('latin-1')
        except OSError as e:
            U.fatal("open %s: %s", s.file, e.strerror)
        nextchar(s)


def scansetloc(loc):
    scanner.file, scanner.line, scanner.col = loc


def scanclose():
    pass


def scan(t):
    global scanner
    scanner.sawspace = False
    while True:
        t.kind = scankind(scanner, t)
        if t.kind != TEOF or not scanner.next:
            break
        scanclose()
        scanner = scanner.next
        scanopen()
    s = scanner
    if s.usebuf:
        t.lit = ''.join(s.buf)
        del s.buf[:]
        s.usebuf = False
    else:
        t.lit = None
    t.space = s.sawspace
    t.hide = False
