"""A C11 preprocessor in pure Python.

cproc does not preprocess: its driver runs the system `cpp` and cproc-qbe reads the result. This
module replaces that step. It produces text with gcc-style line markers (`# <line> "<file>"`) that
cproc-qbe accepts, so cproc's diagnostics point at the original source.

    preprocess(src, filename='<string>', include_dirs=(), defines=None, target='arm64') -> str

Macro expansion follows gcc's model (a stack of expansion contexts; a macro is disabled while its
expansion is being read, and identifiers that name a disabled macro are painted so they never
expand again), which implements the hide-set rules of C11 6.10.3.4 and reproduces the examples in
6.10.3.5. The predefined macros are the ones cproc's config.h passes to `cpp -undef`, plus
__STDC__, __STDC_VERSION__ (201112L) and __STDC_HOSTED__ (0, freestanding).

Deviations from gcc's text output, none of which change the token sequence: whitespace and line
layout differ, digraphs are printed in their canonical spelling (cproc's scanner has no digraphs),
and `#pragma` lines are dropped except `#pragma omp ...`, which is kept (macro-expanded, as
OpenMP specifies) on a line of its own.
"""

import os
import re
import sys
import time

__all__ = ['preprocess', 'Preprocessor', 'PreprocessError', 'TARGETS']

# Token kinds.
IDENT, NUMBER, CHAR, STRING, PUNCT, OTHER, EOF, EOF_INC, EOF_ARG, PLACEMARKER, PASTE = range(11)

_DIGRAPHS = {'<:': '[', ':>': ']', '<%': '{', '%>': '}', '%:': '#', '%:%:': '##'}

_UCN = r'\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8}'
_TOKEN_RE = re.compile(r'''
 (?P<ws>[ \t\f\v\r]+)
|(?P<nl>\n)
|(?P<bc>/\*)
|(?P<lc>//[^\n]*)
|(?P<num>\.?[0-9](?:[eEpP][+-]|[0-9A-Za-z_.$\x80-\U0010ffff]|%(ucn)s)*)
|(?P<str>(?:u8|[uUL])?"(?:[^"\\\n]|\\.)*")
|(?P<chr>[uUL]?'(?:[^'\\\n]|\\.)*')
|(?P<id>(?:[A-Za-z_$\x80-\U0010ffff]|%(ucn)s)(?:[A-Za-z0-9_$\x80-\U0010ffff]|%(ucn)s)*)
|(?P<punct>%%:%%:|\.\.\.|<<=|>>=|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||[*/%%+\-&^|]=|\#\#
    |<:|:>|<%%|%%>|%%:|[\[\](){}.&*+\-~!/%%<>^|?:;=,\#])
|(?P<bad>['"][^\n]*)
|(?P<other>.)
''' % {'ucn': _UCN}, re.X | re.S)

_KIND = {'num': NUMBER, 'str': STRING, 'chr': CHAR, 'id': IDENT, 'punct': PUNCT, 'bad': OTHER,
         'other': OTHER}
_SPLICE_RE = re.compile(r'\\[ \t]*\n')


class PreprocessError(Exception):
    """A preprocessing error; str() is a gcc-like `file:line: error: message`."""

    def __init__(self, msg, filename=None, line=None):
        self.msg = msg
        self.filename = filename
        self.line = line
        if filename is not None:
            text = '%s:%s: error: %s' % (filename, line, msg)
        else:
            text = 'error: %s' % msg
        Exception.__init__(self, text)


class FileInfo(object):
    """A (presumed) source file name, shared by the tokens lexed from it."""
    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name


class Tok(object):
    __slots__ = ('kind', 'text', 'line', 'space', 'bol', 'src', 'noexp', 'col')

    def __init__(self, kind, text, line, space, bol, src, noexp=False, col=0):
        self.kind = kind
        self.text = text
        self.line = line
        self.space = space      # preceded by whitespace (or a comment)
        self.bol = bol          # first token of a logical line
        self.src = src          # FileInfo of the (presumed) file
        self.noexp = noexp      # painted: never macro-expanded again
        self.col = col          # 1-based physical column (0: unknown)

    def copy(self):
        return Tok(self.kind, self.text, self.line, self.space, self.bol, self.src, self.noexp,
                   self.col)

    @property
    def value(self):
        """The spelling with digraphs replaced by their canonical punctuator."""
        if self.kind == PUNCT:
            return _DIGRAPHS.get(self.text, self.text)
        return self.text

    def __repr__(self):
        return 'Tok(%r)' % self.text


_NOSRC = FileInfo('<built-in>')


def lex(text, src):
    """Phases 1-3: splice lines, drop comments, split into pp-tokens (with physical lines)."""
    if '\r' in text:
        text = text.replace('\r\n', '\n')
    splices = []
    if '\\' in text:
        parts = []
        pos = 0
        outlen = 0
        for m in _SPLICE_RE.finditer(text):
            parts.append(text[pos:m.start()])
            outlen += m.start() - pos
            splices.append(outlen)
            pos = m.end()
        if splices:
            parts.append(text[pos:])
            text = ''.join(parts)
    toks = []
    append = toks.append
    match = _TOKEN_RE.match
    line = 1
    si = 0
    nspl = len(splices)
    pos = 0
    n = len(text)
    bol = True
    space = False
    linestart = 0
    while pos < n:
        while si < nspl and splices[si] <= pos:
            line += 1
            linestart = splices[si]
            si += 1
        m = match(text, pos)
        kind = m.lastgroup
        end = m.end()
        if kind == 'ws':
            space = True
        elif kind == 'nl':
            line += 1
            linestart = end
            bol = True
            space = False
        elif kind == 'bc':
            e = text.find('*/', pos + 2)
            if e < 0:
                raise PreprocessError('unterminated comment', src.name, line)
            end = e + 2
            nl = text.count('\n', pos, end)
            if nl:
                # Splices inside the comment are counted by the loop above.
                line += nl
                linestart = text.rfind('\n', pos, end) + 1
            space = True
        elif kind == 'lc':
            space = True
        else:
            append(Tok(_KIND[kind], m.group(), line, space, bol, src, False, pos - linestart + 1))
            bol = False
            space = False
        pos = end
    return toks


def _lex_one(text):
    """Lex `text` as exactly one pp-token, or return None."""
    m = _TOKEN_RE.match(text)
    if m is None or m.end() != len(text) or m.lastgroup not in _KIND or m.lastgroup == 'bad':
        return None
    return _KIND[m.lastgroup]


def _needs_space(prev, cur):
    """Would printing `prev` and `cur` adjacently lex differently? (avoid accidental pastes)"""
    a = prev.text
    m = _TOKEN_RE.match(a + cur.text)
    return m.end() != len(a)


# --- Predefined macros -------------------------------------------------------------------------

_COMMON_DEFINES = [
    ('__STDC_NO_ATOMICS__', None),
    ('__STDC_NO_COMPLEX__', None),
    ('__extension__', ''),
    ('__STDC_ENDIAN_LITTLE__', '1234'),
    ('__STDC_ENDIAN_BIG__', '4321'),
    ('__STDC_ENDIAN_NATIVE__', '__STDC_ENDIAN_LITTLE__'),
    ('__ORDER_LITTLE_ENDIAN__', '__STDC_ENDIAN_LITTLE__'),
    ('__ORDER_BIG_ENDIAN__', '__STDC_ENDIAN_BIG__'),
    ('__BYTE_ORDER__', '__STDC_ENDIAN_NATIVE__'),
    ('__FLT_EVAL_METHOD__', '1'),
    ('__BITINT_MAXWIDTH__', '64'),
    ('__aarch64__', None),
    ('__LP64__', None),
    ('__SIZEOF_LONG__', '8'),
    ('__SIZEOF_POINTER__', '8'),
    ('__SIZEOF_SIZE_T__', '8'),
    ('__SIZEOF_PTRDIFF_T__', '8'),
]
_TAIL_DEFINES = [
    ('__INT_MAX__', '0x7fffffff'),
    ('__LONG_LONG_MAX__', '0x7fffffffffffffffL'),
]

# gcc's type, size and limit macros (not in cproc's list; system headers such as glibc's
# glob.h use them), with gcc's values for aarch64-linux-gnu; _APPLE_TYPES overrides.
_GCC_TYPES = [
    ('__CHAR_BIT__', '8'), ('__SIZEOF_SHORT__', '2'), ('__SIZEOF_INT__', '4'),
    ('__SIZEOF_LONG_LONG__', '8'), ('__SIZEOF_FLOAT__', '4'), ('__SIZEOF_DOUBLE__', '8'),
    ('__SIZEOF_WCHAR_T__', '4'), ('__SIZEOF_WINT_T__', '4'),
    ('__SCHAR_MAX__', '0x7f'), ('__SHRT_MAX__', '0x7fff'),
    ('__LONG_MAX__', '0x7fffffffffffffffL'),
    ('__SCHAR_WIDTH__', '8'), ('__SHRT_WIDTH__', '16'), ('__INT_WIDTH__', '32'),
    ('__LONG_WIDTH__', '64'), ('__LONG_LONG_WIDTH__', '64'),
    ('__SIZE_TYPE__', 'long unsigned int'), ('__SIZE_MAX__', '0xffffffffffffffffUL'),
    ('__SIZE_WIDTH__', '64'),
    ('__PTRDIFF_TYPE__', 'long int'), ('__PTRDIFF_MAX__', '0x7fffffffffffffffL'),
    ('__PTRDIFF_WIDTH__', '64'),
    ('__INTMAX_TYPE__', 'long int'), ('__UINTMAX_TYPE__', 'long unsigned int'),
    ('__INTMAX_MAX__', '0x7fffffffffffffffL'), ('__UINTMAX_MAX__', '0xffffffffffffffffUL'),
    ('__INTMAX_C(c)', 'c ## L'), ('__UINTMAX_C(c)', 'c ## UL'), ('__INTMAX_WIDTH__', '64'),
    ('__INTPTR_TYPE__', 'long int'), ('__UINTPTR_TYPE__', 'long unsigned int'),
    ('__INTPTR_MAX__', '0x7fffffffffffffffL'), ('__UINTPTR_MAX__', '0xffffffffffffffffUL'),
    ('__INTPTR_WIDTH__', '64'),
    ('__CHAR16_TYPE__', 'short unsigned int'), ('__CHAR32_TYPE__', 'unsigned int'),
    ('__INT8_TYPE__', 'signed char'), ('__INT16_TYPE__', 'short int'),
    ('__INT32_TYPE__', 'int'), ('__INT64_TYPE__', 'long int'),
    ('__UINT8_TYPE__', 'unsigned char'), ('__UINT16_TYPE__', 'short unsigned int'),
    ('__UINT32_TYPE__', 'unsigned int'), ('__UINT64_TYPE__', 'long unsigned int'),
    ('__INT8_MAX__', '0x7f'), ('__INT16_MAX__', '0x7fff'), ('__INT32_MAX__', '0x7fffffff'),
    ('__INT64_MAX__', '0x7fffffffffffffffL'), ('__UINT8_MAX__', '0xff'),
    ('__UINT16_MAX__', '0xffff'), ('__UINT32_MAX__', '0xffffffffU'),
    ('__UINT64_MAX__', '0xffffffffffffffffUL'),
    ('__INT8_C(c)', 'c'), ('__INT16_C(c)', 'c'), ('__INT32_C(c)', 'c'), ('__INT64_C(c)', 'c ## L'),
    ('__UINT8_C(c)', 'c'), ('__UINT16_C(c)', 'c'), ('__UINT32_C(c)', 'c ## U'),
    ('__UINT64_C(c)', 'c ## UL'),
    ('__WCHAR_TYPE__', 'unsigned int'), ('__WCHAR_MAX__', '0xffffffffU'), ('__WCHAR_MIN__', '0U'),
    ('__WCHAR_WIDTH__', '32'),
    ('__WINT_TYPE__', 'unsigned int'), ('__WINT_MAX__', '0xffffffffU'), ('__WINT_MIN__', '0U'),
    ('__WINT_WIDTH__', '32'),
]
_LINUX_TYPES = _GCC_TYPES + [('__CHAR_UNSIGNED__', None)]
# Apple arm64: signed char and wchar_t, int64_t is long long (clang's values)
_APPLE_OVERRIDE = {
    '__INT64_TYPE__': 'long long int', '__UINT64_TYPE__': 'long long unsigned int',
    '__INT64_C(c)': 'c ## LL', '__UINT64_C(c)': 'c ## ULL',
    '__INT64_MAX__': '0x7fffffffffffffffLL', '__UINT64_MAX__': '0xffffffffffffffffULL',
    '__WCHAR_TYPE__': 'int', '__WCHAR_MAX__': '2147483647', '__WCHAR_MIN__': '(-2147483647-1)',
    '__WINT_TYPE__': 'int', '__WINT_MAX__': '2147483647', '__WINT_MIN__': '(-2147483647-1)',
}
_APPLE_TYPES = [(k, _APPLE_OVERRIDE.get(k, v)) for k, v in _GCC_TYPES]

#: Predefined macros per target: cproc's config.h list (aarch64-linux-gnu) and a macOS variant,
#: plus gcc's type macros.
TARGETS = {
    'arm64': (_COMMON_DEFINES + [('__linux__', None), ('__unix__', None)] + _TAIL_DEFINES
              + _LINUX_TYPES),
    'arm64_apple': (_COMMON_DEFINES + [('__APPLE__', None), ('__MACH__', None)] + _TAIL_DEFINES
                    + _APPLE_TYPES),
}
# Whether plain char is signed, for character constants in #if.
_CHAR_SIGNED = {'arm64': False, 'arm64_apple': True}
_WCHAR_SIGNED = {'arm64': False, 'arm64_apple': True}

BUNDLED_INCLUDE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include')

# Built-in functions cproc implements, for __has_builtin.
_CPROC_BUILTINS = frozenset('''__builtin_alloca __builtin_constant_p __builtin_expect
__builtin_inff __builtin_nanf __builtin_offsetof __builtin_types_compatible_p
__builtin_unreachable __builtin_va_arg __builtin_va_copy __builtin_va_end __builtin_va_start
__builtin_c23_va_start __builtin_va_list'''.split())

_BUILTIN_MACROS = ('__FILE__', '__LINE__', '__DATE__', '__TIME__', '__COUNTER__',
                   '__INCLUDE_LEVEL__', '__BASE_FILE__', '__has_include', '__has_include_next',
                   '__has_builtin', '__has_attribute', '__has_c_attribute', '__has_feature',
                   '__has_extension')

_MASK = (1 << 64) - 1
_SIGN = 1 << 63


# --- Macros and contexts -----------------------------------------------------------------------

_OP_TOK, _OP_PARAM, _OP_STR, _OP_PASTE = range(4)


class Macro(object):
    __slots__ = ('name', 'params', 'variadic', 'body', 'items', 'builtin', 'haspaste', 'src',
                 'line')

    def __init__(self, name, params=None, variadic=False, body=(), builtin=False):
        self.name = name
        self.params = params        # None for object-like macros
        self.variadic = variadic
        self.body = list(body)
        self.builtin = builtin
        self.items = []
        self.haspaste = False

    def same_as(self, other):
        if self.params != other.params or self.variadic != other.variadic:
            return False
        if len(self.body) != len(other.body):
            return False
        for i, (a, b) in enumerate(zip(self.body, other.body)):
            if a.text != b.text or (i and (a.space or a.bol) != (b.space or b.bol)):
                return False
        return True


class _Context(object):
    __slots__ = ('toks', 'i', 'macro', 'sentinel')

    def __init__(self, toks, macro=None, sentinel=False):
        self.toks = toks
        self.i = 0
        self.macro = macro
        self.sentinel = sentinel


class _Source(object):
    """A file being read: its tokens, read position and conditional stack."""
    __slots__ = ('toks', 'i', 'finfo', 'orig', 'path', 'dir', 'dirindex', 'conds', 'lineoff', 'ended')

    def __init__(self, toks, finfo, path, dirindex):
        self.toks = toks
        self.i = 0
        self.finfo = finfo          # presumed file (changed by #line)
        self.orig = finfo
        self.path = path
        self.dir = os.path.dirname(path) if path else ''
        self.dirindex = dirindex    # index in the search path the file was found at (#include_next)
        self.conds = []             # [taking, taken, seen_else, line, directive]
        self.lineoff = 0            # presumed line - physical line (#line)
        self.ended = False


def _c_string(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


class Preprocessor(object):
    def __init__(self, include_dirs=(), defines=None, target='arm64', warn=None,
                 keep_attributes=False):
        self.keep_attributes = keep_attributes
        if target not in TARGETS:
            raise ValueError('unknown target %r (expected one of %s)' %
                             (target, ', '.join(sorted(TARGETS))))
        self.target = target
        self.include_dirs = [os.path.abspath(d) for d in include_dirs]
        self.search = self.include_dirs + [BUNDLED_INCLUDE]
        self.warn_fn = warn
        self.macros = {}
        self.disabled = set()
        self.ctx = []
        self.files = []
        self.once = set()
        self.cache = {}
        self.counter = 0
        self.pushed_macros = {}
        self.base_file = '<string>'
        self.out = None
        self.pending = []           # line-marker events: (finfo, line, flag)
        self.in_directive = False
        self.peeking = False
        self.warnings = []
        t = time.localtime(int(os.environ['SOURCE_DATE_EPOCH'])) \
            if os.environ.get('SOURCE_DATE_EPOCH', '').isdigit() else time.localtime()
        months = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()
        self.date = '"%s %2d %d"' % (months[t.tm_mon - 1], t.tm_mday, t.tm_year)
        self.time = '"%02d:%02d:%02d"' % (t.tm_hour, t.tm_min, t.tm_sec)
        for name in _BUILTIN_MACROS:
            self.macros[name] = Macro(name, builtin=True)
        pre = [('__STDC__', '1'), ('__STDC_VERSION__', '201112L'), ('__STDC_HOSTED__', '0')]
        pre += TARGETS[target]
        self._define_list(pre, '<built-in>')
        if defines:
            items = defines.items() if hasattr(defines, 'items') else defines
            self._define_list(items, '<command-line>')

    def _define_list(self, items, name):
        lines = []
        for k, v in items:
            if v is None:
                v = '1'
            lines.append('#define %s %s\n' % (k, v))
        self._run_directives(''.join(lines), name)

    def _run_directives(self, text, name):
        finfo = FileInfo(name)
        src = _Source(lex(text, finfo), finfo, None, -1)
        self.files.append(src)
        try:
            while True:
                t = self._lex_token()
                if t.kind == EOF:
                    break
                raise self._error('stray token %r in %s' % (t.text, name), t)
        finally:
            self.files.pop()

    # --- diagnostics ---

    def _loc(self, tok=None):
        if tok is not None and tok.src is not None and tok.src is not _NOSRC:
            return tok.src.name, tok.line
        if self.files:
            f = self.files[-1]
            line = f.toks[min(f.i, len(f.toks)) - 1].line if f.toks and f.i else 1
            return f.finfo.name, line + f.lineoff
        return None, None

    def _error(self, msg, tok=None):
        name, line = self._loc(tok)
        return PreprocessError(msg, name, line)

    def _warn(self, msg, tok=None):
        name, line = self._loc(tok)
        text = '%s:%s: warning: %s' % (name, line, msg) if name else 'warning: %s' % msg
        self.warnings.append(text)
        if self.warn_fn is not None:
            self.warn_fn(text)

    # --- file reading and directives ---

    def _lex_file(self, path):
        hit = self.cache.get(path)
        if hit is None:
            with open(path, 'rb') as f:
                data = f.read()
            text = data.decode('utf-8', 'surrogateescape')
            finfo = FileInfo(path)
            hit = self.cache[path] = (lex(text, finfo), finfo)
        return hit

    def _push_file(self, toks, finfo, path, dirindex):
        if len(self.files) > 200:
            raise self._error('#include nested depth 200 exceeds maximum of 200')
        src = _Source(toks, finfo, path, dirindex)
        self.files.append(src)
        return src

    def _lex_token(self):
        """Next token from the current file, after directives and conditional skipping."""
        while True:
            src = self.files[-1]
            toks = src.toks
            i = src.i
            if i >= len(toks):
                if src.ended:
                    # The include's EOF_INC was returned; now resume the includer.
                    self.files.pop()
                    parent = self.files[-1]
                    pline = (parent.toks[parent.i - 1].line + 1 + parent.lineoff) \
                        if parent.i else 1
                    self.pending.append((parent.finfo, pline, ' 2'))
                    continue
                if src.conds:
                    c = src.conds[-1]
                    raise PreprocessError('unterminated #%s' % c[4], src.finfo.name,
                                          c[3])
                if len(self.files) == 1:
                    return Tok(EOF, '', toks[-1].line if toks else 1, True, True, src.finfo)
                # End of an included file: a function-like macro name or an argument list
                # can't continue past it.
                src.ended = True
                return Tok(EOF_INC, '', 0, True, True, None)
            t = toks[i]
            src.i = i + 1
            if t.bol and t.kind == PUNCT and (t.text == '#' or t.text == '%:'):
                if self.peeking:
                    # Like gcc: a directive between a function-like macro's name and '('
                    # means it is not invoked. Leave the directive for later.
                    src.i = i
                    return Tok(EOF_ARG, '', 0, False, False, None)
                self._directive(src, t)
                continue
            if src.conds and not src.conds[-1][0]:
                # Skipping: jump to the next directive.
                j = i + 1
                n = len(toks)
                while j < n:
                    u = toks[j]
                    if u.bol and u.kind == PUNCT and (u.text == '#' or u.text == '%:'):
                        break
                    j += 1
                src.i = j
                continue
            if src.lineoff or src.finfo is not src.orig:
                t = t.copy()
                t.line += src.lineoff
                t.src = src.finfo
            return t

    def _line_tokens(self, src):
        toks = src.toks
        j = src.i
        n = len(toks)
        while j < n and not toks[j].bol:
            j += 1
        line = toks[src.i:j]
        src.i = j
        if src.lineoff or src.finfo is not src.orig:
            fixed = []
            for t in line:
                t = t.copy()
                t.line += src.lineoff
                t.src = src.finfo
                fixed.append(t)
            line = fixed
        return line

    def _directive(self, src, hashtok):
        line = self._line_tokens(src)
        skipping = bool(src.conds) and not src.conds[-1][0]
        if not line:
            return
        d = line[0]
        name = d.text
        if src.lineoff or src.finfo is not src.orig:
            hashtok = hashtok.copy()
            hashtok.line += src.lineoff
            hashtok.src = src.finfo
        if d.kind == IDENT and name in ('if', 'ifdef', 'ifndef', 'elif', 'else', 'endif'):
            self._conditional(src, name, line, hashtok, skipping)
            return
        if skipping:
            return
        args = line[1:]
        if d.kind == NUMBER:
            self._do_line(src, line, gnu_marker=True)
            return
        if d.kind != IDENT:
            raise self._error('invalid preprocessing directive', d)
        if name == 'define':
            self._do_define(args, d)
        elif name == 'undef':
            if not args or args[0].kind != IDENT:
                raise self._error('macro names must be identifiers' if args else
                                  'no macro name given in #undef directive', d)
            if len(args) > 1:
                self._warn('extra tokens at end of #undef directive', args[1])
            if args[0].text == 'defined':
                raise self._error('"defined" cannot be used as a macro name', args[0])
            self.macros.pop(args[0].text, None)
        elif name in ('include', 'include_next', 'import'):
            self._do_include(src, name, args, d)
        elif name == 'line':
            self._do_line(src, line)
        elif name == 'error':
            raise self._error('#error' + self._join_text(args), d)
        elif name == 'warning':
            self._warn('#warning' + self._join_text(args), d)
        elif name == 'pragma':
            self._do_pragma(args, d)
        elif name in ('ident', 'sccs', 'assert', 'unassert'):
            pass
        else:
            raise self._error('invalid preprocessing directive #%s' % name, d)

    def _join_text(self, toks):
        s = []
        for t in toks:
            s.append((' ' if (t.space or t.bol or not s) else '') + t.text)
        return ''.join(s)

    def _conditional(self, src, name, line, hashtok, skipping):
        conds = src.conds
        args = line[1:]
        d = line[0]
        if name in ('if', 'ifdef', 'ifndef'):
            if skipping:
                conds.append([False, True, False, hashtok.line, name])
                return
            if name == 'if':
                val = self._eval_if(args, d)
            else:
                if not args:
                    raise self._error('no macro name given in #%s directive' % name, d)
                if args[0].kind != IDENT:
                    raise self._error('macro names must be identifiers', args[0])
                if len(args) > 1:
                    self._warn('extra tokens at end of #%s directive' % name, args[1])
                val = (args[0].text in self.macros) == (name == 'ifdef')
            conds.append([val, val, False, hashtok.line, name])
            return
        if not conds:
            raise self._error('#%s without #if' % name, d)
        c = conds[-1]
        if name == 'endif':
            if args:
                self._warn('extra tokens at end of #endif directive', args[0])
            conds.pop()
            return
        if c[2]:
            raise self._error('#%s after #else' % name, d)
        parent_skipping = len(conds) > 1 and not conds[-2][0]
        if name == 'else':
            if args:
                self._warn('extra tokens at end of #else directive', args[0])
            c[2] = True
            c[0] = not c[1] and not parent_skipping
            c[1] = True
            return
        # elif
        if c[1] or parent_skipping:
            c[0] = False
            return
        c[0] = False   # not taking while evaluating (affects nothing, for clarity)
        val = self._eval_if(args, d)
        c[0] = val
        c[1] = val

    # --- #define ---

    def _do_define(self, args, d):
        if not args:
            raise self._error('no macro name given in #define directive', d)
        nt = args[0]
        if nt.kind != IDENT:
            raise self._error('macro names must be identifiers', nt)
        name = nt.text
        if name == 'defined':
            raise self._error('"defined" cannot be used as a macro name', nt)
        if name == '__attribute__' and self.keep_attributes:
            # ignored: glibc's sys/cdefs.h defines it away for compilers without __GNUC__,
            # which would silently drop attributes that cimo must see (vector_size, mode, ...)
            # or reject; cimo's frontend parses all attributes
            return
        params = None
        variadic = False
        i = 1
        if len(args) > 1 and args[1].text == '(' and not args[1].space:
            params = []
            i = 2
            while True:
                if i >= len(args):
                    raise self._error('missing \')\' in macro parameter list', nt)
                t = args[i]
                if t.text == ')' and not params:
                    i += 1
                    break
                if t.text == '...':
                    variadic = True
                    params.append('__VA_ARGS__')
                    i += 1
                elif t.kind == IDENT:
                    if t.text in params:
                        raise self._error('duplicate macro parameter "%s"' % t.text, t)
                    if t.text == '__VA_ARGS__':
                        raise self._error('__VA_ARGS__ can not be used as a parameter name', t)
                    params.append(t.text)
                    i += 1
                    if i < len(args) and args[i].text == '...':
                        variadic = True   # GNU named variadic parameter
                        i += 1
                else:
                    raise self._error('expected parameter name, found "%s"' % t.text, t)
                if i >= len(args):
                    raise self._error('missing \')\' in macro parameter list', nt)
                t = args[i]
                if t.text == ')':
                    i += 1
                    break
                if variadic:
                    raise self._error('missing \')\' after "..."', t)
                if t.text != ',':
                    raise self._error('expected comma in macro parameter list', t)
                i += 1
        elif len(args) > 1 and not args[1].space:
            self._warn('missing whitespace after the macro name', args[1])
        body = []
        for t in args[i:]:
            t = t.copy()
            t.bol = False
            body.append(t)
        if body:
            body[0].space = False
        m = Macro(name, params, variadic, body)
        m.src = nt.src
        m.line = nt.line
        self._compile_macro(m)
        old = self.macros.get(name)
        if old is not None:
            if old.builtin:
                self._warn('redefining builtin macro "%s"' % name, nt)
            elif not old.same_as(m):
                self._warn('"%s" redefined' % name, nt)
        self.macros[name] = m

    def _compile_macro(self, m):
        items = []
        body = m.body
        params = m.params
        n = len(body)
        i = 0
        while i < n:
            t = body[i]
            v = t.value
            if t.kind == PUNCT and v == '##':
                if not items or i == n - 1:
                    raise self._error("'##' cannot appear at either end of a macro expansion", t)
                items.append((_OP_PASTE, None, t))
                m.haspaste = True
                i += 1
                continue
            if params is not None and t.kind == PUNCT and v == '#':
                if i + 1 < n and body[i + 1].kind == IDENT and body[i + 1].text in params:
                    items.append((_OP_STR, params.index(body[i + 1].text), t))
                    i += 2
                    continue
                raise self._error("'#' is not followed by a macro parameter", t)
            if params is not None and t.kind == IDENT and t.text in params:
                items.append((_OP_PARAM, params.index(t.text), t))
            else:
                items.append((_OP_TOK, None, t))
            i += 1
        m.items = items

    # --- #include ---

    def _header_name(self, toks, d):
        """Return (name, is_angled) from the tokens of an #include line."""
        if not toks:
            raise self._error('#include expects "FILENAME" or <FILENAME>', d)
        t = toks[0]
        if t.kind == STRING and t.text.startswith('"'):
            if len(toks) > 1:
                self._warn('extra tokens at end of #include directive', toks[1])
            return t.text[1:-1], False
        if t.kind == PUNCT and t.text == '<':
            parts = []
            for j in range(1, len(toks)):
                u = toks[j]
                if u.kind == PUNCT and u.text == '>':
                    if j + 1 < len(toks):
                        self._warn('extra tokens at end of #include directive', toks[j + 1])
                    return ''.join(parts), True
                parts.append((' ' if (u.space and parts) else '') + u.text)
            raise self._error("missing terminating > character", t)
        return None

    def _do_include(self, src, kind, args, d):
        hn = self._header_name(args, d) if args and (
            (args[0].kind == STRING and args[0].text.startswith('"')) or args[0].text == '<') \
            else None
        if hn is None:
            hn = self._header_name(self._expand_line(args, d), d)
            if hn is None:
                raise self._error('#include expects "FILENAME" or <FILENAME>', d)
        name, angled = hn
        if not name:
            raise self._error('empty filename in #%s' % kind, d)
        found = self._find_include(src, name, angled, kind == 'include_next')
        if found is None:
            raise self._error('%s: No such file or directory' % name, d)
        path, dirindex = found
        real = os.path.realpath(path)
        if real in self.once:
            return
        if kind == 'import':
            self.once.add(real)
        toks, finfo = self._lex_file(path)
        self._push_file(toks, finfo, path, dirindex)
        self.pending.append((finfo, 1, ' 1'))

    def _find_include(self, src, name, angled, include_next=False):
        if os.path.isabs(name):
            return (name, -1) if os.path.isfile(name) else None
        start = 0
        if include_next and src.dirindex >= 0:
            start = src.dirindex + 1
        elif not angled and not include_next:
            base = src.dir
            p = os.path.join(base, name) if base else name
            if os.path.isfile(p):
                return p, -1
        for k in range(start, len(self.search)):
            p = os.path.join(self.search[k], name)
            if os.path.isfile(p):
                return p, k
        return None

    def _has_include(self, name, angled, include_next=False):
        src = self.files[-1]
        return self._find_include(src, name, angled, include_next) is not None

    # --- #line ---

    def _do_line(self, src, line, gnu_marker=False):
        d = line[0]
        toks = line if gnu_marker else line[1:]
        if not gnu_marker:
            toks = self._expand_line(toks, d)
        if not toks or toks[0].kind != NUMBER or not toks[0].text.isdigit():
            raise self._error('"%s" after #line is not a positive integer' %
                              (toks[0].text if toks else ''), d)
        num = int(toks[0].text)
        if not gnu_marker and not 0 <= num <= 2147483647:
            raise self._error('line number out of range', d)
        finfo = src.finfo
        if len(toks) > 1:
            s = toks[1]
            if s.kind != STRING or not s.text.startswith('"'):
                raise self._error('invalid filename "%s"' % s.text, s)
            fname = re.sub(r'\\(.)', r'\1', s.text[1:-1])
            finfo = FileInfo(fname)
            if len(toks) > 2 and not gnu_marker:
                self._warn('extra tokens at end of #line directive', toks[2])
        # The line after the directive has presumed number `num`.
        phys_next = src.toks[src.i - 1].line + 1
        src.lineoff = num - phys_next
        src.finfo = finfo
        self.pending.append((finfo, num, ''))

    # --- #pragma and _Pragma ---

    def _do_pragma(self, args, d):
        if not args:
            return
        first = args[0].text
        if first == 'once' and len(args) == 1:
            path = self.files[-1].path
            if path:
                self.once.add(os.path.realpath(path))
            return
        if first == 'omp':
            toks = self._expand_line(args[1:], d)
            text = '#pragma omp' + self._join_text(toks)
            self._emit_pragma(text, d)
            return
        if first == 'GCC' and len(args) > 1:
            sub = args[1].text
            if sub in ('error', 'warning'):
                msg = self._join_text(args[2:]).strip()
                if len(args) > 2 and args[2].kind == STRING:
                    msg = self._unquote(args[2].text)
                if sub == 'error':
                    raise self._error(msg, d)
                self._warn(msg, d)
            elif sub == 'poison':
                pass
            return
        if first in ('push_macro', 'pop_macro'):
            if len(args) >= 4 and args[1].text == '(' and args[2].kind == STRING:
                mname = self._unquote(args[2].text)
                stack = self.pushed_macros.setdefault(mname, [])
                if first == 'push_macro':
                    stack.append(self.macros.get(mname))
                elif stack:
                    m = stack.pop()
                    if m is None:
                        self.macros.pop(mname, None)
                    else:
                        self.macros[mname] = m
            return
        # Other pragmas (STDC, GCC system_header, pack, ...) are dropped.

    def _unquote(self, s):
        q = s.index('"')
        return re.sub(r'\\(["\\])', r'\1', s[q + 1:-1])

    def _do_Pragma(self, tok):
        """The _Pragma operator: read ( string-literal ) and run it as a #pragma."""
        t1 = self.get_token()
        t2 = self.get_token() if t1.text == '(' else None
        t3 = self.get_token() if t2 is not None and t2.kind == STRING else None
        if t3 is None or t3.text != ')':
            raise self._error('_Pragma takes a parenthesized string literal', tok)
        s = t2.text
        s = s[s.index('"') + 1:-1].replace('\\"', '"').replace('\\\\', '\\')
        finfo = FileInfo(tok.src.name if tok.src else '<_Pragma>')
        toks = lex(s, finfo)
        for t in toks:
            t.line = tok.line
            t.bol = False
        saved = self.in_directive
        self.in_directive = True
        try:
            self._do_pragma(toks, tok)
        finally:
            self.in_directive = saved

    def _emit_pragma(self, text, d):
        if self.out is not None:
            self.out.pragma(text, d)

    # --- macro expansion ---

    def _next_raw(self):
        """Next token without macro expansion. Exhausted contexts are popped (which re-enables
        their macro); an argument sentinel context yields EOF_ARG and stays."""
        ctx = self.ctx
        while ctx:
            c = ctx[-1]
            if c.i < len(c.toks):
                t = c.toks[c.i]
                c.i += 1
                return t
            if c.sentinel:
                return Tok(EOF_ARG, '', 0, False, False, None)
            ctx.pop()
            if c.macro is not None:
                self.disabled.discard(c.macro)
        return self._lex_token()

    def _next_painted(self):
        t = self._next_raw()
        if t.kind == IDENT and not t.noexp and t.text in self.disabled:
            t = t.copy()
            t.noexp = True
        return t

    def _backup(self, t):
        if t.kind == EOF_ARG:
            return
        self.ctx.append(_Context([t]))

    def get_token(self):
        """Next fully macro-expanded token."""
        while True:
            t = self._next_raw()
            if t.kind != IDENT or t.noexp:
                return t
            name = t.text
            m = self.macros.get(name)
            if m is None:
                if name == '_Pragma' and not self.in_directive:
                    self._do_Pragma(t)
                    continue
                return t
            if name in self.disabled:
                t = t.copy()
                t.noexp = True
                return t
            if m.builtin:
                r = self._builtin(m, t)
                if r is None:
                    return t
                return r
            if m.params is None:
                toks = self._substitute(m, None, t)
            else:
                # Look for '(' (newlines don't matter: the lexer doesn't produce them).
                self.peeking = True
                try:
                    p = self._next_painted()
                finally:
                    self.peeking = False
                if p.kind != PUNCT or p.text != '(':
                    self._backup(p)
                    return t
                args = self._collect_args(m, t)
                toks = self._substitute(m, args, t)
            self.disabled.add(name)
            self.ctx.append(_Context(toks, name))

    def _collect_args(self, m, name_tok):
        args = []
        cur = []
        depth = 0
        nparams = len(m.params)
        while True:
            t = self._next_painted()
            k = t.kind
            if k == EOF or k == EOF_INC or k == EOF_ARG:
                raise self._error('unterminated argument list invoking macro "%s"' % m.name,
                                  name_tok)
            if k == PUNCT:
                v = t.text
                if v == '(':
                    depth += 1
                elif v == ')':
                    if depth == 0:
                        args.append(cur)
                        break
                    depth -= 1
                elif v == ',' and depth == 0 and not (m.variadic and len(args) == nparams - 1):
                    args.append(cur)
                    cur = []
                    continue
            cur.append(t)
        if nparams == 0:
            if len(args) == 1 and not args[0]:
                return []
            raise self._error('macro "%s" passed %d arguments, but takes just 0' %
                              (m.name, len(args)), name_tok)
        if len(args) < nparams:
            if m.variadic and len(args) == nparams - 1:
                args.append(None)   # variadic argument omitted entirely (GNU)
            else:
                raise self._error('macro "%s" requires %d arguments, but only %d given' %
                                  (m.name, nparams, len(args)), name_tok)
        elif len(args) > nparams:
            raise self._error('macro "%s" passed %d arguments, but takes just %d' %
                              (m.name, len(args), nparams), name_tok)
        return args

    def _expand_arg(self, toks):
        """Fully macro-replace an argument, as if it formed the rest of the file."""
        if not toks:
            return []
        c = _Context(toks, None, sentinel=True)
        self.ctx.append(c)
        depth = len(self.ctx)
        out = []
        while True:
            t = self.get_token()
            if t.kind == EOF_ARG:
                break
            out.append(t)
        assert len(self.ctx) == depth and self.ctx[-1] is c
        self.ctx.pop()
        return out

    def _expand_line(self, toks, d):
        """Macro-expand the tokens of a directive line (#include, #line, #pragma omp)."""
        saved = self.in_directive
        self.in_directive = True
        try:
            return self._expand_arg(list(toks))
        finally:
            self.in_directive = saved

    def _stringize(self, arg, optok):
        parts = []
        for t in arg or ():
            if parts and (t.space or t.bol):
                parts.append(' ')
            s = t.text
            if t.kind == STRING or t.kind == CHAR:
                s = s.replace('\\', '\\\\').replace('"', '\\"')
            parts.append(s)
        return Tok(STRING, '"' + ''.join(parts) + '"', optok.line, optok.space, False,
                   optok.src)

    def _paste(self, left, right, where):
        text = left.text + right.text
        kind = _lex_one(text)
        if kind is None:
            raise self._error('pasting "%s" and "%s" does not give a valid preprocessing token'
                              % (left.text, right.text), where)
        return Tok(kind, text, left.line, left.space, False, left.src, False, left.col)

    def _substitute(self, m, args, name_tok):
        """The replacement list of `m` with arguments substituted, # and ## applied."""
        items = m.items
        out = []
        nitems = len(items)
        expanded = {}
        line = name_tok.line
        src = name_tok.src
        col = name_tok.col
        lead_empty = False
        carry = False   # an empty argument's whitespace passes to the next token (gcc padding)
        for idx in range(nitems):
            op, val, t = items[idx]
            if op == _OP_TOK:
                c = t.copy()
                c.line = line
                c.src = src
                c.col = col
                if carry:
                    c.space = True
                    carry = False
                out.append(c)
            elif op == _OP_PASTE:
                out.append(t)   # a PUNCT '##' from the body; recognised by identity below
            elif op == _OP_STR:
                s = self._stringize(args[val], t)
                s.line = line
                s.src = src
                s.col = col
                if carry:
                    s.space = True
                    carry = False
                out.append(s)
            else:
                arg = args[val]
                pasted = (idx > 0 and items[idx - 1][0] == _OP_PASTE) or \
                         (idx + 1 < nitems and items[idx + 1][0] == _OP_PASTE)
                if pasted:
                    if (m.variadic and val == len(m.params) - 1 and idx >= 2 and
                            items[idx - 1][0] == _OP_PASTE and items[idx - 2][0] == _OP_TOK
                            and items[idx - 2][2].text == ','):
                        # GNU: `, ## __VA_ARGS__` drops the comma when the variable arguments
                        # are omitted (gcc keeps it for an empty argument, as in `f(a,)`).
                        out.pop()
                        if arg is None:
                            out.pop()
                            continue
                        toks = arg
                    else:
                        toks = arg
                        if not toks:
                            out.append(Tok(PLACEMARKER, '', line, t.space, False, src))
                            continue
                else:
                    if val not in expanded:
                        expanded[val] = self._expand_arg(arg)
                    toks = expanded[val]
                    if not toks:
                        if idx == 0:
                            lead_empty = True
                        elif t.space:
                            carry = True
                        continue
                first = True
                for a in toks:
                    c = a.copy()
                    c.line = line
                    c.src = src
                    c.col = col
                    if c.bol:
                        c.space = True  # a newline inside the arguments is whitespace
                        c.bol = False
                    if first:
                        c.space = t.space or carry
                        carry = False
                        first = False
                    out.append(c)
        if m.haspaste:
            pastes = set(id(it[2]) for it in items if it[0] == _OP_PASTE)
            res = []
            i = 0
            n = len(out)
            while i < n:
                t = out[i]
                if id(t) in pastes and t.kind == PUNCT:
                    left = res.pop()
                    right = out[i + 1]
                    i += 2
                    if left.kind == PLACEMARKER:
                        res.append(right)
                    elif right.kind == PLACEMARKER:
                        res.append(left)
                    else:
                        res.append(self._paste(left, right, name_tok))
                    continue
                res.append(t)
                i += 1
            out = [t for t in res if t.kind != PLACEMARKER]
        if out:
            # The expansion is preceded by whitespace iff the macro name was (gcc's padding
            # rules; after an empty leading argument the next token keeps its own spacing).
            sp = name_tok.space or (lead_empty and out[0].space)
            if out[0].space != sp:
                out[0] = out[0].copy()
                out[0].space = sp
        return out

    def _builtin(self, m, t):
        name = m.name
        f = self.files[-1] if self.files else None

        def num(v):
            return Tok(NUMBER, str(v), t.line, t.space, False, t.src, False, t.col)

        def string(v):
            return Tok(STRING, v, t.line, t.space, False, t.src, False, t.col)

        if name == '__LINE__':
            return num(t.line)
        if name == '__FILE__':
            fname = t.src.name if t.src is not None else (f.finfo.name if f else '')
            return string(_c_string(fname))
        if name == '__BASE_FILE__':
            return string(_c_string(self.base_file))
        if name == '__DATE__':
            return string(self.date)
        if name == '__TIME__':
            return string(self.time)
        if name == '__COUNTER__':
            self.counter += 1
            return num(self.counter - 1)
        if name == '__INCLUDE_LEVEL__':
            return num(max(len(self.files) - 1, 0))
        if not self.in_directive:
            if name.startswith('__has_include'):
                raise self._error('"%s" used outside of preprocessing directive' % name, t)
            return None
        # __has_* operators (only meaningful in #if).
        p = self._next_raw()
        if p.text != '(':
            raise self._error('missing \'(\' after "%s"' % name, t)
        if name.startswith('__has_include'):
            toks = []
            while True:
                u = self._next_raw()
                if u.kind in (EOF, EOF_INC, EOF_ARG):
                    raise self._error('missing \')\' after "%s" operand' % name, t)
                if u.text == ')' and (not toks or toks[0].text != '<' or
                                      any(x.text == '>' for x in toks)):
                    break
                toks.append(u)
            hn = self._header_name(toks, t)
            if hn is None:
                hn = self._header_name(self._expand_arg(toks), t)
            if hn is None:
                raise self._error('operator "%s" requires a header-name' % name, t)
            return num(int(self._has_include(hn[0], hn[1], name == '__has_include_next')))
        depth = 0
        toks = []
        while True:
            u = self._next_raw()
            if u.kind in (EOF, EOF_INC, EOF_ARG):
                raise self._error('missing \')\' after "%s" operand' % name, t)
            if u.text == '(':
                depth += 1
            elif u.text == ')':
                if depth == 0:
                    break
                depth -= 1
            toks.append(u)
        val = 0
        if name == '__has_builtin' and len(toks) == 1:
            val = int(toks[0].text in _CPROC_BUILTINS)
        return num(val)

    def run(self, src, filename='<string>'):
        """Preprocess `src` (the main file) and return the output text."""
        finfo = FileInfo(filename)
        path = filename if not (filename.startswith('<') and filename.endswith('>')) else None
        self.base_file = filename
        toks = lex(src, finfo)
        s = self._push_file(toks, finfo, None, -1)
        s.path = path
        s.dir = os.path.dirname(path) if path else ''
        w = _Writer(self, finfo)
        self.out = w
        while True:
            t = self.get_token()
            k = t.kind
            if k == EOF:
                break
            if k == EOF_INC:
                continue
            w.token(t)
        self.out = None
        self.files.pop()
        return w.text()

    # --- #if expressions ---

    def _eval_if(self, args, d):
        if not args:
            raise self._error('#%s with no expression' % d.text, d)
        saved = self.in_directive
        self.in_directive = True
        c = _Context(list(args), None, sentinel=True)
        self.ctx.append(c)
        toks = []
        try:
            while True:
                t = self.get_token()
                if t.kind == EOF_ARG:
                    break
                if t.kind == IDENT and t.text == 'defined':
                    u = self._next_raw()
                    paren = u.text == '('
                    if paren:
                        u = self._next_raw()
                    if u.kind != IDENT:
                        raise self._error('operator "defined" requires an identifier', d)
                    if paren:
                        v = self._next_raw()
                        if v.text != ')':
                            raise self._error('missing \')\' after "defined"', d)
                    toks.append(Tok(NUMBER, '1' if u.text in self.macros else '0', t.line,
                                    t.space, False, t.src))
                    continue
                toks.append(t)
            self.ctx.pop()
        finally:
            self.in_directive = saved
            if self.ctx and self.ctx[-1] is c:
                self.ctx.pop()
        if not toks:
            raise self._error('#%s with no expression' % d.text, d)
        return _IfParser(self, toks, d).parse() != 0


class _IfParser(object):
    """Integer constant expressions in #if: values are (int, unsigned) with intmax_t/uintmax_t
    semantics; unevaluated operands don't raise division by zero."""

    _BINPREC = {
        '*': 10, '/': 10, '%': 10,
        '+': 9, '-': 9,
        '<<': 8, '>>': 8,
        '<': 7, '>': 7, '<=': 7, '>=': 7,
        '==': 6, '!=': 6,
        '&': 5, '^': 4, '|': 3, '&&': 2, '||': 1,
    }

    def __init__(self, pp, toks, d):
        self.pp = pp
        self.toks = toks
        self.i = 0
        self.d = d

    def error(self, msg, tok=None):
        return self.pp._error(msg, tok if tok is not None else self.d)

    def peek(self):
        if self.i < len(self.toks):
            return self.toks[self.i]
        return None

    def parse(self):
        v, u = self.comma(True)
        t = self.peek()
        if t is not None:
            if t.text == ')':
                raise self.error('missing \'(\' in expression', t)
            raise self.error('missing binary operator before token "%s"' % t.text, t)
        return v

    def comma(self, ev):
        v = self.cond(ev)
        while True:
            t = self.peek()
            if t is None or t.text != ',':
                return v
            self.i += 1
            v = self.cond(ev)

    def cond(self, ev):
        c = self.binary(1, ev)
        t = self.peek()
        if t is None or t.text != '?':
            return c
        self.i += 1
        a = self.comma(ev and c[0] != 0)
        t = self.peek()
        if t is None or t.text != ':':
            raise self.error('\'?\' without following \':\'', t)
        self.i += 1
        b = self.cond(ev and c[0] == 0)
        uns = a[1] or b[1]
        r = a if c[0] != 0 else b
        return (r[0] & _MASK if uns else r[0]), uns

    def binary(self, minprec, ev):
        left = self.unary(ev)
        while True:
            t = self.peek()
            if t is None or t.kind != PUNCT:
                return left
            op = t.value
            prec = self._BINPREC.get(op)
            if prec is None or prec < minprec:
                return left
            self.i += 1
            if op == '&&':
                right = self.binary(prec + 1, ev and left[0] != 0)
                left = (int(left[0] != 0 and right[0] != 0), False)
                continue
            if op == '||':
                right = self.binary(prec + 1, ev and left[0] == 0)
                left = (int(left[0] != 0 or right[0] != 0), False)
                continue
            right = self.binary(prec + 1, ev)
            left = self.apply(op, left, right, ev, t)

    @staticmethod
    def signed(v):
        v &= _MASK
        return v - (1 << 64) if v & _SIGN else v

    def apply(self, op, a, b, ev, t):
        if op in ('<<', '>>'):
            uns = a[1]
            x = a[0]
            n = (b[0] & _MASK) if b[1] else b[0]
            if op == '>>':
                n = -n
            if n >= 0:
                r = (x << n) if n < 128 else 0
            else:
                n = -n
                if uns:
                    r = (x & _MASK) >> n if n < 64 else 0
                else:
                    r = x >> n if n < 64 else (-1 if x < 0 else 0)
            return (r & _MASK, True) if uns else (self.signed(r), False)
        uns = a[1] or b[1]
        x, y = a[0], b[0]
        if uns:
            x &= _MASK
            y &= _MASK
        if op in ('<', '>', '<=', '>=', '==', '!='):
            r = {'<': x < y, '>': x > y, '<=': x <= y, '>=': x >= y, '==': x == y,
                 '!=': x != y}[op]
            return int(r), False
        if op == '+':
            r = x + y
        elif op == '-':
            r = x - y
        elif op == '*':
            r = x * y
        elif op in ('/', '%'):
            if y == 0:
                if ev:
                    raise self.error('division by zero in #%s' % self.d.text, t)
                return 0, uns
            q = abs(x) // abs(y)
            if (x < 0) != (y < 0):
                q = -q
            r = q if op == '/' else x - q * y
        elif op == '&':
            r = x & y
        elif op == '^':
            r = x ^ y
        else:
            r = x | y
        if uns:
            return r & _MASK, True
        s = self.signed(r)
        if s != r and ev:
            self.pp._warn('integer overflow in preprocessor expression', t)
        return s, False

    def unary(self, ev):
        t = self.peek()
        if t is None:
            raise self.error('#%s with no expression' % self.d.text if self.i == 0 else
                             'operator at end of expression has no right operand',
                             self.toks[-1] if self.toks else None)
        self.i += 1
        if t.kind == PUNCT:
            v = t.value
            if v == '(':
                r = self.comma(ev)
                c = self.peek()
                if c is None or c.text != ')':
                    raise self.error('missing \')\' in expression', c)
                self.i += 1
                return r
            if v in ('+', '-', '~', '!'):
                x, uns = self.unary(ev)
                if v == '+':
                    return x, uns
                if v == '!':
                    return int(x == 0), False
                r = -x if v == '-' else ~x
                return (r & _MASK, True) if uns else (self.signed(r), False)
            raise self.error('token "%s" is not valid in preprocessor expressions' % t.text,
                             t)
        if t.kind == NUMBER:
            return self.number(t)
        if t.kind == CHAR:
            return self.char(t)
        if t.kind == IDENT:
            # Identifiers remaining after macro expansion (including keywords) are 0.
            return 0, False
        raise self.error('token "%s" is not valid in preprocessor expressions' % t.text, t)

    _INT_RE = re.compile(r'(0[xX][0-9a-fA-F]+|0[bB][01]+|[0-9]+)([uUlL]*)$')
    _SUFFIX_RE = re.compile(r'[uU]?(?:l|L|ll|LL)?$|(?:l|L|ll|LL)[uU]$')

    def number(self, t):
        m = self._INT_RE.match(t.text)
        if not m:
            if re.match(r'(0[xX][0-9a-fA-F.]*[pP]|[0-9.]*[eE.])', t.text):
                raise self.error('floating constant in preprocessor expression', t)
            raise self.error('invalid integer constant "%s" in #%s' % (t.text, self.d.text), t)
        digits, suffix = m.group(1), m.group(2)
        s = suffix.lower()
        if not self._SUFFIX_RE.match(suffix):
            raise self.error('invalid suffix "%s" on integer constant' % suffix, t)
        if digits[:2] in ('0x', '0X'):
            v = int(digits[2:], 16)
        elif digits[:2] in ('0b', '0B'):
            v = int(digits[2:], 2)
        elif digits[0] == '0':
            if any(ch in '89' for ch in digits):
                raise self.error('invalid digit in octal constant', t)
            v = int(digits, 8)
        else:
            v = int(digits)
        if v > _MASK:
            raise self.error('integer constant is too large for its type', t)
        uns = 'u' in s
        if not uns and v & _SIGN:
            if digits[0] != '0' or digits == '0':
                self.pp._warn('integer constant is so large that it is unsigned', t)
            uns = True
        return v, uns

    _ESC = {'n': 10, 't': 9, 'v': 11, 'b': 8, 'r': 13, 'f': 12, 'a': 7, '\\': 92, "'": 39,
            '"': 34, '?': 63, 'e': 27, 'E': 27}

    def char(self, t):
        text = t.text
        q = text.index("'")
        prefix = text[:q]
        body = text[q + 1:-1]
        if not body:
            raise self.error('empty character constant', t)
        vals = []
        i = 0
        n = len(body)
        while i < n:
            ch = body[i]
            if ch != '\\':
                cp = ord(ch)
                if prefix or cp < 0x80:
                    vals.append(cp)
                else:
                    vals.extend(ch.encode('utf-8', 'surrogateescape'))
                i += 1
                continue
            i += 1
            e = body[i]
            if e in self._ESC:
                vals.append(self._ESC[e])
                i += 1
            elif e in '01234567':
                j = i
                while j < n and j < i + 3 and body[j] in '01234567':
                    j += 1
                vals.append(int(body[i:j], 8))
                i = j
            elif e == 'x':
                j = i + 1
                while j < n and body[j] in '0123456789abcdefABCDEF':
                    j += 1
                if j == i + 1:
                    raise self.error('\\x used with no following hex digits', t)
                vals.append(int(body[i + 1:j], 16))
                i = j
            elif e in 'uU':
                k = 4 if e == 'u' else 8
                cp = int(body[i + 1:i + 1 + k], 16)
                if prefix:
                    vals.append(cp)
                else:
                    vals.extend(chr(cp).encode('utf-8'))
                i += 1 + k
            else:
                self.pp._warn('unknown escape sequence: \'\\%s\'' % e, t)
                vals.append(ord(e))
                i += 1
        if prefix:
            v = vals[-1]
            if len(vals) > 1:
                self.pp._warn('character constant too long for its type', t)
            if prefix == 'L':
                v &= 0xffffffff
                if _WCHAR_SIGNED[self.pp.target] and v & 0x80000000:
                    v -= 1 << 32
            elif prefix == 'u':
                v &= 0xffff
            else:
                v &= 0xffffffff
            return v, False
        if len(vals) == 1:
            v = vals[0] & 0xff
            if _CHAR_SIGNED[self.pp.target] and v & 0x80:
                v -= 0x100
            return v, False
        if len(vals) > 4:
            self.pp._warn('character constant too long for its type', t)
        else:
            self.pp._warn('multi-character character constant', t)
        v = 0
        for c in vals:
            v = (v << 8) | (c & 0xff)
        v &= 0xffffffff
        if v & 0x80000000:
            v -= 1 << 32
        return v, False


# --- Output ------------------------------------------------------------------------------------

class _Writer(object):
    """Turns the token stream into text with gcc-style line markers."""

    def __init__(self, pp, finfo):
        self.pp = pp
        self.parts = []
        self.src = finfo
        self.line = 1
        self.prev = None        # previous token on the current output line
        self.parts.append('# 1 %s\n' % _c_string(finfo.name))
        self.force_marker = False

    def _bol(self):
        if self.prev is not None:
            self.parts.append('\n')
            self.prev = None
            self.line += 1

    def marker(self, finfo, line, flag=''):
        self._bol()
        self.parts.append('# %d %s%s\n' % (line, _c_string(finfo.name), flag))
        self.src = finfo
        self.line = line
        self.force_marker = False

    def flush_pending(self):
        pending = self.pp.pending
        if pending:
            for finfo, line, flag in pending:
                self.marker(finfo, line, flag)
            del pending[:]

    def _sync(self, src, line):
        if self.pp.pending:
            self.flush_pending()
        if src is None or src is _NOSRC:
            return
        if src is not self.src or self.force_marker or line < self.line:
            self.marker(src, line)
        elif line > self.line:
            d = line - self.line
            if d <= 8:
                self.parts.append('\n' * d)
                self.line = line
                self.prev = None
            else:
                self.marker(src, line)

    def token(self, t):
        self._sync(t.src, t.line)
        prev = self.prev
        text = t.text
        if t.kind == PUNCT and text in _DIGRAPHS:
            text = _DIGRAPHS[text]
        if prev is not None:
            if t.space or t.bol or _needs_space(prev, t):
                self.parts.append(' ')
        else:
            # Like gcc: the first token of a line keeps its source column (for diagnostics).
            n = max(t.col - 2, 0) + (1 if t.space else 0)
            if n:
                self.parts.append(' ' * n)
        self.parts.append(text)
        if text is not t.text:
            t = Tok(t.kind, text, t.line, t.space, t.bol, t.src)
        self.prev = t

    def pragma(self, text, d):
        self._sync(d.src, d.line)
        self._bol()
        self.parts.append(text)
        self.parts.append('\n')
        self.line += 1
        if d.src is not self.src:
            self.force_marker = True

    def text(self):
        self.flush_pending()
        if self.prev is not None:
            self.parts.append('\n')
        return ''.join(self.parts)


def preprocess(src, filename='<string>', include_dirs=(), defines=None, target='arm64',
               warn=None, keep_attributes=False):
    """Preprocess C source text; return text with gcc-style line markers.

    include_dirs: searched (after the including file's directory for "...") before the bundled
    headers in cimo/include. defines: dict name -> value (None means 1); a name may carry a
    parameter list, e.g. {'SQ(x)': '((x)*(x))'}. target: 'arm64' (Linux) or 'arm64_apple'.
    warn: callable receiving warning strings (default: print to stderr).
    keep_attributes: ignore `#define __attribute__` (glibc defines it away for compilers other
    than gcc and clang), so that the frontend sees every attribute.
    Raises PreprocessError.
    """
    if warn is None:
        def warn(msg):
            sys.stderr.write(msg + '\n')
    pp = Preprocessor(include_dirs, defines, target, warn, keep_attributes)
    return pp.run(src, filename)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python3 -m cimo.preprocessor', description='cimo C preprocessor')
    ap.add_argument('-I', dest='include', action='append', default=[], metavar='DIR')
    ap.add_argument('-D', dest='define', action='append', default=[], metavar='NAME[=VAL]')
    ap.add_argument('-o', dest='output', default=None)
    ap.add_argument('--target', default='arm64', choices=sorted(TARGETS))
    ap.add_argument('file')
    a = ap.parse_args(argv)
    defines = []
    for d in a.define:
        name, eq, val = d.partition('=')
        defines.append((name, val if eq else None))
    if a.file == '-':
        text = sys.stdin.read()
        name = '<stdin>'
    else:
        with open(a.file, 'rb') as f:
            text = f.read().decode('utf-8', 'surrogateescape')
        name = a.file
    try:
        out = preprocess(text, name, a.include, defines, a.target)
    except PreprocessError as e:
        sys.stderr.write(str(e) + '\n')
        return 1
    if a.output:
        with open(a.output, 'w', encoding='utf-8', errors='surrogateescape') as f:
            f.write(out)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
