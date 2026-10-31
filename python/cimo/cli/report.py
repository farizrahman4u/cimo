"""Rendering of compile errors and kernel errors for the terminal: source excerpts with a caret,
hints rewritten from Python API spelling into command-line flags."""

import re

# file:line[:col]: kind: message (file may start with a Windows drive letter)
_LOC = re.compile(r'^(?P<file>(?:[A-Za-z]:)?[^:\n]+?):(?P<line>\d+)(?::(?P<col>\d+))?: '
                  r'(?P<kind>error|warning|note): (?P<msg>.*)$', re.S)

# hints in driver messages name Python keyword arguments; the CLI spells them as flags
_API_TO_FLAGS = [
    (re.compile(r"host_headers=True, link=\['libc'\]"), '--host-headers --link libc'),
    (re.compile(r"host_headers=True"), '--host-headers'),
    (re.compile(r"link=\['libc'\] to call them"), '--link libc to call them'),
    (re.compile(r"link=\['(\w[\w.+-]*)'\]"), r'--link \1'),
]


def flags_for(text):
    for pat, rep in _API_TO_FLAGS:
        text = pat.sub(rep, text)
    return text


def _split_hint(msg):
    """'message; hint' -> (message, hint or None). Driver hints follow the first '; '."""
    head, sep, tail = msg.partition('; ')
    return (head, tail) if sep else (msg, None)


def compile_error(term, err, sources):
    """Print a CompileError. sources: {filename: text} for excerpts."""
    text = str(err).strip()
    m = _LOC.match(text)
    if not m:
        msg, hint = _split_hint(text)
        term.write('%s %s' % (term.style('error:', 'error'), term.style(msg, 'bold')))
        if hint:
            _hint(term, hint)
        return
    msg, hint = _split_hint(m.group('msg').strip())
    kind = m.group('kind')
    term.write('%s %s' % (term.style(kind + ':', kind), term.style(msg, 'bold')))
    col = m.group('col')
    _excerpt(term, sources, m.group('file'), int(m.group('line')), int(col) if col else None)
    if hint:
        _hint(term, hint)


def _excerpt(term, sources, file, line, col=None, where=''):
    """  → file:line:col, then the source line with a caret under col."""
    loc = '%s:%d%s' % (file, line, ':%d' % col if col else '')
    term.write('  %s %s%s' % (term.style(term.symbol('arrow'), 'muted'), term.style(loc, 'path'),
                              where))
    src = sources.get(file)
    if src is None:
        return
    lines = src.split('\n')
    if not 1 <= line <= len(lines):
        return
    num = str(line)
    gutter = ' ' * len(num)
    bar = term.style(term.symbol('bar'), 'muted')
    code = lines[line - 1].expandtabs(4)
    term.write('   %s %s' % (gutter, bar))
    term.write('   %s %s %s' % (term.style(num, 'muted'), bar, code))
    if col:
        c = len(lines[line - 1][:col - 1].expandtabs(4)) + 1
        if c <= len(code) + 1:
            term.write('   %s %s %s%s' % (gutter, bar, ' ' * (c - 1), term.style('^', 'caret')))


def _hint(term, hint):
    term.write('  %s %s' % (term.style('hint:', 'hint'), term.code(flags_for(hint))))


def kernel_error(term, err, sources=None):
    """Print a KernelError (abort, assert, trap, stack overflow), with the failed assert's
    source line."""
    kind = getattr(err, 'kind', None)
    where = ' in %s()' % err.function if getattr(err, 'function', None) else ''
    if kind == 'assert':
        term.write('%s %s %s' % (term.style('error:', 'error'),
                                 term.style('assertion failed:', 'bold'),
                                 term.style(err.expr or '', 'code')))
        if err.file:
            _excerpt(term, sources or {}, err.file, err.line,
                     where=term.style(where, 'muted'))
        return
    msg = str(err)
    msg = re.sub(r' \(calling \w+\)$', '', msg)
    term.write('%s %s' % (term.style('error:', 'error'), term.style(msg, 'bold')))
    if kind == 'stack overflow':
        _hint(term, 'recursion deeper than the thread stack allows')
