"""test_corpora.py's worker process: reads lines `path` or `path<TAB>prelude<TAB>include dirs`
(os.pathsep-separated) from stdin; for each, compiles the C file with cimo (the prelude first,
if any; include directories: those, then $CIMO_INCLUDE), calls main() and writes one result
line to the original stdout:

    PASS | COMPILE <message> | ABORT <message> | STATUS <n> | OUTPUT <repr> | ERROR <message>

A crash or hang kills only this process; the test runner records it and starts another.

Each main() starts on a stack filled with garbage (a large positive number in every int, an invalid
address in every pointer): on a clean stack, a program reading the wrong memory often reads 0 and
passes. gcc-torture's 20011008-3 hit a miscompiled load that way, and only failed on Windows.
"""

import contextlib
import ctypes
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cimo  # noqa: E402
from cimo.driver import CompileError, KernelError  # noqa: E402


INCLUDE = [d for d in os.environ.get('CIMO_INCLUDE', '').split(os.pathsep) if d]
# in the program itself: Python's own frames would overwrite the part main's locals land on
WRAPPER = '''
void __cimo_poison(void) {
    volatile unsigned char b[1 << 17];
    for (long i = 0; i < sizeof b; i++) b[i] = 0x5a;
}
long __cimo_main(int c, char **v) {
    __cimo_poison();
    return ((long (*)(int, char **))main)(c, v);
}
'''


def run(line):
    path, prelude, inc = (line.split('\t') + ['', ''])[:3]
    inc = [d for d in inc.split(os.pathsep) if d] + INCLUDE
    try:
        first = '#include "%s"\n' % prelude.replace(os.sep, '/') if prelude else ''
        with open(path, 'rb') as f:
            wrapv = b'-fwrapv' in f.read()  # the program's own dg-options
        m = cimo.compile(first + '#include "%s"\n' % path.replace(os.sep, '/') + WRAPPER,
                         include_dirs=inc, wrapv=wrapv)
    except CompileError as e:
        return 'COMPILE ' + str(e).replace(os.path.dirname(path) + os.sep, '').splitlines()[0]
    k = m._kernels.get('main')
    if k is None:
        return 'ERROR no main()'
    argv = (ctypes.c_char_p * 2)(b'prog', None)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            r = m._kernels['__cimo_main'](1, ctypes.addressof(argv))
        status = r if k._ret is not None else 0
    except KernelError as e:
        if e.kind != 'exit':
            return 'ABORT ' + str(e).splitlines()[0]
        status = e.status
    expected = path + '.expected'
    if os.path.exists(expected):
        with open(expected, encoding='utf-8', errors='surrogateescape', newline='') as f:
            if out.getvalue() != f.read():
                return 'OUTPUT %r' % out.getvalue()[:200]
    return 'PASS' if status & 0xff == 0 else 'STATUS %d' % status


def main():
    proto = os.fdopen(os.dup(sys.stdout.fileno()), 'w')
    sys.stdout = sys.stderr  # anything else printed must not reach the protocol
    for line in sys.stdin:
        try:
            res = run(line.rstrip('\n'))
        except Exception as e:  # a compiler crash: Python exception inside cimo
            res = 'ERROR %s: %s' % (type(e).__name__, str(e).splitlines()[0] if str(e) else '')
        proto.write(res.replace('\n', ' ') + '\n')
        proto.flush()


if __name__ == '__main__':
    main()
