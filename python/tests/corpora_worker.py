"""test_corpora.py's worker process: reads C file paths from stdin, one per line; for each,
compiles it with cimo (include directories: $CIMO_INCLUDE, os.pathsep-separated), calls main()
and writes one result line to the original stdout:

    PASS | COMPILE <message> | ABORT <message> | STATUS <n> | OUTPUT <repr> | ERROR <message>

A crash or hang kills only this process; the test runner records it and starts another.
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


def run(path):
    try:
        m = cimo.load(path, include_dirs=INCLUDE)
    except CompileError as e:
        return 'COMPILE ' + str(e).replace(os.path.dirname(path) + os.sep, '').splitlines()[0]
    k = m._kernels.get('main')
    if k is None:
        return 'ERROR no main()'
    argv = (ctypes.c_char_p * 2)(b'prog', None)
    args = [1, ctypes.addressof(argv), 0][:len(k._params)]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            r = k(*args)
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
