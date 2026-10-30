"""Port of main.c: the cproc-qbe command line, plus the compile() entry point.

python -m cimo.cproc [-E] [-t target] [-o output] [input...]
"""

import sys

from .cc import TEOF, TSEMICOLON, PPNEWLINE
from . import util as U
from . import token as TK
from . import scan as S
from . import pp as PP
from . import targ as TG
from . import scope as SC
from . import decl as D
from . import qbe as Q

DEFAULT_TARGET = 'aarch64'


def _reset():
    del U.out[:]
    S._reset()
    PP._reset()
    D._reset()
    Q._reset()


def _run(inputs, target, pponly):
    """inputs: list of (name, latin-1 text or None to read the file)."""
    _reset()
    TG.targinit(target)
    if len(inputs) == 1 and inputs[0][1] is not None:
        S.scanfrom(inputs[0][0], inputs[0][1])
    else:
        for name, text in reversed(inputs):
            S.scanfrom(name, text)
        S.scanopen()
    TK.tokeninit()
    PP.ppinit()
    tok = TK.tok
    if pponly:
        PP.ppflags |= PPNEWLINE
        while tok.kind != TEOF:
            TK.tokenprint(tok, U.out)
            PP.next()
    else:
        SC.scopeinit()
        while tok.kind != TEOF:
            if not D.decl(SC.filescope, None):
                if tok.kind == TSEMICOLON:
                    TK.error(tok.loc, "unexpected ';' at top-level")
                TK.error(tok.loc, "expected declaration or function definition")
        D.emittentativedefns()


class _Recursion(object):
    def __enter__(self):
        self.old = sys.getrecursionlimit()
        if self.old < 20000:
            sys.setrecursionlimit(20000)

    def __exit__(self, *a):
        sys.setrecursionlimit(self.old)


def _tostr(s):
    return s.encode('latin-1').decode('utf-8', 'surrogateescape')


def compile(text, filename='<stdin>', target=DEFAULT_TARGET, pponly=False):
    """Compile preprocessed C (str or bytes) to QBE IR text.

    Raises CprocError with cproc's exact error message on errors."""
    if isinstance(text, str):
        text = text.encode('utf-8', 'surrogateescape')
    text = text.decode('latin-1')
    try:
        with _Recursion():
            _run([(filename, text)], target, pponly)
    except U.CprocError as e:
        e.msg = _tostr(e.msg)
        e.args = (e.msg,)
        e.partial = _tostr(''.join(U.out))
        del U.out[:]
        raise
    res = _tostr(''.join(U.out))
    del U.out[:]
    return res


def usage():
    sys.stderr.write("usage: %s [input]\n" % U.argv0)
    return 2


def main(argv=None):
    if argv is None:
        argv = sys.argv
    pponly = False
    output = None
    target = DEFAULT_TARGET
    args = argv[1:]
    i = 0
    # ARGBEGIN ... ARGEND
    while i < len(args):
        a = args[i]
        if not a.startswith('-') or a == '-':
            break
        i += 1
        if a == '--':
            break
        j = 1
        while j < len(a):
            c = a[j]
            j += 1
            if c == 'E':
                pponly = True
            elif c == 't' or c == 'o':
                if j < len(a):
                    val = a[j:]
                elif i < len(args):
                    val = args[i]
                    i += 1
                else:
                    return usage()
                j = len(a)
                if c == 't':
                    target = val
                else:
                    output = val
            else:
                return usage()
    files = args[i:]
    if files:
        inputs = [(name, None) for name in files]
    else:
        data = sys.stdin.buffer.read()
        inputs = [("<stdin>", data.decode('latin-1'))]
    outf = None
    status = 0
    errmsg = None
    if output:
        try:
            outf = open(output, 'wb')
        except OSError as e:
            sys.stderr.write("%s: open %s: %s\n" % (U.argv0, output, e.strerror))
            return 1
    try:
        with _Recursion():
            _run(inputs, target, pponly)
    except U.CprocError as e:
        errmsg = e.msg
        status = e.status
    except AssertionError:
        # C cproc is built with assertions: assert() prints and abort()s, losing the
        # buffered stdout
        import traceback
        fr = traceback.extract_tb(sys.exc_info()[2])[-1]
        errmsg = "%s: %s:%d: %s: Assertion `%s' failed." % (
            U.argv0, fr.filename.rsplit('/', 1)[-1], fr.lineno, fr.name, fr.line)
        status = -6
        del U.out[:]
    except Exception as e:
        # where C cproc would crash (e.g. a NULL dereference); see PORTING.md
        errmsg = "%s: internal error: %s: %s" % (U.argv0, type(e).__name__, e)
        status = 70
        del U.out[:]
    data = ''.join(U.out).encode('latin-1')
    del U.out[:]
    # C writes diagnostics unbuffered and the (fully buffered) output at exit
    if errmsg is not None:
        sys.stderr.flush()
        sys.stderr.buffer.write(errmsg.encode('latin-1') + b'\n')
        sys.stderr.flush()
    if outf:
        outf.write(data)
        outf.close()
    else:
        sys.stdout.flush()
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    if status == -6:
        import os
        os.abort()
    return status
