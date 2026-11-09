"""Port of util.c, plus the output and error plumbing that replaces stdio.

stdout is the list `out` (pieces of text, joined at the end); stderr text is
collected in `err` and also carried by the exception. `fatal()` and `error()`
(token.py) raise FrontendError instead of calling exit(1).
"""

import sys

argv0 = 'cproc-qbe'

# stdout of the C program, as a list of str pieces (latin-1: one char per byte)
out = []

# cimo extension, off by default (cproc's behavior), set by compile(diagnostics=True): clear
# errors for unsupported C instead of silent acceptance or generic messages (see diag.py)
diagnostics = False
# cimo: `#pragma STDC FENV_ACCESS ON` in effect (pp.directive), and whether it was at any point in the function being
# parsed: the backend then leaves floating-point operations whose result is inexact to run (cimo.backend.fold)
fenvaccess = False
fenvtouched = False
# cimo: `#pragma pack(n)` in effect (gcc's and MSVC's): a struct or union member's alignment is at most pack (0: no
# limit), and the stack of push/pop as (name or None, pack) (pp.directive, decl.addmember)
pack = 0
packstack = []
# cimo: `#pragma STDC CX_LIMITED_RANGE ON` in effect: complex * and / by the usual formulas, inline (qbe._cxbinary)
cxlimited = False
# cimo: `#pragma STDC FP_CONTRACT` (C11 7.12.2) sets qbe.contract: DEFAULT is the compile's fma=, ON is honored only
# where the target has a fused multiply-add (fmaok); a compound statement restores the value it started with (stmt)
fmadefault = False
fmaok = True
# cimo: `#pragma omp simd reduction(+:s)`: the (operator, name) pairs of the pragma before the next statement
# (pp.directive), and {id(decl): operator} for the loop it precedes, while it is parsed (stmt, expr.assignexpr)
ompnext = None
reduce = {}
# cimo: OpenMP (omp.py): parallel loops being parsed (their bodies may not return, break out or nest some
# directives), critical and atomic statements being parsed (no return), the VLA types whose size was first computed
# in the innermost loop being parsed (None outside), parallel loops outlined so far in the unit (their functions'
# numbers), and the names of the critical sections whose locks the unit has defined
ompregion = 0
ompblock = 0
ompvla = None
ompcount = 0
ompcrit = set()


class FrontendError(Exception):
    """cproc exited with an error. `msg` is the exact text cproc writes to stderr
    (without the trailing newline); `status` is the exit status."""

    def __init__(self, msg, status=1):
        Exception.__init__(self, msg)
        self.msg = msg
        self.status = status


def warn(fmt, *args):
    msg = fmt % args
    sys.stderr.write('%s: %s\n' % (argv0, msg))


def fatal(fmt, *args):
    msg = fmt % args
    raise FrontendError('%s: %s' % (argv0, msg))


def progname(name, fallback):
    if not name:
        return fallback
    i = name.rfind('/')
    return name[i + 1:] if i >= 0 else name
