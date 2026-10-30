"""Port of util.c, plus the output and error plumbing that replaces stdio.

stdout is the list `out` (pieces of text, joined at the end); stderr text is
collected in `err` and also carried by the exception. `fatal()` and `error()`
(token.py) raise FrontendError instead of calling exit(1).
"""

import sys

argv0 = 'cproc-qbe'

# stdout of the C program, as a list of str pieces (latin-1: one char per byte)
out = []


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
