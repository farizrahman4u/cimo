"""Port of targ.c.

The target table is rebuilt by targinit() so that per-compile state stored on the
va_list types (Type.value, set by emittype) does not leak between compiles.

Extension (not in cproc): target "aarch64-apple" for Apple arm64 (see PORTING.md):
va_list is `char *`, plain char is signed, wchar_t is int, long double is
double-sized (8 bytes, so it compiles as QBE `d`).

Extension: target "aarch64-windows" for Windows arm64: as aarch64-apple, but wchar_t is
unsigned short. Still LP64 (cimo keeps `long` 64-bit on every target; Windows' own C is LLP64).
"""

from .cc import Type, TYPEARRAY, TYPESTRUCT, TYPEPOINTER, PROPSCALAR, QUALNONE, Ref
from . import type as TY
from . import util as U


class Target(object):
    __slots__ = ('name', 'typevalist', 'typewchar', 'signedchar', 'ldoublesize')

    def __init__(self, name, typevalist, typewchar, signedchar=0, ldoublesize=16):
        self.name = name
        self.typevalist = typevalist
        self.typewchar = typewchar
        self.signedchar = signedchar
        self.ldoublesize = ldoublesize


targ = None


def _alltargs():
    return [
        Target("x86_64-sysv",
               Type(kind=TYPEARRAY, align=8, size=24,
                    base=Type(kind=TYPESTRUCT, align=8, size=24)),
               TY.typeint, 1),
        Target("aarch64",
               Type(kind=TYPESTRUCT, align=8, size=32, tag="va_list"),
               TY.typeuint),
        Target("riscv64",
               Type(kind=TYPEPOINTER, prop=PROPSCALAR, align=8, size=8, base=TY.typevoid),
               TY.typeint),
        Target("aarch64-apple",
               Type(kind=TYPEPOINTER, prop=PROPSCALAR, align=8, size=8, base=TY.typechar),
               TY.typeint, 1, 8),
        Target("aarch64-windows",
               Type(kind=TYPEPOINTER, prop=PROPSCALAR, align=8, size=8, base=TY.typechar),
               TY.typeushort, 1, 8),
    ]


def targinit(name):
    global targ
    targ = None
    alltargs = _alltargs()
    if not name:
        # TODO: provide a way to set this default
        targ = alltargs[0]
    for t in alltargs:
        if targ:
            break
        if t.name == name:
            targ = t
    if not targ:
        U.fatal("unknown target '%s'", name)
    TY.typechar.issigned = bool(targ.signedchar)
    TY.typeldouble.size = TY.typeldouble.align = targ.ldoublesize
    TY.typeadjvalist = TY.typeadjust(targ.typevalist, Ref(QUALNONE))
