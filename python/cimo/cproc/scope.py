"""Port of scope.c. Maps are dicts."""

from .cc import (Scope, Decl, DECLBUILTIN, DECLTYPE, BUILTINALLOCA, BUILTINCONSTANTP,
                 BUILTINEXPECT, BUILTININFF, BUILTINNANF, BUILTINOFFSETOF,
                 BUILTINTYPESCOMPATIBLEP, BUILTINUNREACHABLE, BUILTINVAARG, BUILTINVACOPY,
                 BUILTINVAEND, BUILTINVASTART)
from . import targ as TG

# always access as scope.filescope (it is replaced by scopeinit)
filescope = Scope()


def scopeinit():
    global filescope
    filescope = Scope()
    builtins = [
        ("__builtin_alloca", BUILTINALLOCA),
        ("__builtin_constant_p", BUILTINCONSTANTP),
        ("__builtin_expect", BUILTINEXPECT),
        ("__builtin_inff", BUILTININFF),
        ("__builtin_nanf", BUILTINNANF),
        ("__builtin_offsetof", BUILTINOFFSETOF),
        ("__builtin_types_compatible_p", BUILTINTYPESCOMPATIBLEP),
        ("__builtin_unreachable", BUILTINUNREACHABLE),
        ("__builtin_va_arg", BUILTINVAARG),
        ("__builtin_va_copy", BUILTINVACOPY),
        ("__builtin_va_end", BUILTINVAEND),
        ("__builtin_va_start", BUILTINVASTART),
        ("__builtin_c23_va_start", BUILTINVASTART),
    ]
    for name, b in builtins:
        scopeputdecl(filescope, Decl(name=name, kind=DECLBUILTIN, builtin=b))
    valist = Decl(name="__builtin_va_list", kind=DECLTYPE, type=TG.targ.typevalist)
    scopeputdecl(filescope, valist)


def mkscope(parent):
    s = Scope()
    s.breaklabel = parent.breaklabel
    s.continuelabel = parent.continuelabel
    s.switchcases = parent.switchcases
    s.parent = parent
    return s


def delscope(s):
    return s.parent


def scopegetdecl(s, name, recurse):
    while True:
        d = s.decls.get(name) if s.decls else None
        s = s.parent
        if d or not s or not recurse:
            return d


def scopegettag(s, name, recurse):
    while True:
        t = s.tags.get(name) if s.tags else None
        s = s.parent
        if t or not s or not recurse:
            return t


def scopeputdecl(s, d):
    s.decls[d.name] = d


def scopeputtag(s, name, t):
    s.tags[name] = t
