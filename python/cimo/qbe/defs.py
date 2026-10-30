"""Core IR definitions, ported from QBE all.h and ops.h.

Conventions (see PORTING.md):
  * Ref is an int: (type << 29) | val. R == 0, so req(a, b) is a == b.
  * Unsigned C fields set to -1u hold U32 (0xffffffff) so comparisons stay unsigned.
  * Mutable cross-module globals (T, debug, insb, curi, typ) live in this module;
    access them as defs.X (import the module, not the names).
"""

import struct

NIns = 1 << 20
NAlign = 3
NField = 32
NBit = 64
U32 = 0xffffffff
M64 = 0xffffffffffffffff


def BIT(n):
    return 1 << n


def s64(x):
    """Wrap to a signed 64-bit int (C int64_t overflow)."""
    x &= M64
    return x - (1 << 64) if x >> 63 else x


def s32(x):
    x &= U32
    return x - (1 << 32) if x >> 31 else x


RXX = 0
Tmp0 = NBit

# Ref
RTmp, RCon, RInt, RType, RSlot, RCall, RMem = range(7)
VMASK = 0x1fffffff


def mkref(t, v):
    return (t << 29) | (v & VMASK)


R = 0
UNDEF = mkref(RCon, 0)
CON_Z = mkref(RCon, 1)


def TMP(x):
    return (RTmp << 29) | x


def CON(x):
    return (RCon << 29) | x


def SLOT(x):
    return (RSlot << 29) | (x & VMASK)


def TYPE(x):
    return (RType << 29) | x


def CALL(x):
    return (RCall << 29) | x


def MEM(x):
    return (RMem << 29) | x


def INT(x):
    return (RInt << 29) | (x & VMASK)


def req(a, b):
    return a == b


def rtype(r):
    if r == 0:
        return -1
    return r >> 29


def rval(r):
    """r.val"""
    return r & VMASK


def rkind(r):
    """r.type (raw, no special case for R)"""
    return r >> 29


def rsval(r):
    return ((r & VMASK) ^ 0x10000000) - 0x10000000


# comparisons
(Cieq, Cine, Cisge, Cisgt, Cisle, Cislt, Ciuge, Ciugt, Ciule, Ciult, NCmpI) = range(11)
(Cfeq, Cfge, Cfgt, Cfle, Cflt, Cfne, Cfo, Cfuo, NCmpF) = range(9)
NCmp = NCmpI + NCmpF

# classes
Kx, Kw, Kl, Ks, Kd = -1, 0, 1, 2, 3
Ke = -2
Km = Kl


def KWIDE(k):
    return k & 1


def KBASE(k):
    return k >> 1


# generated from QBE ops.h; (name, argcls, canfold, hasid, idval, commutes, assoc, idemp, cmpeqwl, cmplgtewl, eqval, pinned)
_OPS = [
    ('add', ((Kw, Kl, Ks, Kd), (Kw, Kl, Ks, Kd)), 1, 1, 0, 1, 1, 0, 0, 0, 0, 0),
    ('sub', ((Kw, Kl, Ks, Kd), (Kw, Kl, Ks, Kd)), 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    ('neg', ((Kw, Kl, Ks, Kd), (Kx, Kx, Kx, Kx)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('div', ((Kw, Kl, Ks, Kd), (Kw, Kl, Ks, Kd)), 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    ('rem', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('udiv', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    ('urem', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('mul', ((Kw, Kl, Ks, Kd), (Kw, Kl, Ks, Kd)), 1, 1, 1, 1, 0, 0, 0, 0, 0, 0),
    ('and', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 1, 0, 0, 1, 1, 1, 0, 0, 0, 0),
    ('or', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 1, 1, 0, 1, 1, 1, 0, 0, 0, 0),
    ('xor', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 1, 1, 0, 1, 1, 0, 0, 0, 0, 0),
    ('sar', ((Kw, Kl, Ke, Ke), (Kw, Kw, Ke, Ke)), 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    ('shr', ((Kw, Kl, Ke, Ke), (Kw, Kw, Ke, Ke)), 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    ('shl', ((Kw, Kl, Ke, Ke), (Kw, Kw, Ke, Ke)), 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    ('ceqw', ((Kw, Kw, Ke, Ke), (Kw, Kw, Ke, Ke)), 1, 1, 1, 1, 0, 0, 1, 0, 1, 0),
    ('cnew', ((Kw, Kw, Ke, Ke), (Kw, Kw, Ke, Ke)), 1, 1, 0, 1, 0, 0, 1, 0, 0, 0),
    ('csgew', ((Kw, Kw, Ke, Ke), (Kw, Kw, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 1, 0),
    ('csgtw', ((Kw, Kw, Ke, Ke), (Kw, Kw, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    ('cslew', ((Kw, Kw, Ke, Ke), (Kw, Kw, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 1, 0),
    ('csltw', ((Kw, Kw, Ke, Ke), (Kw, Kw, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    ('cugew', ((Kw, Kw, Ke, Ke), (Kw, Kw, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 1, 0),
    ('cugtw', ((Kw, Kw, Ke, Ke), (Kw, Kw, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    ('culew', ((Kw, Kw, Ke, Ke), (Kw, Kw, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 1, 0),
    ('cultw', ((Kw, Kw, Ke, Ke), (Kw, Kw, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    ('ceql', ((Kl, Kl, Ke, Ke), (Kl, Kl, Ke, Ke)), 1, 0, 0, 1, 0, 0, 1, 0, 1, 0),
    ('cnel', ((Kl, Kl, Ke, Ke), (Kl, Kl, Ke, Ke)), 1, 0, 0, 1, 0, 0, 1, 0, 0, 0),
    ('csgel', ((Kl, Kl, Ke, Ke), (Kl, Kl, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 1, 0),
    ('csgtl', ((Kl, Kl, Ke, Ke), (Kl, Kl, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    ('cslel', ((Kl, Kl, Ke, Ke), (Kl, Kl, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 1, 0),
    ('csltl', ((Kl, Kl, Ke, Ke), (Kl, Kl, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    ('cugel', ((Kl, Kl, Ke, Ke), (Kl, Kl, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 1, 0),
    ('cugtl', ((Kl, Kl, Ke, Ke), (Kl, Kl, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    ('culel', ((Kl, Kl, Ke, Ke), (Kl, Kl, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 1, 0),
    ('cultl', ((Kl, Kl, Ke, Ke), (Kl, Kl, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    ('ceqs', ((Ks, Ks, Ke, Ke), (Ks, Ks, Ke, Ke)), 1, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    ('cges', ((Ks, Ks, Ke, Ke), (Ks, Ks, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('cgts', ((Ks, Ks, Ke, Ke), (Ks, Ks, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('cles', ((Ks, Ks, Ke, Ke), (Ks, Ks, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('clts', ((Ks, Ks, Ke, Ke), (Ks, Ks, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('cnes', ((Ks, Ks, Ke, Ke), (Ks, Ks, Ke, Ke)), 1, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    ('cos', ((Ks, Ks, Ke, Ke), (Ks, Ks, Ke, Ke)), 1, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    ('cuos', ((Ks, Ks, Ke, Ke), (Ks, Ks, Ke, Ke)), 1, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    ('ceqd', ((Kd, Kd, Ke, Ke), (Kd, Kd, Ke, Ke)), 1, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    ('cged', ((Kd, Kd, Ke, Ke), (Kd, Kd, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('cgtd', ((Kd, Kd, Ke, Ke), (Kd, Kd, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('cled', ((Kd, Kd, Ke, Ke), (Kd, Kd, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('cltd', ((Kd, Kd, Ke, Ke), (Kd, Kd, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('cned', ((Kd, Kd, Ke, Ke), (Kd, Kd, Ke, Ke)), 1, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    ('cod', ((Kd, Kd, Ke, Ke), (Kd, Kd, Ke, Ke)), 1, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    ('cuod', ((Kd, Kd, Ke, Ke), (Kd, Kd, Ke, Ke)), 1, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    ('storeb', ((Kw, Ke, Ke, Ke), (Km, Ke, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('storeh', ((Kw, Ke, Ke, Ke), (Km, Ke, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('storew', ((Kw, Ke, Ke, Ke), (Km, Ke, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('storel', ((Kl, Ke, Ke, Ke), (Km, Ke, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('stores', ((Ks, Ke, Ke, Ke), (Km, Ke, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('stored', ((Kd, Ke, Ke, Ke), (Km, Ke, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('loadsb', ((Km, Km, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('loadub', ((Km, Km, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('loadsh', ((Km, Km, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('loaduh', ((Km, Km, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('loadsw', ((Km, Km, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('loaduw', ((Km, Km, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('load', ((Km, Km, Km, Km), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('extsb', ((Kw, Kw, Ke, Ke), (Kx, Kx, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('extub', ((Kw, Kw, Ke, Ke), (Kx, Kx, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('extsh', ((Kw, Kw, Ke, Ke), (Kx, Kx, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('extuh', ((Kw, Kw, Ke, Ke), (Kx, Kx, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('extsw', ((Ke, Kw, Ke, Ke), (Ke, Kx, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('extuw', ((Ke, Kw, Ke, Ke), (Ke, Kx, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('exts', ((Ke, Ke, Ke, Ks), (Ke, Ke, Ke, Kx)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('truncd', ((Ke, Ke, Kd, Ke), (Ke, Ke, Kx, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('stosi', ((Ks, Ks, Ke, Ke), (Kx, Kx, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('stoui', ((Ks, Ks, Ke, Ke), (Kx, Kx, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('dtosi', ((Kd, Kd, Ke, Ke), (Kx, Kx, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('dtoui', ((Kd, Kd, Ke, Ke), (Kx, Kx, Ke, Ke)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('swtof', ((Ke, Ke, Kw, Kw), (Ke, Ke, Kx, Kx)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('uwtof', ((Ke, Ke, Kw, Kw), (Ke, Ke, Kx, Kx)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('sltof', ((Ke, Ke, Kl, Kl), (Ke, Ke, Kx, Kx)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('ultof', ((Ke, Ke, Kl, Kl), (Ke, Ke, Kx, Kx)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('cast', ((Ks, Kd, Kw, Kl), (Kx, Kx, Kx, Kx)), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('alloc4', ((Ke, Kl, Ke, Ke), (Ke, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('alloc8', ((Ke, Kl, Ke, Ke), (Ke, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('alloc16', ((Ke, Kl, Ke, Ke), (Ke, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('vaarg', ((Km, Km, Km, Km), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('vastart', ((Km, Ke, Ke, Ke), (Kx, Ke, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('copy', ((Kw, Kl, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('dbgloc', ((Kw, Ke, Ke, Ke), (Kw, Ke, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('nop', ((Kx, Kx, Kx, Kx), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('addr', ((Km, Km, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('blit0', ((Km, Ke, Ke, Ke), (Km, Ke, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('blit1', ((Kw, Ke, Ke, Ke), (Kx, Ke, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('sel0', ((Kw, Ke, Ke, Ke), (Kx, Ke, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('sel1', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('swap', ((Kw, Kl, Ks, Kd), (Kw, Kl, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('sign', ((Kw, Kl, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('salloc', ((Ke, Kl, Ke, Ke), (Ke, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xidiv', ((Kw, Kl, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xdiv', ((Kw, Kl, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xcmp', ((Kw, Kl, Ks, Kd), (Kw, Kl, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xtest', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('acmp', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('acmn', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('afcmp', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('reqz', ((Kw, Kl, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('rnez', ((Kw, Kl, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('par', ((Kx, Kx, Kx, Kx), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('parsb', ((Kx, Kx, Kx, Kx), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('parub', ((Kx, Kx, Kx, Kx), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('parsh', ((Kx, Kx, Kx, Kx), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('paruh', ((Kx, Kx, Kx, Kx), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('parc', ((Ke, Kx, Ke, Ke), (Ke, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('pare', ((Ke, Kx, Ke, Ke), (Ke, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('arg', ((Kw, Kl, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('argsb', ((Kw, Ke, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('argub', ((Kw, Ke, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('argsh', ((Kw, Ke, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('arguh', ((Kw, Ke, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('argc', ((Ke, Kx, Ke, Ke), (Ke, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('arge', ((Ke, Kl, Ke, Ke), (Ke, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('argv', ((Kx, Kx, Kx, Kx), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('call', ((Km, Km, Km, Km), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('flagieq', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagine', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagisge', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagisgt', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagisle', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagislt', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagiuge', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagiugt', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagiule', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagiult', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagfeq', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagfge', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagfgt', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagfle', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagflt', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagfne', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagfo', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('flagfuo', ((Kx, Kx, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xselieq', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xseline', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xselisge', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xselisgt', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xselisle', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xselislt', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xseliuge', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xseliugt', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xseliule', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xseliult', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xselfeq', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xselfge', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xselfgt', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xselfle', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xselflt', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xselfne', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xselfo', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('xselfuo', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
]



class Op:
    __slots__ = ('name', 'argcls', 'canfold', 'hasid', 'idval', 'commutes', 'assoc',
                 'idemp', 'cmpeqwl', 'cmplgtewl', 'eqval', 'pinned')

    def __init__(self, name, argcls, *flags):
        self.name = name
        self.argcls = argcls
        (self.canfold, self.hasid, self.idval, self.commutes, self.assoc, self.idemp,
         self.cmpeqwl, self.cmplgtewl, self.eqval, self.pinned) = flags


Oxxx = 0
optab = [Op(None, ((Ke,) * 4, (Ke,) * 4), *([0] * 10))]
for _o in _OPS:
    globals()['O' + _o[0]] = len(optab)
    optab.append(Op(*_o))
NOp = len(optab)
del _o

_JMPS = ('retw retl rets retd retsb retub retsh retuh retc ret0 jmp jnz '
         'jfieq jfine jfisge jfisgt jfisle jfislt jfiuge jfiugt jfiule jfiult '
         'jffeq jffge jffgt jffle jfflt jffne jffo jffuo hlt').split()
Jxxx = 0
jtoa = [None] + _JMPS
for _i, _j in enumerate(_JMPS):
    globals()['J' + _j] = _i + 1
NJmp = len(_JMPS) + 1
del _i, _j

Ocmpw = Oceqw
Ocmpw1 = Ocultw
Ocmpl = Oceql
Ocmpl1 = Ocultl
Ocmps = Oceqs
Ocmps1 = Ocuos
Ocmpd = Oceqd
Ocmpd1 = Ocuod
Oalloc = Oalloc4
Oalloc1 = Oalloc16
Oflag = Oflagieq
Oflag1 = Oflagfuo
Oxsel = Oxselieq
Oxsel1 = Oxselfuo
NPubOp = Onop
Jjf = Jjfieq
Jjf1 = Jjffuo


def INRANGE(x, l, u):
    return l <= x <= u


def isstore(o): return Ostoreb <= o <= Ostored
def isload(o): return Oloadsb <= o <= Oload
def isalloc(o): return Oalloc4 <= o <= Oalloc16
def isext(o): return Oextsb <= o <= Oextuw
def ispar(o): return Opar <= o <= Opare
def isarg(o): return Oarg <= o <= Oargv
def isret(j): return Jretw <= j <= Jret0
def isparbh(o): return Oparsb <= o <= Oparuh
def isargbh(o): return Oargsb <= o <= Oarguh
def isretbh(j): return Jretsb <= j <= Jretuh
def isxsel(o): return Oxsel <= o <= Oxsel1


class Target:
    __slots__ = ('name', 'apple', 'windows', 'gpr0', 'ngpr', 'fpr0', 'nfpr', 'rglob',
                 'nrglob', 'rsave', 'nrsave', 'retregs', 'argregs', 'memargs', 'abi0',
                 'abi1', 'isel', 'emitfn', 'emitfin', 'asloc', 'assym', 'cansel')

    def __init__(self, **kw):
        self.name = ''
        self.apple = 0
        self.windows = 0
        self.assym = ''
        self.asloc = ''
        self.cansel = 0
        for k, v in kw.items():
            setattr(self, k, v)


class BSet:
    """Bitset; t is a Python int holding all words, nt the word count."""
    __slots__ = ('nt', 't')

    def __init__(self, nt=0, t=0):
        self.nt = nt
        self.t = t


class Ins:
    __slots__ = ('op', 'cls', 'to', 'arg')

    def __init__(self, op=0, cls=0, to=R, arg0=R, arg1=R):
        self.op = op
        self.cls = cls
        self.to = to
        self.arg = [arg0, arg1]

    def copy(self):
        return Ins(self.op, self.cls, self.to, self.arg[0], self.arg[1])

    def set(self, o):
        """*self = *o"""
        self.op = o.op
        self.cls = o.cls
        self.to = o.to
        self.arg[0] = o.arg[0]
        self.arg[1] = o.arg[1]

    def __repr__(self):
        return 'Ins(%s, %d, %#x, %#x, %#x)' % (optab[self.op].name, self.cls, self.to,
                                               self.arg[0], self.arg[1])


class Phi:
    __slots__ = ('to', 'cls', 'visit', 'narg', 'arg', 'blk', 'link')

    def __init__(self):
        self.to = R
        self.cls = 0
        self.visit = 0
        self.narg = 0
        self.arg = []
        self.blk = []
        self.link = None


class Jmp_:
    __slots__ = ('type', 'arg')

    def __init__(self):
        self.type = Jxxx
        self.arg = R


class Blk:
    __slots__ = ('phi', 'ins', 'nins', 'jmp', 's1', 's2', 'link', 'id', 'visit', 'idom',
                 'dom', 'dlink', 'fron', 'nfron', 'depth', 'pred', 'npred', 'in_', 'out',
                 'gen', 'nlive', 'loop', 'name')

    def __init__(self):
        self.phi = None
        self.ins = []
        self.nins = 0
        self.jmp = Jmp_()
        self.s1 = None
        self.s2 = None
        self.link = None
        self.id = 0
        self.visit = 0
        self.idom = None
        self.dom = None
        self.dlink = None
        self.fron = []
        self.nfron = 0
        self.depth = 0
        self.pred = []
        self.npred = 0
        self.in_ = BSet()
        self.out = BSet()
        self.gen = BSet()
        self.nlive = [0, 0]
        self.loop = 0
        self.name = None

    def __repr__(self):
        return 'Blk(@%s)' % self.name


UXXX, UPhi, UIns, UJmp = range(4)


class Use:
    __slots__ = ('type', 'bid', 'u')

    def __init__(self, type=UXXX, bid=0, u=None):
        self.type = type
        self.bid = bid
        self.u = u  # Ins or Phi


SGlo, SThr, SExt, SExtThr = 0, 1, 2, 3


class Sym:
    __slots__ = ('type', 'id')

    def __init__(self, type=SGlo, id=0):
        self.type = type
        self.id = id

    def copy(self):
        return Sym(self.type, self.id)


class Num:
    __slots__ = ('n', 'nl', 'nr', 'l', 'r')

    def __init__(self):
        self.n = 0
        self.nl = 0
        self.nr = 0
        self.l = R
        self.r = R


NoAlias, MayAlias, MustAlias = range(3)
ABot, ALoc, ACon, AEsc, ASym, AUnk = 0, 1, 2, 3, 4, 6


def astack(t):
    return t & 1


class Alias:
    """u.sym and u.loc share storage in C; here they are separate fields
    (sym, loc_sz, loc_m). slot is an Alias or None."""
    __slots__ = ('type', 'base', 'offset', 'sym', 'loc_sz', 'loc_m', 'slot')

    def __init__(self):
        self.type = ABot
        self.base = 0
        self.offset = 0
        self.sym = Sym()
        self.loc_sz = 0
        self.loc_m = 0
        self.slot = None

    def copy(self):
        a = Alias()
        a.set(self)
        return a

    def set(self, o):
        self.type = o.type
        self.base = o.base
        self.offset = o.offset
        self.sym = o.sym.copy()
        self.loc_sz = o.loc_sz
        self.loc_m = o.loc_m
        self.slot = o.slot


WFull, Wsb, Wub, Wsh, Wuh, Wsw, Wuw = range(7)


class Tmp:
    __slots__ = ('name', 'def_', 'use', 'ndef', 'nuse', 'bid', 'cost', 'slot', 'cls',
                 'hint_r', 'hint_w', 'hint_m', 'phi', 'alias', 'width', 'visit', 'gcmbid')

    def __init__(self):
        self.name = None
        self.def_ = None
        self.use = None  # NULL until filluse() (adduse is a no-op then)
        self.ndef = 0
        self.nuse = 0
        self.bid = 0
        self.cost = 0
        self.slot = 0
        self.cls = 0
        self.hint_r = 0
        self.hint_w = 0
        self.hint_m = 0
        self.phi = 0
        self.alias = Alias()
        self.width = WFull
        self.visit = 0
        self.gcmbid = 0


CUndef, CBits, CAddr = range(3)

_pk_q = struct.Struct('<q').pack
_up_q = struct.Struct('<q').unpack
_pk_d = struct.Struct('<d').pack
_up_d = struct.Struct('<d').unpack
_pk_f = struct.Struct('<f').pack
_up_f = struct.Struct('<f').unpack
_pk_I = struct.Struct('<I').pack
_up_I = struct.Struct('<I').unpack


def f32(x):
    """Round a Python float to float32 precision (overflow gives inf, as in C)."""
    try:
        return _up_f(_pk_f(x))[0]
    except OverflowError:
        return float('inf') if x > 0 else float('-inf')


def dbits(d):
    return _up_q(_pk_d(d))[0]


def bitsd(i):
    return _up_d(_pk_q(s64(i)))[0]


def fbits(f):
    return _up_I(_pk_f(f))[0]


def bitsf(i):
    return _up_f(_pk_I(i & U32))[0]


class Con:
    """bits is a union in C; it is stored as the int64 i, with d and s as views."""
    __slots__ = ('type', 'sym', 'i', 'flt')

    def __init__(self, type=CUndef, i=0):
        self.type = type
        self.sym = Sym()
        self.i = i
        self.flt = 0

    @property
    def d(self):
        return bitsd(self.i)

    @d.setter
    def d(self, v):
        self.i = dbits(v)

    @property
    def s(self):
        return bitsf(self.i)

    @s.setter
    def s(self, v):
        self.i = s64((self.i & ~U32) | fbits(v))

    def copy(self):
        c = Con(self.type, self.i)
        c.sym = self.sym.copy()
        c.flt = self.flt
        return c

    def set(self, o):
        self.type = o.type
        self.sym = o.sym.copy()
        self.i = o.i
        self.flt = o.flt


class Addr:
    __slots__ = ('offset', 'base', 'index', 'scale')

    def __init__(self):
        self.offset = Con()
        self.base = R
        self.index = R
        self.scale = 0

    def copy(self):
        a = Addr()
        a.offset = self.offset.copy()
        a.base = self.base
        a.index = self.index
        a.scale = self.scale
        return a


Mem = Addr


class Lnk:
    __slots__ = ('export', 'thread', 'common', 'align', 'sec', 'secf')

    def __init__(self):
        self.export = 0
        self.thread = 0
        self.common = 0
        self.align = 0
        self.sec = None
        self.secf = None

    def copy(self):
        l = Lnk()
        for k in Lnk.__slots__:
            setattr(l, k, getattr(self, k))
        return l


class Fn:
    __slots__ = ('start', 'tmp', 'con', 'mem', 'ntmp', 'ncon', 'nmem', 'nblk', 'retty',
                 'retr', 'rpo', 'reg', 'slot', 'salign', 'vararg', 'dynalloc', 'leaf',
                 'name', 'lnk')

    def __init__(self):
        self.start = None
        self.tmp = []
        self.con = []
        self.mem = []
        self.ntmp = 0
        self.ncon = 0
        self.nmem = 0
        self.nblk = 0
        self.retty = 0
        self.retr = R
        self.rpo = []
        self.reg = 0
        self.slot = 0
        self.salign = 0
        self.vararg = 0
        self.dynalloc = 0
        self.leaf = 0
        self.name = None
        self.lnk = Lnk()


FEnd, Fb, Fh, Fw, Fl, Fs, Fd, FPad, FTyp = range(9)


class Field:
    __slots__ = ('type', 'len')

    def __init__(self, type=FEnd, len=0):
        self.type = type
        self.len = len


class Typ:
    """fields: list (one per union member) of lists of NField+1 Field."""
    __slots__ = ('name', 'isdark', 'isunion', 'align', 'size', 'nunion', 'fields')

    def __init__(self):
        self.name = None
        self.isdark = 0
        self.isunion = 0
        self.align = 0
        self.size = 0
        self.nunion = 0
        self.fields = []


DStart, DEnd, DB, DH, DW, DL, DZ = range(7)


class Dat:
    """u is a union in C: num/fltd/flts/str/ref_name+ref_off; all kept as fields."""
    __slots__ = ('type', 'name', 'lnk', 'num', 'fltd', 'flts', 'str', 'ref_name',
                 'ref_off', 'isref', 'isstr')

    def __init__(self):
        self.type = DStart
        self.name = None
        self.lnk = None
        self.num = 0
        self.fltd = 0.0
        self.flts = 0.0
        self.str = None
        self.ref_name = None
        self.ref_off = 0
        self.isref = 0
        self.isstr = 0


# globals (main.c / util.c)
T = None
debug = {c: 0 for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
typ = []
insb = [None] * NIns
curi = NIns


class QbeError(Exception):
    """err() / die(): message is printed by the driver, which exits with status 1."""
inpath = ''
lnum = 0
