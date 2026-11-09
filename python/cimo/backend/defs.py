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
Kq = 5  # cimo (ext.vector): 128 bits, in the FP registers; odd, so wide
Ke = -2
Km = Kl


def KWIDE(k):
    return k & 1


def KBASE(k):
    return k >> 1 if k != Kq else 1


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
    ('sel1', ((Kw, Kl, Ks, Kd), (Kw, Kl, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
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
    # cimo extensions (see ext.py); after all QBE ops so QBE's op numbers are unchanged.
    # fused multiply-add: "fma0 c" then "r = fma1 a, b" computes r = a*b + c, rounded once
    ('fma0', ((Ke, Ke, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ('fma1', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    # arm64 after isel: r = afmadd a, MEM{base=b, index=c} (the Mem only carries the third
    # register through liveness, spilling and allocation, which all handle Mem base/index)
    ('afmadd', ((Ke, Ke, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('afmsub', ((Ke, Ke, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('afnmadd', ((Ke, Ke, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('afnmsub', ((Ke, Ke, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    # math builtins (ext.math): C's sqrt, fabs, fmin, fmax, floor, ceil, trunc, round, rint, nearbyint, one
    # instruction each on arm64; pure, so gvn merges and gcm hoists them
    ('fsqrt', ((Ke, Ke, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('fabs', ((Ke, Ke, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('fmin', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('fmax', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('ffloor', ((Ke, Ke, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('fceil', ((Ke, Ke, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('ftrunc', ((Ke, Ke, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('fround', ((Ke, Ke, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('frint', ((Ke, Ke, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ('fnearbyint', ((Ke, Ke, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    # volatile accesses (ext.volatile): loads and stores that no pass may delete, merge, move,
    # forward or promote; pinned, and unknown to the memory passes (so their address escapes).
    # arm64 isel and the wasm emitter turn them into plain loads and stores.
]
_OPS += [('v' + _o[0],) + _o[1:] for _o in _OPS if _o[0].startswith('load')]
_OPS += [('v' + _o[0],) + _o[1:] for _o in _OPS if _o[0].startswith('store')]
# the stack pointer (ext.vla): "r =l getsp" reads it, "setsp r" sets it back to a value getsp
# returned earlier in the function, which frees the variable-length arrays allocated since;
# both pinned and mark the function dynalloc
_OPS += [('getsp', ((Kx, Kx, Kx, Kx), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('setsp', ((Kl, Kl, Kl, Kl), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)]
# bit builtins (ext.math): count leading/trailing zeros, population count, byte swap of the
# operand's class (bswap16: the low 16 bits of a w); pure, like the math builtins
_OPS += [(_n, ((Kw, Kl, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
         for _n in ('bclz', 'bctz', 'bpopcnt', 'bswap')]
_OPS += [('bswap16', ((Kw, Ke, Ke, Ke), (Kx, Kx, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
# _Float16 (ext.math, arm64 for now): "%f =s h2f %w", a half's bits (the low 16 of w) as a float, exact; "%w =w f2h %s",
# a float rounded to a half in the current rounding mode, its bits zero-extended; pure
_OPS += [('h2f', ((Ke, Ke, Kw, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         ('f2h', ((Ks, Ke, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
# the high half of a 64-bit product (ext.cse, division by a constant); pure
_OPS += [(_n, ((Ke, Kl, Ke, Ke), (Ke, Kl, Ke, Ke)), 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
         for _n in ('smulh', 'umulh')]
# arm64 after isel (ext.isel): r = amadd c, MEM{base=a, index=b} is c + a*b, amsub c - a*b (as
# afmadd, the Mem only carries the third register)
_OPS += [(_n, ((Kw, Kl, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
         for _n in ('amadd', 'amsub')]
# r = aalu a, MEM{base=b, scale=alu | shift kind << 4, offset=amount}: see isel.selalu (a may be
# the constant 0: xzr). aubfx, asbfx: r = bits of a at CON(lsb | width << 8). aror a, CON(n). fnmul
# a, b: -(a * b). afcon CON(bits): an FP constant that fits fmov
_OPS += [('aalu', ((Kw, Kl, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         ('aubfx', ((Kw, Kl, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         ('asbfx', ((Kw, Kl, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         ('aror', ((Kw, Kl, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         ('afnmul', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         ('afcon', ((Kx, Kx, Kx, Kx), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         ('aubfiz', ((Kw, Kl, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         # accmp a, MEM{base=b or none with offset=immediate, scale=nzcv | cond << 4 | inverted << 12}
         ('accmp', ((Kw, Kl, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         ('atst', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         ('asmull', ((Ke, Kw, Ke, Ke), (Ke, Kw, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         ('aumull', ((Ke, Kw, Ke, Ke), (Ke, Kw, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         # r = abfi a, MEM{base=b, scale=b's lsb, offset=lsb | width << 8}: a with a field from b
         ('abfi', ((Kw, Kl, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         # add, sub, and that also set the flags (emit._flagsfrom)
         ('aadds', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         ('asubs', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         ('aands', ((Kw, Kl, Ke, Ke), (Kw, Kl, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
# r = acsel a, MEM{base=b, offset=cond}: a if the condition holds on the flags, else b. One op for
# every combination of the result's class and the comparison's kind, unlike xsel*
_OPS += [('acsel', ((Kw, Kl, Ks, Kd), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]



class Op:
    __slots__ = ('name', 'argcls', 'canfold', 'hasid', 'idval', 'commutes', 'assoc',
                 'idemp', 'cmpeqwl', 'cmplgtewl', 'eqval', 'pinned')

    def __init__(self, name, argcls, *flags):
        self.name = name
        # room for class q, whose ops write their own tables
        self.argcls = tuple(a if len(a) == 6 else (a + (Ke, Kq if name in ('copy', 'swap') else Ke)) for a in argcls)
        (self.canfold, self.hasid, self.idval, self.commutes, self.assoc, self.idemp,
         self.cmpeqwl, self.cmplgtewl, self.eqval, self.pinned) = flags

# cimo (ext.vector): operations on a 128-bit value (class q), their lanes in the name: b, h, w, l of 8 to 64
# bits and s, d of floats: qaddw is 4 x i32. Pure, except qload and qstore, which passes treat as volatile.
# qfma0 %c, then %r =q qfma1%lane %a, %b, as fma0 and fma1. qsel0 %mask, then %r =q qsel1 %a, %b: bits of a
# where the mask has them, else of b. qdup: all lanes. qget %v, lane: one lane. Shifts by a constant.
_QQ = lambda: ((Ke, Ke, Ke, Ke, Ke, Kq), (Ke, Ke, Ke, Ke, Ke, Kq))
_Q1 = lambda: ((Ke, Ke, Ke, Ke, Ke, Kq), (Kx,) * 6)
_QS = lambda: ((Ke, Ke, Ke, Ke, Ke, Kq), (Ke, Ke, Ke, Ke, Ke, Kw))
_QOPS = [(n + x, _QQ()) for n, lanes in (('qadd', 'bhwlsd'), ('qsub', 'bhwlsd'), ('qmul', 'bhwsd'), ('qdiv', 'sd'),
                                         ('qmin', 'bhwsd'), ('qmax', 'bhwsd'), ('qumin', 'bhw'), ('qumax', 'bhw'),
                                         ('qceq', 'bhwlsd'), ('qcgt', 'bhwlsd'), ('qcge', 'bhwlsd'),
                                         ('qcugt', 'bhwl'), ('qcuge', 'bhwl')) for x in lanes]
_QOPS += [(n + x, _Q1()) for n, lanes in (('qneg', 'bhwlsd'), ('qabs', 'sd'), ('qsqrt', 'sd')) for x in lanes]
_QOPS += [(n + x, _QS()) for n in ('qshl', 'qshr', 'qsar') for x in 'bhwl']
# conversions (GNU vector types' __builtin_convertvector, arm64 only for now): qitof/quitof s or d, a signed or unsigned
# int of the same width to float; qftoi/qftou, float to int, truncated. qsxl/quxl b, h, w: the low half's lanes sign- or
# zero-extended to twice the width, qsxh/quxh the high half's; qfcvtl/qfcvth, the low or high 2 floats to doubles.
# %r =q qnrwh %lo, %hi: the lanes of lo then of hi, truncated to half the width (h: to bytes); qfcvtn, doubles to floats
_QOPS += [(n + x, _Q1()) for n in ('qitof', 'quitof', 'qftoi', 'qftou') for x in 'sd']
_QOPS += [(n + x, _Q1()) for n in ('qsxl', 'qsxh', 'quxl', 'quxh') for x in 'bhw']
_QOPS += [('qfcvtl', _Q1()), ('qfcvth', _Q1()), ('qfcvtn', _QQ())] + [('qnrw' + x, _QQ()) for x in 'hwl']
# %r =q qsshlvw %v, %n: each lane shifted left by n's lane (signed: right where negative, arithmetic); qushlv, the
# same with logical shifts right (arm64 sshl, ushl; GNU vectors shifted by a variable)
_QOPS += [(n + x, _QQ()) for n in ('qsshlv', 'qushlv') for x in 'bhwl']
# x86-64 AVX2's, by direction: qshlv, qshrv (logical) w and l, qsarvw (vpsllv*, vpsrlv*, vpsravd)
_QOPS += [('qshlvw', _QQ()), ('qshlvl', _QQ()), ('qshrvw', _QQ()), ('qshrvl', _QQ()), ('qsarvw', _QQ())]
# permutations (arm64; __builtin_shufflevector's one-instruction cases), the lane size in the name: qzip1w %a, %b
# interleaves the low halves, qzip2 the high; quzp1/quzp2 the even or odd lanes of a then b; qtrn1/qtrn2 the even or odd
# lanes of each pair; qext<k> %a, %b: bytes k to 15 of a then 0 to k - 1 of b; %r =q qbcastw %v, i: lane i everywhere
_QOPS += [(n + x, _QQ()) for n in ('qzip1', 'qzip2', 'quzp1', 'quzp2', 'qtrn1', 'qtrn2') for x in 'bhwl']
_QOPS += [('qext%d' % k, _QQ()) for k in range(1, 16)] + [('qbcast' + x, _QS()) for x in 'bhwl']
# %r =q qtbl %v, %idx: byte k of r is byte idx[k] of v, or 0 where idx[k] > 15 (arm64 tbl; __builtin_shufflevector)
_QOPS += [('qtbl', _QQ())]
# %r =q qsaddl %a, %b: a's 2 x i64 + the sign-extended low 2 x i32 of b; h: the high ones; u: zero-extended
_QOPS += [('qsaddl', _QQ()), ('qsaddh', _QQ()), ('quaddl', _QQ()), ('quaddh', _QQ())]
_QOPS += [('qand', _QQ()), ('qor', _QQ()), ('qxor', _QQ()), ('qnot', _Q1()),
          ('qdupb', ((Ke, Ke, Ke, Ke, Ke, Kw), (Kx,) * 6)), ('qduph', ((Ke, Ke, Ke, Ke, Ke, Kw), (Kx,) * 6)),
          ('qdupw', ((Ke, Ke, Ke, Ke, Ke, Kw), (Kx,) * 6)), ('qdupl', ((Ke, Ke, Ke, Ke, Ke, Kl), (Kx,) * 6)),
          ('qdups', ((Ke, Ke, Ke, Ke, Ke, Ks), (Kx,) * 6)), ('qdupd', ((Ke, Ke, Ke, Ke, Ke, Kd), (Kx,) * 6))]
_QOPS += [('qget' + x, ((Kq if k == Kw else Ke, Kq if k == Kl else Ke, Kq if k == Ks else Ke, Kq if k == Kd else Ke, Ke, Ke),
                        (Kw, Kw, Kw, Kw, Ke, Ke)))
          for x, k in (('b', Kw), ('h', Kw), ('w', Kw), ('l', Kl), ('s', Ks), ('d', Kd))]
# x86-64's own, which amd64/qlower.py writes what SSE2 lacks with; not in the IR text. qxshufd %v, imm: pshufd. qxmuludq: 2 x
# (low 32 bits of each 64 multiplied). qxandn %a, %b: ~a & b. qxlo/qxhi %a, %b: the low or high 2 x i32 of a and b, interleaved.
_QOPS += [('qxshufd', _QS())]
_QOPS += [(n, _QQ()) for n in ('qxmuludq', 'qxandn', 'qxlo', 'qxhi', 'qxunords', 'qxunordd', 'qxmins', 'qxmind', 'qxmaxs', 'qxmaxd')]
# AVX's vroundpd: toward zero, toward minus infinity (qlower's double to 64-bit int)
_QOPS += [('qxtruncd', _Q1()), ('qxfloord', _Q1())]
_OPS += [(n, a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) for n, a in _QOPS]
_OPS += [('qsel0', ((Ke, Ke, Ke, Ke, Ke, Kq), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qfma0', ((Ke, Ke, Ke, Ke, Ke, Kq), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qfma1s', _QQ(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1), ('qfma1d', _QQ(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         # dot products into 32-bit lanes, as qfma1 after a qfma0 of the accumulator (the CPU intrinsics of the bundled
         # arm_neon.h and immintrin.h): qbfdot, each float lane + the products of its two bf16 pairs (arm64 bfdot, x86-64
         # vdpbf16ps); qsdot, qudot, each int32 lane + the products of its four signed or unsigned bytes (sdot, udot);
         # qusdot, unsigned bytes of a times signed bytes of b (usdot, vpdpbusd)
         ('qbfdot', _QQ(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1), ('qsdot', _QQ(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qudot', _QQ(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1), ('qusdot', _QQ(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         # pinned, as qfma1 and sel1: gvn and gcm must never merge or move it away from its qsel0
         ('qsel1', _QQ(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qload', ((Ke, Ke, Ke, Ke, Ke, Km), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qstore', ((Kq,) * 6, (Km,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         # 16 bytes with AVX2's 32-byte q (x86-64; GNU vectors of 16 bytes): qload16 puts them in both halves
         # (vbroadcasti128: the upper lanes repeat the lower, so they raise no exception the lower don't), qstore16 stores
         # the lower half
         ('qload16', ((Ke, Ke, Ke, Ke, Ke, Km), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qstore16', ((Kq,) * 6, (Km,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         # the same for 32 bytes with AVX-512's 64-byte q (vbroadcasti64x4, and a ymm stored)
         ('qload32', ((Ke, Ke, Ke, Ke, Ke, Km), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qstore32', ((Kq,) * 6, (Km,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         # x86-64's own (amd64 isel, with fma): qfma1 with its second operand loaded from memory, %r =q qxfmams %a, %addr
         ('qxfmams', ((Ke, Ke, Ke, Ke, Ke, Kq), (Ke, Ke, Ke, Ke, Ke, Km)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qxfmamd', ((Ke, Ke, Ke, Ke, Ke, Kq), (Ke, Ke, Ke, Ke, Ke, Km)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         # made by isel (util.dupfold): qdup of a value loaded only for it, from memory, %r =q qdupmw %addr
         # (x86-64 AVX2 vpbroadcastd mem, arm64 ldr s + dup: no move from a general register)
         ('qdupmb', ((Ke, Ke, Ke, Ke, Ke, Km), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qdupmh', ((Ke, Ke, Ke, Ke, Ke, Km), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qdupmw', ((Ke, Ke, Ke, Ke, Ke, Km), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qdupml', ((Ke, Ke, Ke, Ke, Ke, Km), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qdupms', ((Ke, Ke, Ke, Ke, Ke, Km), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qdupmd', ((Ke, Ke, Ke, Ke, Ke, Km), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         # bytes to and from 16-bit lanes (vectorize's narrow loops): %r =q qldbh %addr, a byte a lane zero-extended;
         # %r =q qld3bh %addr, k, byte k of each 3 (rgb pixels, channel k); qstbh %v, %addr, each lane's low byte
         ('qldbh', ((Ke, Ke, Ke, Ke, Ke, Km), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qld3bh', ((Ke, Ke, Ke, Ke, Ke, Km), (Ke, Ke, Ke, Ke, Ke, Kw)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qstbh', ((Kq,) * 6, (Km,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         # _Float16 arrays (arm64): %r =q qldh %addr, 4 halves widened to floats (ldr d, fcvtl); qsth %v, %addr, the
         # 4 floats rounded to halves and stored (fcvtn, str d)
         ('qldh', ((Ke, Ke, Ke, Ke, Ke, Km), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qsth', ((Kq,) * 6, (Km,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         # arm64's own: qld3x %addr, ld3 of 8 pixels into v28-v30; %r =q qwide3 k, channel k zero-extended to 16 bits;
         # %r =q qwmul3 %c, k, channel k times the bytes of %c (umull); qwmla0 %acc, %r =q qwmla3 %c, k: + %acc (umlal)
         ('qld3x', ((Km,) * 6, (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qwide3', ((Ke, Ke, Ke, Ke, Ke, Kw), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qwmul3', ((Ke, Ke, Ke, Ke, Ke, Kq), (Ke, Ke, Ke, Ke, Ke, Kw)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qwmla0', ((Ke, Ke, Ke, Ke, Ke, Kq), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qwmla3', ((Ke, Ke, Ke, Ke, Ke, Kq), (Ke, Ke, Ke, Ke, Ke, Kw)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         # qshrn0 %v, then qstshrn %addr, s: each 16-bit lane of %v shifted right by s, its low byte stored (shrn)
         ('qshrn0', ((Kq,) * 6, (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qstshrn', ((Km,) * 6, (Kw,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         # x86-64's own, for pixels: %r =q qxwin %addr, off, the 16 bytes at addr + off (with AVX2 those at + 24 too, in the
         # upper half); %r =q qxshufb %v, m, vpshufb with mask m (2 * channel + 0 or 1 for the window at 0 or 8)
         ('qxwin', ((Ke, Ke, Ke, Ke, Ke, Km), (Ke, Ke, Ke, Ke, Ke, Kw)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qxshufb', ((Ke, Ke, Ke, Ke, Ke, Kq), (Ke, Ke, Ke, Ke, Ke, Kw)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         # arm64, 16 pixels: qld3x16 %addr, ld3 of 16 pixels into v28-v30; qwide3h, qwmul3h, qwmla3h as qwide3, qwmul3,
         # qwmla3 on the upper 8 (ushll2, umull2, umlal2)
         ('qld3x16', ((Km,) * 6, (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qwide3h', ((Ke, Ke, Ke, Ke, Ke, Kw), (Kx,) * 6), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qwmul3h', ((Ke, Ke, Ke, Ke, Ke, Kq), (Ke, Ke, Ke, Ke, Ke, Kw)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('qwmla3h', ((Ke, Ke, Ke, Ke, Ke, Kq), (Ke, Ke, Ke, Ke, Ke, Kw)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)]
# cimo: x86-64's read-modify-write, made by amd64 isel from a load, an operation and a store to one address: xaddm %v, %a
# is add %v to the word or long at %a, of the instruction's class
_OPS += [('x%sm' % n, ((Kw, Kl, Ke, Ke, Ke, Ke), (Km, Km, Ke, Ke, Ke, Ke)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
         for n in ('add', 'sub', 'and', 'or', 'xor')]
# cimo: the floating-point environment (ext.math, <fenv.h>): "%r =w rdfpcr" reads the control register, rdfpsr the
# status (arm64 FPCR and FPSR; x86-64 MXCSR for both), "wrfpcr %v" and "wrfpsr %v" write them; pinned. xcmpe (x86-64
# comis) and afcmpq (arm64 fcmp): the float comparisons, signaling and quiet, beside xcmp (quiet) and afcmp (signaling)
_OPS += [('rdfpcr', ((Kx, Kx, Kx, Kx), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('rdfpsr', ((Kx, Kx, Kx, Kx), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('wrfpcr', ((Kw, Kw, Kw, Kw), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('wrfpsr', ((Kw, Kw, Kw, Kw), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
         ('xcmpe', ((Kw, Kl, Ks, Kd), (Kw, Kl, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
         ('afcmpq', ((Ke, Ke, Ks, Kd), (Ke, Ke, Ks, Kd)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
# cimo (cimo.gpu): "%r =w gpuid N", a thread, block or grid index or size on the GPU (cimo.gpu.backend.ptx); pure
_OPS += [('gpuid', ((Kw, Ke, Ke, Ke), (Kx, Kx, Kx, Kx)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
del _QQ, _Q1, _QS


Oxxx = 0
optab = [Op(None, ((Ke,) * 6, (Ke,) * 6), *([0] * 10))]
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
def isvload(o): return Ovloadsb <= o <= Ovload or o in (Oqload, Oqldbh, Oqld3bh, Oqld3x, Oqxwin, Oqld3x16, Oqldh, Oqload16, Oqload32)  # vector memory: as careful as volatile, for now
def isvstore(o): return Ovstoreb <= o <= Ovstored or o in (Oqstore, Oqstbh, Oqstshrn, Oqsth, Oqstore16, Oqstore32)
def isqop(o): return Oqaddb <= o <= Oqwmla3h


# cimo: the second of a qfma0 pair, which reads its destination too (isel made it a copy of the accumulator)
QACC = frozenset((Oqfma1s, Oqfma1d, Oqbfdot, Oqsdot, Oqudot, Oqusdot))


def unvolatile(fn, sink=None):
    """Turn the volatile loads and stores of fn into plain ones (ext.volatile), once the
    optimizations are done (uses filled in). sink: a register that register allocation never
    treats as dead (arm64: x17); a volatile load whose result is unused loads into it as an
    integer of the same size, since rega deletes instructions that define dead temporaries.
    sink='use' (amd64, which has no such register): the result is kept, and a `copy` to
    nothing after the load uses it (as amd64 isel keeps a flag result live)."""
    b = fn.start
    while b:
        extra = None
        for k, i in enumerate(b.ins[:b.nins]):
            if Ovloadsb <= i.op <= Ovload:
                fn.hasvolatile = True
                i.op = Oloadsb + i.op - Ovloadsb
                if sink is not None and rtype(i.to) == RTmp and fn.tmp[rval(i.to)].nuse == 0:
                    if sink == 'use':
                        fn.tmp[rval(i.to)].nuse = 1
                        extra = extra or []
                        extra.append(k)
                    else:
                        i.to = sink
                        i.cls = Kw if i.cls in (Kw, Ks) else Kl
            elif Ovstoreb <= i.op <= Ovstored:
                fn.hasvolatile = True
                i.op = Ostoreb + i.op - Ovstoreb
        if extra:
            ins = []
            for k, i in enumerate(b.ins[:b.nins]):
                ins.append(i)
                if extra and k == extra[0]:
                    extra.pop(0)
                    ins.append(Ins(Ocopy, i.cls, R, i.to))
            b.ins = ins
            b.nins = len(ins)
        b = b.link
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
                 'abi1', 'isel', 'emitfn', 'emitfin', 'asloc', 'assym', 'cansel', 'rqavoid')

    def __init__(self, **kw):
        self.rqavoid = 0  # cimo (ext.vector): registers of which a call keeps only part
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
    __slots__ = ('op', 'cls', 'to', 'arg', 'nsw', 'contract', 'reassoc')

    def __init__(self, op=0, cls=0, to=R, arg0=R, arg1=R):
        self.op = op
        self.cls = cls
        self.to = to
        self.arg = [arg0, arg1]
        self.nsw = False  # cimo (ext.nowrap): C says this signed operation doesn't overflow
        self.contract = False  # cimo (ext.fma), on fma1: contracted from a*b + c (IR "fmac"), so it may be split again;
        # an fma() call ("fma") never is. Copies keep it; a lost mark only leaves a product fused
        self.reassoc = False  # cimo, on a float add, sub, mul or fma1: a reduction `#pragma omp simd reduction` names
        # (IR "addassoc", ...), which the vectorizer may sum in any order. A lost mark only keeps C's order

    def copy(self):
        i = Ins(self.op, self.cls, self.to, self.arg[0], self.arg[1])
        i.nsw = self.nsw
        i.contract = self.contract
        i.reassoc = self.reassoc
        return i

    def set(self, o):
        """*self = *o"""
        self.op = o.op
        self.cls = o.cls
        self.to = o.to
        self.arg[0] = o.arg[0]
        self.arg[1] = o.arg[1]
        self.nsw = o.nsw
        self.contract = o.contract
        self.reassoc = o.reassoc

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
    __slots__ = ('export', 'thread', 'common', 'align', 'sec', 'secf', 'qbytes')

    def __init__(self):
        self.qbytes = 0  # cimo (ext.vector): "vector 64 function": the bytes of this function's q values (0: ext's)
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
                 'name', 'lnk', 'concache', 'hasvolatile', 'retp')

    def __init__(self):
        self.retp = False  # cimo: returns a pointer (class p: wasm32's i32 at the boundary)
        self.start = None
        self.tmp = []
        self.con = []
        self.concache = None  # util._concache: lookup index over con (not in C)
        # cimo (ext.volatile): set by unvolatile: volatile accesses are plain ones from there on,
        # so arm64 pairs (ldp/stp) no access in this function, which could reorder or merge them
        self.hasvolatile = False
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
    __slots__ = ('name', 'isdark', 'isunion', 'align', 'size', 'nunion', 'fields', 'isvec')

    def __init__(self):
        self.name = None
        self.isvec = 0  # cimo: a GNU vector type ("vector" after align): AAPCS64 passes 8 and 16 bytes in one V register
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


class BackendError(Exception):
    """err() / die(): message is printed by the driver, which exits with status 1."""
inpath = ''
lnum = 0
