"""Helpers for editing the backend IR (`backend.defs`) from the middle end.

Conventions: blocks keep `ins` as a plain list with `nins == len(ins)` once the middle end has
touched them; every function here leaves the CFG (preds, rpo, dominators) stale, so callers
finish with `refresh(fn)`.
"""

from ..backend import defs as D
from ..backend.defs import (Blk, Ins, Phi, R, RTmp, RCon, CBits, CAddr, Kw, Kl, Ks, Kd,
                            Jjmp, Jjnz, rtype, rval, s64, s32, U32)
from ..backend.util import newtmp, getcon, newcon
from ..backend import cfg as C
from ..backend import ssa as S

_nblk = [0]


class Bail(Exception):
    """A transformation found something it can't handle; the function is left unchanged."""


def refresh(fn):
    """Recompute rpo, preds, uses and dominators after an edit."""
    for b in blocks(fn):
        b.ins = b.ins[:b.nins]
    C.fillcfg(fn)
    S.filluse(fn)
    C.filldom(fn)


def blocks(fn):
    b = fn.start
    while b:
        yield b
        b = b.link


def newblock(fn, after, tag='m'):
    """A new empty block linked into fn's block list right after `after`."""
    _nblk[0] += 1
    b = Blk()
    b.name = '%s.%d' % (tag, _nblk[0])
    b.ins = []
    b.nins = 0
    b.link = after.link
    after.link = b
    return b


def phis(b):
    p = b.phi
    while p:
        yield p
        p = p.link


def ins(b):
    return b.ins[:b.nins]


def setins(b, lst):
    b.ins = lst
    b.nins = len(lst)


def succs(b):
    out = []
    if b.s1:
        out.append(b.s1)
    if b.s2 and b.s2 is not b.s1:
        out.append(b.s2)
    return out


def setjmp(b, target):
    b.jmp.type = Jjmp
    b.jmp.arg = R
    b.s1 = target
    b.s2 = None


def setjnz(b, cond, t, f):
    b.jmp.type = Jjnz
    b.jmp.arg = cond
    b.s1 = t
    b.s2 = f


def retarget(b, old, new):
    """Make b's jumps to `old` go to `new`."""
    if b.s1 is old:
        b.s1 = new
    if b.s2 is old:
        b.s2 = new


def addphi(b, cls, to, pairs):
    """Prepend a phi `to = phi [(pred, val)...]` to b."""
    p = Phi()
    p.to = to
    p.cls = cls
    p.narg = len(pairs)
    p.blk = [x for x, _ in pairs]
    p.arg = [v for _, v in pairs]
    p.link = b.phi
    b.phi = p
    return p


def phiargof(p, blk):
    for n in range(p.narg):
        if p.blk[n] is blk:
            return p.arg[n]
    raise KeyError(blk.name)


def setphiarg(p, blk, val):
    for n in range(p.narg):
        if p.blk[n] is blk:
            p.arg[n] = val
            return
    raise KeyError(blk.name)


def renamepred(b, old, new):
    """Phis of b: the argument coming from `old` now comes from `new`."""
    for p in phis(b):
        for n in range(p.narg):
            if p.blk[n] is old:
                p.blk[n] = new


class Emitter:
    """Appends instructions to a block, with constant folding of integer arithmetic."""

    def __init__(self, fn, blk):
        self.fn = fn
        self.blk = blk
        self.out = blk.ins[:blk.nins]

    def flush(self):
        setins(self.blk, self.out)

    def tmp(self, cls):
        return newtmp('m', cls, self.fn)

    def con(self, v):
        return getcon(s64(v), self.fn)

    def raw(self, op, cls, a0=R, a1=R, to=None):
        if to is None:
            to = self.tmp(cls) if cls is not None else R
        self.out.append(Ins(op, cls if cls is not None else Kw, to, a0, a1))
        return to

    def ins(self, i):
        self.out.append(i)

    def op(self, op, cls, a, b=R):
        """Integer op with folding when both operands are constants (and identities)."""
        ca = conval(self.fn, a)
        cb = conval(self.fn, b) if b != R else None
        if cls == Kw and op in (D.Oadd, D.Osub, D.Omul, D.Oand, D.Oor, D.Oxor, D.Oneg):
            # w constants as small signed values (cproc writes -1 as 4294967295)
            if ca is not None:
                ca = s32(ca)
                a = self.con(ca)
            if cb is not None:
                cb = s32(cb)
                b = self.con(cb)
        if cls in (Kw, Kl):
            if ca is not None and (b == R or cb is not None):
                v = fold(op, cls, ca, cb)
                if v is not None:
                    return self.con(v)
            if ca is not None and cb is None and b != R:
                # constants go second (the arm64 isel only takes immediates there)
                if op in (D.Oadd, D.Omul, D.Oand, D.Oor, D.Oxor):
                    a, b, ca, cb = b, a, cb, ca
                elif op == D.Osub:
                    return self.op(D.Oadd, cls, self.raw(D.Oneg, cls, b), a)
            if (cls == Kw and op in (D.Oadd, D.Osub) and cb is not None
                    and -(1 << 31) < cb < 0):
                # see normalize_wcons
                return self.raw(D.Osub if op == D.Oadd else D.Oadd, cls, a, self.con(-cb))
            if op == D.Oadd:
                if cb == 0:
                    return a if cls == Kl or self.fn_cls(a) == Kw else self.raw(D.Ocopy, cls, a)
                if ca == 0:
                    return b if cls == Kl or self.fn_cls(b) == Kw else self.raw(D.Ocopy, cls, b)
            if op == D.Osub and cb == 0:
                return a if cls == Kl or self.fn_cls(a) == Kw else self.raw(D.Ocopy, cls, a)
            if op == D.Omul:
                if cb == 1:
                    return a if cls == Kl or self.fn_cls(a) == Kw else self.raw(D.Ocopy, cls, a)
                if ca == 1:
                    return b if cls == Kl or self.fn_cls(b) == Kw else self.raw(D.Ocopy, cls, b)
        return self.raw(op, cls, a, b)

    def cmp(self, op, a, b):
        """Comparison (w result), folded when both operands are constants."""
        ca, cb = conval(self.fn, a), conval(self.fn, b)
        if ca is not None and cb is not None:
            from ..backend.util import iscmp
            k, c = iscmp(op)
            r = cmpfold(c, k, ca, cb)
            if r is not None:
                return self.con(r)
        return self.raw(op, Kw, a, b)

    def fn_cls(self, r):
        if rtype(r) == RTmp:
            return self.fn.tmp[rval(r)].cls
        return None

    def add(self, cls, a, b):
        return self.op(D.Oadd, cls, a, b)

    def addc(self, cls, a, c):
        if c == 0:
            return a
        return self.op(D.Oadd, cls, a, self.con(c))

    def sub(self, cls, a, b):
        return self.op(D.Osub, cls, a, b)

    def mul(self, cls, a, b):
        return self.op(D.Omul, cls, a, b)

    def mulc(self, cls, a, c):
        if c == 1:
            return a
        if c == 0:
            return self.con(0)
        if c > 0 and c & (c - 1) == 0:
            return self.op(D.Oshl, cls, a, self.con(c.bit_length() - 1))
        return self.op(D.Omul, cls, a, self.con(c))


def conval(fn, r):
    """Integer value of a CBits constant ref, else None."""
    if rtype(r) == RCon:
        c = fn.con[rval(r)]
        if c.type == CBits:
            return c.i
    return None


def fold(op, cls, a, b):
    w = cls == Kw
    norm = s32 if w else s64
    a = norm(a)
    if b is not None:
        b = norm(b)
    if op == D.Oadd:
        return norm(a + b)
    if op == D.Osub:
        return norm(a - b)
    if op == D.Omul:
        return norm(a * b)
    if op == D.Oneg:
        return norm(-a)
    if op == D.Oand:
        return norm(a & b)
    if op == D.Oor:
        return norm(a | b)
    if op == D.Oxor:
        return norm(a ^ b)
    if op == D.Oshl:
        return norm(a << (b & (31 if w else 63)))
    if op == D.Oextsw:
        return s32(a)
    if op == D.Oextuw:
        return a & U32
    return None


def cmpfold(c, k, a, b):
    """Integer comparison Ci* c of class k on constants: 1/0, or None."""
    W = 32 if k == Kw else 64
    if k not in (Kw, Kl):
        return None
    sa, sb = (s32(a), s32(b)) if W == 32 else (s64(a), s64(b))
    m = (1 << W) - 1
    ua, ub = a & m, b & m
    r = {D.Cieq: sa == sb, D.Cine: sa != sb, D.Cisge: sa >= sb, D.Cisgt: sa > sb,
         D.Cisle: sa <= sb, D.Cislt: sa < sb, D.Ciuge: ua >= ub, D.Ciugt: ua > ub,
         D.Ciule: ua <= ub, D.Ciult: ua < ub}.get(c)
    return None if r is None else int(r)


def defblock(fn, r):
    """Block id of the definition of tmp r (U32 if none: a parameter-less undefined tmp)."""
    return fn.tmp[rval(r)].bid


def istmp(r):
    return rtype(r) == RTmp


# ops the middle end knows how to copy: pure computations and memory accesses
PURE_SAFE = None    # pure and never traps (can run speculatively)
PURE = None         # pure, may trap on some targets (div/rem)


def _init_ops():
    global PURE_SAFE, PURE
    s = set()
    for name in ('add sub neg mul and or xor sar shr shl extsb extub extsh extuh extsw extuw '
                 'exts truncd stosi stoui dtosi dtoui swtof uwtof sltof ultof cast copy '
                 'fsqrt fabs fmin fmax ffloor fceil ftrunc fround').split():
        s.add(getattr(D, 'O' + name))
    for o in range(D.Ocmpw, D.Ocmpd1 + 1):
        s.add(o)
    PURE_SAFE = frozenset(s)
    PURE = frozenset(s | {D.Odiv, D.Orem, D.Oudiv, D.Ourem})


_init_ops()


def isstore(op):
    return D.Ostoreb <= op <= D.Ostored


def isload(op):
    return D.Oloadsb <= op <= D.Oload


def isfma(op):
    return op in (getattr(D, 'Ofma0', -1), getattr(D, 'Ofma1', -1))


def copyable(i):
    """Can this instruction be duplicated (unrolling, versioning)?"""
    op = i.op
    return op in PURE or isload(op) or isstore(op) or isfma(op) or op in (D.Onop, D.Odbgloc)


STORE_SIZE = None


def access_size(op):
    if isstore(op):
        return {D.Ostoreb: 1, D.Ostoreh: 2, D.Ostorew: 4, D.Ostorel: 8, D.Ostores: 4,
                D.Ostored: 8}[op]
    return None


def dce(fn):
    """Delete pure instructions and phis whose results are never used by anything with an
    effect (mark and sweep, so dead induction cycles go too). Doesn't need filluse."""
    defs = {}
    for b in blocks(fn):
        for p in phis(b):
            defs[rval(p.to)] = p
        for i in ins(b):
            if istmp(i.to):
                defs[rval(i.to)] = i
    live = set()
    work = []

    def mark(r):
        if istmp(r):
            t = rval(r)
            if t not in live:
                live.add(t)
                work.append(t)

    for b in blocks(fn):
        mark(b.jmp.arg)
        for i in ins(b):
            if not (i.op in PURE or i.op == D.Onop):
                if istmp(i.to):
                    mark(i.to)
                for a in i.arg:
                    mark(a)
    while work:
        t = work.pop()
        d = defs.get(t)
        if d is None:
            continue
        if isinstance(d, Phi):
            for a in d.arg[:d.narg]:
                mark(a)
        else:
            for a in d.arg:
                mark(a)
    for b in blocks(fn):
        setins(b, [i for i in ins(b) if not ((i.op in PURE or i.op == D.Onop) and not (
            istmp(i.to) and rval(i.to) in live))])
        prev = None
        p = b.phi
        while p:
            if rval(p.to) not in live:
                if prev is None:
                    b.phi = p.link
                else:
                    prev.link = p.link
            else:
                prev = p
            p = p.link


def merge_blocks(fn):
    """Merge b -> s when b ends in `jmp s`, s has b as its only predecessor and no phis
    (needs fillpreds)."""
    changed = False
    for b in list(blocks(fn)):
        while True:
            s = b.s1
            if (b.jmp.type != Jjmp or s is None or s is b or s is fn.start or s.npred != 1
                    or s.phi is not None):
                break
            setins(b, ins(b) + ins(s))
            b.jmp.type = s.jmp.type
            b.jmp.arg = s.jmp.arg
            b.s1 = s.s1
            b.s2 = s.s2
            for x in succs(s):
                renamepred(x, s, b)
                for k in range(x.npred):
                    if x.pred[k] is s:
                        x.pred[k] = b
            s.npred = 0
            setins(s, [])
            s.jmp.type = Jjmp
            s.s1 = s.s2 = None
            changed = True
    return changed


_WOPS = None


def normalize_wcons(fn):
    """Write the constant operands of 32-bit arithmetic as small signed values, and add/sub
    of a negative constant as sub/add of a positive one (cproc writes -1 as 4294967295).
    Works around a backend bug: QBE's gvn folds constant chains like (x + c1) + c2 in
    64-bit arithmetic without truncating w results, and the arm64 isel's immediate check
    doesn't truncate either, so e.g. (x + -1) + -30 gives an unencodable immediate. Chains
    of positive constants fold to values that stay in range."""
    global _WOPS
    if _WOPS is None:
        _WOPS = (D.Oadd, D.Osub, D.Omul, D.Oand, D.Oor, D.Oxor)
    cache = {}

    def con(v):
        if v not in cache:
            cache[v] = getcon(v, fn)
        return cache[v]

    for b in blocks(fn):
        for i in ins(b):
            if i.cls != Kw or i.op not in _WOPS:
                continue
            for n in (0, 1):
                c = conval(fn, i.arg[n])
                if c is None:
                    continue
                v = s32(c)
                if n == 1 and i.op in (D.Oadd, D.Osub) and -(1 << 31) < v < 0:
                    i.op = D.Osub if i.op == D.Oadd else D.Oadd
                    v = -v
                if v != c:
                    i.arg[n] = con(v)
