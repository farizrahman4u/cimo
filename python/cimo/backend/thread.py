"""cimo: jump threading (ext.cse; not in QBE), from LLVM's JumpThreading and GCC's thread_jumps.

    x = a ? 1 : b;  if (x) ...         a true goes straight to the arm

A block that only joins values and branches on one of them is skipped by each predecessor whose value
is known: a constant, or a condition that predecessor (or a branch above it) has already decided. It
is what `a && b`, `if (ok) ...` and flag variables compile to. Values the block defines reach its
successors only through their phis, or it is left alone, so that no other temporary needs a new phi.

Requires rpo, preds, dom, use. Breaks them; run() returns True if it changed anything and leaves them
recomputed.
"""

from . import defs as D
from .defs import RCon, RTmp, CBits, KBASE, Kw, Kl, UPhi, rtype, rval
from .util import iscmp, phiarg, isconbits, newtmp, cmpwlneg
from .simplify import _facts


_CHEAP = frozenset((D.Oadd, D.Osub, D.Oand, D.Oor, D.Oxor, D.Oshl, D.Oshr, D.Osar, D.Oextsw, D.Oextuw, D.Ocopy))


def _value(fn, q, b, v, k):
    """Whether v, flowing from q into b, is nonzero when read in class k (a bare jnz reads a word,
    a comparison its class); or None if unknown."""
    if rtype(v) == RCon:
        c = isconbits(fn, v)
        return None if c is None else bool(c & (0xffffffffffffffff if k == Kl else 0xffffffff))
    if rtype(v) != RTmp:
        return None
    # a jnz reads a word: its low half nonzero says the value is, but zero says nothing of the high half
    wide = k == Kl and fn.tmp[rval(v)].cls == Kl
    if q.jmp.type == D.Jjnz and q.jmp.arg == v and q.s1 is not q.s2:
        if q.s1 is b:
            return True
        return None if wide else False
    for c, sense in _facts(q):
        if c == v:
            if not sense and wide:
                return None
            return sense
    return None


def _cond(fn, b):
    """(phi of b the branch depends on, inverted, class it's read in) or None."""
    r = b.jmp.arg
    if rtype(r) != RTmp or fn.tmp[rval(r)].nuse != 1:
        return None
    ins = [i for i in b.ins[:b.nins] if i.op not in (D.Onop, D.Odbgloc)]
    inv = False
    k = Kw
    if ins:
        cm = iscmp(ins[0].op)
        if len(ins) > 1 or ins[0].to != r or cm is None or cm[1] not in (D.Cieq, D.Cine) \
                or isconbits(fn, ins[0].arg[1]) != 0:
            return None
        inv = cm[1] == D.Cieq
        k = cm[0]
        r = ins[0].arg[0]
        if rtype(r) != RTmp or fn.tmp[rval(r)].nuse != 1:
            return None
    p = b.phi
    while p:
        if p.to == r:
            return p, inv, k
        p = p.link
    return None


def _thread(fn, b):
    if b.jmp.type != D.Jjnz or b.s1 is b.s2 or b.phi is None or b.s1 is b or b.s2 is b:
        return False
    if any(q.id >= b.id for q in b.pred[:b.npred]):  # a loop header: the loop would change shape
        return False
    c = _cond(fn, b)
    if c is None:
        return False
    p, inv, k = c
    phis = {}
    x = b.phi
    while x:
        if x is not p:
            for u in fn.tmp[rval(x.to)].use[:fn.tmp[rval(x.to)].nuse]:  # only into a successor's phi
                if u.type != UPhi or u.bid not in (b.s1.id, b.s2.id):
                    return False
                if any(a == x.to and bp is not b for a, bp in zip(u.u.arg[:u.u.narg], u.u.blk)):
                    return False
            phis[x.to] = x
        x = x.link
    changed = False
    for n in range(p.narg):
        q, v = p.blk[n], p.arg[n]
        if q.s1 is not b and q.s2 is not b:
            continue  # taken already, in this round
        val = _value(fn, q, b, v, k)
        if val is None:
            continue
        t = b.s1 if val != inv else b.s2
        if t.id <= b.id or q.s1 is t or q.s2 is t:
            continue
        x = t.phi
        while x:
            a = phiarg(x, b)
            if a in phis:
                a = phiarg(phis[a], q)
            x.arg.append(a)
            x.blk.append(q)
            x.narg += 1
            x = x.link
        if q.s1 is b:
            q.s1 = t
        if q.s2 is b:
            q.s2 = t
        changed = True
    return changed


def _combinable(fn, x, n):
    """The instructions of x, if it only computes a condition from what's there before it: then
    computing it on the other path too costs nothing and can't fault. None if not."""
    ins = [i for i in x.ins[:x.nins] if i.op not in (D.Onop, D.Odbgloc)]
    if x.phi is not None or x.npred != 1 or x.jmp.type != D.Jjnz or x.s1 is x.s2 or len(ins) > n:
        return None
    if not ins or ins[-1].to != x.jmp.arg or iscmp(ins[-1].op) is None:
        return None
    if KBASE(iscmp(ins[-1].op)[0]) != 0:  # no fccmp yet
        return None
    if any(i.op not in _CHEAP and iscmp(i.op) is None for i in ins):
        return None
    if fn.tmp[rval(x.jmp.arg)].nuse != 1:
        return None
    return ins


def _samephis(t, p, q):
    """t's phis take the same value from p and q."""
    x = t.phi
    while x:
        if phiarg(x, p) != phiarg(x, q):
            return False
        x = x.link
    return True


def _ifcombine(fn, h):
    """GCC's tree_ifcombine, LLVM's SimplifyCFG: `if (a) { if (b) Y; else N; } else N;` is
    `if (a & b) Y; else N;`, and likewise for `or`. Both a and b are comparisons, of 0 or 1. arm64
    selects ccmp for it (isel._ccmp)."""
    if h.jmp.type != D.Jjnz or h.s1 is h.s2 or rtype(h.jmp.arg) != RTmp:
        return False
    d = fn.tmp[rval(h.jmp.arg)].def_
    if d is None or iscmp(d.op) is None or fn.tmp[rval(h.jmp.arg)].nuse != 1:
        return False
    for x, other, op in ((h.s1, h.s2, D.Oand), (h.s2, h.s1, D.Oor)):
        # one comparison, nothing else: LLVM allows one instruction more, which took scramjit's flood_fill
        # from 2.21 to 2.52 ms (`m != 0 && (m < 0) != (b < 0)`, whose branches predict well)
        ins = _combinable(fn, x, 1)
        if ins is None or x is h:
            continue
        # a way out of the loop is rarely taken (LLVM's BranchProbabilityInfo): bounds checks predict
        # perfectly, and scramjit's flood_fill ran 25% slower combined. Otherwise combining pays: at
        # 50% taken 1.04 ns per element against 4.6, and when always taken 1.03 against 0.79
        if h.loop > other.loop:
            continue
        y, n = (x.s1, x.s2) if op == D.Oand else (x.s2, x.s1)
        if n is not other:
            if y is not other:
                continue
            # the arm's other way goes there: test its opposite
            ins[-1].op = cmpwlneg(ins[-1].op)
            x.s1, x.s2 = x.s2, x.s1
            y, n = n, y
        if y is x or y is other or not _samephis(other, h, x):
            continue
        r = newtmp('cb', Kw, fn)
        h.ins = h.ins[:h.nins] + ins + [D.Ins(op, Kw, r, h.jmp.arg, x.jmp.arg)]
        h.nins = len(h.ins)
        for i in ins:
            if i.to != D.R:
                fn.tmp[rval(i.to)].bid = h.id
        x.ins = []
        x.nins = 0
        x.jmp.type = D.Jhlt  # unreachable now; fillcfg removes it
        x.jmp.arg = D.R
        s1, s2 = (y, other) if op == D.Oand else (other, y)
        for t in (y, other):
            p = t.phi
            while p:
                p.blk = [h if b is x else b for b in p.blk] if t is y else p.blk
                p = p.link
        x.s1 = x.s2 = None
        h.jmp.arg = r
        h.s1, h.s2 = s1, s2
        return True
    return False


def run(fn):
    from . import cfg, ssa
    any_ = False
    cfg.fillloop(fn)
    for _ in range(4):
        cur = False
        for b in list(fn.rpo[:fn.nblk]):
            if _thread(fn, b):
                cur = True
        if not cur:
            for b in list(fn.rpo[:fn.nblk]):
                if _ifcombine(fn, b):
                    cur = True
                    break
        if not cur:
            break
        any_ = True
        cfg.fillcfg(fn)
        ssa.filluse(fn)
        cfg.filldom(fn)
    return any_
