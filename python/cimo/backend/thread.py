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
from .util import getcon, iscmp, phiarg, isconbits, newtmp, cmpwlneg
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


BITTEST_MIN = 6


def _equality(fn, b):
    """(x, class, constant, where it goes when equal, else) when b ends in x == constant, used only there."""
    d = fn.tmp[rval(b.jmp.arg)].def_ if rtype(b.jmp.arg) == RTmp else None
    cm = iscmp(d.op) if d is not None else None
    if cm is None or KBASE(cm[0]) != 0 or cm[1] not in (D.Cieq, D.Cine) or fn.tmp[rval(b.jmp.arg)].nuse != 1:
        return None
    c = isconbits(fn, d.arg[1])
    if c is None or rtype(d.arg[0]) != RTmp:
        return None
    eq, ne = (b.s1, b.s2) if cm[1] == D.Cieq else (b.s2, b.s1)
    return d.arg[0], cm[0], c, eq, ne


def _bittest(fn, h):
    """c == k1 || c == k2 || ..., as branches to one place, is (unsigned)(c - lo) <= hi - lo & mask >> (c - lo).
    LLVM's form has a branch for each half, from 3 values, and against the ccmp chains here it took up to 2.66x
    at 4 values and 1.43x at 6. Without those branches it costs about the same on any data. Against the chains,
    best to worst on Snapdragon X and Neoverse N1, V1, V2: at 4 values 0.60-1.07, 0.77-1.28, 0.61-1.03, 0.87-1.19;
    at 6 0.24-1.04, 0.39-1.12, 0.32-0.98, 0.40-1.34; at 10 0.43-0.83, 0.54-0.85, 0.52-0.78, 0.46-0.78."""
    if h.jmp.type != D.Jjnz or h.s1 is h.s2:
        return False
    e = _equality(fn, h)
    if e is None:
        return False
    x, k, c, hit, b = e
    w = 64 if k == D.Kl else 32
    full = (1 << w) - 1
    vals = {c & full}
    chain = []
    while b is not h and b not in chain and _combinable(fn, b, 1) is not None:
        e = _equality(fn, b)
        if e is None or e[0] != x or e[1] != k or e[3] is not hit or not _samephis(hit, h, b):
            break
        vals.add(e[2] & full)
        chain.append(b)
        b = e[4]
    miss = b
    if len(vals) < BITTEST_MIN or max(vals) - min(vals) >= 64 or hit is h or hit in chain or miss is hit or miss is h:
        return False
    lo, span = min(vals), max(vals) - min(vals)
    mask = sum(1 << (v - lo) for v in vals)
    t = x
    new = []
    if lo:
        t = newtmp('bt', k, fn)
        new.append(D.Ins(D.Osub, k, t, x, getcon(D.s64(lo), fn)))
    inr, sh, bit, bw, r = (newtmp('bt', cl, fn) for cl in (Kw, D.Kl, D.Kl, Kw, Kw))
    new += [D.Ins((D.Ocmpw if k == Kw else D.Ocmpl) + D.Ciule, Kw, inr, t, getcon(span, fn)),
            D.Ins(D.Oshr, D.Kl, sh, getcon(D.s64(mask), fn), t),
            D.Ins(D.Oand, D.Kl, bit, sh, getcon(1, fn)),
            D.Ins(D.Ocopy, Kw, bw, bit, D.R),
            D.Ins(D.Oand, Kw, r, inr, bw)]
    h.ins = h.ins[:h.nins] + new
    h.nins = len(h.ins)
    last = chain[-1]
    p = miss.phi
    while p:
        p.blk = [h if q is last else q for q in p.blk]
        p = p.link
    for q in chain:
        q.ins = []
        q.nins = 0
        q.jmp.type = D.Jhlt  # unreachable now; fillcfg removes it
        q.jmp.arg = D.R
        q.s1 = q.s2 = None
    h.jmp.arg = r
    h.s1, h.s2 = hit, miss
    return True


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


def _deadphis(fn):
    """Threading leaves phis nothing reads, and their arguments would look used elsewhere."""
    dead = False
    for b in fn.rpo[:fn.nblk]:
        kept = []
        p = b.phi
        while p:
            if fn.tmp[rval(p.to)].nuse:
                kept.append(p)
            else:
                dead = True
            p = p.link
        if dead:
            for x, y in zip(kept, kept[1:] + [None]):
                x.link = y
            b.phi = kept[0] if kept else None
    return dead


def _rename(fn, old, new):
    for b in fn.rpo[:fn.nblk]:
        p = b.phi
        while p:
            p.arg = [new if a == old else a for a in p.arg]
            p = p.link
        for i in b.ins[:b.nins]:
            if i.arg[0] == old:
                i.arg[0] = new
            if i.arg[1] == old:
                i.arg[1] = new
        if b.jmp.arg == old:
            b.jmp.arg = new


def _mergeline(fn):
    """A block whose only successor has no other predecessor, as one block: threading leaves these, and
    a condition behind one isn't seen as computed where it's tested."""
    merged = False
    for b in fn.rpo[:fn.nblk]:
        if b.s1 is None:  # merged already
            continue
        while b.jmp.type == D.Jjmp and b.s1 is not b and b.s1 is not fn.start and b.s1.npred == 1:
            s = b.s1
            p = s.phi
            while p:
                _rename(fn, p.to, p.arg[0])
                p = p.link
            b.ins = b.ins[:b.nins] + s.ins[:s.nins]
            b.nins = len(b.ins)
            for i in s.ins[:s.nins]:
                if i.to != D.R:
                    fn.tmp[rval(i.to)].bid = b.id
            b.jmp.type, b.jmp.arg = s.jmp.type, s.jmp.arg
            b.s1, b.s2 = s.s1, s.s2
            for t in (s.s1, s.s2):
                p = t.phi if t else None
                while p:
                    p.blk = [b if x is s else x for x in p.blk]
                    p = p.link
            s.phi, s.ins, s.nins = None, [], 0
            s.jmp.type, s.jmp.arg = D.Jhlt, D.R
            s.s1 = s.s2 = None
            merged = True
    return merged


def run(fn):
    from . import cfg, ssa
    any_ = False
    cfg.fillloop(fn)
    bits = False
    if D.T.name.startswith('arm64'):
        for b in list(fn.rpo[:fn.nblk]):
            if b.jmp.type != D.Jhlt and _bittest(fn, b):
                bits = True
    if bits:
        any_ = True
        cfg.fillcfg(fn)
        ssa.filluse(fn)
        cfg.filldom(fn)
    for _ in range(6):
        cur = False
        for b in list(fn.rpo[:fn.nblk]):
            if _thread(fn, b):
                cur = True
        if not cur:
            if _deadphis(fn):
                ssa.filluse(fn)
            if _mergeline(fn):
                cur = True
                cfg.fillcfg(fn)
                ssa.filluse(fn)
                cfg.filldom(fn)
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
    if any_:
        cfg.fillloop(fn)  # fix: the loop depths of the blocks it moved (loadopt asserts on them; Csmith, x86-64)
    return any_
