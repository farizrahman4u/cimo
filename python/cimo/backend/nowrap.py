"""What C's promise that signed +, - and * don't overflow allows (ext.nowrap).

The frontend marks those instructions (Ins.nsw). This pass is the only reader, and it removes the
marks: from here on every other pass may rewrite an add as it likes, the mistake LLVM has to guard
against all over. Before it, only gvn merges or rewrites arithmetic, and it clears or ands the mark.

- Comparisons, from InstSimplify and InstCombine: x + 1 > x, x + c < y + c, x + c1 < c2, x - y < 0,
  x * c < y * c, x * c < 0.
- Divisions: x * c / c, x * c % c, (x * 2^n) >> n, x * 6 / 2.
- WidenIV of LLVM's IndVarSimplify: a w induction variable with an extension in the loop becomes an l
  one, which removes the extensions of it and of what's computed from it (a[i + 1], a[i * m + j]).
  Comparing it with something invariant becomes an l comparison, as LFTR leaves it. What still reads the
  w value reads a `copy w`, which is free when it's only read.

The mark is a promise about the executions that reach the instruction. gcm may move an instruction to
where it runs more often, so a mark says nothing about a value nobody uses: what's rewritten here is
always a use of the marked result, which is reached only when C would have computed it.
"""

from . import defs as D
from .defs import R, RTmp, RCon, CBits, Ins, Kw, Kl, Ocopy, Oadd, Osub, Omul, Odiv, Orem, Osar, Oextsw, rtype, rval, s32, s64
from .defs import Cieq, Cine, Cisge, Cisgt, Cisle, Cislt
from .cfg import dom
from .util import getcon, iscmp, newtmp

_SWAP = {Cieq: Cieq, Cine: Cine, Cisge: Cisle, Cisgt: Cislt, Cisle: Cisge, Cislt: Cisgt}
_SIGNED = (Cisge, Cisgt, Cisle, Cislt)
_MAXDEPTH = 6
_MAXLEAVES = 3


def _sval(k, v):
    return s32(v & 0xffffffff) if k == Kw else s64(v)


def _fits(k, v):
    return -(1 << 31) <= v < (1 << 31) if k == Kw else -(1 << 63) <= v < (1 << 63)


def _bits(fn, r, k):
    """The signed value of constant r in class k, or None."""
    if rtype(r) == RCon and fn.con[rval(r)].type == CBits:
        return _sval(k, fn.con[rval(r)].i)
    return None


def _marked(fn, r, k):
    """The instruction with C's promise that defines r in class k, or None."""
    if rtype(r) == RTmp:
        d = fn.tmp[rval(r)].def_
        if d is not None and d.nsw and d.cls == k:
            return d
    return None


def _addcon(fn, r, k):
    """(x, c) when r is the marked x + c or x - c, else (r, 0)."""
    d = _marked(fn, r, k)
    if d is not None and d.op in (Oadd, Osub):
        c = _bits(fn, d.arg[1], k)
        if c is not None and rtype(d.arg[0]) == RTmp and _fits(k, -c):
            return d.arg[0], c if d.op == Oadd else -c
    return r, 0


def _mulcon(fn, r, k):
    """(x, c) when r is the marked x * c, else None."""
    d = _marked(fn, r, k)
    if d is not None and d.op == Omul:
        c = _bits(fn, d.arg[1], k)
        if c is not None and c != 0 and rtype(d.arg[0]) == RTmp:
            return d.arg[0], c
    return None


def _holds(cc, a, b):
    return {Cieq: a == b, Cine: a != b, Cisge: a >= b, Cisgt: a > b, Cisle: a <= b, Cislt: a < b}[cc]


def _cmp(fn, i, k, cc):
    a, b = i.arg
    if rtype(a) == RCon and rtype(b) == RTmp and cc in _SWAP:
        a, b, cc = b, a, _SWAP[cc]
    if cc not in _SWAP:
        return None
    x, c1 = _addcon(fn, a, k)
    y, c2 = _addcon(fn, b, k)
    if x == y and (c1 or c2) and rtype(x) == RTmp:
        return Ins(Ocopy, i.cls, i.to, getcon(int(_holds(cc, c1, c2)), fn), R)
    if c1 and c1 == c2:
        return cc, x, y
    if cc in _SIGNED or c1:
        kb = _bits(fn, b, k)
        if kb is not None and c1 and _fits(k, kb - c1):  # x + c1 < kb
            return cc, x, getcon(s64(_sval(k, kb - c1)), fn)
    kb = _bits(fn, b, k)
    d = _marked(fn, a, k)
    if d is not None and d.op == Osub and _bits(fn, d.arg[1], k) is None and kb is not None:  # x - y < 0
        if kb == 0 and (cc in _SIGNED or fn.tmp[rval(a)].nuse == 1):  # else subs, which needs no promise for == 0
            return cc, d.arg[0], d.arg[1]
        if kb == 1 and cc == Cislt:
            return Cisle, d.arg[0], d.arg[1]
        if kb == -1 and cc == Cisgt:
            return Cisge, d.arg[0], d.arg[1]
    m = _mulcon(fn, a, k)
    if m is not None:
        x, c = m
        if kb == 0:  # x * c < 0
            return (cc if c > 0 else _SWAP[cc]), x, b
        my = _mulcon(fn, b, k)
        if my is not None and my[1] == c:  # x * c < y * c
            return (cc if c > 0 else _SWAP[cc]), x, my[0]
    return None


def _div(fn, i):
    """A division of a marked product by what's in its constant: what it becomes, or None."""
    m = _mulcon(fn, i.arg[0], i.cls)
    if m is None:
        return None
    x, c = m
    n = _bits(fn, i.arg[1], i.cls)
    if n is None or n == 0:
        return None
    if i.op == Osar:
        if not 0 < n < (32 if i.cls == Kw else 64):
            return None
        n = 1 << n
    if c % n:
        return None
    if i.op == Orem:
        return Ins(Ocopy, i.cls, i.to, getcon(0, fn), R)
    q = c // n
    if q == 1:
        return Ins(Ocopy, i.cls, i.to, x, R)
    return Ins(Omul, i.cls, i.to, x, getcon(s64(_sval(i.cls, q)), fn))


def _folds(fn):
    changed = False
    for b in fn.rpo[:fn.nblk]:
        for i in b.ins[:b.nins]:
            if i.op in (Odiv, Orem, Osar):
                new = _div(fn, i)
                if new is not None:
                    i.set(new)
                    changed = True
                continue
            c = iscmp(i.op)
            if c is None or c[0] not in (Kw, Kl):
                continue
            new = _cmp(fn, i, c[0], c[1])
            if new is None:
                continue
            if isinstance(new, Ins):
                i.set(new)
            else:
                cc, a, b1 = new
                i.op = (D.Ocmpw if c[0] == Kw else D.Ocmpl) + cc
                i.arg[0], i.arg[1] = a, b1
            changed = True
    return changed


class _IV:
    __slots__ = ('phi', 'head', 'next', 'def_', 'wide', 'widenext')

    def __init__(self, phi, head, nxt, d):
        self.phi, self.head, self.next, self.def_ = phi, head, nxt, d
        self.wide = self.widenext = R


def _ivs(fn):
    """The w induction variables: by the temporary of the phi and of its step, the same object."""
    ivs = {}
    for h in fn.rpo[:fn.nblk]:
        p = h.phi
        while p:
            if p.cls == Kw:
                back = set(p.arg[n] for n in range(p.narg) if dom(h, p.blk[n]))
                if len(back) == 1 and any(not dom(h, p.blk[n]) for n in range(p.narg)):
                    nxt = back.pop()
                    d = _marked(fn, nxt, Kw)
                    if d is not None and d.op in (Oadd, Osub) and d.arg[0] == p.to and _bits(fn, d.arg[1], Kw) is not None:
                        iv = _IV(p, h, nxt, d)
                        ivs[p.to] = ivs[nxt] = iv
            p = p.link
    return ivs


def _tree(fn, r, ivs, seen, depth=0):
    """How many leaves widening r takes, counting those that are induction variables in seen: None when too many."""
    if r in ivs:
        seen.append(ivs[r])
        return 0
    if rtype(r) != RTmp:
        return 0
    d = _marked(fn, r, Kw)
    if d is None or d.op not in (Oadd, Osub, Omul) or depth >= _MAXDEPTH:
        return 1
    n = 0
    for a in d.arg[:2]:
        m = _tree(fn, a, ivs, seen, depth + 1)
        if m is None:
            return None
        n += m
    return n if n <= _MAXLEAVES else None


def _invariant(fn, r, h):
    """r has one value during loop h: it's a constant or defined before."""
    if rtype(r) == RCon:
        return fn.con[rval(r)].type == CBits
    if rtype(r) != RTmp:
        return False
    d = fn.rpo[fn.tmp[rval(r)].bid]
    return d is not h and dom(d, h)


def _widen(fn):
    ivs = _ivs(fn)
    if not ivs:
        return False
    live = {}  # the induction variables something is extended from
    sites = []
    for b in fn.rpo[:fn.nblk]:
        for i in b.ins[:b.nins]:
            if i.op == Oextsw and i.cls == Kl and rtype(i.arg[0]) == RTmp:
                seen = []
                if _tree(fn, i.arg[0], ivs, seen) is not None and seen:
                    for iv in seen:
                        live[iv.phi.to] = iv
                    sites.append((b, i))
    if not live:
        return False
    used = {}
    for iv in live.values():
        used[iv.phi.to] = used[iv.next] = iv
        iv.wide = newtmp('wide', Kl, fn)
        iv.widenext = newtmp('wide', Kl, fn)
    before = {}
    tails = {}
    heads = {}
    after = {}

    wide = {}
    for r, iv in used.items():
        wide[r] = iv.wide if r == iv.phi.to else iv.widenext

    def build(r, out):
        if r in wide:
            return wide[r]
        if rtype(r) == RCon and fn.con[rval(r)].type == CBits:
            return getcon(s32(fn.con[rval(r)].i & 0xffffffff), fn)
        d = _marked(fn, r, Kw)
        if rtype(r) == RTmp and d is not None and d.op in (Oadd, Osub, Omul):
            a = build(d.arg[0], out)
            c = build(d.arg[1], out)
            t = newtmp('wide', Kl, fn)
            out.append(Ins(d.op, Kl, t, a, c))
            return t
        t = newtmp('wide', Kl, fn)
        out.append(Ins(Oextsw, Kl, t, r, R))
        return t

    # comparisons of what's being widened with something invariant, before any instruction changes
    for b in fn.rpo[:fn.nblk]:
        for i in b.ins[:b.nins]:
            c = iscmp(i.op)
            if c is None or c[0] != Kw or c[1] not in _SWAP:
                continue
            a0, a1 = i.arg
            for x, y in ((a0, a1), (a1, a0)):
                if rtype(x) != RTmp:
                    continue
                seen = []
                if _tree(fn, x, used, seen) is None or not seen:
                    continue
                if all(_invariant(fn, y, iv.head) for iv in seen) or y in used:
                    out = before.setdefault(id(i), [])
                    p0 = build(a0, out)
                    p1 = build(a1, out)
                    i.op = D.Ocmpl + c[1]
                    i.arg[0], i.arg[1] = p0, p1
                    break
    for b, i in sites:
        out = before.setdefault(id(i), [])
        r = build(i.arg[0], out)
        i.set(Ins(Ocopy, Kl, i.to, r, R))
    # the variables themselves
    for iv in live.values():
        p = iv.phi
        narrow = p.to
        d = iv.def_
        c = _bits(fn, d.arg[1], Kw)
        c = c if d.op == Oadd else -c
        op, c = (Oadd, c) if c >= 0 else (Osub, -c)
        after.setdefault(id(d), []).append(Ins(op, Kl, iv.widenext, iv.wide, getcon(c, fn)))
        heads.setdefault(iv.head.id, []).append(Ins(Ocopy, Kw, narrow, iv.wide, R))
        for n in range(p.narg):
            if p.arg[n] == iv.next and dom(iv.head, p.blk[n]):
                p.arg[n] = iv.widenext
            else:
                out = tails.setdefault(p.blk[n].id, [])
                p.arg[n] = build(p.arg[n], out)
        p.to = iv.wide
        p.cls = Kl
    for b in fn.rpo[:fn.nblk]:
        new = list(heads.get(b.id, ()))
        for i in b.ins[:b.nins]:
            new += before.get(id(i), ())
            new.append(i)
            new += after.get(id(i), ())
        new += tails.get(b.id, ())
        b.ins = new
        b.nins = len(new)
    return True


def run(fn):
    """Needs filluse and filldom. Returns whether something changed; no instruction has a mark after."""
    if not any(i.nsw for b in fn.rpo[:fn.nblk] for i in b.ins[:b.nins]):
        return False
    changed = _folds(fn)
    changed = _widen(fn) or changed
    for b in fn.rpo[:fn.nblk]:
        for i in b.ins[:b.nins]:
            i.nsw = False
    return changed
