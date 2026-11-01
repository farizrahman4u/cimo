"""cimo: common subexpressions QBE leaves behind (ext.cse; not in QBE).

gvn merges pure instructions, but QBE pins integer divisions (they may trap), and load
elimination runs before gvn, when two computations of an address are still two temporaries (and
gcm's sink copies them again). With this flag:

- main runs loadopt again after gvn, on one temporary per address (domcse);
- on arm64, where sdiv and udiv never trap, divisions are pure (gcm.isdivwl), so they merge across
  sibling branches and leave loops, and divrem computes a remainder from the division of the same
  operands: a - (a / b) * b is what the rem instruction's own sdiv + msub computes, also for
  b == 0 and INT_MIN / -1;
- on arm64, a division by a constant is a multiplication by its inverse (divconst; wasm engines
  do this themselves);
- on arm64, a large multiplier used in a loop is built once, out of it (hoistcons);
- on wasm, where they trap, gvn merges a division with an identical one that dominates it (if the
  first did not trap, neither does the second);
- gvn merges phis that join the two ways of the same condition with the same values, and removes
  a jnz that a dominating jnz on the same temporary has decided: a bounds check repeated on an
  index computed the same way (gvn.dedupselect, gvn.knownjmp).
"""

from . import defs as D
from .defs import KBASE
from .cfg import dom
from .gcm import pinned
from .gvn import replaceuses
from .util import newtmp, getcon



_DIVOPS = (D.Odiv, D.Orem, D.Oudiv, D.Ourem)
_MULS = (D.Omul, D.Osmulh, D.Oumulh)


def nload(fn):
    """How many loads fn has: load elimination removed some if it went down."""
    n = 0
    b = fn.start
    while b:
        n += sum(1 for i in b.ins[:b.nins] if D.isload(i.op))
        b = b.link
    return n


def domcse(fn):
    """Merge a pure instruction with an identical one that dominates it, which undoes gcm's sink
    (it copies cheap address arithmetic to each load and store): loads can only match on one
    temporary. Requires rpo, dom and use, maintains them. Returns True if anything changed."""
    seen = {}
    changed = False
    for b in fn.rpo[:fn.nblk]:
        for i in b.ins[:b.nins]:
            if i.op == D.Onop or pinned(i) or i.to == D.R:
                continue
            for b1, r in seen.setdefault((i.op, i.cls, i.arg[0], i.arg[1]), []):
                if dom(b1, b):
                    replaceuses(fn, i.to, r)
                    i.set(D.Ins(D.Onop))
                    changed = True
                    break
            else:
                seen[(i.op, i.cls, i.arg[0], i.arg[1])].append((b, i.to))
    return changed


def divrem(fn):
    """arm64: a remainder whose operands also have a division becomes a - (a / b) * b, and gvn
    then merges the divisions. Requires args as gvn left them; breaks use. Returns True if
    anything changed."""
    divs = set()
    rems = False
    for b in fn.rpo[:fn.nblk]:
        for i in b.ins[:b.nins]:
            if i.op in (D.Odiv, D.Oudiv) and KBASE(i.cls) == 0:
                divs.add((i.op, i.cls, i.arg[0], i.arg[1]))
            elif i.op in (D.Orem, D.Ourem) and KBASE(i.cls) == 0:
                rems = True
    if not (divs and rems):
        return False
    changed = False
    for b in fn.rpo[:fn.nblk]:
        out = []
        for i in b.ins[:b.nins]:
            op = D.Odiv if i.op == D.Orem else D.Oudiv if i.op == D.Ourem else None
            if op is None or (op, i.cls, i.arg[0], i.arg[1]) not in divs:
                out.append(i)
                continue
            d = newtmp('div', i.cls, fn)
            t = newtmp('mul', i.cls, fn)
            out.append(D.Ins(op, i.cls, d, i.arg[0], i.arg[1]))
            out.append(D.Ins(D.Omul, i.cls, t, d, i.arg[1]))
            out.append(D.Ins(D.Osub, i.cls, i.to, i.arg[0], t))
            changed = True
        b.ins = out
        b.nins = len(out)
    return changed


def smagic(d, w):
    """(multiplier as a signed w-bit value, shift) so that a / d, for |d| >= 2 and not a power of
    two, is the high half of a * multiplier, plus a if d > 0 > multiplier, minus a if
    d < 0 < multiplier, shifted right by shift, plus its own sign bit (Hacker's Delight 10-1)."""
    ad = abs(d)
    t = (1 << (w - 1)) + (1 if d < 0 else 0)
    anc = t - 1 - t % ad
    p = w - 1
    while True:
        p += 1
        if (1 << p) > anc * (ad - (1 << p) % ad):
            break
    m = (1 << p) // ad + 1
    if m >= 1 << (w - 1):
        m -= 1 << w
    return (-m if d < 0 else m), p - w


def umagic(d, w):
    """(multiplier, shift, add) so that a / d, unsigned, for 1 < d < 2**w and not a power of two,
    is hi(a * multiplier) >> shift; with add, t = hi(a * multiplier) and the quotient is
    (t + ((a - t) >> 1)) >> shift (Granlund & Montgomery, Hacker's Delight 10-9)."""
    l = 0
    while (1 << l) < d:
        l += 1
    for s in range(l):
        m = -(-(1 << (w + s)) // d)  # rounded up
        if m < 1 << w and m * d - (1 << (w + s)) <= 1 << s:
            return m, s, False
    return ((1 << w) * ((1 << l) - d)) // d + 1, l - 1, True


def _expand(i, fn, out):
    """Instructions for the division or remainder i by a constant, appended to out; False if i
    isn't one this handles."""
    c = fn.con[i.arg[1] & D.VMASK] if D.rtype(i.arg[1]) == D.RCon else None
    if c is None or c.type != D.CBits or D.rtype(i.arg[0]) != D.RTmp:
        return False
    k = i.cls
    wide = k == D.Kl
    w = 64 if wide else 32
    signed = i.op in (D.Odiv, D.Orem)
    d = c.i & ((1 << w) - 1)
    if signed and d >> (w - 1):
        d -= 1 << w
    ad = abs(d)
    pow2 = ad & (ad - 1) == 0
    if ad < 2 or (pow2 and not signed):  # 0 and 1 stay; simpl has the unsigned powers of two
        return False
    a = i.arg[0]

    def op(o, x, y=D.R, cls=k):
        r = newtmp('dc', cls, fn)
        out.append(D.Ins(o, cls, r, x, y))
        return r

    def con(n):
        return getcon(D.s64(n), fn)

    if pow2:
        s = 0
        while (1 << s) < ad:
            s += 1
        q = op(D.Oshr, op(D.Osar, a, con(w - 1)), con(w - s))
        q = op(D.Osar, op(D.Oadd, a, q), con(s))
        if d < 0:
            q = op(D.Oneg, q)
    elif signed:
        m, s = smagic(d, w)
        if wide:
            h = op(D.Osmulh, a, con(m))
        else:
            h = op(D.Osar, op(D.Omul, op(D.Oextsw, a, cls=D.Kl), con(m), cls=D.Kl), con(32), cls=D.Kl)
        q = h
        if d > 0 > m:
            q = op(D.Oadd, q, a)
        elif d < 0 < m:
            q = op(D.Osub, q, a)
        if s:
            q = op(D.Osar, q, con(s))
        q = op(D.Oadd, q, op(D.Oshr, q, con(w - 1)))
    else:
        m, s, add = umagic(d, w)
        if wide:
            h = op(D.Oumulh, a, con(m))
        else:
            h = op(D.Oshr, op(D.Omul, op(D.Oextuw, a, cls=D.Kl), con(m), cls=D.Kl), con(32), cls=D.Kl)
        q = h
        if add:
            q = op(D.Oadd, op(D.Oshr, op(D.Osub, a, h), con(1)), h)
        if s:
            q = op(D.Oshr, q, con(s))
    if i.op in (D.Orem, D.Ourem):
        out.append(D.Ins(D.Osub, k, i.to, a, op(D.Omul, q, i.arg[1])))
    else:
        out.append(D.Ins(D.Ocopy, k, i.to, q, D.R))
    return True


def divconst(fn):
    """arm64: integer divisions and remainders by a constant, as multiplications. Breaks use.
    Returns True if anything changed."""
    changed = False
    for b in fn.rpo[:fn.nblk]:
        if not any(i.op in _DIVOPS and KBASE(i.cls) == 0 for i in b.ins[:b.nins]):
            continue
        out = []
        for i in b.ins[:b.nins]:
            if i.op in _DIVOPS and KBASE(i.cls) == 0 and _expand(i, fn, out):
                changed = True
            else:
                out.append(i)
        b.ins = out
        b.nins = len(out)
    return changed


def _bigcon(c):
    """A 64-bit constant that takes more than one mov to build (arm64: nonzero 16-bit parts,
    counted on the value and on its complement for movn)."""
    n = c.i & D.M64
    parts = sum(1 for k in range(0, 64, 16) if (n >> k) & 0xffff)
    inv = sum(1 for k in range(0, 64, 16) if (~n >> k) & 0xffff)
    return min(parts, inv) > 1


def hoistcons(fn):
    """arm64: a multiplier that takes several instructions to build, used in a loop, is built once
    at the end of the start block, which dominates every loop (MachineLICM). isel would build it
    at each use, once per unrolled copy, since gvn folds copies of constants: so this runs after
    the last gvn. Requires rpo and loop depths; breaks use. Returns True if anything changed."""
    regs = {}
    for b in fn.rpo[1:fn.nblk]:
        if b.loop <= 1:
            continue
        for i in b.ins[:b.nins]:
            if i.op not in _MULS or KBASE(i.cls) != 0:
                continue
            for n in range(2):
                r = i.arg[n]
                if D.rtype(r) != D.RCon:
                    continue
                c = fn.con[r & D.VMASK]
                if c.type != D.CBits or not _bigcon(c):
                    continue
                k = (r, i.cls)
                if k not in regs:
                    regs[k] = newtmp('con', i.cls, fn)
                i.arg[n] = regs[k]
    if not regs:
        return False
    st = fn.start
    st.ins = st.ins[:st.nins] + [D.Ins(D.Ocopy, k, t, r, D.R) for (r, k), t in regs.items()]
    st.nins = len(st.ins)
    return True


def _adddef(fn, r, k):
    """The add or sub-of-a-constant defining r in class k, if r is only used once; or None."""
    if D.rtype(r) != D.RTmp or (r & D.VMASK) < D.Tmp0:
        return None
    t = fn.tmp[r & D.VMASK]
    d = t.def_
    if d is None or d.cls != k or t.nuse != 1 or (d.op != D.Oadd and not (
            d.op == D.Osub and D.rtype(d.arg[1]) == D.RCon)):
        return None
    return d


def _conof(fn, r):
    if D.rtype(r) == D.RCon and fn.con[r & D.VMASK].type == D.CBits:
        return fn.con[r & D.VMASK].i
    return None


def _sum(fn, r, k, leaves, depth=0):
    """Adds r's terms to leaves; returns its constant part."""
    c = _conof(fn, r)
    if c is not None:
        return c
    d = _adddef(fn, r, k) if depth < 8 else None
    if d is None:
        leaves.append(r)
        return 0
    if d.op == D.Osub:
        return _sum(fn, d.arg[0], k, leaves, depth + 1) - _conof(fn, d.arg[1])
    return _sum(fn, d.arg[0], k, leaves, depth + 1) + _sum(fn, d.arg[1], k, leaves, depth + 1)


def reassoc(fn):
    """LLVM's Reassociate and InstCombine's distribution, for integers, which wrap: (x + c) * k is
    x * k + c * k, and a sum of sums is its terms in a fixed order, then the constant. So p[i + 1] and
    p[i + 2] share p + i * 8 and differ by a constant, which gvn merges and addressing folds. Requires
    use; breaks it. Returns True if anything changed."""
    changed = False
    for b in fn.rpo[:fn.nblk]:
        if not any(i.op in (D.Oadd, D.Omul, D.Oshl) and KBASE(i.cls) == 0 for i in b.ins[:b.nins]):
            continue
        out = []
        for i in b.ins[:b.nins]:
            k = i.cls
            if KBASE(k) == 0 and i.op in (D.Omul, D.Oshl):
                n = _conof(fn, i.arg[1])
                d = _adddef(fn, i.arg[0], k)
                c = _conof(fn, d.arg[1]) if d is not None else None
                if n is not None and c is not None:
                    if i.op == D.Oshl:
                        n = 1 << (n & (63 if k == D.Kl else 31))
                    if d.op == D.Osub:
                        c = -c
                    t = newtmp('ra', k, fn)
                    x = D.Ins(i.op, k, t, d.arg[0], i.arg[1])
                    out.append(x)
                    fn.tmp[t & D.VMASK].def_ = x
                    fn.tmp[t & D.VMASK].nuse = 1
                    i.op = D.Oadd
                    i.arg[0] = t
                    i.arg[1] = getcon(D.s64(c * n), fn)
                    changed = True
            elif KBASE(k) == 0 and i.op == D.Oadd:
                nested = _adddef(fn, i.arg[0], k) or _adddef(fn, i.arg[1], k)
                if nested is not None:
                    leaves = []
                    c = _sum(fn, i.arg[0], k, leaves) + _sum(fn, i.arg[1], k, leaves)
                    leaves.sort()
                    if len(leaves) >= 2 or c:
                        r = leaves[0] if leaves else None
                        for l in leaves[1:-1] if not c else leaves[1:]:
                            t = newtmp('ra', k, fn)
                            x = D.Ins(D.Oadd, k, t, r, l)
                            out.append(x)
                            fn.tmp[t & D.VMASK].def_ = x
                            fn.tmp[t & D.VMASK].nuse = 1
                            r = t
                        if r is not None:
                            i.arg[0] = r
                            i.arg[1] = getcon(D.s64(c), fn) if c else leaves[-1]
                            changed = True
            out.append(i)
        b.ins = out
        b.nins = len(out)
    return changed
