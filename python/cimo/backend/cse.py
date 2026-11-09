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
  do this themselves); on x86-64, a signed one by a power of two is shifts;
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
from .util import newtmp, getcon, chuse, iscmp
from .ssa import filluse



_DIVOPS = (D.Odiv, D.Orem, D.Oudiv, D.Ourem)
_MULS = (D.Omul, D.Osmulh, D.Oumulh)
_LOGIC = (D.Oand, D.Oor, D.Oxor)
MAXHOISTED = 16  # addresses kept in registers across a function's loops
_ARITH = (D.Oadd, D.Osub) + _LOGIC


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


def _expand(i, fn, out, pow2only=False):
    """Instructions for the division or remainder i by a constant, appended to out; False if i
    isn't one this handles (pow2only: not a signed one by a power of two)."""
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
    if pow2only and not pow2:
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


def divconst(fn, pow2only=False):
    """arm64: integer divisions and remainders by a constant, as multiplications. Breaks use.
    pow2only (x86-64): only signed ones by a power of two, as shifts. Returns True if anything changed."""
    changed = False
    for b in fn.rpo[:fn.nblk]:
        if not any(i.op in _DIVOPS and KBASE(i.cls) == 0 for i in b.ins[:b.nins]):
            continue
        out = []
        for i in b.ins[:b.nins]:
            if i.op in _DIVOPS and KBASE(i.cls) == 0 and _expand(i, fn, out, pow2only):
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
    """arm64: a constant that takes several instructions to build, used in a loop, is built once
    at the end of the start block, which dominates every loop (MachineLICM). isel would build it
    at each use, once per unrolled copy, since gvn folds copies of constants: so this runs after
    the last gvn. Requires rpo and loop depths; breaks use. Returns True if anything changed."""
    from .arm64.isel import arm64_logimm
    regs = {}
    for b in fn.rpo[1:fn.nblk]:
        if b.loop <= 1:
            continue
        for i in b.ins[:b.nins]:
            if D.isload(i.op) or D.isstore(i.op) or i.op == D.Oadd:  # a global's address: adrp, and a GOT load
                for n in range(2):
                    r = i.arg[n]
                    if D.rtype(r) != D.RCon:
                        continue
                    c = fn.con[r & D.VMASK]
                    if c.type == D.CAddr and c.sym.type in (D.SGlo, D.SExt) and (r, D.Kl) not in regs \
                            and len(regs) < MAXHOISTED:
                        regs[(r, D.Kl)] = newtmp('con', D.Kl, fn)
                    if (r, D.Kl) in regs and c.type == D.CAddr:
                        i.arg[n] = regs[(r, D.Kl)]
            if KBASE(i.cls) != 0 or not (i.op in _MULS or i.op in _ARITH or D.Ocmpw <= i.op <= D.Ocmpw1
                                         or D.Ocmpl <= i.op <= D.Ocmpl1):
                continue
            for n in range(2):
                r = i.arg[n]
                if D.rtype(r) != D.RCon:
                    continue
                c = fn.con[r & D.VMASK]
                if c.type != D.CBits or not _bigcon(c) or (i.op in _LOGIC and arm64_logimm(c.i, i.cls)):
                    continue
                cc = iscmp(i.op)  # a comparison's own class is its result's
                k = (r, cc[0] if cc else i.cls)
                if k not in regs:
                    regs[k] = newtmp('con', k[1], fn)
                i.arg[n] = regs[k]
    if not regs:
        return False
    for b in fn.rpo[1:fn.nblk]:  # now that it's in a register, the other uses of an address too
        if b.loop > 1:
            continue
        for i in b.ins[:b.nins]:
            if D.isload(i.op) or D.isstore(i.op) or i.op == D.Oadd:
                for n in range(2):
                    r = i.arg[n]
                    if D.rtype(r) == D.RCon and (r, D.Kl) in regs and fn.con[r & D.VMASK].type == D.CAddr:
                        i.arg[n] = regs[(r, D.Kl)]
    st = fn.start
    st.ins = st.ins[:st.nins] + [D.Ins(D.Ocopy, k, t, r, D.R) for (r, k), t in regs.items()]
    st.nins = len(st.ins)
    return True


def _adddef(fn, r, k):
    """The add or sub-of-an-integer-constant defining r in class k, if r is only used once; or None."""
    if D.rtype(r) != D.RTmp or (r & D.VMASK) < D.Tmp0:
        return None
    t = fn.tmp[r & D.VMASK]
    d = t.def_
    if d is None or d.cls != k or t.nuse != 1 or (d.op != D.Oadd and not (
            d.op == D.Osub and _conof(fn, d.arg[1]) is not None)):  # not an address's: `x - $t`
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


def _distribute(fn, i, k, out):
    """LLVM's Reassociate: a * b + a * c as a * (b + c), each product only used here."""
    d = []
    for r in i.arg:
        x = fn.tmp[r & D.VMASK].def_ if D.rtype(r) == D.RTmp and (r & D.VMASK) >= D.Tmp0 else None
        if x is None or x.op != D.Omul or x.cls != k or fn.tmp[r & D.VMASK].nuse != 1:
            return False
        d.append(x)
    for p in (0, 1):
        for q in (0, 1):
            if d[0].arg[p] == d[1].arg[q] and D.rtype(d[0].arg[p]) == D.RTmp:
                u, v = d[0].arg[p ^ 1], d[1].arg[q ^ 1]
                if D.rtype(u) == D.RCon and D.rtype(v) == D.RCon:
                    return False
                t = newtmp('ra', k, fn)
                x = D.Ins(i.op, k, t, u, v)
                out.append(x)
                fn.tmp[t & D.VMASK].def_ = x
                fn.tmp[t & D.VMASK].nuse = 1
                i.op, i.arg[0], i.arg[1] = D.Omul, d[0].arg[p], t
                return True
    return False


def ranges(fn):
    """LLVM's InstCombine: k0 <= x && x <= k1 as one unsigned comparison, (unsigned)(x - k0) <= k1 - k0;
    x < k0 || x > k1 as its opposite; x == k || x == k + 1 the same way; x == k || x == k | m, one bit
    apart, as (x | m) == (k | m). Each comparison only used here."""
    changed = False
    for b in fn.rpo[:fn.nblk]:
        if not any(i.op in (D.Oand, D.Oor) and i.cls == D.Kw for i in b.ins[:b.nins]):
            continue
        out = []
        for i in b.ins[:b.nins]:
            got = _range(fn, i) if i.op in (D.Oand, D.Oor) and i.cls == D.Kw else None
            if got is not None:
                out += got
                changed = True
            else:
                out.append(i)
        b.ins = out
        b.nins = len(out)
    return changed


def _bound(fn, r):
    """(x, class, condition, constant) of a single-use integer comparison with a constant, x on the left."""
    if D.rtype(r) != D.RTmp or (r & D.VMASK) < D.Tmp0 or fn.tmp[r & D.VMASK].nuse != 1:
        return None
    d = fn.tmp[r & D.VMASK].def_
    cm = iscmp(d.op) if d is not None else None
    if cm is None or KBASE(cm[0]) != 0:
        return None
    c = _conof(fn, d.arg[1])
    if c is None or D.rtype(d.arg[0]) != D.RTmp:
        return None
    return d.arg[0], cm[0], cm[1], c


def _range(fn, i):
    a, c = _bound(fn, i.arg[0]), _bound(fn, i.arg[1])
    if a is None or c is None or a[0] != c[0] or a[1] != c[1]:
        return None
    x, k = a[0], a[1]
    w = 64 if k == D.Kl else 32
    full = (1 << w) - 1
    if i.op == D.Oor and a[2] == D.Cieq and c[2] == D.Cieq:
        lo, hi = sorted((a[3] & full, c[3] & full))
        if hi == lo + 1:
            return _inrange(fn, i, x, k, lo, hi, False, D.Ciule)
        m = lo ^ hi
        if m & (m - 1) == 0 and m:
            t = newtmp('rg', k, fn)
            return [D.Ins(D.Oor, k, t, x, getcon(m, fn)), D.Ins(_cmpop(k, D.Cieq), D.Kw, i.to, t, getcon(hi | m, fn))]
        return None
    if a[2] not in _BOUNDS or c[2] not in _BOUNDS or (a[2] in _SIGNED) != (c[2] in _SIGNED):
        return None
    signed = a[2] in _SIGNED
    lo = hi = None
    for cc, n in ((a[2], a[3]), (c[2], c[3])):
        n = _sign(n, w) if signed else n & full
        low, by = _BOUNDS[cc][i.op == D.Oor]
        if low:
            lo = n + by
        else:
            hi = n + by
    if lo is None or hi is None or lo > hi:
        return None
    if (signed and not (-(1 << (w - 1)) <= lo and hi < 1 << (w - 1))) or (not signed and not (0 <= lo and hi <= full)):
        return None
    return _inrange(fn, i, x, k, lo, hi, i.op == D.Oor, D.Ciule)


def _sign(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if v >> (w - 1) else v


# what a comparison with n bounds: (is it the lower bound, n's distance from it), for x in the range when
# it holds (&&) and for x out of it when it holds (||)
_BOUNDS = {D.Cisge: ((True, 0), (False, -1)), D.Cisgt: ((True, 1), (False, 0)),
           D.Cisle: ((False, 0), (True, 1)), D.Cislt: ((False, -1), (True, 0)),
           D.Ciuge: ((True, 0), (False, -1)), D.Ciugt: ((True, 1), (False, 0)),
           D.Ciule: ((False, 0), (True, 1)), D.Ciult: ((False, -1), (True, 0))}
_SIGNED = frozenset((D.Cisge, D.Cisgt, D.Cisle, D.Cislt))


def _cmpop(k, cc):
    return (D.Ocmpw if k == D.Kw else D.Ocmpl) + cc


def _inrange(fn, i, x, k, lo, hi, out, cc):
    w = 64 if k == D.Kl else 32
    full = (1 << w) - 1
    t = newtmp('rg', k, fn)
    op = _cmpop(k, D.Ciugt if out else cc)
    return [D.Ins(D.Osub, k, t, x, getcon(D.s64(lo & full), fn)),
            D.Ins(op, D.Kw, i.to, t, getcon(D.s64((hi - lo) & full), fn))]


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
            elif KBASE(k) == 0 and i.op in (D.Oadd, D.Osub) and _distribute(fn, i, k, out):
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


_BYTEDEAD = {D.Oor, D.Oshl, D.Oshr, D.Oand, D.Oadd, D.Oextub, D.Oextuh, D.Oextuw, D.Oextsw, D.Oloadub, D.Oloaduh, D.Oloadsw,
             D.Oloaduw, D.Oload, D.Ocopy}


def _byteof(fn, r, n, st):
    """Where each of the n low bytes of r comes from (LLVM's calculateByteProvider, AggressiveInstCombine):
    a list of 0 (known zero), ('v', tmp, byte) or ('m', base, offset, load); or None if unknown."""
    st[0] -= 1
    if st[0] < 0:
        return None
    if D.rtype(r) == D.RCon:
        c = fn.con[D.rval(r)]
        if c.type != D.CBits:
            return None
        return [0 if not c.i >> (8 * j) & 255 else None for j in range(n)]
    if D.rtype(r) != D.RTmp or D.rval(r) < D.Tmp0:
        return None
    t = fn.tmp[D.rval(r)]
    d = t.def_
    width = 8 if t.cls == D.Kl else 4
    leaf = [('v', r, j) if j < width else None for j in range(n)]
    if d is None:
        return leaf
    op = d.op
    if op == D.Ocopy:
        return _byteof(fn, d.arg[0], n, st)
    if op == D.Oor:
        a = _byteof(fn, d.arg[0], n, st)
        c = _byteof(fn, d.arg[1], n, st)
        if a is None or c is None:
            return leaf
        out = []
        for x, y in zip(a, c):
            if x == 0:
                out.append(y)
            elif y == 0:
                out.append(x)
            else:
                out.append(None)
        return out
    if op in (D.Oshl, D.Oshr) and D.rtype(d.arg[1]) == D.RCon:
        s = _conof(fn, d.arg[1])
        if s is None or s % 8 or not 0 <= s < width * 8:
            return leaf
        s //= 8
        a = _byteof(fn, d.arg[0], width, st)
        if a is None:
            return leaf
        if op == D.Oshl:
            out = [0] * s + a[:width - s]
        else:
            out = a[s:] + [0] * s
        return (out + [None] * n)[:n]
    if op == D.Oand and _conof(fn, d.arg[1]) is not None:
        m = _conof(fn, d.arg[1])
        a = _byteof(fn, d.arg[0], width, st)
        if a is None or any((m >> (8 * j) & 255) not in (0, 255) for j in range(width)):
            return leaf
        return ([x if m >> (8 * j) & 255 else 0 for j, x in enumerate(a)] + [None] * n)[:n]
    if op in (D.Oextub, D.Oextuh, D.Oextuw, D.Oextsw):
        src = {D.Oextub: 1, D.Oextuh: 2}.get(op, 4)
        a = _byteof(fn, d.arg[0], src, st)
        if a is None or (op == D.Oextsw and a[3] != 0):
            return leaf
        return ([a[j] if j < src else 0 for j in range(8)])[:n]
    if (op in (D.Oloadub, D.Oloaduh, D.Oloaduw, D.Oload) or op == D.Oloadsw and d.cls == D.Kw) and KBASE(d.cls) == 0:
        size = {D.Oloadub: 1, D.Oloaduh: 2, D.Oloaduw: 4, D.Oloadsw: 4}.get(op, width)
        base, off = _baseoff(fn, d.arg[0])
        if base is None:
            return leaf
        return ([('m', base, off + j, d, off) if j < size else 0 for j in range(8)])[:n]
    return leaf


def _baseoff(fn, r):
    """An address as (what it adds up to, without constants; the constants), or (None, 0). reassoc left
    copies of its sums for the next gvn, so the sum is compared and not the temporary."""
    leaves = []
    off = 0
    todo = [r]
    while todo:
        x = todo.pop()
        if D.rtype(x) == D.RCon:
            c = fn.con[D.rval(x)]
            if c.type == D.CBits:
                off += c.i
            elif c.type == D.CAddr and c.sym.type in (D.SGlo, D.SExt):
                off += c.i
                leaves.append(('s', c.sym.id))
            else:
                return None, 0
        elif D.rtype(x) == D.RTmp and D.rval(x) >= D.Tmp0:
            d = fn.tmp[D.rval(x)].def_
            if d is not None and d.op == D.Oadd and d.cls == D.Kl and len(leaves) + len(todo) < 8:
                todo += [d.arg[0], d.arg[1]]
            else:
                leaves.append(('t', D.rval(x)))
        else:
            return None, 0
    return tuple(sorted(leaves)), off


def _combineor(fn, b, n):
    """Instructions replacing the or at b.ins[n], and where they go; or None."""
    i = b.ins[n]
    k = i.cls
    size = 8 if k == D.Kl else 4
    src = _byteof(fn, i.to, size, [64])
    if src is None or None in src:
        return None
    w = size
    while w and src[w - 1] == 0:
        w -= 1
    if w not in (2, 4, 8) or 0 in src[:w]:
        return None
    kinds = {s[0] for s in src[:w]}
    if kinds == {'v'}:
        x = src[0][1]
        if any(s[1] != x or s[2] != w - 1 - j for j, s in enumerate(src[:w])):
            return None
        if w == 2 and k == D.Kw:
            return n, [(D.Obswap16, D.Kw, i.to, x)]
        if w == size and fn.tmp[D.rval(x)].cls == k:
            return n, [(D.Obswap, k, i.to, x)]
        return None
    if kinds != {'m'} or len({id(s[3]) for s in src[:w]}) < 2 or any(s[1] != src[0][1] for s in src[:w]):
        return None
    lo = min(s[2] for s in src[:w])
    if all(s[2] == lo + j for j, s in enumerate(src[:w])):
        swap = False
    elif all(s[2] == lo + w - 1 - j for j, s in enumerate(src[:w])):
        swap = True
    else:
        return None
    first = next(s for s in src[:w] if s[2] == lo)  # the lowest byte; its load may begin before it
    addr, ahead = first[3].arg[0], lo - first[4]
    pos = {}
    for j, x in enumerate(b.ins[:n]):
        pos[id(x)] = j
    if any(id(s[3]) not in pos for s in src[:w]):
        return None  # loaded in another block
    at = [pos[id(s[3])] for s in src[:w]]
    for x in b.ins[min(at):max(at)]:
        if D.isstore(x.op) or x.op in (D.Ocall, D.Oblit0, D.Oblit1) or D.isvstore(x.op) or D.isvload(x.op) or D.isamx(x.op):
            return None
    last = max(at) + 1
    pre = []
    if ahead:
        a = newtmp('lb', D.Kl, fn)
        pre = [(D.Oadd, D.Kl, a, addr, getcon(ahead, fn))]
        addr = a
    load = {2: D.Oloaduh, 4: D.Oload if k == D.Kw else D.Oloaduw, 8: D.Oload}[w]
    if not swap:
        return last, pre + [(load, k, i.to, addr)]
    if w == size:
        t = newtmp('lb', k, fn)
        return last, pre + [(load, k, t, addr), (D.Obswap, k, i.to, t)]
    t = newtmp('lb', D.Kw, fn)
    if w == 2 and k == D.Kw:
        return last, pre + [(D.Oloaduh, D.Kw, t, addr), (D.Obswap16, D.Kw, i.to, t)]
    if w == 4:  # 32 bits, zero-extended
        u = newtmp('lb', D.Kw, fn)
        return last, pre + [(D.Oload, D.Kw, t, addr), (D.Obswap, D.Kw, u, t), (D.Oextuw, D.Kl, i.to, u)]
    return None


def bytepatterns(fn):
    """LLVM's bswap matching and load combining, GCC's optimize_bswap: an or of bytes that are another
    value's in reverse is rev; of bytes that are consecutive in memory is one load, perhaps then rev. Only
    when nothing can write between the loads. Returns True if anything changed."""
    changed = False
    for b in fn.rpo[:fn.nblk]:
        n = b.nins - 1
        while n >= 0:
            i = b.ins[n]
            if i.op != D.Oor or KBASE(i.cls) != 0 or D.rtype(i.to) != D.RTmp:
                n -= 1
                continue
            got = _combineor(fn, b, n)
            if got is None:
                n -= 1
                continue
            at, new = got
            ins = [D.Ins(*x) if len(x) == 5 else D.Ins(x[0], x[1], x[2], x[3], D.R) for x in new]
            b.ins = b.ins[:b.nins]
            if at == n:
                b.ins[n:n + 1] = ins
            else:  # the root stays, a copy of what is loaded earlier: nothing moves past it
                t = newtmp('lb', i.cls, fn)
                for x in ins:
                    if x.to == i.to:
                        x.to = t
                i.op, i.arg[0], i.arg[1] = D.Ocopy, t, D.R
                b.ins[at:at] = ins
            for x in ins:
                fn.tmp[D.rval(x.to)].def_ = x
            b.nins = len(b.ins)
            changed = True
            n = (n if at == n else at) - 1
    if changed:  # the byte loads go now, or the next load elimination builds the new load from them again
        filluse(fn)
        for b in reversed(fn.rpo[:fn.nblk]):
            for i in reversed(b.ins[:b.nins]):
                if i.op in _BYTEDEAD and D.rtype(i.to) == D.RTmp and fn.tmp[D.rval(i.to)].nuse == 0:
                    for a in i.arg:
                        if D.rtype(a) == D.RTmp:
                            chuse(a, -1, fn)
                    i.set(D.Ins(D.Onop))
    return changed
