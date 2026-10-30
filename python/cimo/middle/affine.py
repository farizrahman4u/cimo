"""Induction variables and affine forms of the values computed in a loop.

The value of an integer temporary in iteration t (t = 0, 1, ...) of a loop is described, when
possible, as an affine form `L + S*t (mod 2^W)`, W = 32 or 64 (the temporary's class), where
L and S are linear combinations of loop-invariant leaves with integer coefficients (`Lin`).

Leaves are keyed (ref, kind): kind 'l' is a 64-bit value, 'w' the low 32 bits of a value,
's' / 'u' the sign / zero extension of the low 32 bits of a value to 64 bits; 'x' marks a
non-integer invariant (only ever a whole form, never combined). Two's-complement add, sub,
mul and shl are ring homomorphisms mod 2^W, so a form computed with exact integers is the
temporary's value mod 2^W no matter where the C program's intermediate values wrapped.

Extensions are the only non-homomorphic step: `extsw v` with v = L + S*t (32-bit) equals the
64-bit form `ext(L) + S*t` exactly when that exact value lies in the int32 range (uint32 for
`extuw`) for every t that runs. It is affine in t, so checking t = 0 and t = T-1 at run time
is enough; such forms carry the ids of the checks they rely on (`AV.chk`), and the loop
transformation emits those checks in the versioning guard. A compile-time fact (C's
"signed overflow is undefined") is deliberately not used: QBE IR doesn't say whether a `w`
add came from a signed or an unsigned C operation.
"""

from ..backend import defs as D
from ..backend.defs import R, RTmp, RCon, CBits, CAddr, Kw, Kl, rtype, rval, s32, s64
from . import ir

MAXCOEF = 1 << 16
MAXSTEP = 1 << 20


def norm(w, v):
    return s32(v) if w == 32 else s64(v)


class Lin:
    """sum(coef * leaf) + c, immutable."""
    __slots__ = ('t', 'c')

    def __init__(self, t=None, c=0):
        self.t = t or {}
        self.c = c

    @staticmethod
    def leaf(key):
        return Lin({key: 1}, 0)

    def isconst(self):
        return not self.t

    def add(self, o, w, k=1):
        t = dict(self.t)
        for key, v in o.t.items():
            n = norm(w, t.get(key, 0) + k * v)
            if n:
                t[key] = n
            else:
                t.pop(key, None)
        return Lin(t, norm(w, self.c + k * o.c))

    def scale(self, k, w):
        if norm(w, k) == 0:
            return Lin()
        return Lin({key: norm(w, v * k) for key, v in self.t.items() if norm(w, v * k)},
                   norm(w, self.c * k))

    def noconst(self):
        return Lin(self.t, 0)

    def ident(self):
        return tuple(sorted(self.t.items())), self.c

    def mapkeys(self, f, w):
        t = {}
        for key, v in self.t.items():
            nk = f(key)
            n = norm(w, t.get(nk, 0) + v)
            if n:
                t[nk] = n
            else:
                t.pop(nk, None)
        return Lin(t, norm(w, self.c))

    def __eq__(self, o):
        return isinstance(o, Lin) and self.t == o.t and self.c == o.c

    def __hash__(self):
        return hash(self.ident())

    def __repr__(self):
        s = ' + '.join('%d*%s' % (v, _kname(k)) for k, v in sorted(self.t.items()))
        return '(%s%s%d)' % (s, ' + ' if s else '', self.c)


def _kname(k):
    r, kind = k
    return '%s%x:%s' % ('t' if rtype(r) == RTmp else 'c', rval(r), kind)


ZERO = Lin()


class AV:
    """Affine value L + S*t mod 2^w (w = 32 or 64; w = 0 for an opaque invariant)."""
    __slots__ = ('w', 'L', 'S', 'chk')

    def __init__(self, w, L, S=ZERO, chk=frozenset()):
        self.w = w
        self.L = L
        self.S = S
        self.chk = chk

    def inv(self):
        return self.S.isconst() and self.S.c == 0

    def __repr__(self):
        return 'AV%d(%r + %r*t)' % (self.w, self.L, self.S)


def trunc(f):
    """64-bit form used as a 32-bit operand."""
    if f.w == 32:
        return f
    k = lambda key: (key[0], 'w') if key[1] in ('l', 's', 'u', 'w') else key
    return AV(32, f.L.mapkeys(k, 32), f.S.mapkeys(k, 32), f.chk)


class Check:
    """Run-time condition: ext(L + S*t) is exact for t in [0, T-1] (kind 's' or 'u')."""
    __slots__ = ('kind', 'L', 'S')

    def __init__(self, kind, L, S):
        self.kind = kind
        self.L = L
        self.S = S

    def ident(self):
        return self.kind, self.L.ident(), self.S.c


def _haskind(lin, kind):
    return any(k[1] == kind for k in lin.t)


class Analysis:
    """Affine forms of everything computed in loop `lp`, whose blocks run in the order
    `order` each iteration (straight-line: header, then the body)."""

    def __init__(self, fn, lp, order, hoistloads=frozenset()):
        self.fn = fn
        self.lp = lp
        self.order = order
        self.inloop_ids = set(b.id for b in lp.blocks)
        self.hoistloads = hoistloads   # tmp indices of loads proven invariant
        self.forms = {}
        self.level = {}                # in-loop invariant tmp -> 'G' (speculatable) or 'F'
        self.defins = {}               # tmp -> defining Ins (in the loop)
        self.checks = []
        self._checkids = {}
        self.biv = {}                  # phi tmp -> (init ref, step Lin)
        self.headerins = set(ir.ins(order[0]))
        for b in order:
            for i in ir.ins(b):
                if ir.istmp(i.to):
                    self.defins[rval(i.to)] = i
        self._find_bivs()
        self._evaluate()

    # -- helpers

    def inloop(self, r):
        return rtype(r) == RTmp and self.fn.tmp[rval(r)].bid in self.inloop_ids

    def leaflevel(self, key):
        r = key[0]
        if not self.inloop(r):
            return 'G'
        return self.level.get(rval(r), 'X')

    def linlevel(self, lin):
        lv = 'G'
        for key in lin.t:
            l = self.leaflevel(key)
            if l == 'X':
                return 'X'
            if l == 'F':
                lv = 'F'
        return lv

    def avlevel(self, f):
        a = self.linlevel(f.L)
        b = self.linlevel(f.S)
        if 'X' in (a, b):
            return 'X'
        return 'F' if 'F' in (a, b) else 'G'

    def _check(self, kind, L, S):
        c = Check(kind, L, S)
        k = c.ident()
        if k not in self._checkids:
            self._checkids[k] = len(self.checks)
            self.checks.append(c)
        return self._checkids[k]

    # -- operand forms

    def opnd(self, r, W, phiforms):
        """Form of r used in a W-bit integer context (W=0: any class), or None."""
        if rtype(r) == RCon:
            c = self.fn.con[rval(r)]
            if c.type == CBits:
                if W == 0:
                    return AV(0, Lin.leaf((r, 'x')))
                return AV(W, Lin(c=norm(W, c.i)))
            if c.type == CAddr:
                f = AV(64, Lin.leaf((r, 'l')))
                return trunc(f) if W == 32 else f if W == 64 else AV(0, Lin.leaf((r, 'x')))
            return None
        if rtype(r) != RTmp:
            return None
        t = rval(r)
        if t in phiforms:
            f = phiforms[t]
        elif t in self.forms:
            f = self.forms[t]
        elif not self.inloop(r):
            k = self.fn.tmp[t].cls
            if k == Kl:
                f = AV(64, Lin.leaf((r, 'l')))
            elif k == Kw:
                f = AV(32, Lin.leaf((r, 'w')))
            else:
                f = AV(0, Lin.leaf((r, 'x')))
        else:
            return None
        if f is None:
            return None
        if W == 0:
            if f.w == 0:
                return f
            return f if f.inv() else None
        if f.w == 0:
            return None
        if f.w == 64 and W == 32:
            return trunc(f)
        if f.w == 32 and W == 64:
            return None
        return f

    # -- evaluation

    def _eval_ins(self, i, phiforms, symbolic):
        """Form of i's result, or None if it varies non-affinely. In non-symbolic mode,
        records invariant instructions in self.level."""
        op = i.op
        cls = i.cls
        W = 32 if cls == Kw else 64 if cls == Kl else 0
        t = rval(i.to)
        if ir.isload(op):
            if not symbolic and t in self.hoistloads:
                a = self.opnd(i.arg[0], 64, phiforms)
                if a is not None and a.inv() and self.avlevel(a) != 'X':
                    self.level[t] = 'F'
                    return AV(W, Lin.leaf((i.to, 'l' if W == 64 else 'w'))) if W else \
                        AV(0, Lin.leaf((i.to, 'x')))
            return None
        if op not in ir.PURE:
            return None
        if W and op in (D.Oadd, D.Osub, D.Oneg, D.Omul, D.Oshl, D.Ocopy):
            a = self.opnd(i.arg[0], W, phiforms)
            b = self.opnd(i.arg[1], W, phiforms) if op not in (D.Oneg, D.Ocopy) else None
            r = self._arith(op, W, a, b)
            if r is not None:
                return r
        elif op in (D.Oextsw, D.Oextuw) and cls == Kl:
            a = self.opnd(i.arg[0], 32, phiforms)
            if a is not None and not a.inv() and not symbolic:
                if (a.S.isconst() and abs(a.S.c) < MAXSTEP and not _haskind(a.L, 'φ')
                        and sum(abs(v) for v in a.L.t.values()) < MAXCOEF):
                    kind = 's' if op == D.Oextsw else 'u'
                    L = a.L.mapkeys(lambda key: (key[0], kind), 64)
                    L = Lin(L.t, s32(a.L.c))
                    if self.linlevel(L) == 'G':
                        cid = self._check(kind, L, a.S)
                        return AV(64, L, Lin(c=s32(a.S.c)), a.chk | {cid})
        # generic: invariant if every operand is invariant
        if symbolic:
            return None
        # pure instructions of the header run at least once (P(0)) before anything else
        # of the loop, so even a trapping one can move to the guard
        lv = 'G' if op in ir.PURE_SAFE or i in self.headerins else 'F'
        for a in i.arg:
            if a == R:
                continue
            f = self.opnd(a, 0, phiforms)
            if f is None or not f.inv():
                return None
            l = self.avlevel(f)
            if l == 'X':
                return None
            if l == 'F':
                lv = 'F'
        self.level[t] = lv
        if W:
            return AV(W, Lin.leaf((i.to, 'l' if W == 64 else 'w')))
        return AV(0, Lin.leaf((i.to, 'x')))

    def _arith(self, op, W, a, b):
        if a is None:
            return None
        if op == D.Ocopy:
            return a
        if op == D.Oneg:
            return AV(W, a.L.scale(-1, W), a.S.scale(-1, W), a.chk)
        if b is None:
            return None
        chk = a.chk | b.chk
        if op == D.Oadd:
            return AV(W, a.L.add(b.L, W), a.S.add(b.S, W), chk)
        if op == D.Osub:
            return AV(W, a.L.add(b.L, W, -1), a.S.add(b.S, W, -1), chk)
        if op == D.Oshl:
            if b.inv() and b.L.isconst() and 0 <= b.L.c < W:
                k = 1 << b.L.c
                return AV(W, a.L.scale(k, W), a.S.scale(k, W), chk)
            return None
        if op == D.Omul:
            for x, y in ((a, b), (b, a)):
                if x.inv() and x.L.isconst():
                    k = x.L.c
                    return AV(W, y.L.scale(k, W), y.S.scale(k, W), chk)
            for x, y in ((a, b), (b, a)):
                # (c0 + s0*t) * X with X an invariant linear combination
                if (y.inv() and not x.inv() and x.L.isconst() and x.S.isconst()
                        and not _haskind(y.L, 'φ')):
                    return AV(W, y.L.scale(x.L.c, W), y.L.scale(x.S.c, W), chk)
            return None
        return None

    def _run(self, phiforms, symbolic):
        forms = {}
        self.forms = forms
        for b in self.order:
            for i in ir.ins(b):
                if ir.istmp(i.to):
                    forms[rval(i.to)] = self._eval_ins(i, phiforms, symbolic)
        return forms

    def _find_bivs(self):
        h = self.lp.header
        latch = self.lp.latches[0]
        sym = {}
        for p in ir.phis(h):
            W = 32 if p.cls == Kw else 64 if p.cls == Kl else 0
            if W:
                sym[rval(p.to)] = AV(W, Lin.leaf((p.to, 'φ')))
        forms = self._run(sym, True)
        self.phiforms = {}
        for p in ir.phis(h):
            t = rval(p.to)
            self.phiforms[t] = None
            if t not in sym:
                continue
            W = sym[t].w
            nxt = ir.phiargof(p, latch)
            init = [a for bb, a in zip(p.blk[:p.narg], p.arg[:p.narg]) if bb is not latch]
            if len(init) != 1:
                continue
            f = self.opnd(nxt, W, sym)
            if f is None or not f.inv():
                continue
            d = f.L.add(Lin.leaf((p.to, 'φ')), W, -1)
            if _haskind(d, 'φ') or (d.isconst() and d.c == 0) or f.chk:
                continue
            fi = self.opnd(init[0], W, {})
            if fi is None or not fi.inv():
                continue
            self.biv[t] = (init[0], d)
            self.phiforms[t] = (fi, d)

    def _evaluate(self):
        phiforms = {}
        for t, v in self.phiforms.items():
            if v is None:
                phiforms[t] = None
            else:
                fi, d = v
                W = 32 if self.fn.tmp[t].cls == Kw else 64
                if self.linlevel(d) == 'X' or self.avlevel(fi) == 'X':
                    phiforms[t] = None
                    self.biv.pop(t, None)
                else:
                    phiforms[t] = AV(W, fi.L, d)
        self.phiforms = phiforms
        self._run(phiforms, False)
        self.forms.update(phiforms)

    def form(self, r, W=None):
        """Form of a ref as seen inside the loop; W defaults to the ref's class."""
        if W is None:
            W = 64
            if rtype(r) == RTmp:
                k = self.fn.tmp[rval(r)].cls
                W = 32 if k == Kw else 64 if k == Kl else 0
        return self.opnd(r, W, self.phiforms)
