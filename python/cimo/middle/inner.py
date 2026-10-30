"""Innermost counted loops: versioning, rotation, strength reduction, LICM, unrolling with a
remainder loop, scalar replacement of loads reused across iterations, full unrolling.

Shape handled: a loop whose header H ends in `jnz c` with one successor outside (the only
exit) and whose body is either H alone (bottom-tested) or H plus one block B that jumps back
to H. Each iteration runs P (H's instructions, including the exit test), then Q (B's); with T
iterations the loop runs P(0) Q(0) ... P(T-1) Q(T-1) P(T) and leaves after P(T)'s test.

The exit test must compare an affine value x = x0 + s*t (s a non-zero constant) with an
invariant bound b. The trip count T is computed exactly at run time; when it is a
compile-time constant and small, the loop is fully unrolled instead.

Versioned form (the original loop is kept, untouched, as the slow path):

    G1: cont0 = x0 REL b                          ; first test; else slow path
    G2: T, overflow and extension checks          ; else slow path
    FP: hoisted invariants, IV inits, end values
    M:  U copies of P+Q per trip                  ; bottom-tested: iv != end4
    RC: remainder check                           ; iv != end
    R:  1 copy of P+Q per trip                    ; bottom-tested: iv != end
    FX: P(T), for the values the loop leaves with
    X:  join of the slow and fast paths, then the original exit block

Copies run in the original order, so memory accesses and floating-point operations happen
exactly as in C. Values affine in t (addresses, counters) are not copied: they are computed
from one induction variable per (base, stride) group, so unrolled accesses become
`base + constant` (folded into [reg, #imm] by isel).
"""

from ..backend import defs as D
from ..backend.defs import R, RTmp, RCon, Kw, Kl, Ks, Kd, Ins, rtype, rval, s32, s64
from ..backend.util import iscmp, newtmp
from . import ir
from .ir import Bail, Emitter
from .affine import Analysis, Lin, AV, norm
import cimo.middle as M

MAXFULL = 16         # full unrolling: at most this many iterations
MAXFULLINS = 320     # ... and this many copied instructions
MAXBODY = 160        # unrolled main-loop body size budget (instructions)
SR_MAXSPAN = 8

_REL = {D.Cieq: 'eq', D.Cine: 'ne', D.Cisge: 'sge', D.Cisgt: 'sgt', D.Cisle: 'sle',
        D.Cislt: 'slt', D.Ciuge: 'uge', D.Ciugt: 'ugt', D.Ciule: 'ule', D.Ciult: 'ult'}
_NEG = {'eq': 'ne', 'ne': 'eq', 'sge': 'slt', 'slt': 'sge', 'sgt': 'sle', 'sle': 'sgt',
        'uge': 'ult', 'ult': 'uge', 'ugt': 'ule', 'ule': 'ugt'}
_SWAP = {'eq': 'eq', 'ne': 'ne', 'sge': 'sle', 'sle': 'sge', 'sgt': 'slt', 'slt': 'sgt',
         'uge': 'ule', 'ule': 'uge', 'ugt': 'ult', 'ult': 'ugt'}


def cmpop(rel, W):
    return getattr(D, 'Oc%s%s' % (rel, 'w' if W == 32 else 'l'))


def evalrel(rel, W, a, b):
    if rel[0] == 'u':
        m = (1 << W) - 1
        a &= m
        b &= m
    else:
        a = norm(W, a)
        b = norm(W, b)
    return {'eq': a == b, 'ne': a != b, 'sge': a >= b, 'sgt': a > b, 'sle': a <= b,
            'slt': a < b, 'uge': a >= b, 'ugt': a > b, 'ule': a <= b, 'ult': a < b}[rel]


class Shape:
    """The parts of a loop in the handled shape."""

    def __init__(self, fn, lp):
        if len(lp.latches) != 1:
            raise Bail('several latches')
        h = lp.header
        self.H = h
        self.latch = lp.latches[0]
        if len(lp.blocks) == 1:
            self.B = None
        elif len(lp.blocks) == 2:
            (b,) = [x for x in lp.blocks if x is not h]
            if b.npred != 1 or b.jmp.type != D.Jjmp or b.s1 is not h or b.phi:
                raise Bail('body shape')
            self.B = b
        else:
            raise Bail('multi-block body')
        if h.jmp.type != D.Jjnz:
            raise Bail('header does not branch')
        if h.s1 in lp.blocks and h.s2 not in lp.blocks:
            self.cont_true = True
            self.E = h.s2
        elif h.s2 in lp.blocks and h.s1 not in lp.blocks:
            self.cont_true = False
            self.E = h.s1
        else:
            raise Bail('exit shape')
        if self.B is not None and (h.s1 if self.cont_true else h.s2) is not self.B:
            raise Bail('exit shape')
        outs = lp.outside_preds()
        if len(outs) != 1 or outs[0].jmp.type != D.Jjmp:
            raise Bail('no preheader')
        self.pre = outs[0]
        self.order = [h] if self.B is None else [h, self.B]
        for b in self.order:
            for i in ir.ins(b):
                if not ir.copyable(i):
                    raise Bail('uncopyable instruction %s' % D.optab[i.op].name)
        c = h.jmp.arg
        if rtype(c) != RTmp or fn.tmp[rval(c)].bid != h.id:
            raise Bail('exit test not computed in the header')
        self.test = None        # None: `jnz c` on an integer c, i.e. c != 0
        for i in ir.ins(h):
            if i.to == c and iscmp(i.op) is not None:
                self.test = i
        self.cond = c


class Control:
    """Normalized exit test: continue while x REL b, x = x0 + s*t."""

    def __init__(self, A, sh):
        if sh.test is None:
            k, c = Kw, D.Cine
            a, b = sh.cond, A.fn_con0
        else:
            k, c = iscmp(sh.test.op)
            a, b = sh.test.arg
        if k not in (Kw, Kl):
            raise Bail('float exit test')
        W = 32 if k == Kw else 64
        rel = _REL[c]
        if not sh.cont_true:
            rel = _NEG[rel]
        fa, fb = A.form(a, W), A.form(b, W)
        if fa is None or fb is None:
            raise Bail('exit test not affine')
        if fa.inv() and not fb.inv():
            fa, fb = fb, fa
            rel = _SWAP[rel]
        if fa.inv() or not fb.inv():
            raise Bail('exit test: no induction variable')
        self.sym = not fa.S.isconst()
        if self.sym:
            # symbolic step: its sign comes from the test, checked at run time
            if A.linlevel(fa.S) != 'G' or rel not in ('slt', 'ult', 'sle', 'ule', 'sgt',
                                                       'ugt', 'sge', 'uge'):
                raise Bail('exit test: symbolic step')
            s = 1 if rel[1:] in ('lt', 'le') else -1
        else:
            s = norm(W, fa.S.c)
        if s == 0:
            raise Bail('zero step')
        if A.avlevel(fa) != 'G' or A.avlevel(fb) != 'G' or fb.chk:
            raise Bail('exit test operands not computable before the loop')
        if rel in ('slt', 'ult') and s > 0:
            kind = 'up_lt'
        elif rel in ('sle', 'ule') and s > 0:
            kind = 'up_le'
        elif rel in ('sgt', 'ugt') and s < 0:
            kind = 'down_gt'
        elif rel in ('sge', 'uge') and s < 0:
            kind = 'down_ge'
        elif rel == 'ne' and abs(s) & (abs(s) - 1) == 0:
            kind = 'ne'
        else:
            raise Bail('exit test direction')
        if abs(s) >= (1 << 20):
            raise Bail('huge step')
        self.W, self.rel, self.kind, self.s = W, rel, kind, s
        self.x, self.b = fa, fb

    def const_trips(self, limit):
        """Trip count if x0 and b are constants and it is at most limit, else None."""
        if self.sym or not (self.x.L.isconst() and self.b.L.isconst()):
            return None
        W = self.W
        x = self.x.L.c
        bb = self.b.L.c
        n = 0
        while evalrel(self.rel, W, x, bb):
            n += 1
            if n > limit:
                return None
            x = norm(W, x + self.s)
        return n


class Group:
    __slots__ = ('W', 'key', 'L0', 'S', 'c0', 'init', 'ks', 'idx')


class Chain:
    """Loads of one array at positions qmin..qmax (stride S apart) in every iteration."""
    __slots__ = ('op', 'cls', 'Lr', 'S', 'qmin', 'qmax', 'pos', 'fresh')


class Fast:
    def __init__(self, fn, lp, sh, A, rts):
        self.fn = fn
        self.lp = lp
        self.sh = sh
        self.A = A
        self.rts = rts
        self.memo = {}
        self.em = None
        self.groups = {}
        self.chains = []
        self.srload = {}       # load tmp -> (chain, position)
        self.chk = set()       # ids of the extension checks the fast loop relies on

    # ---------- materialization of invariant values (in self.em, memoized)

    def leaf(self, key):
        r, kind = key
        if key in self.memo:
            return self.memo[key]
        if self.A.inloop(r):
            base = self.hoist(r)
        else:
            base = r
        em = self.em
        if kind in ('l', 'w', 'x'):
            v = base
        elif kind == 's':
            v = em.op(D.Oextsw, Kl, base)
        elif kind == 'u':
            v = em.op(D.Oextuw, Kl, base)
        else:
            raise AssertionError(kind)
        self.memo[key] = v
        return v

    def hoist(self, r):
        """An in-loop invariant instruction, copied before the loop."""
        key = ('h', r)
        if key in self.memo:
            return self.memo[key]
        i = self.A.defins[rval(r)]
        args = [self.inv_operand(a) if a != R else R for a in i.arg]
        if i.op in ir.PURE and i.cls in (Kw, Kl):
            v = self.em.op(i.op, i.cls, args[0], args[1])
        else:
            v = self.em.raw(i.op, i.cls, args[0], args[1])
        self.memo[key] = v
        return v

    def inv_operand(self, a):
        if not self.A.inloop(a):
            return a
        f = self.A.forms.get(rval(a))
        assert f is not None and f.inv(), 'hoisting a variant operand'
        if f.w == 0:
            return self.leaf(next(iter(f.L.t)))
        return self.lin(f.L, f.w)

    def lin(self, L, W):
        """Materialize a Lin in W-bit arithmetic."""
        key = ('L', W, L.ident())
        if key in self.memo:
            return self.memo[key]
        em = self.em
        k = Kw if W == 32 else Kl
        acc = None
        for lk, coef in sorted(L.t.items()):
            v = self.leaf(lk)
            if W == 64 and lk[1] == 'w':
                raise AssertionError('w leaf in 64-bit form')
            term = em.mulc(k, v, coef) if coef > 0 else em.mulc(k, v, -coef)
            if acc is None:
                acc = term if coef > 0 else em.op(D.Osub, k, em.con(0), term)
            elif coef > 0:
                acc = em.add(k, acc, term)
            else:
                acc = em.sub(k, acc, term)
        if acc is None:
            acc = em.con(L.c)
        elif L.c:
            acc = em.addc(k, acc, L.c)
        if rtype(acc) == RCon and W == 32:
            pass
        self.memo[key] = acc
        return acc

    # ---------- planning

    def plan(self, U):
        """Decide what gets copied, hoisted, scalar-replaced; collect groups and checks."""
        A = self.A
        sh = self.sh
        self.U = U
        self.cloned = []          # instructions copied per iteration, in order
        test_only = sh.test is not None and self.fn.tmp[rval(sh.test.to)].nuse == 1
        for b in sh.order:
            for i in ir.ins(b):
                if ir.istmp(i.to):
                    t = rval(i.to)
                    f = A.forms.get(t)
                    if f is not None:
                        continue          # affine or invariant: computed, not copied
                    if i is sh.test and test_only:
                        continue
                self.cloned.append(i)
        for i in self.cloned:
            for a in i.arg:
                self._need(a)
        for p in ir.phis(sh.H):
            if rval(p.to) not in A.biv:
                self._need(ir.phiargof(p, sh.latch))
        self.chk |= self.ctl.x.chk

    def _need(self, a):
        A = self.A
        if not A.inloop(a):
            return
        t = rval(a)
        f = A.forms.get(t)
        if f is None or f.inv():
            return
        if t in self.srload:
            return
        self.chk |= f.chk
        self.group(f)

    def group(self, f):
        key = (f.w, f.L.noconst().ident(), f.S.ident())
        g = self.groups.get(key)
        if g is None:
            g = Group()
            g.W = f.w
            g.key = key
            g.L0 = f.L.noconst()
            g.S = f.S
            g.c0 = f.L.c
            g.init = None
            g.ks = {}
            g.idx = len(self.groups)
            self.groups[key] = g
        return g

    def find_chains(self, noalias_load):
        """Scalar replacement candidates among the loads (noalias_load(i) is true when no
        store in the loop may write what i reads)."""
        A = self.A
        cands = {}
        for b in self.sh.order:
            for i in ir.ins(b):
                if not ir.isload(i.op) or not ir.istmp(i.to):
                    continue
                if A.forms.get(rval(i.to)) is not None:
                    continue
                f = A.form(i.arg[0], 64)
                if f is None or f.inv() or not f.S.isconst() or A.avlevel(f) != 'G':
                    continue
                if not noalias_load(i):
                    continue
                S = f.S.c
                r = f.L.c % abs(S)
                key = (i.op, i.cls, f.L.noconst().ident(), S, r)
                cands.setdefault(key, []).append((i, f))
        for key, lst in cands.items():
            op, cls, _, S, r = key
            pos = {}
            for i, f in lst:
                q = (f.L.c - r) // S
                pos.setdefault(q, []).append(i)
            qs = sorted(pos)
            if len(qs) < 2 or qs[-1] - qs[0] + 1 != len(qs) or len(qs) > SR_MAXSPAN:
                continue
            ch = Chain()
            ch.op, ch.cls, ch.S = op, cls, S
            f0 = lst[0][1]
            q0 = (f0.L.c - r) // S
            ch.Lr = Lin(f0.L.t, f0.L.c - q0 * S)       # address at position 0, t = 0
            ch.qmin, ch.qmax = qs[0], qs[-1]
            ch.pos = pos
            ch.fresh = pos[ch.qmax][0]
            self.chains.append(ch)
            for q, ins_ in pos.items():
                for i in ins_:
                    self.srload[rval(i.to)] = (ch, q)
            self.chk |= f0.chk

    # ---------- body copies

    def opmap(self, a, k, env, state, closed):
        """Value of original ref a in copy k (relative to the state's iteration)."""
        if not ir.istmp(a):
            return a
        t = rval(a)
        if t in env:
            return env[t]
        if not self.A.inloop(a):
            return a
        sv = ('phi', t)
        if sv in state:
            return state[sv]
        f = self.A.forms.get(t)
        if f is None:
            raise AssertionError('no value for %s' % self.fn.tmp[t].name)
        if f.inv():
            return self.inv_value(f)
        return self.affine_value(f, k, state, closed)

    def inv_value(self, f):
        em, self.em = self.em, self.fp
        try:
            if f.w == 0:
                return self.leaf(next(iter(f.L.t)))
            return self.lin(f.L, f.w)
        finally:
            self.em = em

    def affine_value(self, f, k, state, closed):
        em = self.em
        kc = Kw if f.w == 32 else Kl
        if closed is not None:
            # closed form at iteration T (in FX)
            L = self.inv_value(AV(f.w, f.L))
            if f.S.isconst():
                st = em.mulc(kc, closed, f.S.c)
            else:
                st = em.mul(kc, closed, self.inv_value(AV(f.w, f.S)))
            return em.add(kc, L, st)
        g = self.groups[(f.w, f.L.noconst().ident(), f.S.ident())]
        base = state[('iv', g.idx)]
        dc = norm(f.w, f.L.c - g.c0)
        if f.S.isconst():
            return em.addc(kc, base, norm(f.w, dc + k * f.S.c))
        if k:
            base = em.add(kc, base, self.kmul(g, k))
        return em.addc(kc, base, dc)

    def kmul(self, g, k):
        if k not in g.ks:
            em, self.em = self.em, self.fp
            try:
                g.ks[k] = self.lin(g.S.scale(k, g.W), g.W)
            finally:
                self.em = em
        return g.ks[k]

    def copies(self, em, state, n, closed=None, final=False):
        """Emit n copies of P+Q (final: only P, for iteration T) into em; returns the state
        after them (without the group IV updates) and the env of the last copy."""
        self.em = em
        sh = self.sh
        A = self.A
        state = dict(state)
        env = {}
        lm = LocalMem(self, em)
        for k in range(n):
            env = {}
            fresh = {}
            for i in self.cloned:
                if final and sh.B is not None and i in self._qset:
                    continue
                t = rval(i.to) if ir.istmp(i.to) else None
                if t is not None and t in self.srload:
                    ch, q = self.srload[t]
                    if q != ch.qmax:
                        env[t] = state[('sr', id(ch), q)]
                        continue
                    if id(ch) in fresh:
                        env[t] = fresh[id(ch)]
                        continue
                args = [self.opmap(a, k, env, state, closed) if a != R else R for a in i.arg]
                if ir.isstore(i.op):
                    lm.store(i, args, k, closed)
                    continue
                if ir.isload(i.op):
                    v = lm.load(i, args, k, closed)
                    if v is not None:
                        env[t] = v
                        if t in self.srload:
                            fresh[id(self.srload[t][0])] = v
                        continue
                if t is None:
                    em.ins(Ins(i.op, i.cls, R, args[0], args[1]))
                    continue
                if i.op in ir.PURE and i.cls in (Kw, Kl) and ir.fold(i.op, i.cls, 0, 0) is not None:
                    v = em.op(i.op, i.cls, args[0], args[1])
                else:
                    v = em.raw(i.op, i.cls, args[0], args[1])
                env[t] = v
                if t in self.srload:
                    fresh[id(self.srload[t][0])] = v
            if final:
                break
            # loop-carried values for the next iteration
            new = {}
            for p in ir.phis(sh.H):
                t = rval(p.to)
                if ('phi', t) in state:
                    new[('phi', t)] = self.opmap(ir.phiargof(p, sh.latch), k, env, state,
                                                 closed)
            for ch in self.chains:
                c = id(ch)
                for q in range(ch.qmin, ch.qmax - 1):
                    new[('sr', c, q)] = state[('sr', c, q + 1)]
                new[('sr', c, ch.qmax - 1)] = fresh[c]
            state.update(new)
        lm.finish()
        return state, env

    def advance(self, em, state, n):
        """Group IVs after n iterations."""
        out = dict(state)
        for g in self.groups.values():
            kc = Kw if g.W == 32 else Kl
            cur = state[('iv', g.idx)]
            if g.S.isconst():
                out[('iv', g.idx)] = em.addc(kc, cur, norm(g.W, n * g.S.c))
            else:
                out[('iv', g.idx)] = em.add(kc, cur, self.kmul(g, n))
        return out


_STORE = None


def _storeinfo(op):
    global _STORE
    if _STORE is None:
        _STORE = {D.Ostoreb: (Kw, 1), D.Ostoreh: (Kw, 2), D.Ostorew: (Kw, 4),
                  D.Ostorel: (Kl, 8), D.Ostores: (Ks, 4), D.Ostored: (Kd, 8)}
    return _STORE[op]


def _loadinfo(op, cls):
    """(size, extension op or None) of a load; None if unknown."""
    if op == D.Oload:
        return ({Kw: 4, Kl: 8, Ks: 4, Kd: 8}[cls], None)
    n = {D.Oloadsb: (1, D.Oextsb), D.Oloadub: (1, D.Oextub), D.Oloadsh: (2, D.Oextsh),
         D.Oloaduh: (2, D.Oextuh), D.Oloadsw: (4, D.Oextsw), D.Oloaduw: (4, D.Oextuw)}
    return n.get(op)


class LocalMem:
    """Store-to-load forwarding, redundant load elimination and dead store elimination
    within one straight-line block of copies. Addresses are compared by affine form (the
    same group and offset is the same address, different offsets of a group are disjoint
    when the bytes don't overlap); anything else goes through the alias analysis on the
    original address temporaries."""

    def __init__(self, fast, em):
        self.f = fast
        self.em = em
        self.avail = []      # (key, size, cls, value, orig addr)  values readable
        self.pending = []    # (key, size, orig addr, Ins)  stores not yet read

    def key(self, i, addr_orig, addr_new, k, closed):
        A = self.f.A
        f = A.form(addr_orig, 64)
        if f is not None and f.S.isconst() and closed is None:
            return ('a', f.L.noconst().ident(), f.S.c, norm(64, f.L.c + k * f.S.c))
        return ('r', addr_new)

    def overlap(self, k1, s1, a1, k2, s2, a2):
        if k1 == k2:
            return True
        if k1[0] == 'a' and k2[0] == 'a' and k1[1:3] == k2[1:3]:
            o1, o2 = k1[3], k2[3]
            return o1 < o2 + s2 and o2 < o1 + s1
        return self.f.rts.may_alias(a1, a2)

    def store(self, i, args, k, closed):
        cls, size = _storeinfo(i.op)
        ao = i.arg[1]
        key = self.key(i, ao, args[1], k, closed)
        ins = Ins(i.op, i.cls, R, args[0], args[1])
        # an earlier store to exactly these bytes, not read since, is dead
        keep = []
        for p in self.pending:
            if p[0] == key and p[1] <= size:
                p[3].op = D.Onop
                p[3].arg[0] = p[3].arg[1] = R
                M.count('dead_store')
            else:
                keep.append(p)
        self.pending = keep + [(key, size, ao, ins)]
        self.avail = [a for a in self.avail
                      if not self.overlap(a[0], a[1], a[4], key, size, ao)]
        self.avail.append((key, size, cls, args[0], ao))
        self.em.ins(ins)

    def load(self, i, args, k, closed):
        info = _loadinfo(i.op, i.cls)
        ao = i.arg[0]
        key = self.key(i, ao, args[0], k, closed)
        size = info[0] if info else 8
        if info is not None:
            v = self._forward(i, info, key, size)
            if v is not None:
                return v       # memory isn't read: pending stores stay removable
        self.pending = [p for p in self.pending
                        if not self.overlap(p[0], p[1], p[2], key, size, ao)]
        if info is None:
            return None
        v = self.em.raw(i.op, i.cls, args[0], args[1])
        self.avail.append((key, size, i.cls, v, ao))
        return v

    def _forward(self, i, info, key, size):
        for a in reversed(self.avail):
            if a[0] == key and a[1] == size:
                v = a[3]
                fn = self.f.fn
                if ir.istmp(v) and fn.tmp[rval(v)].cls != a[2]:
                    return None     # e.g. storew of a 64-bit value: keep the load
                ext = info[1]
                if i.op == D.Oload or (ext in (D.Oextsw, D.Oextuw) and i.cls == Kw):
                    if a[2] != i.cls:
                        return None
                    M.count('forward')
                    return v
                if a[2] != Kw:
                    return None
                M.count('forward')
                return self.em.op(ext, i.cls, v)
        return None

    def finish(self):
        self.em.out = [x for x in self.em.out if x.op != D.Onop]


def full_unroll(fn, lp, sh, A, ctl, T):
    """Replace the loop by T copies of P+Q and a final P (T is exact)."""
    pre, H, E = sh.pre, sh.H, sh.E
    blk = ir.newblock(fn, pre, 'unr')
    em = Emitter(fn, blk)
    latch = sh.latch
    env_state = {}
    cur = {}
    for p in ir.phis(H):
        cur[rval(p.to)] = ir.phiargof(p, pre)
    W = None
    last = {}
    for t in range(T + 1):
        env = dict(cur)
        for b in sh.order:
            if t == T and b is not H:
                break
            for i in ir.ins(b):
                args = [_m(env, a) for a in i.arg]
                if not ir.istmp(i.to):
                    em.ins(Ins(i.op, i.cls, R, args[0], args[1]))
                    continue
                if i.op in ir.PURE and i.cls in (Kw, Kl) and ir.fold(i.op, i.cls, 0, 0) is not None:
                    v = em.op(i.op, i.cls, args[0], args[1])
                else:
                    v = em.raw(i.op, i.cls, args[0], args[1])
                env[rval(i.to)] = v
        last = env
        if t < T:
            cur = {rval(p.to): _m(env, ir.phiargof(p, latch)) for p in ir.phis(H)}
    em.flush()
    ir.setjmp(blk, E)
    ir.retarget(pre, H, blk)
    _replace_outside(fn, lp, H, blk, E, last, set([blk]))
    M.count('full_unroll')


def _m(env, a):
    if ir.istmp(a) and rval(a) in env:
        return env[rval(a)]
    return a


def _replace_outside(fn, lp, H, newexit, E, env, newblocks):
    """The loop's exit edge H -> E is replaced by newexit -> E; uses of loop values outside
    the loop now see env's values."""
    ir.renamepred(E, H, newexit)
    for b in ir.blocks(fn):
        if b in lp.blocks or b in newblocks:
            continue
        for p in ir.phis(b):
            for n in range(p.narg):
                p.arg[n] = _m(env, p.arg[n])
        for i in ir.ins(b):
            i.arg[0] = _m(env, i.arg[0])
            i.arg[1] = _m(env, i.arg[1])
        b.jmp.arg = _m(env, b.jmp.arg)


def transform(fn, lp, rts, done, mode='opt'):
    """Optimize innermost loop lp; returns True if the function changed. mode 'full': only
    full unrolling of small constant-trip loops; 'opt': everything else."""
    sh = Shape(fn, lp)
    A = Analysis(fn, lp, sh.order)
    A.fn_con0 = ir.getcon(0, fn)
    ctl = Control(A, sh)
    nbody = sum(ir.ins(b).__len__() for b in sh.order)
    T = ctl.const_trips(MAXFULL)
    if mode == 'full':
        if T is not None and T * nbody <= MAXFULLINS:
            full_unroll(fn, lp, sh, A, ctl, T)
            return True
        return False
    if T is not None and T <= 1:
        return False

    # memory: hoistable loads, stores
    loads = []
    stores = []
    for b in sh.order:
        for i in ir.ins(b):
            if ir.isload(i.op):
                loads.append(i)
            elif ir.isstore(i.op):
                stores.append(i)

    def noalias_load(i):
        return all(not rts.may_alias(i.arg[0], s.arg[1]) for s in stores)

    hoist = set()
    for _ in range(4):
        new = set()
        for i in loads:
            f = A.form(i.arg[0], 64)
            if f is not None and f.inv() and A.avlevel(f) != 'X' and noalias_load(i):
                new.add(rval(i.to))
        if new == hoist:
            break
        hoist = new
        A = Analysis(fn, lp, sh.order, frozenset(hoist))
        A.fn_con0 = ir.getcon(0, fn)
        ctl = Control(A, sh)

    fast = Fast(fn, lp, sh, A, rts)
    fast.ctl = ctl
    fast.find_chains(noalias_load)
    ncl = 0
    U = max(1, M.unroll)
    fast.plan(U)
    ncl = len(fast.cloned)
    while U > 1 and ncl * U > MAXBODY:
        U //= 2
    fast.U = U
    if U == 1 and not fast.groups and not hoist and not fast.chains:
        return False
    emit_versioned(fn, lp, sh, A, ctl, fast, U, done)
    M.count('versioned')
    return True


def emit_versioned(fn, lp, sh, A, ctl, fast, U, done):
    pre, H, E = sh.pre, sh.H, sh.E
    W = ctl.W
    kw = Kw if W == 32 else Kl
    fast._qset = set(ir.ins(sh.B)) if sh.B is not None else set()

    G1 = ir.newblock(fn, pre, 'g')
    G2 = ir.newblock(fn, G1, 'g')
    SL = ir.newblock(fn, G2, 'slow')
    FP = ir.newblock(fn, G2, 'fp')
    MB = ir.newblock(fn, FP, 'main')
    RC = ir.newblock(fn, MB, 'rc')
    RB = ir.newblock(fn, RC, 'rem')
    FX = ir.newblock(fn, RB, 'fx')
    X = ir.newblock(fn, FX, 'x')
    newblocks = {G1, G2, SL, FP, MB, RC, RB, FX, X}

    # entry: pre -> G1; slow path through SL into the original header
    ir.retarget(pre, H, G1)
    ir.setjmp(SL, H)
    ir.renamepred(H, pre, SL)

    e1 = Emitter(fn, G1)
    fast.em = e1
    x0 = fast.lin(ctl.x.L, W)
    b = fast.lin(ctl.b.L, W)
    cont0 = e1.cmp(cmpop(ctl.rel, W), x0, b)
    s = ctl.s
    up = ctl.kind in ('up_lt', 'up_le') or (ctl.kind == 'ne' and s > 0)
    if ctl.sym:
        # the step's magnitude a, checked to be in [1, 2^20) before any division by it
        sv = fast.lin(ctl.x.S, W)
        sl = e1.op(D.Oextsw, Kl, sv) if W == 32 else sv
        a = Sym(sl if up else e1.op(D.Oneg, Kl, sl))
        sok = e1.raw(D.Oand, Kw, e1.cmp(D.Ocsgel, a.r, e1.con(1)),
                     e1.cmp(D.Ocsltl, a.r, e1.con(1 << 20)))
        cont0 = e1.raw(D.Oand, Kw, cont0, sok)
    else:
        a = abs(s)
    e1.flush()
    ir.setjnz(G1, cont0, G2, SL)

    e2 = Emitter(fn, G2)
    fast.em = e2
    d = e2.sub(kw, b, x0) if up else e2.sub(kw, x0, b)
    dl = e2.op(D.Oextuw, Kl, d) if W == 32 else d
    if ctl.kind in ('up_lt', 'down_gt'):
        if a == 1:
            T = dl
        else:
            T = e2.addc(Kl, _udivc(e2, e2.addc(Kl, dl, -1), a), 1)
    elif ctl.kind in ('up_le', 'down_ge'):
        T = e2.addc(Kl, _udivc(e2, dl, a), 1)
    else:
        T = _udivc(e2, dl, a)
    oks = []
    if ctl.kind == 'ne' and a > 1:
        # x reaches b only if the distance is a multiple of the step
        oks.append(e2.cmp(D.Oceql, e2.op(D.Oand, Kl, dl, e2.con(a - 1)), e2.con(0)))
    signed = ctl.rel[0] == 's'
    if ctl.kind != 'ne' and (a != 1 or ctl.kind in ('up_le', 'down_ge')):
        # x(T) = x0 + s*T must not wrap past the end of the type
        r = e2.sub(Kl, e2.mul(Kl, T, a.r) if isinstance(a, Sym) else e2.mulc(Kl, T, a), dl)
        # (in 64-bit arithmetic: exact for 32-bit values; for 64-bit ones the distance is
        # in [0, 2^64) and the unsigned comparison below reads it right)
        if W == 32:
            bl = e2.op(D.Oextsw if signed else D.Oextuw, Kl, b)
        else:
            bl = b
        if up:
            mx = ((1 << (W - 1)) - 1) if signed else ((1 << W) - 1)
            ml = e2.sub(Kl, e2.con(s64(mx)), bl)
        else:
            mn = (-(1 << (W - 1))) if signed else 0
            ml = e2.sub(Kl, bl, e2.con(mn))
        oks.append(e2.cmp(D.Ocugel, ml, r))
    if W == 64:
        oks.append(e2.cmp(D.Ocultl, T, e2.con(1 << 32)))
    Tm1 = e2.addc(Kl, T, -1)
    # extension checks: ext(N + c + S*t) exact for every c of a group and t in [0, T-1]:
    # enough to check the smallest and largest values
    cg = {}
    for cid in sorted(fast.chk):
        c = A.checks[cid]
        key = (c.kind, c.L.noconst().ident(), c.S.c)
        lo, hi, _ = cg.get(key, (c.L.c, c.L.c, c))
        cg[key] = (min(lo, c.L.c), max(hi, c.L.c), c)
    for key in sorted(cg, key=str):
        lo, hi, c = cg[key]
        N = fast.lin(c.L.noconst(), 64)
        st = e2.mulc(Kl, Tm1, c.S.c) if c.S.c else e2.con(0)
        vlo = e2.addc(Kl, N, lo)
        vhi = e2.addc(Kl, N, hi)
        if c.S.c > 0:
            vhi = e2.add(Kl, vhi, st)
        elif c.S.c < 0:
            vlo = e2.add(Kl, vlo, st)
        for v in (vlo, vhi):
            ext = e2.op(D.Oextsw if c.kind == 's' else D.Oextuw, Kl, v)
            oks.append(e2.cmp(D.Oceql, ext, v))
    ok = None
    for o in oks:
        co = ir.conval(fn, o)
        if co is not None:
            if co:
                continue
            ok = o
            break
        ok = o if ok is None else e2.raw(D.Oand, Kw, ok, o)
    e2.flush()
    if ok is None:
        ir.setjmp(G2, FP)
    else:
        ir.setjnz(G2, ok, FP, SL)

    # fast preheader
    ef = Emitter(fn, FP)
    fast.fp = ef
    fast.em = ef
    groups = list(fast.groups.values())
    # the IV that controls the loop: a 64-bit group with a constant stride, else a counter
    ctlg = None
    for g in sorted(groups, key=lambda g: (g.W != 64, g.idx)):
        if g.S.isconst() and (g.W == 64 or abs(g.S.c) == 1):
            ctlg = g
            break
    if ctlg is None:
        ctlg = fast.group(AV(64, Lin(), Lin(c=1)))
        groups = list(fast.groups.values())
    state0 = {}
    for g in groups:
        g.init = fast.lin(Lin(g.L0.t, g.c0), g.W)
        state0[('iv', g.idx)] = g.init
    for p in ir.phis(H):
        t = rval(p.to)
        if t not in A.biv:
            state0[('phi', t)] = ir.phiargof(p, SL)
    for ch in fast.chains:
        for q in range(ch.qmin, ch.qmax):
            addr = fast.lin(Lin(ch.Lr.t, ch.Lr.c + q * ch.S), 64)
            state0[('sr', id(ch), q)] = ef.raw(ch.op, ch.cls, addr)
    # hoisted invariant loads are materialized lazily into FP while copying
    kc = Kw if ctlg.W == 32 else Kl
    T4 = ef.op(D.Oand, Kl, T, ef.con(-U)) if U > 1 else T
    end4 = ef.add(kc, ctlg.init, ef.mulc(kc, T4, ctlg.S.c))
    end = ef.add(kc, ctlg.init, ef.mulc(kc, T, ctlg.S.c))
    big = ef.cmp(D.Ocugel, T, ef.con(U)) if U > 1 else None

    svs = sorted(state0, key=str)

    def phis_for(blk, ins_):
        """New phi refs in blk for every state variable; ins_ is [(pred, state)]."""
        st = {}
        for sv in svs:
            vals = [(pb, s_[sv]) for pb, s_ in ins_]
            cls = _svcls(fn, sv, fast, vals)
            v = newtmp('m', cls, fn)
            ir.addphi(blk, cls, v, vals)
            st[sv] = v
        return st

    # main loop
    if U > 1:
        stM = {sv: newtmp('m', _svcls(fn, sv, fast, [(None, state0[sv])]), fn) for sv in svs}
        em = Emitter(fn, MB)
        out, _ = fast.copies(em, stM, U)
        out = fast.advance(em, out, U)
        c = em.raw(D.Ocnel if kc == Kl else D.Ocnew, Kw, out[('iv', ctlg.idx)], end4)
        em.flush()
        ir.setjnz(MB, c, MB, RC)
        for sv in svs:
            ir.addphi(MB, _svcls(fn, sv, fast, [(None, state0[sv])]), stM[sv],
                      [(FP, state0[sv]), (MB, out[sv])])
        stRC = phis_for(RC, [(FP, state0), (MB, out)])
        fast.em = ef
        ef.flush()
        ir.setjnz(FP, big, MB, RC)
    else:
        ef.flush()
        ir.setjmp(FP, RC)
        stRC = phis_for(RC, [(FP, state0)])
    er = Emitter(fn, RC)
    c = er.raw(D.Ocnel if kc == Kl else D.Ocnew, Kw, stRC[('iv', ctlg.idx)], end)
    er.flush()
    ir.setjnz(RC, c, RB, FX)

    stR = {sv: newtmp('m', _svcls(fn, sv, fast, [(None, stRC[sv])]), fn) for sv in svs}
    em = Emitter(fn, RB)
    outR, _ = fast.copies(em, stR, 1)
    outR = fast.advance(em, outR, 1)
    c = em.raw(D.Ocnel if kc == Kl else D.Ocnew, Kw, outR[('iv', ctlg.idx)], end)
    em.flush()
    ir.setjnz(RB, c, RB, FX)
    for sv in svs:
        ir.addphi(RB, _svcls(fn, sv, fast, [(None, stRC[sv])]), stR[sv],
                  [(RC, stRC[sv]), (RB, outR[sv])])
    stFX = phis_for(FX, [(RC, stRC), (RB, outR)])

    # FX: P(T) for the values the loop leaves with
    ex = Emitter(fn, FX)
    fast.em = ex
    # the FP emitter may still receive hoisted invariants: they must come before FP's jump,
    # which is fine since FP's instructions are flushed again below
    _, env = fast.copies(ex, stFX, 1, closed=T, final=True)
    liveout = {}
    for p in ir.phis(H):
        t = rval(p.to)
        liveout[t] = fast.opmap(p.to, 0, env, stFX, T)
    for i in ir.ins(H):
        if ir.istmp(i.to):
            t = rval(i.to)
            if t in env:
                liveout[t] = env[t]
            else:
                try:
                    liveout[t] = fast.opmap(i.to, 0, env, stFX, T)
                except AssertionError:
                    pass
    fast.em = ex
    ex.flush()
    ir.setjmp(FX, X)
    ef.flush()

    # X: join slow and fast exits
    xenv = {}
    used = _used_outside(fn, lp, newblocks)
    for t in sorted(used):
        if t not in liveout:
            raise AssertionError('live-out without fast value')
        cls = fn.tmp[t].cls
        v = newtmp('m', cls, fn)
        ir.addphi(X, cls, v, [(H, D.TMP(t)), (FX, liveout[t])])
        xenv[t] = v
    ir.retarget(H, E, X)
    ir.setjmp(X, E)
    _replace_outside(fn, lp, H, X, E, xenv, newblocks)
    done.update([H, MB, RB])


def _used_outside(fn, lp, newblocks):
    ids = set()
    for b in ir.blocks(fn):
        if b in lp.blocks or b in newblocks:
            continue
        refs = []
        for p in ir.phis(b):
            refs.extend(p.arg[:p.narg])
        for i in ir.ins(b):
            refs.extend(i.arg)
        refs.append(b.jmp.arg)
        for r in refs:
            if ir.istmp(r) and fn.tmp[rval(r)].bid == lp.header.id:
                ids.add(rval(r))
    return ids


def _svcls(fn, sv, fast, vals):
    if sv[0] == 'iv':
        g = [g for g in fast.groups.values() if g.idx == sv[1]][0]
        return Kw if g.W == 32 else Kl
    if sv[0] == 'phi':
        return fn.tmp[sv[1]].cls
    if sv[0] == 'sr':
        ch = [c for c in fast.chains if id(c) == sv[1]][0]
        return ch.cls
    raise AssertionError(sv)


class Sym:
    """A run-time value (a Ref is an int, so it needs a wrapper to be told from a constant)."""
    __slots__ = ('r',)

    def __init__(self, r):
        self.r = r


def _udivc(em, v, a):
    if isinstance(a, Sym):
        return em.raw(D.Oudiv, Kl, v, a.r)
    if a & (a - 1) == 0:
        return em.op(D.Oshr, Kl, v, em.con(a.bit_length() - 1))
    return em.raw(D.Oudiv, Kl, v, em.con(a))
