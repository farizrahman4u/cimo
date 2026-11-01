"""cimo: algebraic simplification and known bits (ext.cse; not in QBE), from LLVM's InstCombine and
ValueTracking. gvn calls simplify() on each instruction, operands first.

IR integers wrap and carry no "no signed overflow" flag, so only rules that hold in modular
arithmetic are here: (x + c) == (y + c) folds, but x * 2 == 8 can't become x == 4, nor x + 1 > x be
true. Float rules are the ones exact for every value, NaN included: x * 1.0, -(-x).
"""

from . import defs as D
from .defs import R, RTmp, RCon, Kw, Kl, Ks, Kd, CBits, KBASE, UIns, rtype, rval
from .util import getcon, iscmp, cmpwlneg
from .ssa import adduse

_SIMPLE = frozenset((D.Oadd, D.Osub, D.Oneg, D.Omul, D.Odiv, D.Oudiv, D.Orem, D.Ourem, D.Oand, D.Oor,
                     D.Oxor, D.Oshl, D.Oshr, D.Osar, D.Oextsb, D.Oextub, D.Oextsh, D.Oextuh, D.Oextsw,
                     D.Oextuw, D.Ofsqrt, D.Ofabs, D.Offloor, D.Ofceil, D.Oftrunc)) \
    | frozenset(range(D.Ocmpw, D.Ocmpl1 + 1))
_ONE = {Ks: 0x3f800000, Kd: 0x3ff0000000000000}
_ZEXT = {D.Oextub: 8, D.Oextuh: 16, D.Oextuw: 32, D.Oloadub: 8, D.Oloaduh: 16, D.Oloaduw: 32,
         D.Oparub: 8, D.Oparuh: 16}  # the last two: Apple's ABI, where the caller extends
_SEXT = {D.Oextsb: 8, D.Oextsh: 16, D.Oextsw: 32, D.Oloadsb: 8, D.Oloadsh: 16}
MAXDEPTH = 6


def _mask(k):
    return 0xffffffffffffffff if k == Kl else 0xffffffff


def _defn(fn, r):
    """The instruction defining a temporary, or None (a phi, a parameter, a constant)."""
    if rtype(r) != RTmp:
        return None
    d = fn.tmp[rval(r)].def_
    return None if d is None or d.op == D.Onop else d


def _con(fn, r, k):
    """A constant's bits in class k, or None."""
    if rtype(r) == RCon and fn.con[rval(r)].type == CBits:
        return fn.con[rval(r)].i & _mask(k)
    return None


def _ref(v, k, fn):
    return getcon(D.s64(v & _mask(k)), fn)


def known(fn, r, k, depth=0):
    """(bits known zero, bits known one) of a value in class k's width."""
    m = _mask(k)
    c = _con(fn, r, k)
    if c is not None:
        return m & ~c, c
    d = _defn(fn, r)
    if d is None or depth >= MAXDEPTH or KBASE(d.cls) != 0:
        return 0, 0
    if d.cls == Kw and k == Kl:  # the high half of a word is undefined
        return 0, 0
    o = d.op
    dm = _mask(d.cls)
    if iscmp(o) is not None:
        return m & ~1, 0
    if o == D.Ocopy:
        return known(fn, d.arg[0], k, depth + 1)
    if o in _ZEXT:
        return m & ~((1 << _ZEXT[o]) - 1), 0
    if o in _SEXT or o == D.Oextsw:
        n = _SEXT[o]
        if o in (D.Oloadsb, D.Oloadsh):
            return 0, 0
        z, n1 = known(fn, d.arg[0], Kw, depth + 1)
        low = (1 << n) - 1
        sign = 1 << (n - 1)
        hi = dm & ~low
        if z & sign:
            return (z & low | hi) & m, n1 & low & m
        if n1 & sign:
            return z & low & m, (n1 & low | hi) & m
        return z & (low >> 1) & m, n1 & (low >> 1) & m
    if o in (D.Oand, D.Oor, D.Oxor):
        z0, o0 = known(fn, d.arg[0], d.cls, depth + 1)
        z1, o1 = known(fn, d.arg[1], d.cls, depth + 1)
        if o == D.Oand:
            return (z0 | z1) & m, o0 & o1 & m
        if o == D.Oor:
            return z0 & z1 & m, (o0 | o1) & m
        return ((z0 & z1) | (o0 & o1)) & m, ((z0 & o1) | (o0 & z1)) & m
    if o in (D.Oshl, D.Oshr, D.Osar):
        n = _con(fn, d.arg[1], Kw)
        if n is None:
            return 0, 0
        w = 64 if d.cls == Kl else 32
        n &= w - 1
        z, on = known(fn, d.arg[0], d.cls, depth + 1)
        if o == D.Oshl:
            return ((z << n) | ((1 << n) - 1)) & dm & m, (on << n) & dm & m
        hi = dm & ~(dm >> n)
        if o == D.Oshr:
            return ((z >> n) | hi) & m, (on >> n) & m
        sign = 1 << (w - 1)
        if z & sign:
            return ((z >> n) | hi) & m, (on >> n) & m
        if on & sign:
            return (z >> n) & m, ((on >> n) | hi) & m
        return (z >> n) & ~hi & m, (on >> n) & ~hi & m
    if o in (D.Oadd, D.Osub, D.Omul):  # low zeros stay zeros, as far as both have them
        z0, _ = known(fn, d.arg[0], d.cls, depth + 1)
        z1, _ = known(fn, d.arg[1], d.cls, depth + 1)
        def tz(z):
            n = 0
            while (z >> n) & 1:
                n += 1
            return n
        n = tz(z0) + tz(z1) if o == D.Omul else min(tz(z0), tz(z1))
        return ((1 << n) - 1) & m, 0
    return 0, 0


def _srange(z, o, w):
    """(lowest, highest) signed value with those known bits."""
    m = (1 << w) - 1
    sign = 1 << (w - 1)
    lo, hi = o, m & ~z
    if z & sign:
        return lo, hi
    if o & sign:
        return lo - (1 << w), hi - (1 << w)
    return (lo | sign) - (1 << w), hi & ~sign


def _foldcmp(fn, i, ck, cc):
    """0, 1 or None: a comparison the known bits decide."""
    w = 64 if ck == Kl else 32
    z0, o0 = known(fn, i.arg[0], ck)
    z1, o1 = known(fn, i.arg[1], ck)
    if not (z0 | o0 | z1 | o1):
        return None
    if cc in (D.Cieq, D.Cine):
        if (z0 & o1) | (o0 & z1):
            return int(cc == D.Cine)
        return None
    m = (1 << w) - 1
    if cc in (D.Ciuge, D.Ciugt, D.Ciule, D.Ciult):
        lo0, hi0, lo1, hi1 = o0, m & ~z0, o1, m & ~z1
    else:
        lo0, hi0 = _srange(z0, o0, w)
        lo1, hi1 = _srange(z1, o1, w)
    if cc in (D.Cisge, D.Ciuge):
        return 1 if lo0 >= hi1 else 0 if hi0 < lo1 else None
    if cc in (D.Cisgt, D.Ciugt):
        return 1 if lo0 > hi1 else 0 if hi0 <= lo1 else None
    if cc in (D.Cisle, D.Ciule):
        return 1 if hi0 <= lo1 else 0 if lo0 > hi1 else None
    return 1 if hi0 < lo1 else 0 if lo0 >= hi1 else None


def _set(fn, b, i, op, a, c=R):
    """Rewrite i, keeping the use lists right for what gvn replaces later."""
    i.op = op
    i.arg[0] = a
    i.arg[1] = c
    for r in (a, c):
        if rtype(r) == RTmp:
            adduse(fn.tmp[rval(r)], UIns, b, i)


def _isnot(fn, r, k):
    """x if r is ~x in class k."""
    d = _defn(fn, r)
    if d is not None and d.op == D.Oxor and d.cls == k and _con(fn, d.arg[1], k) == _mask(k):
        return d.arg[0]
    return None


_MATH = frozenset((D.Ofsqrt, D.Ofabs, D.Offloor, D.Ofceil, D.Oftrunc))


def _fold1(o, x):
    """A math instruction on a finite constant, exactly as the instruction rounds; or None."""
    import math
    if not math.isfinite(x):
        return None
    if o == D.Ofsqrt:
        return math.sqrt(x) if x >= 0 else None
    if o == D.Ofabs:
        return abs(x)
    f = {D.Offloor: math.floor, D.Ofceil: math.ceil, D.Oftrunc: math.trunc}[o](x)
    return math.copysign(float(f), x)


def _float(fn, b, i):
    k = i.cls
    if i.op in _MATH:
        v = _con(fn, i.arg[0], Kl if k == Kd else Kw)
        if v is None:
            return R
        x = D.bitsd(v) if k == Kd else D.bitsf(v)
        y = _fold1(i.op, x)
        if y is None:
            return R
        return getcon(D.dbits(y), fn) if k == Kd else getcon(D.fbits(D.f32(y)), fn)
    if i.op == D.Omul:  # x * 2.0: an add is cheaper, and exact the same
        two = {Ks: 0x40000000, Kd: 0x4000000000000000}[k]
        if _con(fn, i.arg[1], Kl if k == Kd else Kw) == two:
            _set(fn, b, i, D.Oadd, i.arg[0], i.arg[0])
            return R
    if i.op == D.Oneg:
        d = _defn(fn, i.arg[0])
        if d is not None and d.op == D.Oneg and d.cls == k:
            return d.arg[0]
    elif i.op in (D.Omul, D.Odiv):
        one = _ONE.get(k)
        if _con(fn, i.arg[1], Kl) == one or (k == Ks and _con(fn, i.arg[1], Kw) == one):
            return i.arg[0]
        if i.op == D.Omul and (_con(fn, i.arg[0], Kl) == one or (k == Ks and _con(fn, i.arg[0], Kw) == one)):
            return i.arg[1]
    return R


def simplify(fn, b, i):
    """A value i is equal to, or R; i may be rewritten to something simpler. Runs on every
    instruction, on the operands before their uses."""
    o = i.op
    if o not in _SIMPLE:
        return R
    k = i.cls
    if KBASE(k) != 0:
        return _float(fn, b, i)
    if o in _MATH:
        return R
    a, c = i.arg
    cm = iscmp(o)
    if cm is not None:
        return _compare(fn, b, i, cm[0], cm[1])
    n = _con(fn, c, k)
    d0, d1 = _defn(fn, a), _defn(fn, c)
    if o == D.Oadd:
        for x, y, dy in ((a, c, d1), (c, a, d0)):
            if dy is None or dy.cls != k:
                continue
            if dy.op == D.Osub and dy.arg[1] == x:  # x + (z - x)
                return dy.arg[0]
            if dy.op == D.Oneg:  # x + -z
                _set(fn, b, i, D.Osub, x, dy.arg[0])
                return R
            if dy.arg[0] == x and dy.op in (D.Omul, D.Oshl):  # x + x * n, x + (x << n)
                f = _con(fn, dy.arg[1], k)
                if f is not None:
                    f = f + 1 if dy.op == D.Omul else (1 << (f & (63 if k == Kl else 31))) + 1
                    _set(fn, b, i, D.Omul, x, _ref(f, k, fn))
                    return R
        if a == c:
            _set(fn, b, i, D.Omul, a, getcon(2, fn))
            return R
    elif o == D.Osub:
        if a == c:
            return getcon(0, fn)
        if d0 is not None and d0.cls == k:
            if d0.op == D.Oadd:  # (x + y) - y, (x + y) - x
                if d0.arg[1] == c:
                    return d0.arg[0]
                if d0.arg[0] == c:
                    return d0.arg[1]
            if d0.op == D.Osub and d0.arg[0] == c:  # x - (x ... : (x - y) - x is -y
                _set(fn, b, i, D.Oneg, d0.arg[1])
                return R
        if d1 is not None and d1.cls == k:
            if d1.op == D.Osub and d1.arg[0] == a:  # x - (x - y)
                return d1.arg[1]
            if d1.op == D.Oneg:  # x - -y
                _set(fn, b, i, D.Oadd, a, d1.arg[0])
                return R
            if d1.op == D.Oadd:  # x - (x + y)
                if d1.arg[0] == a:
                    _set(fn, b, i, D.Oneg, d1.arg[1])
                    return R
                if d1.arg[1] == a:
                    _set(fn, b, i, D.Oneg, d1.arg[0])
                    return R
        if d0 is not None and d1 is not None and d0.op == D.Oor and d1.op == D.Oand and d0.cls == k \
                and d1.cls == k and {d0.arg[0], d0.arg[1]} == {d1.arg[0], d1.arg[1]}:  # (x | y) - (x & y)
            _set(fn, b, i, D.Oxor, d0.arg[0], d0.arg[1])
            return R
        if _con(fn, a, k) == 0:
            _set(fn, b, i, D.Oneg, c)
            return R
    elif o == D.Oneg:
        if d0 is not None and d0.cls == k:
            if d0.op == D.Oneg:
                return d0.arg[0]
            if d0.op == D.Osub:
                _set(fn, b, i, D.Osub, d0.arg[1], d0.arg[0])
                return R
    elif o == D.Omul:
        if n == 0:
            return getcon(0, fn)
        if n == 1:
            return a
        if n == _mask(k):
            _set(fn, b, i, D.Oneg, a)
            return R
    elif o in (D.Odiv, D.Oudiv):
        if n == 1:
            return a
    elif o in (D.Orem, D.Ourem):
        if n == 1:
            return getcon(0, fn)
    elif o == D.Oshl:
        if n is not None and d0 is not None and d0.cls == k and d0.op in (D.Oshr, D.Osar) \
                and _con(fn, d0.arg[1], k) == n:  # (x >> n) << n
            n &= 63 if k == Kl else 31
            _set(fn, b, i, D.Oand, d0.arg[0], _ref(_mask(k) << n, k, fn))
            return R
    elif o == D.Oor or o == D.Oand or o == D.Oxor:
        if o != D.Oxor:
            if _isnot(fn, a, k) == c or _isnot(fn, c, k) == a:  # x & ~x, x | ~x
                return getcon(0, fn) if o == D.Oand else _ref(_mask(k), k, fn)
        if o == D.Oor and d0 is not None and d1 is not None and d0.op == D.Oand and d1.op == D.Oand \
                and d0.cls == k and d1.cls == k:  # (x & y) | (x & ~y)
            for p, q in ((0, 0), (0, 1), (1, 0), (1, 1)):
                if d0.arg[p] == d1.arg[q]:
                    u, v = d0.arg[p ^ 1], d1.arg[q ^ 1]
                    if _isnot(fn, u, k) == v or _isnot(fn, v, k) == u:
                        return d0.arg[p]
    # known bits
    if o == D.Oand and n is not None:
        z, _ = known(fn, a, k)
        if (z | n) & _mask(k) == _mask(k):  # no bit it clears can be set
            return a
    elif o == D.Oor and n is not None:
        _, on = known(fn, a, k)
        if n & ~on == 0:
            return a
    elif o == D.Osar and n is not None:
        z, _ = known(fn, a, k)
        if z >> ((64 if k == Kl else 32) - 1) & 1:
            _set(fn, b, i, D.Oshr, a, c)
    elif D.isext(o):
        dd = _defn(fn, a)
        z, on = known(fn, a, Kw)
        w = {D.Oextub: 8, D.Oextuh: 16, D.Oextuw: 32, D.Oextsb: 8, D.Oextsh: 16, D.Oextsw: 32}[o]
        if w < 32 and dd is not None and dd.cls == k and k == Kw:
            sign = 1 << (w - 1)
            high = 0xffffffff & ~((1 << w) - 1)
            if z & high == high and (o in (D.Oextub, D.Oextuh) or z & sign):
                return a
    z, on = known(fn, i.to, k)
    if (z | on) & _mask(k) == _mask(k):
        return _ref(on, k, fn)
    return R


def _compare(fn, b, i, ck, cc):
    a, c = i.arg
    v = _foldcmp(fn, i, ck, cc)
    if v is None:
        v = decide(fn, b, ck, cc, a, c)
    if v is not None:
        return getcon(v, fn)
    if cc in (D.Cieq, D.Cine):
        d0, d1 = _defn(fn, a), _defn(fn, c)
        n = _con(fn, c, ck)
        if n == 0 and d0 is not None and d0.op in (D.Oextuw, D.Oextsw):  # a 0 or 1, widened
            d2 = _defn(fn, d0.arg[0])
            cm = iscmp(d2.op) if d2 is not None else None
            if cm is not None and (cc == D.Cine or KBASE(cm[0]) == 0):
                _set(fn, b, i, d2.op if cc == D.Cine else cmpwlneg(d2.op), d2.arg[0], d2.arg[1])
                return R
        if n == 0 and d0 is not None and iscmp(d0.op) is not None:
            if cc == D.Cine:  # (x < y) != 0, in whichever class it is read
                if i.cls == d0.cls and ck == d0.cls:
                    return a
                _set(fn, b, i, d0.op, d0.arg[0], d0.arg[1])
            elif KBASE(iscmp(d0.op)[0]) == 0:  # (x < y) == 0
                _set(fn, b, i, cmpwlneg(d0.op), d0.arg[0], d0.arg[1])
            return R
        if n == 0 and d0 is not None and d0.op in (D.Orem, D.Ourem) and d0.cls == ck \
                and fn.tmp[rval(a)].nuse == 1:  # x % 2**k == 0 looks at the low bits, whatever x's sign
            c0 = _con(fn, d0.arg[1], ck)
            if c0 and c0 & (c0 - 1) == 0:
                d0.op = D.Oand
                d0.arg[1] = _ref(c0 - 1, ck, fn)
                return R
        if d0 is not None and d0.op in (D.Oadd, D.Osub) and d0.cls == ck:
            c0 = _con(fn, d0.arg[1], ck)
            if c0 is not None:
                if n is not None:  # x + c0 == n
                    v = n - c0 if d0.op == D.Oadd else n + c0
                    _set(fn, b, i, i.op, d0.arg[0], _ref(v, ck, fn))
                    return R
                if d1 is not None and d1.op == d0.op and d1.cls == ck and _con(fn, d1.arg[1], ck) == c0:
                    _set(fn, b, i, i.op, d0.arg[0], d1.arg[0])  # x + c == y + c
                    return R
    return R


# Facts from dominating branches (LLVM's ConstraintElimination and CorrelatedValuePropagation, GCC's
# dominator and vrp): inside an arm of `if (x < y)`, x < y holds. Every temporary has one value per
# execution of its definition, and the arm is dominated by the branch, so that value is the one the
# branch tested. Ranges are of the wrapping values: x > 0 gives x - 1 >= 0 because the subtraction
# cannot wrap for x in [1, MAX], but x + 1 > 0 doesn't follow from x >= 0.

_LT, _EQ, _GT = 1, 2, 4
_SET = {D.Cieq: _EQ, D.Cine: _LT | _GT, D.Cisge: _EQ | _GT, D.Cisgt: _GT, D.Cisle: _LT | _EQ, D.Cislt: _LT,
        D.Ciuge: _EQ | _GT, D.Ciugt: _GT, D.Ciule: _LT | _EQ, D.Ciult: _LT}
_UNS = frozenset((D.Ciuge, D.Ciugt, D.Ciule, D.Ciult))
_TOSIGNED = {D.Ciuge: D.Cisge, D.Ciugt: D.Cisgt, D.Ciule: D.Cisle, D.Ciult: D.Cislt}
_MIRROR = {_LT: _GT, _GT: _LT, _EQ: _EQ, _LT | _EQ: _EQ | _GT, _EQ | _GT: _LT | _EQ, _LT | _GT: _LT | _GT}
MAXWALK = 24  # dominators looked at per question: enough for checks close together


def _facts(b):
    """(temporary or condition, true or false) for each branch that decides how b is reached."""
    cur = b
    for _ in range(MAXWALK):
        p = cur.idom
        if p is None:
            return
        if cur.npred == 1 and p.jmp.type == D.Jjnz and p.s1 is not p.s2 and rtype(p.jmp.arg) == RTmp:
            if p.s1 is cur:
                yield p.jmp.arg, True
            elif p.s2 is cur:
                yield p.jmp.arg, False
        cur = p


def _fact(fn, c, sense, ck):
    """(x, relation set, y, unsigned) of a condition with that outcome, in class ck; or None."""
    d = _defn(fn, c)
    if d is None:
        return None
    cm = iscmp(d.op)
    if cm is None or cm[0] != ck:
        return None
    s = _SET[cm[1]]
    if not sense:
        s = (_LT | _EQ | _GT) & ~s
    return d.arg[0], s, d.arg[1], cm[1] in _UNS


def _sval(v, w):
    return v - (1 << w) if v >> (w - 1) else v


def _narrow(lo, hi, s, n, uns, w):
    """[lo, hi] signed, for a value related to the constant n by the set s."""
    smax = (1 << (w - 1)) - 1
    if uns:  # x u< n, with n a nonnegative number, says 0 <= x < n; else nothing signed
        if n > smax or not (s & (_LT | _EQ)) or s & _GT:
            return lo, hi
        lo = max(lo, 0)
        hi = min(hi, n if s & _EQ else n - 1)
        return lo, hi
    n = _sval(n, w)
    if s == _EQ:
        return max(lo, n), min(hi, n)
    if s == _LT | _GT:
        if n == lo:
            lo += 1
        if n == hi:
            hi -= 1
        return lo, hi
    if s & _LT and not s & _GT:
        hi = min(hi, n if s & _EQ else n - 1)
    if s & _GT and not s & _LT:
        lo = max(lo, n if s & _EQ else n + 1)
    return lo, hi


def srange(fn, b, r, ck, depth=0):
    """[lo, hi]: the signed values r can have in block b, in class ck."""
    w = 64 if ck == Kl else 32
    z, on = known(fn, r, ck)
    lo, hi = _srange(z, on, w)
    if rtype(r) != RTmp or depth >= 3:
        return lo, hi
    for c, sense in _facts(b):
        f = _fact(fn, c, sense, ck)
        if f is None:
            continue
        x, s, y, uns = f
        if x == r and _con(fn, y, ck) is not None:
            lo, hi = _narrow(lo, hi, s, _con(fn, y, ck), uns, w)
        elif y == r and _con(fn, x, ck) is not None:
            lo, hi = _narrow(lo, hi, _MIRROR[s], _con(fn, x, ck), uns, w)
    d = _defn(fn, r)
    if d is not None and d.cls == ck:
        if d.op in (D.Oadd, D.Osub):
            n = _con(fn, d.arg[1], ck)
            if n is not None:
                n = _sval(n, w)
                if d.op == D.Osub:
                    n = -n
                lo1, hi1 = srange(fn, b, d.arg[0], ck, depth + 1)
                lo1 += n
                hi1 += n
                if -(1 << (w - 1)) <= lo1 and hi1 < 1 << (w - 1):  # it doesn't wrap
                    lo, hi = max(lo, lo1), min(hi, hi1)
        elif d.op == D.Oextsw and ck == Kl:
            lo1, hi1 = srange(fn, b, d.arg[0], Kw, depth + 1)
            lo, hi = max(lo, lo1), min(hi, hi1)
    elif d is None:
        lo1 = _induction(fn, b, r, ck)
        if lo1 is not None:
            lo = max(lo, lo1)
    return lo, hi


def _strip(fn, r):
    """Through copies: gvn reaches a loop's latch last, when it still reads what loadopt left."""
    for _ in range(8):
        d = _defn(fn, r)
        if d is None or d.op != D.Ocopy:
            return r
        r = d.arg[0]
    return r


def _induction(fn, b, r, ck):
    """The first value of a counter that only grows: i = phi(c, i + n), n > 0, in which a signed
    test on i guards the addition, so it can't wrap (LLVM's IndVarSimplify, without SCEV). Else None.
    Only for n == 1: i < x means i + 1 <= x fits."""
    p = _PHI.get(rval(r))
    if p is None or p.narg != 2 or p.cls != ck:
        return None
    for n in (0, 1):
        init, nxt = _strip(fn, p.arg[n]), _strip(fn, p.arg[n ^ 1])
        c = _con(fn, init, ck)
        d = _defn(fn, nxt)
        if c is None or d is None or d.op != D.Oadd or d.cls != ck or _strip(fn, d.arg[0]) != r \
                or _con(fn, d.arg[1], ck) != 1:
            continue
        # the block with the addition is under a branch that said i < something
        for f, sense in _facts(fn.rpo[fn.tmp[rval(nxt)].bid]):
            q = _fact(fn, f, sense, ck)
            if q is None or q[3]:
                continue
            x, s, y, _ = q
            if (_strip(fn, x) == r and s == _LT) or (_strip(fn, y) == r and s == _GT):
                return _sval(c, 64 if ck == Kl else 32)
    return None


_PHI = {}


def startfn(fn):
    """The phis of fn by temporary, for _induction."""
    _PHI.clear()
    b = fn.start
    while b:
        p = b.phi
        while p:
            _PHI[rval(p.to)] = p
            p = p.link
        b = b.link


def decide(fn, b, ck, cc, a, c):
    """0, 1 or None: what the branches dominating b, and the ranges, say about "a cc c"."""
    q = _SET[cc]
    uns = cc in _UNS
    w = 64 if ck == Kl else 32
    la, ha = srange(fn, b, a, ck)
    lc, hc = srange(fn, b, c, ck)
    if uns and la >= 0 and lc >= 0:
        uns = False  # both nonnegative: unsigned and signed order agree
    if not uns:
        if q == _EQ or q == _LT | _GT:
            if ha < lc or hc < la:
                return int(q != _EQ)
            if la == ha == lc == hc:
                return int(q == _EQ)
        else:
            if q == _LT:
                r = 1 if ha < lc else 0 if la >= hc else None
            elif q == _LT | _EQ:
                r = 1 if ha <= lc else 0 if la > hc else None
            elif q == _GT:
                r = 1 if la > hc else 0 if ha <= lc else None
            else:
                r = 1 if la >= hc else 0 if ha < lc else None
            if r is not None:
                return r
    if rtype(a) != RTmp and rtype(c) != RTmp:
        return None
    equality = q == _EQ or q == _LT | _GT
    for f, sense in _facts(b):
        g = _fact(fn, f, sense, ck)
        if g is None:
            continue
        x, s, y, fu = g
        if x == c and y == a:
            s = _MIRROR.get(s, s)
        elif not (x == a and y == c):
            continue
        # order in one sense says nothing about the other, except when it's equality
        if not equality and fu != uns and s not in (_EQ, _LT | _GT):
            if cc in _UNS and fu and not uns:
                pass  # the query became signed, on nonnegative values: same order
            else:
                continue
        if s & ~q == 0:
            return 1
        if s & q == 0:
            return 0
    return None
