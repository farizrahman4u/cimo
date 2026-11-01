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
                     D.Oextuw)) | frozenset(range(D.Ocmpw, D.Ocmpl1 + 1))
_ONE = {Ks: 0x3f800000, Kd: 0x3ff0000000000000}
_ZEXT = {D.Oextub: 8, D.Oextuh: 16, D.Oextuw: 32, D.Oloadub: 8, D.Oloaduh: 16, D.Oloaduw: 32}
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


def _float(fn, b, i):
    k = i.cls
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
    if v is not None:
        return getcon(v, fn)
    if cc in (D.Cieq, D.Cine):
        d0, d1 = _defn(fn, a), _defn(fn, c)
        n = _con(fn, c, ck)
        if n == 0 and d0 is not None and iscmp(d0.op) is not None and KBASE(iscmp(d0.op)[0]) == 0 \
                and d0.cls == ck:
            if cc == D.Cine:  # (x < y) != 0
                if i.cls == d0.cls:
                    return a
            else:  # (x < y) == 0
                _set(fn, b, i, cmpwlneg(d0.op), d0.arg[0], d0.arg[1])
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
