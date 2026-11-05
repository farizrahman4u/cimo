"""cimo (ext.vector): the q operations SSE2 has no instruction for, written with those it has, before the last cleanup:
gvn makes one temporary of a repeated mask, gcm takes it out of the loop. The sequences are the usual ones: 4 x i32 multiply
through pmuludq, 16 x i8 multiply through pmullw on odd and even bytes, 2 x i64 compares from 32-bit ones, unsigned
compares by flipping the sign, byte shifts as 16-bit ones with a mask. 256-bit AVX2 or SSE4.1, behind a check at load,
would shorten most of them. Nothing here can do a fused multiply-add, which SSE2 doesn't have and C's rounding forbids faking.
"""

from .. import defs as D
from .. import ext as _ext
from ..defs import R, Kq, Ins, rval, optab
from ..util import newtmp, getcon, die

_SIGN = {'b': 0x80, 'h': 0x8000, 'w': 0x80000000, 'l': -(1 << 63), 's': 0x80000000, 'd': -(1 << 63)}
_DUP = {'b': 'qdupb', 'h': 'qduph', 'w': 'qdupw', 'l': 'qdupl', 's': 'qdupw', 'd': 'qdupl'}


class _L:
    def __init__(self, fn):
        self.fn = fn
        self.out = []

    def op(self, name, a, b=R, to=None):
        to = to if to is not None else newtmp('qx', Kq, self.fn)
        self.out.append(Ins(getattr(D, 'O' + name), Kq, to, a, b))
        return to

    def copy(self, a, to):
        self.out.append(Ins(D.Ocopy, Kq, to, a, R))

    def con(self, v):
        return getcon(v, self.fn)

    def dup(self, x, v):
        n = {'b': 8, 'h': 16, 'w': 32, 'l': 64}[x]
        v &= (1 << n) - 1
        if v == (1 << n) - 1:
            v = -1
        return self.op(_DUP[x], self.con(v))

    def zero(self):
        return self.dup('w', 0)

    def ones(self):
        return self.dup('w', -1)

    def lnot(self, a):
        return self.op('qxor', a, self.ones())

    def sel(self, m, a, b):
        """bits of a where m has them, else of b."""
        return self.op('qor', self.op('qand', a, m), self.op('qxandn', m, b))

    def cgt(self, x, a, b):
        if x != 'l' or _ext.qbytes() == 32:  # AVX2 has vpcmpgtq
            return self.op('qcgt' + x, a, b)
        # a > b is b < a: the sign of d ^ ((b ^ a) & (d ^ b)), d = b - a (Hacker's Delight 2-12)
        d = self.op('qsubl', b, a)
        m = self.op('qxor', d, self.op('qand', self.op('qxor', b, a), self.op('qxor', d, b)))
        return self.op('qxshufd', self.op('qsarw', m, self.con(31)), self.con(0xf5))

    def cugt(self, x, a, b):
        s = self.dup(x, _SIGN[x])
        return self.cgt(x, self.op('qxor', a, s), self.op('qxor', b, s))


# what AVX2 has an instruction for (amd64/emit.py's _QY), on 256 bits
_AVX2 = frozenset(('qmulw', 'qceql', 'qcgtl', 'qminb', 'qminw', 'qmaxb', 'qmaxw', 'quminh', 'quminw', 'qumaxh', 'qumaxw'))


def _lower(L, i, third):
    """Writes i in L if it has to be; False if it has an instruction."""
    name = optab[i.op].name
    if name in _AVX2 and _ext.qbytes() == 32:
        return False
    a, b, to = i.arg[0], i.arg[1], i.to
    base, x = name[:-1], name[-1]
    if name == 'qnot':
        L.op('qxor', a, L.ones(), to)
    elif name == 'qsel1':
        L.op('qor', L.op('qand', a, third), L.op('qxandn', third, b), to)
    elif name in ('qfma1s', 'qfma1d'):
        if 'fma' not in _ext.x86:
            die("x86-64 has no fused multiply-add without FMA")
        return False  # vfmadd231ps/pd (amd64/isel.py)
    elif name == 'qsaddl' or name == 'qsaddh' or name == 'quaddl' or name == 'quaddh':
        high = L.zero() if name[1] == 'u' else L.op('qcgtw', L.zero(), b)
        L.op('qaddl', a, L.op('qxhi' if x == 'h' else 'qxlo', b, high), to)
    elif base == 'qneg':
        if x in 'sd':
            L.op('qxor', a, L.dup(_DUP[x][-1], _SIGN[x]), to)
        else:
            L.op('qsub' + x, L.zero(), a, to)
    elif base == 'qabs':
        L.op('qand', a, L.dup(_DUP[x][-1], ~_SIGN[x]), to)
    elif name == 'qmulw':
        odd = L.op('qxmuludq', L.op('qxshufd', a, L.con(0xf5)), L.op('qxshufd', b, L.con(0xf5)))
        even = L.op('qxmuludq', a, b)
        L.op('qxlo', L.op('qxshufd', even, L.con(8)), L.op('qxshufd', odd, L.con(8)), to)
    elif name == 'qmulb':
        even = L.op('qand', L.op('qmulh', a, b), L.dup('h', 0xff))
        odd = L.op('qshlh', L.op('qmulh', L.op('qshrh', a, L.con(8)), L.op('qshrh', b, L.con(8))), L.con(8))
        L.op('qor', even, odd, to)
    elif name == 'qceql':
        e = L.op('qceqw', a, b)
        L.op('qand', e, L.op('qxshufd', e, L.con(0xb1)), to)
    elif name == 'qcgtl':
        L.copy(L.cgt('l', a, b), to)
    elif base == 'qcge' and x in 'bhwl':
        L.op('qxor', L.cgt(x, b, a), L.ones(), to)
    elif base == 'qcugt':
        L.copy(L.cugt(x, a, b), to)
    elif base == 'qcuge':
        L.op('qxor', L.cugt(x, b, a), L.ones(), to)
    elif name in ('qmins', 'qmind', 'qmaxs', 'qmaxd'):
        # the number wins over a NaN, and -0 < +0: fminnm and fmaxnm, on what returns its second operand for both
        low = base == 'qmin'
        mm = ('qxmin' if low else 'qxmax') + x
        t, u = L.op(mm, a, b), L.op(mm, b, a)
        both = L.op('qor' if low else 'qand', t, u)
        na, nb = L.op('qxunord' + x, a, a), L.op('qxunord' + x, b, b)
        L.op('qor', L.op('qand', t, na), L.op('qxandn', na, L.sel(nb, u, both)), to)
    elif (base in ('qmin', 'qmax') and x in 'bw') or (base in ('qumin', 'qumax') and x in 'hw'):
        m = L.cugt(x, a, b) if base[1] == 'u' else L.cgt(x, a, b)
        hi = base.endswith('max')
        L.op('qor', L.op('qand', a if hi else b, m), L.op('qxandn', m, b if hi else a), to)
    elif name == 'qshlb':
        L.op('qand', L.op('qshlh', a, b), L.dup('b', 0xff << L.fn.con[rval(b)].i), to)
    elif name == 'qshrb':
        L.op('qand', L.op('qshrh', a, b), L.dup('b', 0xff >> L.fn.con[rval(b)].i), to)
    elif name == 'qsarb':
        m = L.dup('b', 0x80 >> L.fn.con[rval(b)].i)
        L.op('qsubb', L.op('qxor', L.op('qand', L.op('qshrh', a, b), L.dup('b', 0xff >> L.fn.con[rval(b)].i)), m), m, to)
    elif name == 'qsarl':
        s = L.op('qxshufd', L.op('qsarw', a, L.con(31)), L.con(0xf5))
        L.op('qxor', L.op('qshrl', L.op('qxor', a, s), b), s, to)
    else:
        return False
    return True


def run(fn):
    """Whether anything changed. Needs rpo."""
    changed = False
    for blk in fn.rpo[:fn.nblk]:
        if not any(D.isqop(i.op) for i in blk.ins[:blk.nins]):
            continue
        out = []
        third = R
        for i in blk.ins[:blk.nins]:
            if i.op == D.Oqfma0 or i.op == D.Oqsel0:
                third = i.arg[0]
                if i.op == D.Oqfma0 and 'fma' in _ext.x86:
                    out.append(i)  # kept with its qfma1
                    continue
                changed = True
                continue
            L = _L(fn)
            if D.isqop(i.op) and _lower(L, i, third):
                out += L.out
                changed = True
            else:
                out.append(i)
        blk.ins = out
        blk.nins = len(out)
    return changed
