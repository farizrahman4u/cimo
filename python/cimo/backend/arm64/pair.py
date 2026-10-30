"""cimo extension (ext.ldp): ldp/stp pairing on arm64.

Two parts:
  * schedule(fn), after instruction selection (SSA temps, before register allocation): moves
    the second access of a pairable couple next to the first, so that register allocation
    gives them distinct registers and they come out adjacent. A load moves up to its partner;
    a store moves down to its partner (its value is defined before it, so it stays valid).
  * pairof(i1, i2, e), in emit: two adjacent accesses that fit one ldp/stp are emitted as one.
    That needs no alias reasoning (nothing moves), so it also pairs spill slots and the
    callee-save saves/restores.

Pairable: plain 32/64-bit integer or float loads/stores (ldr/str w, x, s, d; not the sign-
extending or byte/half forms) off the same base, at offsets o and o+size, the lower one a
multiple of size in [-64*size, 63*size].

Moving an access past another instruction is allowed only if that instruction cannot touch
the same memory: loads move past loads; any access moves past a store/load of the same base
whose range doesn't overlap; nothing moves past calls or other memory side effects, or past
an access off another base, unless ext.disjoint(fn, base1, base2) says the two bases never
overlap (hook for alias knowledge such as restrict; default: they may).
"""

from .. import defs as D
from .. import ext
from ..defs import (R, RTmp, RMem, Tmp0, Kw, Kl, Ks, Kd, Oload, Oloadsw, Oloaduw, Ostorew,
                    Ostorel, Ostores, Ostored, Ocall, Ovastart, Ovaarg, Oblit0, Oblit1,
                    Osalloc, Onop, rtype, rval, isload, isstore)

WINDOW = 24  # how far ahead to look for a partner

_LOADK = {Kw: 'w', Kl: 'x', Ks: 's', Kd: 'd'}
_STOREK = {Ostorew: 'w', Ostorel: 'x', Ostores: 's', Ostored: 'd'}
SIZE = {'w': 4, 'x': 8, 's': 4, 'd': 8}


def kind(i):
    """('l'|'s', register letter) for a pairable access, else None."""
    op = i.op
    if op == Oload:
        return 'l', _LOADK[i.cls]
    if op == Oloaduw or (op == Oloadsw and i.cls == Kw):
        return 'l', 'w'
    k = _STOREK.get(op)
    if k:
        return 's', k
    return None


def inrange(lo, sz):
    return lo % sz == 0 and -64 * sz <= lo <= 63 * sz


def _addr(r, fn):
    """(base temp, offset) of an address operand before register allocation, or None."""
    t = rtype(r)
    if t == RTmp:
        return r, 0
    if t == RMem:
        m = fn.mem[rval(r)]
        if m.index == R and rtype(m.base) == RTmp and m.offset.type in (D.CBits, D.CUndef):
            return m.base, m.offset.i if m.offset.type == D.CBits else 0
    return None


def access(i, fn):
    """(isload, base, off, size, letter) for any load/store with a simple address; else None.
    Byte/half/sign-extending accesses are described too (they block moves) with letter None."""
    if isload(i.op):
        a = _addr(i.arg[0], fn)
        ld = True
    elif isstore(i.op):
        a = _addr(i.arg[1], fn)
        ld = False
    else:
        return None
    if a is None:
        return False  # unknown address
    from ..load import loadsz, storesz
    k = kind(i)
    sz = loadsz(i) if ld else storesz(i)
    return ld, a[0], a[1], sz, k[1] if k else None


_BARRIER = (Ocall, Ovastart, Ovaarg, Oblit0, Oblit1, Osalloc)


def _disjoint(fn, a, b):
    """Can access a (descr) and access b never overlap?"""
    if a[1] == b[1]:
        return a[2] + a[3] <= b[2] or b[2] + b[3] <= a[2]
    f = ext.disjoint
    return bool(f and f(fn, a[1], b[1]))


def _cross(fn, mover, i, d):
    """Can access `mover` move past instruction i (access descr d)?"""
    if i.op in _BARRIER:
        return False
    if d is None:
        return True
    if d is False:
        return False
    if mover[0] and d[0]:
        return True  # load past load
    return _disjoint(fn, mover, d)


def _phys(r):
    return rtype(r) == RTmp and rval(r) < Tmp0


def schedule(fn):
    b = fn.start
    while b:
        _schedblk(b, fn)
        b = b.link


def _schedblk(b, fn):
    ins = b.ins[:b.nins]
    n = len(ins)
    info = [access(i, fn) for i in ins]
    done = [False] * n
    k = 0
    while k < n:
        d = info[k]
        if not d or not d[4] or done[k] or _phys(d[1]) or _phys(ins[k].to) \
                or (not d[0] and _phys(ins[k].arg[0])):
            k += 1
            continue
        for j in range(k + 1, min(n, k + WINDOW)):
            dj = info[j]
            if ins[j].op in _BARRIER:
                break
            if (not dj or done[j] or dj[0] != d[0] or dj[1] != d[1] or dj[4] != d[4]
                    or abs(dj[2] - d[2]) != d[3] or not inrange(min(d[2], dj[2]), d[3])):
                continue
            if _phys(ins[j].to) or (not dj[0] and _phys(ins[j].arg[0])):
                continue
            if d[0]:  # hoist load j to k+1
                if all(_cross(fn, dj, ins[m], info[m]) for m in range(k + 1, j)):
                    for a in (ins, info, done):
                        a.insert(k + 1, a.pop(j))
                    done[k] = done[k + 1] = True
                    break
            else:  # sink store k to j-1
                if all(_cross(fn, d, ins[m], info[m]) for m in range(k + 1, j)):
                    for a in (ins, info, done):
                        a.insert(j - 1, a.pop(k))
                    done[j - 1] = done[j] = True
                    k -= 1  # ins[k] is a new instruction now
                    break
        k += 1
    b.ins[:n] = ins
