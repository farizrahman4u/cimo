"""Which memory accesses may touch the same bytes: base objects ("roots") of addresses.

Roots are the values an address is computed from by add/sub/copy/phi: 64-bit function
parameters, stack slots (alloc) and global symbols. 32-bit values and the results of other
integer arithmetic (mul, shifts, extensions...) are offsets, with no roots. For each
temporary we compute
  may(r):  every root r may be based on (union over phi arguments), or None if unknown
           (a pointer loaded from memory, a call result...);
  must(r): roots r is certainly based on (intersection over phi arguments).

Two accesses A and B don't alias when may(A) and may(B) are known and non-empty and
  * a `restrict` parameter p is in must(A) but not in may(B), or vice versa (C11 6.7.3.1:
    if an object accessed through a restrict pointer is modified, every access to it goes
    through a pointer based on it); or
  * every pair (a, b) from may(A) x may(B) are distinct identified objects: two different
    stack slots or global symbols, or a stack slot and a parameter (a caller can't point
    into our frame; our stack slots only escape through stores, after which pointers
    loaded from memory are unknown anyway).
Anything else may alias.
"""

from ..backend import defs as D
from ..backend.defs import R, RTmp, RCon, CAddr, Kl, rtype, rval
from . import ir


class Roots:
    def __init__(self, fn, restrict_names):
        self.fn = fn
        self.restrict = set()
        self.kind = {}      # root key -> 'par' | 'slot' | 'sym'
        self.defs = {}
        for b in ir.blocks(fn):
            for p in ir.phis(b):
                self.defs[rval(p.to)] = p
            for i in ir.ins(b):
                if ir.istmp(i.to):
                    self.defs[rval(i.to)] = i
        for t, d in self.defs.items():
            if isinstance(d, D.Ins) and D.ispar(d.op):
                if fn.tmp[t].name in restrict_names:
                    self.restrict.add(('t', t))
        # may: grows from empty (optimistic for phi cycles) up to None (unknown)
        self.may = {}
        changed = True
        while changed:
            changed = False
            for t, d in self.defs.items():
                v = self._may(t, d)
                if t not in self.may or self.may[t] != v:
                    self.may[t] = v
                    changed = True
        # must: shrinks from "everything" (0 = not computed yet) by intersection
        self.must = {}
        changed = True
        while changed:
            changed = False
            for t, d in self.defs.items():
                v = self._must(t, d)
                if self.must.get(t, 0) != v:
                    self.must[t] = v
                    changed = True

    def _rootkey(self, r):
        if rtype(r) == RCon:
            c = self.fn.con[rval(r)]
            if c.type == CAddr:
                k = ('s', c.sym.type, c.sym.id)
                self.kind[k] = 'sym'
                return frozenset([k])
            return frozenset()
        return None

    def mayof(self, r):
        if rtype(r) == RTmp:
            return self.may.get(rval(r), frozenset())
        return self._rootkey(r)

    def mustof(self, r):
        if rtype(r) == RTmp:
            return self.must.get(rval(r), 0)
        return self._rootkey(r) or frozenset()

    def _union(self, args):
        acc = frozenset()
        for a in args:
            if a == R:
                continue
            ra = self.mayof(a)
            if ra is None:
                return None
            acc = acc | ra
        return acc

    def _may(self, t, d):
        if isinstance(d, D.Phi):
            return self._union(d.arg[:d.narg])
        op = d.op
        if D.ispar(op):
            if d.cls != Kl:
                return frozenset()
            k = ('t', t)
            self.kind[k] = 'par'
            return frozenset([k])
        if D.isalloc(op):
            k = ('t', t)
            self.kind[k] = 'slot'
            return frozenset([k])
        if op in (D.Oadd, D.Osub, D.Ocopy):
            return self._union(d.arg)
        if op in ir.PURE:
            return frozenset()      # an offset
        return None

    def _must(self, t, d):
        if isinstance(d, D.Phi):
            acc = 0
            for a in d.arg[:d.narg]:
                ra = self.mustof(a)
                if ra is None:
                    return frozenset()
                if ra == 0:
                    continue
                acc = ra if acc == 0 else acc & ra
            return frozenset() if acc == 0 else acc
        op = d.op
        if D.ispar(op) or D.isalloc(op):
            m = self.may.get(t)
            return m if m is not None else frozenset()
        if op in (D.Oadd, D.Osub, D.Ocopy):
            acc = frozenset()
            for a in d.arg:
                if a == R:
                    continue
                ra = self.mustof(a)
                if ra == 0:
                    return 0
                if ra is None:
                    continue
                acc = acc | ra
            return acc
        return frozenset()

    def may_alias(self, a, b):
        ma, mb = self.mayof(a), self.mayof(b)
        if not ma or not mb:
            return True
        sa, sb = self.mustof(a) or frozenset(), self.mustof(b) or frozenset()
        for x in sa:
            if x in self.restrict and x not in mb:
                return False
        for x in sb:
            if x in self.restrict and x not in ma:
                return False
        for x in ma:
            for y in mb:
                if x == y:
                    return True
                kx = self.kind.get(x)
                ky = self.kind.get(y)
                if kx == 'par' and ky == 'par':
                    return True
                if (kx == 'sym' and ky == 'par') or (kx == 'par' and ky == 'sym'):
                    return True
        return False
