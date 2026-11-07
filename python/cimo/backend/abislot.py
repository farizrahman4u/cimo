"""cimo: after the ABI (abi1), a struct passed or returned by value goes through a slot: the parameter's registers are
stored to one the body loads from, a call's result registers to one the caller then loads. Nothing after the ABI
forwards stores to loads (alias analysis doesn't take its physical registers), so __int128 and small structs paid a
store and a load at every call boundary. Here, a slot that doesn't escape, written once at each offset by a store that
dominates every load of that offset and that no other store overlaps, has each such load replaced by the stored value (a cast where the class differs);
a slot left without loads loses its stores and its alloc."""

from . import defs as D
from .defs import R, RTmp, RCon, Ins, rtype, rval, UIns
from . import cfg as _cfg

_STORE = {D.Ostoreb: (1, None), D.Ostoreh: (2, None), D.Ostorew: (4, D.Kw), D.Ostorel: (8, D.Kl),
          D.Ostores: (4, D.Ks), D.Ostored: (8, D.Kd)}
_LOAD = {D.Oloadsw: 4, D.Oloaduw: 4, D.Oload: None}  # Oload: by its class


def forward(fn):
    """Whether anything changed. Needs filluse; fills dominators."""
    b0 = fn.start
    allocs = [i for i in b0.ins[:b0.nins] if D.Oalloc <= i.op <= D.Oalloc1 and rtype(i.to) == RTmp]
    if not allocs:
        return False
    _cfg.filldom(fn)
    where = {}  # id(ins) -> (block, index)
    for b in fn.rpo[:fn.nblk]:
        for k, i in enumerate(b.ins[:b.nins]):
            where[id(i)] = (b, k)
    changed = False
    for a in allocs:
        acc = _accesses(fn, a.to)
        if acc is None:
            continue
        stores, loads, adds = acc
        byoff = {}
        for off, st in stores:
            if off in byoff:
                byoff = None
                break
            byoff[off] = st
        if byoff is None:
            continue
        done = []
        for off, ld in loads:
            st = byoff.get(off)
            if st is None:
                continue
            size, k = _STORE[st.op]
            lsize = _LOAD.get(ld.op, 0)
            if lsize is None:
                lsize = 8 if D.KWIDE(ld.cls) else 4
            if k is None or lsize != size or rtype(st.arg[0]) not in (RTmp, RCon):
                continue
            if any(o2 < off + size and off < o2 + _STORE[s2.op][0] for o2, s2 in stores if s2 is not st):
                continue  # another store writes some of these bytes
            if ld.op in (D.Oloadsw, D.Oloaduw) and ld.cls != D.Kw:
                continue  # an extension, not the value
            sb, si = where[id(st)]
            lb, li = where[id(ld)]
            if not (_cfg.dom(sb, lb) and (sb is not lb or si < li)):
                continue
            done.append((ld, st, k))
        if not done:
            continue
        for ld, st, k in done:
            v = st.arg[0]
            ld.op = D.Ocopy if D.KBASE(k) == D.KBASE(ld.cls) else D.Ocast
            ld.arg[0], ld.arg[1] = v, R
        changed = True
        if len(done) == len(loads):  # nothing reads the slot now: its stores, address arithmetic and alloc go
            for off, st in stores:
                st.set(Ins(D.Onop))
            for x in adds:
                x.set(Ins(D.Onop))
            a.set(Ins(D.Onop))
    return changed


def _accesses(fn, slot):
    """([(offset, store)], [(offset, load)], [address adds]) of every use of the slot, or None if one is anything else
    (an escape: a call argument, a store of the address, arithmetic other than a constant offset)."""
    stores, loads, adds = [], [], []
    work = [(slot, 0)]
    while work:
        t, off = work.pop()
        tm = fn.tmp[rval(t)]
        for u in tm.use[:tm.nuse]:
            if u.type != UIns:
                return None
            i = u.u
            if i.op in _STORE and i.arg[1] == t and i.arg[0] != t:
                stores.append((off, i))
            elif (i.op in _LOAD) and i.arg[0] == t:
                loads.append((off, i))
            elif i.op == D.Oadd and i.cls == D.Kl and rtype(i.to) == RTmp and i.arg[0] == t and rtype(i.arg[1]) == RCon \
                    and fn.con[rval(i.arg[1])].type == D.CBits:
                adds.append(i)
                work.append((i.to, off + fn.con[rval(i.arg[1])].i))
            else:
                return None
    return stores, loads, adds
