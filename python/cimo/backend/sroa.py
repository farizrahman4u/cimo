"""cimo: LLVM's SROA, GCC's sra (ext.cse; not in QBE).

QBE promotes a stack slot to a temporary when every access is the whole slot. A struct or a small array
whose address never leaves the function, read and written at constant offsets, is several such slots."""

from . import defs as D
from .defs import R, RTmp, RCon, CBits, Kw, Kl, Ks, Kd, KBASE, rtype, rval, Oalloc, Oalloc1, Ocast, Oadd, Ocopy
from .util import newtmp, getcon

_LOADSIZE = {D.Oloadsb: 1, D.Oloadub: 1, D.Oloadsh: 2, D.Oloaduh: 2, D.Oloadsw: 4, D.Oloaduw: 4}
_STORESIZE = {D.Ostoreb: 1, D.Ostoreh: 2, D.Ostorew: 4, D.Ostorel: 8, D.Ostores: 4, D.Ostored: 8}
_STORECLS = {D.Ostoreb: Kw, D.Ostoreh: Kw, D.Ostorew: Kw, D.Ostorel: Kl, D.Ostores: Ks, D.Ostored: Kd}
_STOREOF = {Kw: D.Ostorew, Kl: D.Ostorel, Ks: D.Ostores, Kd: D.Ostored}
MAXSIZE = 256
_QLOADS = (D.Oqload, D.Oqload16)
_QSTORES = (D.Oqstore, D.Oqstore16)


def _qsize(op):
    from . import ext
    return 16 if op in (D.Oqload16, D.Oqstore16) else ext.qbytes()


def _loadsize(i):
    if i.op in _LOADSIZE:
        return _LOADSIZE[i.op]
    return 8 if i.cls in (Kl, Kd) else 4


def sroa(fn):
    """Returns True if a slot was split. Requires the temporaries' definition counts (filluse)."""
    roots = {}
    for i in fn.start.ins[:fn.start.nins]:
        if Oalloc <= i.op <= Oalloc1 and rtype(i.arg[0]) == RCon and rtype(i.to) == RTmp \
                and fn.tmp[rval(i.to)].ndef == 1:
            size = fn.con[rval(i.arg[0])].i
            if 0 < size <= MAXSIZE:
                roots[i.to] = (size, i)
    if not roots:
        return False
    at = {r: (r, 0) for r in roots}  # temporary: (slot, offset)
    adds = []
    # addresses are computed before they are used, in the order of the blocks
    known = {}  # temporaries that hold a constant: the frontend's p + extsw(1) * 4

    def value(r):
        if rtype(r) == RCon and fn.con[rval(r)].type == CBits:
            return fn.con[rval(r)].i
        return known.get(r)
    for b in fn.rpo[:fn.nblk]:
        for i in b.ins[:b.nins]:
            if rtype(i.to) != RTmp or fn.tmp[rval(i.to)].ndef != 1:
                continue
            if i.op == Oadd and i.cls == Kl and (i.arg[0] in at or i.arg[1] in at):
                x, y = i.arg
                if y in at:
                    x, y = y, x
                c = value(y)
                if c is not None:
                    at[i.to] = (at[x][0], at[x][1] + c)
                    adds.append(i)
            elif i.op == Ocopy and i.cls == Kl and i.arg[0] in at:
                # the same address under another name: an inlined callee's pointer parameter (scramjit passes an
                # int64_t io[] for results, which stayed in memory)
                at[i.to] = at[i.arg[0]]
                adds.append(i)
            elif i.op in (D.Oextsw, D.Oextuw, Ocopy) and value(i.arg[0]) is not None:
                c = value(i.arg[0])
                if i.op == D.Oextsw:
                    c = (c & 0xffffffff ^ 0x80000000) - 0x80000000
                elif i.op == D.Oextuw:
                    c &= 0xffffffff
                known[i.to] = c
            elif i.op in (D.Omul, Oadd) and i.cls == Kl and value(i.arg[0]) is not None and value(i.arg[1]) is not None:
                known[i.to] = value(i.arg[0]) * value(i.arg[1]) if i.op == D.Omul else value(i.arg[0]) + value(i.arg[1])
    bad = set()
    access = {r: {} for r in roots}  # slot: {offset: size}
    uses = []
    for b in fn.rpo[:fn.nblk]:
        p = b.phi
        while p:
            bad.update(at[a][0] for a in p.arg if a in at)
            p = p.link
        if b.jmp.arg in at:
            bad.add(at[b.jmp.arg][0])
        for i in b.ins[:b.nins]:
            if Oalloc <= i.op <= Oalloc1:
                continue
            if i.op in (Oadd, Ocopy) and i.to in at:
                continue
            if D.isload(i.op) and i.arg[0] in at:
                root, off = at[i.arg[0]]
                size = _loadsize(i)
            elif i.op in _QLOADS and i.arg[0] in at:  # ext.vector: a vector of several q values, each its own
                root, off = at[i.arg[0]]
                size = _qsize(i.op)
            elif i.op in _QSTORES and i.arg[1] in at:
                if i.arg[0] in at:
                    bad.add(at[i.arg[0]][0])
                root, off = at[i.arg[1]]
                size = _qsize(i.op)
            elif D.isstore(i.op) and i.arg[1] in at:
                if i.arg[0] in at:
                    bad.add(at[i.arg[0]][0])
                root, off = at[i.arg[1]]
                size = _STORESIZE[i.op]
            else:
                bad.update(at[a][0] for a in i.arg if a in at)
                continue
            if access[root].setdefault(off, size) != size or off < 0 or off + size > roots[root][0]:
                bad.add(root)
            uses.append((b, i, root, off))
    for r, acc in access.items():
        end = 0
        for off in sorted(acc):
            if off < end:  # overlapping accesses of different extent
                bad.add(r)
            end = max(end, off + acc[off])
        if len(acc) == 1 and 0 in acc and acc[0] == roots[r][0]:
            bad.add(r)  # the whole slot: promotion needs nothing
    good = {r for r in roots if r not in bad}
    if not good:
        return False
    slots = {}
    cast = {}  # store whose value needs another class: slots hold one
    classes = {}
    for b, i, root, off in uses:
        if root in good and D.isstore(i.op):
            classes.setdefault((root, off), _STORECLS[i.op])
    newalloc = []
    for root in good:
        for off, size in sorted(access[root].items()):
            t = newtmp('sr', Kl, fn)
            slots[(root, off)] = t
            newalloc.append(D.Ins(Oalloc if size <= 4 else Oalloc + 1 if size <= 8 else Oalloc1, Kl, t, getcon(size, fn), R))
    perblock = {}
    for b, i, root, off in uses:
        if root not in good:
            continue
        t = slots[(root, off)]
        if D.isload(i.op) or i.op in _QLOADS:
            i.arg[0] = t
        elif i.op in _QSTORES:
            i.arg[1] = t
        else:
            i.arg[1] = t
            k = _STORECLS[i.op]
            want = classes[(root, off)]
            if k != want:  # the same size in another class: a struct copied by words
                c = newtmp('sr', want, fn)
                perblock.setdefault(id(b), (b, {}))[1][id(i)] = D.Ins(Ocast, want, c, i.arg[0], R)
                i.op = _STOREOF[want]
                i.arg[0] = c
    for b, extra in perblock.values():
        out = []
        for i in b.ins[:b.nins]:
            if id(i) in extra:
                out.append(extra[id(i)])
            out.append(i)
        b.ins = out
        b.nins = len(out)
    for i in adds:
        if at[i.to][0] in good:
            i.set(D.Ins(D.Onop))
    s = fn.start
    for r in good:
        roots[r][1].set(D.Ins(D.Onop))
    npar = 0  # after the parameters, which must come first (promote makes a q slot's alloc an instruction)
    while npar < s.nins and D.ispar(s.ins[npar].op):
        npar += 1
    s.ins = s.ins[:npar] + newalloc + s.ins[npar:s.nins]
    s.nins = len(s.ins)
    return True
