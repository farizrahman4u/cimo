"""Port of QBE live.c."""

import sys

from . import defs as D
from .defs import BSet, R, RTmp, RCall, RMem, M64, Ocall, KBASE, rtype, rval
from .util import (bsinit, bscopy, bsclr, bsset, bshas, bsunion, bsequal, bscount,
                   bselems, dumpts)


def liveon(v, b, s):
    bscopy(v, s.in_)
    p = s.phi
    while p:
        if rtype(p.to) == RTmp:
            bsclr(v, rval(p.to))
        p = p.link
    p = s.phi
    while p:
        for a in range(p.narg):
            if p.blk[a] is b:
                if rtype(p.arg[a]) == RTmp:
                    bsset(v, rval(p.arg[a]))
                    bsset(b.gen, rval(p.arg[a]))
        p = p.link


def bset(r, b, nlv, tmp):
    if rtype(r) != RTmp:
        return
    bsset(b.gen, rval(r))
    if not bshas(b.in_, rval(r)):
        nlv[KBASE(tmp[rval(r)].cls)] += 1
        bsset(b.in_, rval(r))


def filllive(f):
    """liveness analysis
    requires rpo computation"""
    T = D.T
    u = BSet()
    v = BSet()
    bsinit(u, f.ntmp)
    bsinit(v, f.ntmp)
    b = f.start
    while b:
        bsinit(b.in_, f.ntmp)
        bsinit(b.out, f.ntmp)
        bsinit(b.gen, f.ntmp)
        b = b.link
    chg = 1
    while True:
        for n in range(f.nblk - 1, -1, -1):
            b = f.rpo[n]

            bscopy(u, b.out)
            if b.s1:
                liveon(v, b, b.s1)
                bsunion(b.out, v)
            if b.s2:
                liveon(v, b, b.s2)
                bsunion(b.out, v)
            chg |= int(not bsequal(b.out, u))

            nlv = [0, 0]
            b.out.t |= T.rglob
            bscopy(b.in_, b.out)
            for t in bselems(b.in_):
                nlv[KBASE(f.tmp[t].cls)] += 1
            if rtype(b.jmp.arg) == RCall:
                assert (bscount(b.in_) == T.nrglob and (b.in_.t & M64) == T.rglob)
                b.in_.t |= T.retregs(b.jmp.arg, nlv)
            else:
                bset(b.jmp.arg, b, nlv, f.tmp)
            for k in range(2):
                b.nlive[k] = nlv[k]
            for ii in range(b.nins - 1, -1, -1):
                i = b.ins[ii]
                if i.op == Ocall and rtype(i.arg[1]) == RCall:
                    m = [0, 0]
                    b.in_.t &= ~T.retregs(i.arg[1], m)
                    for k in range(2):
                        nlv[k] -= m[k]
                        # caller-save registers are used
                        # by the callee, in that sense,
                        # right in the middle of the call,
                        # they are live:
                        nlv[k] += T.nrsave[k]
                        if nlv[k] > b.nlive[k]:
                            b.nlive[k] = nlv[k]
                    b.in_.t |= T.argregs(i.arg[1], m)
                    for k in range(2):
                        nlv[k] -= T.nrsave[k]
                        nlv[k] += m[k]
                if i.to != R:
                    assert rtype(i.to) == RTmp
                    t = rval(i.to)
                    if bshas(b.in_, t):
                        nlv[KBASE(f.tmp[t].cls)] -= 1
                    bsset(b.gen, t)
                    bsclr(b.in_, t)
                for k in range(2):
                    if rtype(i.arg[k]) == RMem:
                        ma = f.mem[rval(i.arg[k])]
                        bset(ma.base, b, nlv, f.tmp)
                        bset(ma.index, b, nlv, f.tmp)
                    else:
                        bset(i.arg[k], b, nlv, f.tmp)
                for k in range(2):
                    if nlv[k] > b.nlive[k]:
                        b.nlive[k] = nlv[k]
        if chg:
            chg = 0
            continue
        break

    if D.debug['L']:
        err = sys.stderr
        err.write("\n> Liveness analysis:\n")
        b = f.start
        while b:
            err.write("\t%-10sin:   " % b.name)
            dumpts(b.in_, f.tmp, err)
            err.write("\t          out:  ")
            dumpts(b.out, f.tmp, err)
            err.write("\t          gen:  ")
            dumpts(b.gen, f.tmp, err)
            err.write("\t          live: ")
            err.write("%d %d\n" % (b.nlive[0], b.nlive[1]))
            b = b.link
