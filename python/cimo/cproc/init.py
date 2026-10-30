"""Port of init.c.

The initparser's `obj[32]` array is a list of Object; `cur` and `sub` are indices
(cur is None for NULL). `last` (struct init **) is the Init whose `next` is the
insertion point, or None for &p->init. Object.u is the C union {mem, idx}: one
attribute, so stale values carry over exactly as in C.
"""

from .cc import (Init, BITS0, M64, TYPEARRAY, TYPESTRUCT, TYPEUNION, PROPSCALAR,
                 PROPINT, PROPCHAR, EXPRSTRING, TLBRACK, TPERIOD, TASSIGN, TIDENT, TRBRACK,
                 TLBRACE, TRBRACE, TCOMMA)
from .token import tok, error
from .pp import next, expect, consume
from .type import typecompatible
from . import expr as E
from . import util as U

NOBJ = 32


class Object(object):
    __slots__ = ('offset', 'type', 'u', 'iscur')

    def __init__(self):
        self.offset = 0
        self.type = None
        self.u = None
        self.iscur = False


class InitParser(object):
    __slots__ = ('obj', 'cur', 'sub', 'init', 'last')


def mkinit(start, end, bits, expr):
    return Init(start, end, bits, expr)


def initadd(p, new):
    prev = p.last
    old = p.init if prev is None else prev.next
    while old is not None:
        if (old.end * 8 - old.bits[1]) & M64 <= (new.start * 8 + new.bits[0]) & M64:
            prev = old
            old = old.next
            continue
        # no overlap, insert before `old`
        if (new.end * 8 - new.bits[1]) & M64 <= (old.start * 8 + old.bits[0]) & M64:
            break
        # replace any initializers that `new` covers
        if (old.end * 8 - old.bits[1]) & M64 <= (new.end * 8 - new.bits[1]) & M64:
            while True:
                old = old.next
                if not (old and (old.end * 8 - old.bits[1]) & M64 <= (new.end * 8 - new.bits[1]) & M64):
                    break
            break
        # `old` covers `new`, keep looking
        prev = old
        old = old.next
    new.next = old
    if prev is None:
        p.init = new
    else:
        prev.next = new
    p.last = new


def subobj(p, t, off):
    # the subobject could be incomplete if it's a flexible array member
    if t.incomplete:
        error(tok.loc, "initializer specified for incomplete type")
    off = (off + p.obj[p.sub].offset) & M64
    p.sub += 1
    if p.sub == NOBJ:
        U.fatal("internal error: too many designators")
    o = p.obj[p.sub]
    o.type = t
    o.offset = off
    o.iscur = False


def findmember(p, name):
    m = p.obj[p.sub].type.members
    while m:
        if m.name is not None:
            if m.name == name:
                p.obj[p.sub].u = m
                subobj(p, m.type, m.offset)
                return True
        else:
            subobj(p, m.type, m.offset)
            if findmember(p, name):
                return True
            p.sub -= 1
        m = m.next
    return False


def designator(s, p):
    p.last = None
    p.sub = p.cur
    while True:
        o = p.obj[p.sub]
        t = o.type
        if tok.kind == TLBRACK:
            if t.kind != TYPEARRAY:
                error(tok.loc, "index designator is only valid for array types")
            next()
            o.u = (E.intconstexpr(s, False) * t.base.size) & M64
            if o.u >= t.size:
                if not t.incomplete:
                    error(tok.loc, "index designator is larger than array length")
                t.size = (o.u + t.base.size) & M64
            expect(TRBRACK, "for index designator")
            subobj(p, t.base, o.u)
        elif tok.kind == TPERIOD:
            if t.kind != TYPESTRUCT and t.kind != TYPEUNION:
                error(tok.loc, "member designator only valid for struct/union types")
            next()
            name = expect(TIDENT, "for member designator")
            if not findmember(p, name):
                error(tok.loc, "%s has no member named '%s'", "union" if t.kind == TYPEUNION else "struct", name)
        else:
            expect(TASSIGN, "after designator")
            return


def focus(p):
    o = p.obj[p.sub]
    k = o.type.kind
    if k == TYPEARRAY:
        t = o.type.base
        o.u = 0
        if o.type.incomplete:
            o.type.size = t.size
    elif k == TYPESTRUCT or k == TYPEUNION:
        o.u = o.type.members
        t = o.u.type
    else:
        U.fatal("internal error: init cursor has unexpected type")
    subobj(p, t, 0)


def advance(p):
    while True:
        p.sub -= 1
        o = p.obj[p.sub]
        t = o.type
        if t.kind == TYPEARRAY:
            o.u = (o.u + t.base.size) & M64
            if o.u == t.size:
                if t.incomplete:
                    t.size = (t.size + t.base.size) & M64
                    subobj(p, t.base, o.u)
                    return
            else:
                subobj(p, t.base, o.u)
                return
        elif t.kind == TYPESTRUCT:
            o.u = o.u.next
            if o.u:
                subobj(p, o.u.type, o.u.offset)
                return
        if p.sub == p.cur:
            error(tok.loc, "too many initializers for type")


def parseinit(s, t):
    """6.7.9 Initialization"""
    p = InitParser()
    p.obj = [Object() for _ in range(NOBJ)]
    p.cur = None
    p.sub = 0
    p.obj[0].offset = 0
    p.obj[0].type = t
    p.obj[0].iscur = False
    p.init = None
    p.last = None
    if t.incomplete and t.kind != TYPEARRAY:
        error(tok.loc, "initializer specified for incomplete type")
    if t.kind == TYPEARRAY and t.base.size == 0:
        error(tok.loc, "initializer specified for variable length array type")
    obj = p.obj
    while True:
        skip_to_next = False
        if p.cur is not None:
            if tok.kind == TLBRACK or tok.kind == TPERIOD:
                designator(s, p)
            elif p.sub != p.cur:
                advance(p)
            elif obj[p.cur].type.kind == TYPESTRUCT or obj[p.cur].type.kind == TYPEUNION:
                focus(p)
        if consume(TLBRACE):
            if consume(TRBRACE):
                if obj[p.sub].type.incomplete:
                    error(tok.loc, "array of unknown size has empty initializer")
                skip_to_next = True
            else:
                if p.cur == p.sub:
                    if obj[p.cur].type.prop & PROPSCALAR:
                        error(tok.loc, "nested braces around scalar initializer")
                    assert obj[p.cur].type.kind == TYPEARRAY
                    focus(p)
                p.cur = p.sub
                obj[p.cur].iscur = True
                continue
        if not skip_to_next:
            expr = E.assignexpr(s)
            while True:
                t = obj[p.sub].type
                if t.kind == TYPEARRAY:
                    if expr.decayed and expr.base.kind == EXPRSTRING and t.base.prop & PROPINT:
                        base = t.base
                        expr = expr.base
                        if not (base.prop & PROPCHAR and expr.type.base.prop & PROPCHAR) and not typecompatible(base, expr.type.base):
                            error(tok.loc, "cannot initialize array with string literal of different width")
                        if t.incomplete:
                            t.size = expr.type.size
                        break
                elif t.kind == TYPESTRUCT or t.kind == TYPEUNION:
                    if typecompatible(expr.type, t):
                        break
                else:  # scalar type
                    assert t.prop & PROPSCALAR
                    expr = E.exprassign(expr, t)
                    break
                focus(p)
            # add:
            if p.sub > 0 and (obj[p.sub - 1].type.kind == TYPESTRUCT or obj[p.sub - 1].type.kind == TYPEUNION):
                bits = obj[p.sub - 1].u.bits
            else:
                bits = BITS0
            o = obj[p.sub]
            initadd(p, mkinit(o.offset, (o.offset + o.type.size) & M64, bits, expr))
        while True:
            if not skip_to_next:
                if obj[p.sub].type.incomplete:
                    obj[p.sub].type.incomplete = False
            skip_to_next = False
            # next:
            if p.cur is None:
                return p.init
            if tok.kind == TCOMMA:
                next()
                if tok.kind != TRBRACE:
                    break
            elif tok.kind != TRBRACE:
                error(tok.loc, "expected ',' or '}' after initializer")
            next()
            p.sub = p.cur
            while True:
                p.cur = None if p.cur == 0 else p.cur - 1
                if not (p.cur is not None and not obj[p.cur].iscur):
                    break
