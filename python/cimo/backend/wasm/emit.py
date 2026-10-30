"""wasm code generation: QBE IR (after the target-independent passes, before any machine
lowering) to cimo.backend.wasm.obj functions and data.

- Every SSA temporary is a wasm local of its class; phis become copies on the incoming edges,
  done through the operand stack (push every argument, then pop into the phi locals), which is
  a parallel copy for free.
- Control flow is structured by Ramsey's algorithm ("Beyond Relooper", JFP 2022): over the
  dominator tree, a loop header opens a `loop`, every dominator-tree child with two or more
  forward predecessors (a merge node) is placed after a `block` that ends right before it, and
  branches become `br` to the enclosing loop or block, or the target's code inline. Irreducible
  control flow (a goto into a loop) is rejected.
- ABI (cimo's own; both sides of every call are compiled by cimo): wasm parameters in order;
  a struct returned by value through a hidden first pointer argument; a struct argument by
  pointer to a copy the caller makes; variadic arguments in 8-byte slots of a caller buffer
  whose address is passed as a last i64 argument (va_list keeps the next slot's address in its
  first 8 bytes). The C stack is in linear memory below `__stack_pointer` (Emscripten's), with
  a 16-byte aligned frame per function that needs one.
"""

import struct

from .. import defs as D
from ..defs import (R, RTmp, RCon, RInt, RType, Tmp0, Kw, Kl, Ks, Kd, CAddr, rtype, rval, rsval,
                    s32, s64, fbits, dbits, bitsf, bitsd, f32)
from ..util import str_
from .. import parse as _parse
from ..cfg import fillcfg, filldom
from .obj import (I32, I64, F32, F64, VT, Func, Data, CALL, ADDR, ADDR32, TYPE, PTR64, PTR32,
                  sigkey, uleb, sleb, uleb5, sleb5)

G_SP, G_MEM, G_TAB = 0, 1, 2  # imported globals: __stack_pointer, __memory_base, __table_base

_BIN = {  # op -> (i32, i64, f32, f64)
    D.Oadd: (0x6A, 0x7C, 0x92, 0xA0), D.Osub: (0x6B, 0x7D, 0x93, 0xA1),
    D.Omul: (0x6C, 0x7E, 0x94, 0xA2), D.Odiv: (0x6D, 0x7F, 0x95, 0xA3),
    D.Oudiv: (0x6E, 0x80, None, None), D.Orem: (0x6F, 0x81, None, None),
    D.Ourem: (0x70, 0x82, None, None), D.Oand: (0x71, 0x83, None, None),
    D.Oor: (0x72, 0x84, None, None), D.Oxor: (0x73, 0x85, None, None),
    D.Oshl: (0x74, 0x86, None, None), D.Osar: (0x75, 0x87, None, None),
    D.Oshr: (0x76, 0x88, None, None),
}
# integer comparisons in QBE order: eq ne sge sgt sle slt uge ugt ule ult
_ICMP = {Kw: (0x46, 0x47, 0x4E, 0x4A, 0x4C, 0x48, 0x4F, 0x4B, 0x4D, 0x49),
         Kl: (0x51, 0x52, 0x59, 0x55, 0x57, 0x53, 0x5A, 0x56, 0x58, 0x54)}
# float comparisons in QBE order: eq ge gt le lt ne (o, uo are special)
_FCMP = {Ks: (0x5B, 0x60, 0x5E, 0x5F, 0x5D, 0x5C), Kd: (0x61, 0x66, 0x64, 0x65, 0x63, 0x62)}
_LOAD = {  # op -> (opcode for a w result, opcode for an l result, log2 align)
    D.Oloadsb: (0x2C, 0x30, 0), D.Oloadub: (0x2D, 0x31, 0), D.Oloadsh: (0x2E, 0x32, 1),
    D.Oloaduh: (0x2F, 0x33, 1), D.Oloadsw: (0x28, 0x34, 2), D.Oloaduw: (0x28, 0x35, 2),
}
_STORE = {D.Ostoreb: (0x3A, Kw, 0), D.Ostoreh: (0x3B, Kw, 1), D.Ostorew: (0x36, Kw, 2),
          D.Ostorel: (0x37, Kl, 3), D.Ostores: (0x38, Ks, 2), D.Ostored: (0x39, Kd, 3)}
_UNF = {  # math builtins: op -> (f32, f64)
    D.Ofsqrt: (0x91, 0x9F), D.Ofabs: (0x8B, 0x99), D.Offloor: (0x8E, 0x9C),
    D.Ofceil: (0x8D, 0x9B), D.Oftrunc: (0x8F, 0x9D),
}


class WasmError(D.BackendError):
    pass


def symname(sym):
    s = str_(sym.id)
    return s[1:-1] if s.startswith('"') else s


def retvt(fname):
    """(results, sret) of a function from its declared return class."""
    k = _parse.retcls.get(fname, _parse.K0)
    if k == _parse.K0:
        return [], False
    if k == _parse.Kc:
        return [], True
    return [VT[Kw if k >= _parse.Ksb else k]], False


class _Gen:
    def __init__(self, fn):
        self.fn = fn
        self.c = bytearray()
        self.relocs = []
        self.calls = {}

    # ---- bytes

    def op(self, *bs):
        self.c.extend(bs)

    def lget(self, i):
        self.c.append(0x20)
        self.c += uleb(i)

    def lset(self, i):
        self.c.append(0x21)
        self.c += uleb(i)

    def ltee(self, i):
        self.c.append(0x22)
        self.c += uleb(i)

    def gget(self, g):
        self.c += bytes([0x23, g])

    def gset(self, g):
        self.c += bytes([0x24, g])

    def i32c(self, v):
        self.c.append(0x41)
        self.c += sleb(s32(v))

    def i64c(self, v):
        self.c.append(0x42)
        self.c += sleb(s64(v))

    def mem(self, opc, align, off=0):
        self.c.append(opc)
        self.c += uleb(align) + uleb(off)

    def addr(self, name, addend, wide):
        """The address of a symbol: i64 if wide, else i32."""
        self.relocs.append((len(self.c), ADDR if wide else ADDR32, name, addend))
        if wide:
            self.c += bytes([0x23, G_MEM, 0xAD, 0x42]) + sleb5(0) + bytes([0x7C])
        else:
            self.c += bytes([0x23, G_MEM, 0x41]) + sleb5(0) + bytes([0x6A])

    # ---- values

    def tcls(self, r):
        return self.fn.tmp[rval(r)].cls

    def push(self, r, k):
        t = rtype(r)
        if t == RTmp:
            n = rval(r)
            self.lget(self.loc[n])
            tk = self.fn.tmp[n].cls
            if tk == Kl and k == Kw:
                self.op(0xA7)
            elif tk == Kw and k == Kl:
                self.op(0xAD)
            elif tk != k:
                raise WasmError('wasm: class mismatch for %%%s' % self.fn.tmp[n].name)
        elif t == RCon:
            c = self.fn.con[rval(r)]
            if c.type == CAddr:
                self.addr(symname(c.sym), s64(c.i), k != Kw)
            elif k == Kw:
                self.i32c(c.i)
            elif k == Kl:
                self.i64c(c.i)
            elif k == Ks:
                bits = fbits(f32(c.d)) if c.flt == 2 else c.i & 0xffffffff
                self.c += bytes([0x43]) + struct.pack('<I', bits)
            else:
                bits = dbits(c.s) if c.flt == 1 else c.i
                self.c += bytes([0x44]) + struct.pack('<Q', bits & 0xffffffffffffffff)
        elif t == RInt:
            (self.i32c if k == Kw else self.i64c)(rsval(r))
        else:
            raise WasmError('wasm: unexpected operand')

    def paddr(self, r):
        """An address operand as an i32."""
        if rtype(r) == RCon and self.fn.con[rval(r)].type == CAddr:
            self.push(r, Kw)
        elif rtype(r) == RTmp and self.tcls(r) == Kw:
            self.push(r, Kw)
        else:
            self.push(r, Kl)
            self.op(0xA7)

    def setto(self, i):
        self.lset(self.loc[rval(i.to)])

    # ---- function

    def run(self):
        fn = self.fn
        fillcfg(fn)
        filldom(fn)
        self.check_reducible()
        self.results, sret = retvt(fn.name)
        params = []
        self.loc = {}
        if sret:
            self.sret = len(params)
            params.append(I64)
        start = fn.start
        for i in start.ins[:start.nins]:
            if i.op == D.Opar:
                self.loc[rval(i.to)] = len(params)
                params.append(VT[i.cls])
            elif i.op == D.Oparc:
                self.loc[rval(i.to)] = len(params)
                params.append(I64)
            elif i.op == D.Opare:
                raise WasmError('wasm: env parameters are not supported')
        if fn.vararg:
            self.vaptr = len(params)
            params.append(I64)
        self.params = params
        # locals, grouped by type after the parameters
        groups = {I32: [], I64: [], F32: [], F64: []}
        for n in range(Tmp0, fn.ntmp):
            if n not in self.loc:
                groups[VT[fn.tmp[n].cls]].append(n)
        extra = {I32: ['sp0', 'fp', 'x0', 'x1'], I64: ['l0', 'l1'], F32: ['f0', 'f1'],
                 F64: ['d0', 'd1']}
        idx = len(params)
        decls = []
        for t in (I32, I64, F32, F64):
            names = groups[t] + extra[t]
            decls.append((len(names), t))
            for n in names:
                self.loc[n] = idx
                idx += 1
        self.layout_frame()
        if self.frame:
            self.gget(G_SP)
            self.ltee(self.loc['sp0'])
            self.i32c(self.frame)
            self.op(0x6B)  # i32.sub
            self.ltee(self.loc['fp'])
            self.gset(G_SP)
        self.ctx = []
        self.tree(fn.start)
        if self.results:
            self.op(0x00)  # unreachable: every path returned
        self.op(0x0B)
        head = uleb(len(decls)) + b''.join(uleb(n) + bytes([t]) for n, t in decls)
        relocs = [(o + len(head), k, s, a) for o, k, s, a in self.relocs]
        return Func(fn.name, bool(fn.lnk.export), params, self.results,
                    head + bytes(self.c), relocs, self.calls)

    def check_reducible(self):
        for b in self.fn.rpo:
            for s in (b.s1, b.s2):
                if s is not None and s.id <= b.id:
                    d = b
                    while d is not None and d is not s:
                        d = d.idom
                    if d is None:
                        raise WasmError('%s: irreducible control flow (a goto into a loop) is '
                                        'not supported by the wasm backend' % self.fn.name)

    # ---- frame: static allocas, and per-call areas for varargs and struct copies

    def layout_frame(self):
        fn = self.fn
        off = 0
        self.slots = {}   # alloc Ins (id) -> offset
        self.areas = {}   # call Ins (id) -> {'va': off, 'sret': off, 'copies': [off...]}
        self.dyn = False

        def take(size, align):
            nonlocal off
            off = (off + align - 1) & -align
            o = off
            off += size
            return o

        for b in fn.rpo:
            args = []
            for i in b.ins[:b.nins]:
                if D.isalloc(i.op):
                    a = 4 << (i.op - D.Oalloc4)
                    if b is fn.start and rtype(i.arg[0]) == RCon and \
                            fn.con[rval(i.arg[0])].type != CAddr:
                        self.slots[id(i)] = take(max(fn.con[rval(i.arg[0])].i, 0), max(a, 4))
                    else:
                        self.dyn = True
                elif D.isarg(i.op):
                    args.append(i)
                elif i.op == D.Ocall:
                    area = {'copies': []}
                    nva, va = 0, False
                    for a in args:
                        if a.op == D.Oargv:
                            va = True
                        elif va:
                            if a.op == D.Oargc:
                                raise WasmError('%s: struct arguments to variadic functions are'
                                                ' not supported by the wasm backend' % fn.name)
                            nva += 1
                        elif a.op == D.Oargc:
                            t = D.typ[rval(a.arg[0])]
                            area['copies'].append(take(t.size, max(t.align, 8)))
                    if va:
                        area['va'] = take(8 * max(nva, 1), 16)
                    if rtype(i.arg[1]) == RType:
                        t = D.typ[rval(i.arg[1])]
                        area['sret'] = take(t.size, max(t.align, 8))
                    self.areas[id(i)] = area
                    args = []
        self.frame = (off + 15) & -16
        if self.dyn and not self.frame:
            self.frame = 16

    def epilogue(self):
        if self.frame:
            self.lget(self.loc['sp0'])
            self.gset(G_SP)

    # ---- structured control flow

    def merge(self, b):
        return sum(1 for p in b.pred if p.id < b.id) >= 2

    def loophead(self, b):
        return any(p.id >= b.id for p in b.pred)

    def tree(self, x):
        ys = []
        c = x.dom
        while c is not None:
            if self.merge(c):
                ys.append(c)
            c = c.dlink
        ys.sort(key=lambda b: -b.id)
        if self.loophead(x):
            self.op(0x03, 0x40)  # loop
            self.ctx.append(('loop', x))
            self.within(x, ys)
            self.ctx.pop()
            self.op(0x0B)
        else:
            self.within(x, ys)

    def within(self, x, ys):
        if ys:
            y = ys[0]
            self.op(0x02, 0x40)  # block
            self.ctx.append(('block', y))
            self.within(x, ys[1:])
            self.ctx.pop()
            self.op(0x0B)
            self.tree(y)
            return
        self.body(x)
        j = x.jmp
        if j.type == D.Jjmp:
            self.branch(x, x.s1)
        elif j.type == D.Jjnz:
            if x.s1 is x.s2:
                self.branch(x, x.s1)
                return
            self.push(j.arg, Kw if rtype(j.arg) != RTmp or self.tcls(j.arg) == Kw else Kl)
            if rtype(j.arg) == RTmp and self.tcls(j.arg) == Kl:
                self.i64c(0)
                self.op(0x52)  # i64.ne
            self.op(0x04, 0x40)  # if
            self.ctx.append(('if', None))
            self.branch(x, x.s1)
            self.op(0x05)  # else
            self.branch(x, x.s2)
            self.ctx.pop()
            self.op(0x0B)
        elif D.isret(j.type):
            self.ret(j)
        elif j.type == D.Jhlt:
            self.op(0x00)
        else:
            raise WasmError('wasm: unexpected jump %s' % D.jtoa[j.type])

    def branch(self, x, y):
        self.phicopies(x, y)
        if y.id <= x.id or self.merge(y):
            want = 'loop' if y.id <= x.id else 'block'
            for depth, (kind, b) in enumerate(reversed(self.ctx)):
                if kind == want and b is y:
                    self.op(0x0C)
                    self.c += uleb(depth)
                    return
            raise WasmError('wasm: no label for a branch (internal error)')
        self.tree(y)

    def phicopies(self, x, y):
        moves = []
        p = y.phi
        while p is not None:
            for a, b in zip(p.arg[:p.narg], p.blk[:p.narg]):
                if b is x:
                    if a != p.to:
                        moves.append((p.to, a, p.cls))
                    break
            p = p.link
        for to, a, k in moves:
            self.push(a, k)
        for to, a, k in reversed(moves):
            self.lset(self.loc[rval(to)])

    def ret(self, j):
        fn = self.fn
        if j.type == D.Jretc:
            self.lget(self.sret)
            self.op(0xA7)
            self.paddr(j.arg)
            self.i32c(D.typ[fn.retty].size)
            self.op(0xFC, 0x0A, 0x00, 0x00)  # memory.copy
        elif j.type != D.Jret0:
            k = (Kw, Kl, Ks, Kd, Kw, Kw, Kw, Kw)[j.type - D.Jretw]
            if self.results:
                self.push(j.arg, k)
        if j.type == D.Jret0 and self.results:  # `return;` in a function declared non-void
            self.zero(self.results[0])
        self.epilogue()
        self.op(0x0F)

    def zero(self, t):
        if t == I32:
            self.i32c(0)
        elif t == I64:
            self.i64c(0)
        elif t == F32:
            self.c += b'\x43' + bytes(4)
        else:
            self.c += b'\x44' + bytes(8)

    # ---- instructions

    def body(self, b):
        args = []
        for i in b.ins[:b.nins]:
            o = i.op
            if D.ispar(o) or o in (D.Onop, D.Odbgloc):
                continue
            if D.isarg(o):
                args.append(i)
                continue
            if o == D.Ocall:
                self.call(i, args)
                args = []
                continue
            self.ins(i)

    def ins(self, i):
        fn = self.fn
        o, k = i.op, i.cls
        a0, a1 = i.arg
        if o in _BIN:
            ops = _BIN[o]
            opc = ops[k]
            if opc is None:
                raise WasmError('wasm: bad class for %s' % D.optab[o].name)
            self.push(a0, k)
            if o in (D.Oshl, D.Osar, D.Oshr):
                self.push(a1, Kw)
                if k == Kl:
                    self.op(0xAD)
            else:
                self.push(a1, k)
            self.op(opc)
        elif o == D.Oneg:
            if k in (Kw, Kl):
                (self.i32c if k == Kw else self.i64c)(0)
                self.push(a0, k)
                self.op(0x6B if k == Kw else 0x7D)
            else:
                self.push(a0, k)
                self.op(0x8C if k == Ks else 0x9A)
        elif o == D.Ocopy:
            self.push(a0, k)
        elif D.Ocmpw <= o <= D.Ocmpl1:
            ak = Kw if o <= D.Ocmpw1 else Kl
            self.push(a0, ak)
            self.push(a1, ak)
            self.op(_ICMP[ak][o - (D.Ocmpw if ak == Kw else D.Ocmpl)])
            if k == Kl:
                self.op(0xAD)
        elif D.Ocmps <= o <= D.Ocmpd1:
            ak = Ks if o <= D.Ocmps1 else Kd
            c = o - (D.Ocmps if ak == Ks else D.Ocmpd)
            if c < 6:
                self.push(a0, ak)
                self.push(a1, ak)
                self.op(_FCMP[ak][c])
            else:  # ordered (co) / unordered (cuo)
                eq, ne = _FCMP[ak][0], _FCMP[ak][5]
                for r in (a0, a1):
                    self.push(r, ak)
                    self.push(r, ak)
                    self.op(eq if c == 6 else ne)
                self.op(0x71 if c == 6 else 0x72)  # i32.and / i32.or
            if k == Kl:
                self.op(0xAD)
        elif D.isload(o):
            self.paddr(a0)
            if o == D.Oload:
                self.mem((0x28, 0x29, 0x2A, 0x2B)[k], (2, 3, 2, 3)[k])
            else:
                w, l, al = _LOAD[o]
                self.mem(w if k == Kw else l, al)
        elif D.isstore(o):
            opc, vk, al = _STORE[o]
            self.paddr(a1)
            self.push(a0, vk)
            self.mem(opc, al)
            return
        elif o in (D.Oextsb, D.Oextub, D.Oextsh, D.Oextuh):
            self.push(a0, Kw)
            if o == D.Oextsb:
                self.op(0xC0)
            elif o == D.Oextsh:
                self.op(0xC1)
            else:
                self.i32c(0xff if o == D.Oextub else 0xffff)
                self.op(0x71)
            if k == Kl:
                self.op(0xAC if o in (D.Oextsb, D.Oextsh) else 0xAD)
        elif o in (D.Oextsw, D.Oextuw):
            self.push(a0, Kw)
            if k == Kl:
                self.op(0xAC if o == D.Oextsw else 0xAD)
        elif o == D.Oexts:
            self.push(a0, Ks)
            self.op(0xBB)
        elif o == D.Otruncd:
            self.push(a0, Kd)
            self.op(0xB6)
        elif o in (D.Ostosi, D.Ostoui, D.Odtosi, D.Odtoui):
            src = Ks if o in (D.Ostosi, D.Ostoui) else Kd
            self.push(a0, src)
            n = {D.Ostosi: 0, D.Ostoui: 1, D.Odtosi: 2, D.Odtoui: 3}[o] + (4 if k == Kl else 0)
            self.op(0xFC, n)  # saturating, like arm64's fcvtzs/fcvtzu
        elif o in (D.Oswtof, D.Ouwtof, D.Osltof, D.Oultof):
            src = Kw if o in (D.Oswtof, D.Ouwtof) else Kl
            self.push(a0, src)
            n = {D.Oswtof: 0, D.Ouwtof: 1, D.Osltof: 2, D.Oultof: 3}[o]
            self.op((0xB2 if k == Ks else 0xB7) + n)
        elif o == D.Ocast:
            src = {Ks: Kw, Kd: Kl, Kw: Ks, Kl: Kd}[k]
            self.push(a0, src)
            self.op({Ks: 0xBE, Kd: 0xBF, Kw: 0xBC, Kl: 0xBD}[k])
        elif D.isalloc(o):
            off = self.slots.get(id(i))
            if off is not None:
                self.lget(self.loc['fp'])
                if off:
                    self.i32c(off)
                    self.op(0x6A)
            else:
                al = 4 << (o - D.Oalloc4)
                self.gget(G_SP)
                self.paddr(a0) if rtype(a0) == RTmp else self.push(a0, Kw)
                self.op(0x6B)
                self.i32c(-max(al, 16))
                self.op(0x71)
                self.ltee(self.loc['x0'])
                self.gset(G_SP)
                self.lget(self.loc['x0'])
            self.op(0xAD)
        elif o == D.Ofma0:  # wasm has no scalar fma: call the runtime's (exact) fma/fmaf
            self.fmaadd = a0
            return
        elif o == D.Ofma1:
            self.push(a0, k)
            self.push(a1, k)
            self.push(self.fmaadd, k)
            name = 'fmaf' if k == Ks else 'fma'
            self.c.append(0x10)
            self.relocs.append((len(self.c), CALL, name, 0))
            self.c += uleb5(0)
            self.calls[name] = 'fff:f' if k == Ks else 'ddd:d'
        elif o == D.Oblit0:
            self.blit = (a0, a1)
            return
        elif o == D.Oblit1:
            src, dst = self.blit
            self.paddr(dst)
            self.paddr(src)
            self.i32c(abs(rsval(a0)))
            self.op(0xFC, 0x0A, 0x00, 0x00)
            return
        elif o in _UNF:
            self.push(a0, k)
            self.op(_UNF[o][k == Kd])
        elif o == D.Ofround:  # half away from zero: trunc(|x| + (0.5 - ulp)) with x's sign
            self.push(a0, k)
            self.ltee(self.loc['f0' if k == Ks else 'd0'])
            if k == Ks:
                self.op(0x8B)
                self.c += b'\x43' + struct.pack('<f', 0.49999997)
                self.op(0x92, 0x8F)
                self.lget(self.loc['f0'])
                self.op(0x98)
            else:
                self.op(0x99)
                self.c += b'\x44' + struct.pack('<d', 0.49999999999999994)
                self.op(0xA0, 0x9D)
                self.lget(self.loc['d0'])
                self.op(0xA6)
        elif o in (D.Ofmin, D.Ofmax):  # C: a NaN operand yields the other one
            x, y = ('f0', 'f1') if k == Ks else ('d0', 'd1')
            x, y = self.loc[x], self.loc[y]
            ne = 0x5C if k == Ks else 0x62
            self.push(a0, k)
            self.lset(x)
            self.push(a1, k)
            self.lset(y)
            for p, q in ((x, y), (y, x)):
                self.lget(q)
                self.lget(p)
                self.lget(p)
                self.lget(p)
                self.op(ne, 0x1B)  # p != p ? q : p
            self.op({(D.Ofmin, Ks): 0x96, (D.Ofmin, Kd): 0xA4,
                     (D.Ofmax, Ks): 0x97, (D.Ofmax, Kd): 0xA5}[o, k])
        elif o == D.Ovastart:
            self.paddr(a0)
            self.lget(self.vaptr)
            self.mem(0x37, 3)
            return
        elif o == D.Ovaarg:
            x = self.loc['x0']
            self.paddr(a0)
            self.ltee(x)
            self.lget(x)
            self.mem(0x29, 3)  # the slot's address
            self.op(0xA7)
            self.ltee(self.loc['x1'])
            self.mem((0x28, 0x29, 0x2A, 0x2B)[k], (2, 3, 2, 3)[k])
            self.setto(i)
            self.lget(x)
            self.lget(self.loc['x1'])
            self.i32c(8)
            self.op(0x6A, 0xAD)
            self.mem(0x37, 3)
            return
        else:
            raise WasmError('wasm: unsupported instruction %s' % D.optab[o].name)
        if i.to != R:
            self.setto(i)
        else:
            self.op(0x1A)  # drop

    def call(self, i, args):
        fn = self.fn
        area = self.areas[id(i)]
        ptypes = []
        if 'sret' in area:
            self.lget(self.loc['fp'])
            self.i32c(area['sret'])
            self.op(0x6A, 0xAD)
            ptypes.append(I64)
        va = False
        nva = 0
        copies = iter(area['copies'])
        for a in args:
            if a.op == D.Oargv:
                va = True
                continue
            if a.op == D.Oarge:
                raise WasmError('wasm: env arguments are not supported')
            if va:  # store into the caller's slot buffer
                self.lget(self.loc['fp'])
                self.mem_store_va(a, area['va'] + 8 * nva)
                nva += 1
                continue
            if a.op == D.Oargc:
                off = next(copies)
                t = D.typ[rval(a.arg[0])]
                self.lget(self.loc['fp'])
                self.i32c(off)
                self.op(0x6A)
                self.paddr(a.arg[1])
                self.i32c(t.size)
                self.op(0xFC, 0x0A, 0x00, 0x00)
                self.lget(self.loc['fp'])
                self.i32c(off)
                self.op(0x6A, 0xAD)
                ptypes.append(I64)
            else:
                k = a.cls if a.op == D.Oarg else Kw
                self.push(a.arg[0], k)
                ptypes.append(VT[k])
        if va:
            self.lget(self.loc['fp'])
            self.i32c(area['va'])
            self.op(0x6A, 0xAD)
            ptypes.append(I64)
        results = [] if (i.to == R or 'sret' in area) else [VT[i.cls]]
        key = sigkey(ptypes, results)
        callee = i.arg[0]
        if rtype(callee) == RCon and fn.con[rval(callee)].type == CAddr and \
                fn.con[rval(callee)].i == 0:
            name = symname(fn.con[rval(callee)].sym)
            self.c.append(0x10)
            self.relocs.append((len(self.c), CALL, name, 0))
            self.c += uleb5(0)
            self.calls[name] = key
        else:
            self.paddr(callee)
            self.c.append(0x11)
            self.relocs.append((len(self.c), TYPE, key, 0))
            self.c += uleb5(0) + b'\x00'
        if 'sret' in area:
            if i.to != R:
                self.lget(self.loc['fp'])
                self.i32c(area['sret'])
                self.op(0x6A, 0xAD)
                self.setto(i)
        elif i.to != R:
            self.setto(i)

    def mem_store_va(self, a, off):
        """fp is on the stack: store vararg a at fp + off."""
        k = a.cls if a.op == D.Oarg else Kw
        self.push(a.arg[0], k)
        self.mem({Kw: 0x36, Kl: 0x37, Ks: 0x38, Kd: 0x39}[k], (2, 3, 2, 3)[k], off)


def emitfn(fn, obj):
    obj.funcs.append(_Gen(fn).run())


class DataBuilder:
    """Collects QBE's data stream (DStart, items, DEnd) into obj.Data entries."""

    def __init__(self, obj):
        self.obj = obj
        self.cur = None

    def __call__(self, d):
        if d.type == D.DStart:
            name = d.name[1:-1] if d.name.startswith('"') else d.name
            self.cur = Data(name, bool(d.lnk.export), max(d.lnk.align, 1), 0, bytearray(), [],
                            bool(d.lnk.common))
            self.zeros = 0
            return
        cur = self.cur
        if d.type == D.DEnd:
            cur.size = len(cur.init) + self.zeros
            cur.init = bytes(cur.init)
            self.obj.datas.append(cur)
            self.cur = None
            return
        if d.type == D.DZ:
            self.zeros += s64(d.num)
            return
        if self.zeros:
            cur.init += bytes(self.zeros)
            self.zeros = 0
        size = {D.DB: 1, D.DH: 2, D.DW: 4, D.DL: 8}[d.type]
        if d.isstr:
            from ...assembler import _string
            cur.init += _string(d.str, 0)[0]
        elif d.isref:
            name = d.ref_name[1:-1] if d.ref_name.startswith('"') else d.ref_name
            if size not in (4, 8):
                raise WasmError('wasm: a %d-byte address in data' % size)
            cur.relocs.append((len(cur.init), PTR64 if size == 8 else PTR32, name,
                               s64(d.ref_off)))
            cur.init += bytes(size)
        else:
            cur.init += (d.num & ((1 << (8 * size)) - 1)).to_bytes(size, 'little')
