"""PTX code generation: QBE IR (after the target-independent passes, before any machine lowering) to PTX text, which
the NVIDIA driver compiles (cuModuleLoadDataEx). No register allocation or instruction selection here: the driver's
PTX compiler does both, as the wasm engine does for cimo.backend.wasm.

- Every SSA temporary is a virtual register of its class (w %r, l %rd, s %f, d %fd); a comparison also leaves its
  predicate in %p of the same number, which a conditional branch on it uses. Phis become copies on the incoming
  edges, through fresh registers (a parallel copy); a conditional branch to a block with phis goes through a label of
  its own that does the copies.
- Memory: registers hold generic addresses, as C pointers are. Where an address is known to point into one space (a
  kernel's pointer parameter: global; a __gpu_shared object: shared; a local array: local), the access uses that
  space, converting with cvta.to, so the driver's compiler need not prove it.
- Kernels (__gpu_kernel, frontend paraminfo['#gpukernels']) are .entry; other functions are .func, called with the PTX
  calling sequence. Structs by value go by address, to a copy the caller makes (as on wasm).
- Calls of __cimo_gpu_asm (and __cimo_gpu_asmout for its outputs) are asm statements: the template, its operands
  replaced by registers. __cimo_gpu_barrier is bar.sync 0.
"""

import struct

from ....backend import defs as D
from ....backend.defs import (R, RTmp, RCon, RInt, RType, Tmp0, Kw, Kl, Ks, Kd, CAddr, rtype, rval, rsval,
                              s32, s64, fbits, dbits, f32)
from ....backend.util import str_
from ....backend import parse as _parse
from ....backend.cfg import fillcfg, filldom


class PtxError(D.BackendError):
    pass


def symname(sym):
    s = str_(sym.id)
    return mangle(s[1:-1] if s.startswith('"') else s)


def mangle(name):
    """A PTX identifier for an IR name: '.' (local statics, `.Ltile.3`) is not allowed; '$' is."""
    return name.replace('.', '$')


REG = {Kw: '%r', Kl: '%rd', Ks: '%f', Kd: '%fd'}
RTYPE = {Kw: '.b32', Kl: '.b64', Ks: '.f32', Kd: '.f64'}
ITY = {Kw: '32', Kl: '64'}
FTY = {Ks: 'f32', Kd: 'f64'}
PARAMTY = {Kw: '.b32', Kl: '.b64', Ks: '.f32', Kd: '.f64'}

# integer comparisons in QBE order: eq ne sge sgt sle slt uge ugt ule ult
_ICMP = (('eq', 's'), ('ne', 's'), ('ge', 's'), ('gt', 's'), ('le', 's'), ('lt', 's'),
         ('ge', 'u'), ('gt', 'u'), ('le', 'u'), ('lt', 'u'))
# float comparisons in QBE order: eq ge gt le lt ne o uo (ne is C's !=: true for a NaN)
_FCMP = ('eq', 'ge', 'gt', 'le', 'lt', 'neu', 'num', 'nan')
_LOADTY = {D.Oloadsb: 's8', D.Oloadub: 'u8', D.Oloadsh: 's16', D.Oloaduh: 'u16', D.Oloadsw: 's32',
           D.Oloaduw: 'u32'}
_STORE = {D.Ostoreb: ('u8', Kw), D.Ostoreh: ('u16', Kw), D.Ostorew: ('u32', Kw), D.Ostorel: ('u64', Kl),
          D.Ostores: ('f32', Ks), D.Ostored: ('f64', Kd)}
_VLOAD = {D.Ovloadsb + n: D.Oloadsb + n for n in range(D.Oload - D.Oloadsb + 1)}
_VSTORE = {D.Ovstoreb + n: D.Ostoreb + n for n in range(D.Ostored - D.Ostoreb + 1)}
_ROUND = {D.Offloor: 'rmi', D.Ofceil: 'rpi', D.Oftrunc: 'rzi', D.Ofrint: 'rni', D.Ofnearbyint: 'rni'}
# gpuid N (cimo/gpu.h): the special register read
GPUID = {0: '%tid.x', 1: '%tid.y', 2: '%tid.z', 4: '%ntid.x', 5: '%ntid.y', 6: '%ntid.z',
         8: '%ctaid.x', 9: '%ctaid.y', 10: '%ctaid.z', 12: '%nctaid.x', 13: '%nctaid.y', 14: '%nctaid.z',
         16: '%laneid', 17: '%warpid', 18: '%smid'}
SPECIAL = ('__cimo_gpu_asm', '__cimo_gpu_asmout', '__cimo_gpu_barrier')

# address spaces of a value (_spaces): N an integer, not an address; G generic (unknown); g global, s shared, l local
N, G = 'N', 'G'


def fltbits(c, k):
    if k == Ks:
        return fbits(f32(c.d)) if c.flt == 2 else c.i & 0xffffffff
    return (dbits(c.s) if c.flt == 1 else c.i) & 0xffffffffffffffff


class Func:
    """One function's PTX: its declaration (for forward references) and its definition."""

    def __init__(self, name, entry, decl, text, calls, features, params):
        self.name, self.entry, self.decl, self.text, self.calls = name, entry, decl, text, calls
        self.features = features
        self.params = params  # 'w', 'l', 's', 'd' or ('struct', size, align), in order


class _Gen:
    def __init__(self, fn, mod):
        self.fn = fn
        self.mod = mod
        self.out = []
        self.nx = {}  # scratch registers per prefix
        self.calls = set()
        self.features = set()
        self.entry = mod.iskernel(fn.name)
        self.name = mangle(fn.name)
        self.preds = {}
        self.amemo = {}  # (block, symbol, base) -> (space, address register): conversions made in the block
        self.curblock = None

    # ---- registers and operands

    def w(self, s):
        self.out.append('\t' + s + ';')

    def label(self, s):
        self.out.append(s + ':')

    def x(self, prefix):
        """A fresh scratch register: %rx, %rdx, %fx, %fdx, %px, %hx."""
        n = self.nx.get(prefix, 0)
        self.nx[prefix] = n + 1
        return '%s%d' % (prefix, n)

    def xk(self, k):
        return self.x(REG[k] + 'x')

    def tcls(self, r):
        return self.fn.tmp[rval(r)].cls

    def reg(self, r):
        """The register of temporary r, in its own class."""
        n = rval(r)
        return '%s%d' % (REG[self.fn.tmp[n].cls], n)

    def val(self, r, k):
        """Operand r as class k: a register or an immediate."""
        t = rtype(r)
        if t == RTmp:
            n = rval(r)
            tk = self.fn.tmp[n].cls
            name = '%s%d' % (REG[tk], n)
            if tk == k:
                return name
            if tk == Kl and k == Kw:
                x = self.xk(Kw)
                self.w('cvt.u32.u64 %s, %s' % (x, name))
                return x
            if tk == Kw and k == Kl:
                x = self.xk(Kl)
                self.w('cvt.u64.u32 %s, %s' % (x, name))
                return x
            raise PtxError('ptx: class mismatch for %%%s' % self.fn.tmp[n].name)
        if t == RCon:
            c = self.fn.con[rval(r)]
            if c.type == CAddr:
                a = self.symaddr(c)
                if k == Kw:
                    x = self.xk(Kw)
                    self.w('cvt.u32.u64 %s, %s' % (x, a))
                    return x
                return a
            if k == Kw:
                return str(s32(c.i))
            if k == Kl:
                return str(s64(c.i))
            if k == Ks:
                return '0f%08X' % fltbits(c, Ks)
            return '0d%016X' % fltbits(c, Kd)
        if t == RInt:
            return str(rsval(r))
        raise PtxError('ptx: unexpected operand')

    def rv(self, r, k):
        """Operand r as class k, in a register (an immediate is moved into one)."""
        v = self.val(r, k)
        if v.startswith('%'):
            return v
        x = self.xk(k)
        self.w('mov%s %s, %s' % (RTYPE[k], x, v))
        return x

    def symaddr(self, c):
        """The generic address of a symbol (plus its offset), in a scratch register."""
        name = symname(c.sym)
        sp = self.mod.space_of(name)
        if sp is None:
            if self.mod.isfunc(name):
                raise PtxError('%s: function pointers are not supported in GPU code (the address of %s)'
                               % (self.fn.name, name))
            raise PtxError('%s: %s is not defined in GPU code' % (self.fn.name, name))
        x = self.xk(Kl)
        self.w('cvta.%s.u64 %s, %s' % ('shared' if sp == 's' else 'global', x, name))
        if c.i:
            self.w('add.s64 %s, %s, %d' % (x, x, s64(c.i)))
        return x

    # ---- address spaces

    def spaces(self):
        """Address space of every temporary of class l (N, G, g, s, l), to a fixed point over the phis."""
        fn = self.fn
        sp = {}

        def of(r):
            t = rtype(r)
            if t == RTmp:
                return sp.get(rval(r))  # None: not known yet
            if t == RCon:
                c = fn.con[rval(r)]
                if c.type == CAddr:
                    return self.mod.space_of(symname(c.sym)) or G
                return N
            return N

        def meet(a, b):
            if a is None:
                return b
            if b is None:
                return a
            if a == N:
                return b
            if b == N:
                return a
            return a if a == b else G

        ptrpar = self.mod.pointer_params(fn.name) if self.entry else None
        changed = True
        while changed:
            changed = False
            for b in fn.rpo[:fn.nblk]:
                p = b.phi
                while p:
                    v = None
                    for a in p.arg[:p.narg]:
                        v = meet(v, of(a))
                    if v is not None and sp.get(rval(p.to)) != v:
                        sp[rval(p.to)] = v
                        changed = True
                    p = p.link
                npar = 0
                for i in b.ins[:b.nins]:
                    if rtype(i.to) != RTmp:
                        if D.ispar(i.op):
                            npar += 1
                        continue
                    o = i.op
                    if D.ispar(o):
                        v = 'g' if ptrpar and npar < len(ptrpar) and ptrpar[npar] else G if i.cls == Kl else N
                        if o == D.Oparc:
                            v = 'l'
                        npar += 1
                    elif o in (D.Oadd, D.Osub, D.Ocopy):
                        v = meet(of(i.arg[0]), of(i.arg[1]) if o != D.Ocopy else N)
                        if o == D.Osub and of(i.arg[1]) not in (N, None):
                            v = N  # a difference of addresses
                        if v is None:
                            continue
                    elif D.isalloc(o):
                        v = 'l'
                    elif i.cls == Kl and (D.isload(o) or o == D.Ocall or o in _VLOAD):
                        v = G
                    else:
                        v = N
                    if sp.get(rval(i.to)) != v:
                        sp[rval(i.to)] = v
                        changed = True
        return sp

    def decompose(self, r):
        """An address as (symbol or None, temporary or None, constant): adds of constants peeled off r."""
        fn = self.fn
        sym, c = None, 0
        for _ in range(8):
            t = rtype(r)
            if t == RCon:
                k = fn.con[rval(r)]
                if k.type == CAddr:
                    return symname(k.sym), None, c + s64(k.i)
                return None, None, c + s64(k.i)
            if t != RTmp:
                break
            d = fn.tmp[rval(r)].def_
            if d is None or d.op != D.Oadd or d.cls != Kl:
                break
            x, y = d.arg
            if rtype(y) == RCon:
                k = fn.con[rval(y)]
            elif rtype(x) == RCon:
                k, x = fn.con[rval(x)], y
            else:
                break
            if k.type == CAddr:
                if sym is not None:
                    break
                sym = symname(k.sym)
            c += s64(k.i)
            r = x
        return sym, r, c

    def address(self, r):
        """(state space suffix, address operand) for a load or store through r: a symbol, or a register in the
        space's own addressing (32-bit for shared), plus a constant offset."""
        sym, base, c = self.decompose(r)
        if not -0x80000000 <= c < 0x80000000:
            sym, base, c = None, r, 0
        if sym is not None and self.mod.space_of(sym) is None:
            sym, base, c = None, r, 0
        off = ('+%d' % c if c > 0 else '%d' % c) if c else ''
        sp = self.mod.space_of(sym) if sym is not None else (self.space.get(rval(base)) if base is not None and
                                                              rtype(base) == RTmp else None)
        if base is None:  # a symbol (and offset)
            return ('.shared' if sp == 's' else '.global'), '[%s%s]' % (sym, off)
        key = (self.curblock, sym, base)
        hit = self.amemo.get(key)
        if hit is not None:
            return hit[0], '[%s%s]' % (hit[1], off)
        if rtype(base) == RTmp and sym is None:
            d = self.fn.tmp[rval(base)].def_
            if d is not None and D.isalloc(d.op) and id(d) in self.slots:  # a local array: the depot itself
                c2 = c + self.slots[id(d)]
                return '.local', '[__local_depot%s]' % (('+%d' % c2) if c2 else '')
        if sym is not None and sp == 's' and (rval(base), sym) in self.sbases:  # converted where base is defined
            res = ('.shared', self.sbases[rval(base), sym])
        elif sym is not None and sp == 's':  # a shared object plus an offset: 32 bits
            x = self.x('%rx')
            self.w('mov.u32 %s, %s' % (x, sym))
            self.w('add.s32 %s, %s, %s' % (x, x, self.val(base, Kw)))
            res = ('.shared', x)
        elif sym is not None:  # a global object plus an offset
            x = self.xk(Kl)
            self.w('mov.u64 %s, %s' % (x, sym))
            self.w('add.s64 %s, %s, %s' % (x, x, self.val(base, Kl)))
            res = ('.global', x)
        elif sp == 's':
            y, x = self.xk(Kl), self.x('%rx')
            self.w('cvta.to.shared.u64 %s, %s' % (y, self.rv(base, Kl)))
            self.w('cvt.u32.u64 %s, %s' % (x, y))
            res = ('.shared', x)
        elif sp in ('g', 'l'):
            x = self.xk(Kl)
            self.w('cvta.to.%s.u64 %s, %s' % ('global' if sp == 'g' else 'local', x, self.rv(base, Kl)))
            res = ('.global' if sp == 'g' else '.local', x)
        else:
            res = ('', self.rv(base, Kl))
        self.amemo[key] = res
        return res[0], '[%s%s]' % (res[1], off)

    # ---- function

    def run(self):
        fn = self.fn
        fillcfg(fn)
        filldom(fn)
        self.space = self.spaces()
        self.sbases = self.sharedbases()
        self.layout_frame()
        start = fn.start
        ret = _parse.retcls.get(fn.name, _parse.K0)
        self.sret = ret == _parse.Kc
        self.retk = None if ret in (_parse.K0, _parse.Kc) else (Kw if ret >= _parse.Ksb else ret)
        if self.entry and (self.retk is not None or self.sret):
            raise PtxError('%s: a GPU kernel must return void' % fn.name)
        if fn.vararg:
            raise PtxError('%s: variadic functions are not supported in GPU code' % fn.name)
        params = []
        pinfo = []
        prologue = []
        if self.sret:
            params.append(('.b64', 'sret', None))
            self.sretreg = '%rdsret'
        np = 0
        for i in start.ins[:start.nins]:
            if not D.ispar(i.op):
                continue
            pname = '%s_param_%d' % (self.name, np)
            np += 1
            if i.op == D.Opar:
                pinfo.append('wlsd'[i.cls])
                params.append((PARAMTY[i.cls], pname, None))
                prologue.append('ld.param%s %s, [%s]' % (PARAMTY[i.cls], self.reg(i.to), pname))
            elif i.op == D.Oparc:
                t = D.typ[rval(i.arg[0])]
                pinfo.append(('struct', t.size, 1 << t.align))
                if self.entry:  # the struct's bytes are the parameter: copied to the frame
                    params.append(('.align %d .b8' % (1 << t.align), '%s[%d]' % (pname, t.size), None))
                    off = self.parcopies[id(i)]
                    prologue.append('add.s64 %s, %%rdfp, %d' % (self.reg(i.to), off))
                    for n in range(t.size):
                        prologue.append('ld.param.u8 %%rsbyte, [%s+%d]' % (pname, n))
                        prologue.append('st.local.u8 [__local_depot+%d], %%rsbyte' % (off + n))
                    self.needbyte = True
                else:  # by address, to the caller's copy
                    params.append(('.b64', pname, None))
                    prologue.append('ld.param.b64 %s, [%s]' % (self.reg(i.to), pname))
            else:
                raise PtxError('%s: env parameters are not supported in GPU code' % fn.name)
        self.body_all()
        # registers: temporaries by class, then the scratch ones
        decl = ['.reg .pred %p<{}>;'.format(fn.ntmp)]
        if self.sbases:
            decl.append('.reg .b32 %%sa<%d>;' % len(self.sbases))
        for k in (Kw, Kl, Ks, Kd):
            decl.append('.reg %s %s<%d>;' % (RTYPE[k], REG[k], fn.ntmp))
        for prefix, n in sorted(self.nx.items()):
            ty = {'%rx': '.b32', '%rdx': '.b64', '%fx': '.f32', '%fdx': '.f64', '%px': '.pred', '%hx': '.b16'}[prefix]
            decl.append('.reg %s %s<%d>;' % (ty, prefix, n))
        if self.frame:
            decl.append('.local .align 16 .b8 __local_depot[%d];' % self.frame)
            decl.append('.reg .b64 %rdfp;')
        if getattr(self, 'needbyte', False):
            decl.append('.reg .b16 %rsbyte;')
        if self.sret:
            decl.append('.reg .b64 %rdsret;')
        head = []
        if self.frame:
            head.append('\tcvta.local.u64 %rdfp, __local_depot;')
        if self.sret:
            head.append('\tld.param.b64 %%rdsret, [%s_param_sret];' % self.name)
        head += ['\t' + s + ';' for s in prologue]
        plist = ',\n'.join('\t.param %s %s' % (ty, (self.name + '_param_sret') if n == 'sret' else n)
                           for ty, n, _ in params)
        if self.entry:
            sig = '.visible .entry %s(\n%s\n)' % (self.name, plist) if params else '.visible .entry %s()' % self.name
        else:
            r = '(.param %s %s_retval) ' % (PARAMTY[self.retk], self.name) if self.retk is not None else ''
            sig = '.func %s%s(\n%s\n)' % (r, self.name, plist) if params else '.func %s%s()' % (r, self.name)
        text = sig + '\n{\n' + '\n'.join('\t' + d for d in decl) + '\n\n' + '\n'.join(head + self.out) + '\n}\n'
        return Func(self.name, self.entry, sig + ';', text, self.calls, self.features, pinfo)

    # ---- frame: static allocas, copies of struct arguments and results

    def layout_frame(self):
        fn = self.fn
        off = 0
        self.slots = {}
        self.areas = {}
        self.parcopies = {}

        def take(size, align):
            nonlocal off
            off = (off + align - 1) & -align
            o = off
            off += size
            return o

        for b in fn.rpo[:fn.nblk]:
            args = []
            for i in b.ins[:b.nins]:
                if i.op in (D.Ogetsp, D.Osetsp):
                    raise PtxError('%s: variable-length arrays and alloca are not supported in GPU code' % fn.name)
                if D.isalloc(i.op):
                    a = 4 << (i.op - D.Oalloc4)
                    if b is fn.start and rtype(i.arg[0]) == RCon and fn.con[rval(i.arg[0])].type != CAddr:
                        self.slots[id(i)] = take(max(fn.con[rval(i.arg[0])].i, 0), max(a, 4))
                    else:
                        raise PtxError('%s: variable-length arrays and alloca are not supported in GPU code'
                                       % fn.name)
                elif i.op == D.Oparc and self.entry:
                    t = D.typ[rval(i.arg[0])]
                    self.parcopies[id(i)] = take(t.size, max(1 << t.align, 8))
                elif D.isarg(i.op):
                    args.append(i)
                elif i.op == D.Ocall:
                    area = {'copies': []}
                    for a in args:
                        if a.op == D.Oargc:
                            t = D.typ[rval(a.arg[0])]
                            area['copies'].append(take(t.size, max(1 << t.align, 8)))
                    if rtype(i.arg[1]) == RType:
                        t = D.typ[rval(i.arg[1])]
                        area['sret'] = take(t.size, max(1 << t.align, 8))
                    self.areas[id(i)] = area
                    args = []
        self.frame = (off + 15) & -16

    # ---- control flow

    def blabel(self, b):
        return '$L__%s_%d' % (self.name, b.id)

    def body_all(self):
        fn = self.fn
        blocks = fn.rpo[:fn.nblk]
        self.edge = 0
        self.stubs = []
        for n, b in enumerate(blocks):
            nxt = blocks[n + 1] if n + 1 < len(blocks) else None
            self.label(self.blabel(b))
            self.curblock = b
            self.body(b)
            j = b.jmp
            if j.type == D.Jjmp:
                self.phicopies(b, b.s1)
                if b.s1 is not nxt:
                    self.w('bra.uni %s' % self.blabel(b.s1))
            elif j.type == D.Jjnz:
                if b.s1 is b.s2:
                    self.phicopies(b, b.s1)
                    if b.s1 is not nxt:
                        self.w('bra.uni %s' % self.blabel(b.s1))
                    continue
                p = self.pred(j.arg)
                t1, t2 = self.target(b, b.s1), self.target(b, b.s2)
                if b.s1 is nxt and t1 == self.blabel(b.s1):
                    self.w('@!%s bra %s' % (p, t2))
                else:
                    self.w('@%s bra %s' % (p, t1))
                    if not (b.s2 is nxt and t2 == self.blabel(b.s2)):
                        self.w('bra.uni %s' % t2)
            elif D.isret(j.type):
                self.ret(j)
            elif j.type == D.Jhlt:
                self.w('trap')
            else:
                raise PtxError('ptx: unexpected jump %s' % D.jtoa[j.type])
        for e, x, y in self.stubs:  # the copies of an edge with phis, out of line
            self.label(e)
            self.phicopies(x, y)
            self.w('bra.uni %s' % self.blabel(y))

    def target(self, x, y):
        """The label a branch from x to y goes to: y's, or a stub doing y's phi copies first."""
        if not self.hasphi(y, x):
            return self.blabel(y)
        self.edge += 1
        e = '$L__%s_e%d' % (self.name, self.edge)
        self.stubs.append((e, x, y))
        return e

    def pred(self, r):
        """A predicate register for `r != 0`."""
        if rtype(r) == RTmp and rval(r) in self.preds:
            return self.preds[rval(r)]
        p = self.x('%px')
        if rtype(r) == RTmp:
            k = self.tcls(r)
            self.w('setp.ne.%s %s, %s, 0' % ('s32' if k == Kw else 's64', p, self.reg(r)))
        else:
            self.w('setp.ne.s32 %s, %s, 0' % (p, self.val(r, Kw)))
        return p

    def hasphi(self, y, x):
        p = y.phi
        while p:
            for a, b in zip(p.arg[:p.narg], p.blk[:p.narg]):
                if b is x and a != p.to:
                    return True
            p = p.link
        return False

    def phicopies(self, x, y):
        moves = []
        p = y.phi
        while p:
            for a, b in zip(p.arg[:p.narg], p.blk[:p.narg]):
                if b is x:
                    if a != p.to:
                        moves.append((p.to, a, p.cls))
                    break
            p = p.link
        if len(moves) == 1:
            to, a, k = moves[0]
            self.w('mov%s %s, %s' % (RTYPE[k], self.reg(to), self.val(a, k)))
            return
        tmps = []
        for to, a, k in moves:
            t = self.xk(k)
            self.w('mov%s %s, %s' % (RTYPE[k], t, self.val(a, k)))
            tmps.append(t)
        for (to, a, k), t in zip(moves, tmps):
            self.w('mov%s %s, %s' % (RTYPE[k], self.reg(to), t))

    def ret(self, j):
        if j.type == D.Jretc:
            self.copy(self.sretreg, self.rv(j.arg, Kl), D.typ[self.fn.retty].size, '', '')
        elif j.type != D.Jret0 and self.retk is not None:
            k = (Kw, Kl, Ks, Kd, Kw, Kw, Kw, Kw)[j.type - D.Jretw]
            self.w('st.param%s [%s_retval], %s' % (PARAMTY[self.retk], self.name, self.rv(j.arg, self.retk)))
        self.w('ret')

    def copy(self, dst, src, n, dsp, ssp):
        """n bytes from address register src to dst (generic unless the spaces are given), forward."""
        self.needbyte = True
        for o in range(n):
            self.w('ld%s.u8 %%rsbyte, [%s+%d]' % (ssp, src, o))
            self.w('st%s.u8 [%s+%d], %%rsbyte' % (dsp, dst, o))

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
            else:
                self.ins(i)
            if rtype(i.to) == RTmp and rval(i.to) in self.sdefs:
                for sym in self.sdefs[rval(i.to)]:  # its shared address, 32 bits, which is what stays live
                    x = self.sbases[rval(i.to), sym]
                    self.w('mov.u32 %s, %s' % (x, sym))
                    self.w('add.s32 %s, %s, %s' % (x, x, self.val(i.to, Kw)))

    def sharedbases(self):
        """{(temporary, shared symbol): register} for the offsets that index shared objects (symbol + offset +
        constant): the 32-bit address is made right after the offset, so a loop that uses it (the offset hoisted out
        of it) keeps one 32-bit register live, not a 64-bit one."""
        fn = self.fn
        out, self.sdefs = {}, {}
        for b in fn.rpo[:fn.nblk]:
            for i in b.ins[:b.nins]:
                a = i.arg[0] if D.isload(i.op) or i.op in _VLOAD else i.arg[1] if D.isstore(i.op) or i.op in _VSTORE \
                    else None
                if a is None:
                    continue
                sym, base, c = self.decompose(a)
                if sym is None or base is None or rtype(base) != RTmp or self.mod.space_of(sym) != 's' \
                        or fn.tmp[rval(base)].def_ is None or D.ispar(fn.tmp[rval(base)].def_.op):
                    continue
                key = (rval(base), sym)
                if key not in out:
                    out[key] = '%%sa%d' % len(out)
                    self.sdefs.setdefault(rval(base), []).append(sym)
        return out

    def setp(self, i, cmp):
        """A comparison: its predicate, kept for branches, and its 0 or 1."""
        n = rval(i.to)
        p = '%%p%d' % n
        self.w(cmp.replace('@P', p))
        self.preds[n] = p
        self.w('selp.u%s %s, 1, 0, %s' % (ITY[i.cls], self.reg(i.to), p))

    def ins(self, i):
        fn = self.fn
        o, k = i.op, i.cls
        a0, a1 = i.arg
        d = self.reg(i.to) if rtype(i.to) == RTmp else None
        if o in (D.Oadd, D.Osub, D.Omul, D.Odiv):
            name = {D.Oadd: 'add', D.Osub: 'sub', D.Omul: 'mul', D.Odiv: 'div'}[o]
            if k in (Kw, Kl):
                ty = ('lo.s' if o == D.Omul else 's') + ITY[k]
            else:
                ty = 'rn.' + FTY[k]
            self.w('%s.%s %s, %s, %s' % (name, ty, d, self.rv(a0, k), self.val(a1, k)))
        elif o in (D.Orem, D.Oudiv, D.Ourem):
            name = 'div' if o == D.Oudiv else 'rem'
            self.w('%s.%s%s %s, %s, %s' % (name, 'u' if o != D.Orem else 's', ITY[k], d, self.rv(a0, k),
                                           self.val(a1, k)))
        elif o in (D.Oand, D.Oor, D.Oxor):
            name = {D.Oand: 'and', D.Oor: 'or', D.Oxor: 'xor'}[o]
            self.w('%s.b%s %s, %s, %s' % (name, ITY[k], d, self.rv(a0, k), self.val(a1, k)))
        elif o in (D.Oshl, D.Osar, D.Oshr):
            name = {D.Oshl: 'shl.b', D.Osar: 'shr.s', D.Oshr: 'shr.u'}[o]
            mask = 31 if k == Kw else 63
            if rtype(a1) in (RCon, RInt):
                sh = str(int(self.val(a1, Kw)) & mask)
            else:
                sh = self.xk(Kw)
                self.w('and.b32 %s, %s, %d' % (sh, self.val(a1, Kw), mask))
            self.w('%s%s %s, %s, %s' % (name, ITY[k], d, self.rv(a0, k), sh))
        elif o == D.Oneg:
            self.w('neg.%s %s, %s' % ('s' + ITY[k] if k in (Kw, Kl) else FTY[k], d, self.rv(a0, k)))
        elif o == D.Ocopy:
            self.w('mov%s %s, %s' % (RTYPE[k], d, self.val(a0, k)))
        elif D.Ocmpw <= o <= D.Ocmpl1:
            ak = Kw if o <= D.Ocmpw1 else Kl
            c, s = _ICMP[o - (D.Ocmpw if ak == Kw else D.Ocmpl)]
            self.setp(i, 'setp.%s.%s%s @P, %s, %s' % (c, s, ITY[ak], self.rv(a0, ak), self.val(a1, ak)))
        elif D.Ocmps <= o <= D.Ocmpd1:
            ak = Ks if o <= D.Ocmps1 else Kd
            c = _FCMP[o - (D.Ocmps if ak == Ks else D.Ocmpd)]
            self.setp(i, 'setp.%s.%s @P, %s, %s' % (c, FTY[ak], self.rv(a0, ak), self.val(a1, ak)))
        elif D.isload(o) or o in _VLOAD:
            vol = '.volatile' if o in _VLOAD else ''
            o = _VLOAD.get(o, o)
            sp, addr = self.address(a0)
            ty = _LOADTY.get(o) or {Kw: 'u32', Kl: 'u64', Ks: 'f32', Kd: 'f64'}[k]
            if vol and sp == '.local':
                vol = ''
            self.w('ld%s%s.%s %s, %s' % (vol, sp, ty, d, addr))
        elif D.isstore(o) or o in _VSTORE:
            vol = '.volatile' if o in _VSTORE else ''
            o = _VSTORE.get(o, o)
            ty, vk = _STORE[o]
            v = self.rv(a0, vk)
            sp, addr = self.address(a1)
            if vol and sp == '.local':
                vol = ''
            self.w('st%s%s.%s %s, %s' % (vol, sp, ty, addr, v))
        elif o in (D.Oextsb, D.Oextub, D.Oextsh, D.Oextuh):
            a = self.rv(a0, Kw)
            t = d if k == Kw else self.xk(Kw)
            if o == D.Oextsb:
                self.w('bfe.s32 %s, %s, 0, 8' % (t, a))
            elif o == D.Oextsh:
                self.w('bfe.s32 %s, %s, 0, 16' % (t, a))
            else:
                self.w('and.b32 %s, %s, %d' % (t, a, 0xff if o == D.Oextub else 0xffff))
            if k == Kl:
                self.w('cvt.%s64.%s32 %s, %s' % ((('s', 's') if o in (D.Oextsb, D.Oextsh) else ('u', 'u')) + (d, t)))
        elif o in (D.Oextsw, D.Oextuw):
            a = self.rv(a0, Kw)
            if k == Kl:
                self.w('cvt.%s64.%s32 %s, %s' % ((('s', 's') if o == D.Oextsw else ('u', 'u')) + (d, a)))
            else:
                self.w('mov.b32 %s, %s' % (d, a))
        elif o == D.Oexts:
            self.w('cvt.f64.f32 %s, %s' % (d, self.rv(a0, Ks)))
        elif o == D.Otruncd:
            self.w('cvt.rn.f32.f64 %s, %s' % (d, self.rv(a0, Kd)))
        elif o in (D.Ostosi, D.Ostoui, D.Odtosi, D.Odtoui):
            src = Ks if o in (D.Ostosi, D.Ostoui) else Kd
            sg = 's' if o in (D.Ostosi, D.Odtosi) else 'u'
            self.w('cvt.rzi.%s%s.%s %s, %s' % (sg, ITY[k], FTY[src], d, self.rv(a0, src)))
        elif o in (D.Oswtof, D.Ouwtof, D.Osltof, D.Oultof):
            src = Kw if o in (D.Oswtof, D.Ouwtof) else Kl
            sg = 's' if o in (D.Oswtof, D.Osltof) else 'u'
            self.w('cvt.rn.%s.%s%s %s, %s' % (FTY[k], sg, ITY[src], d, self.rv(a0, src)))
        elif o == D.Ocast:
            src = {Ks: Kw, Kd: Kl, Kw: Ks, Kl: Kd}[k]
            self.w('mov.b%s %s, %s' % ('32' if k in (Kw, Ks) else '64', d, self.rv(a0, src)))
        elif o in (D.Ordfpcr, D.Ordfpsr):  # one rounding mode, no exception flags
            self.w('mov.b32 %s, 0' % d)
        elif o in (D.Owrfpcr, D.Owrfpsr):
            return
        elif D.isalloc(o):
            off = self.slots[id(i)]
            self.w('add.s64 %s, %%rdfp, %d' % (d, off))
        elif o == D.Ofma0:
            self.fmaadd = a0
        elif o == D.Ofma1:
            self.w('fma.rn.%s %s, %s, %s, %s' % (FTY[k], d, self.rv(a0, k), self.val(a1, k),
                                                 self.val(self.fmaadd, k)))
        elif o == D.Osel0:
            self.selc = a0
        elif o == D.Osel1:
            p = self.pred(self.selc)
            self.w('selp%s %s, %s, %s, %s' % (RTYPE[k], d, self.val(a0, k), self.val(a1, k), p))
        elif o == D.Oblit0:
            self.blit = (a0, a1)
        elif o == D.Oblit1:
            src, dst = self.blit
            self.copy(self.rv(dst, Kl), self.rv(src, Kl), abs(rsval(a0)), '', '')
        elif o == D.Ofsqrt:
            self.w('sqrt.rn.%s %s, %s' % (FTY[k], d, self.rv(a0, k)))
        elif o == D.Ofabs:
            self.w('abs.%s %s, %s' % (FTY[k], d, self.rv(a0, k)))
        elif o in (D.Ofmin, D.Ofmax):  # a NaN operand yields the other one, as C's
            self.w('%s.%s %s, %s, %s' % ('min' if o == D.Ofmin else 'max', FTY[k], d, self.rv(a0, k),
                                         self.val(a1, k)))
        elif o in _ROUND:
            self.w('cvt.%s.%s.%s %s, %s' % (_ROUND[o], FTY[k], FTY[k], d, self.rv(a0, k)))
        elif o == D.Ofround:  # half away from zero: trunc(|x| + (0.5 - ulp)) with x's sign
            a = self.rv(a0, k)
            t = self.xk(k)
            half = '0f3EFFFFFF' if k == Ks else '0d3FDFFFFFFFFFFFFF'
            self.w('abs.%s %s, %s' % (FTY[k], t, a))
            self.w('add.rn.%s %s, %s, %s' % (FTY[k], t, t, half))
            self.w('cvt.rzi.%s.%s %s, %s' % (FTY[k], FTY[k], t, t))
            self.w('copysign.%s %s, %s, %s' % (FTY[k], d, a, t))
        elif o in (D.Obclz, D.Obpopcnt):
            name = 'clz' if o == D.Obclz else 'popc'
            if k == Kw:
                self.w('%s.b32 %s, %s' % (name, d, self.rv(a0, Kw)))
            else:
                t = self.xk(Kw)
                self.w('%s.b64 %s, %s' % (name, t, self.rv(a0, Kl)))
                self.w('cvt.u64.u32 %s, %s' % (d, t))
        elif o == D.Obctz:
            t = self.xk(k)
            self.w('brev.b%s %s, %s' % (ITY[k], t, self.rv(a0, k)))
            if k == Kw:
                self.w('clz.b32 %s, %s' % (d, t))
            else:
                u = self.xk(Kw)
                self.w('clz.b64 %s, %s' % (u, t))
                self.w('cvt.u64.u32 %s, %s' % (d, u))
        elif o == D.Obswap:
            if k == Kw:
                self.w('prmt.b32 %s, %s, 0, 0x0123' % (d, self.rv(a0, Kw)))
            else:
                lo, hi, lo2, hi2 = (self.xk(Kw) for _ in range(4))
                self.w('mov.b64 {%s, %s}, %s' % (lo, hi, self.rv(a0, Kl)))
                self.w('prmt.b32 %s, %s, 0, 0x0123' % (lo2, hi))
                self.w('prmt.b32 %s, %s, 0, 0x0123' % (hi2, lo))
                self.w('mov.b64 %s, {%s, %s}' % (d, lo2, hi2))
        elif o == D.Obswap16:
            self.w('prmt.b32 %s, %s, 0, 0x4401' % (d, self.rv(a0, Kw)))
        elif o in (D.Osmulh, D.Oumulh):
            self.w('mul.hi.%s64 %s, %s, %s' % ('s' if o == D.Osmulh else 'u', d, self.rv(a0, Kl), self.val(a1, Kl)))
        elif o == D.Oh2f:
            h = self.x('%hx')
            self.w('cvt.u16.u32 %s, %s' % (h, self.rv(a0, Kw)))
            self.w('cvt.f32.f16 %s, %s' % (d, h))
        elif o == D.Of2h:
            h = self.x('%hx')
            self.w('cvt.rn.f16.f32 %s, %s' % (h, self.rv(a0, Ks)))
            self.w('cvt.u32.u16 %s, %s' % (d, h))
        elif o == D.Ogpuid:
            n = rsval(a0) if rtype(a0) == RInt else self.fn.con[rval(a0)].i
            if n not in GPUID:
                raise PtxError('%s: no GPU query %d' % (fn.name, n))
            self.w('mov.u32 %s, %s' % (d, GPUID[n]))
        elif o in (D.Ovastart, D.Ovaarg):
            raise PtxError('%s: variadic functions are not supported in GPU code' % fn.name)
        else:
            raise PtxError('ptx: unsupported instruction %s' % D.optab[o].name)

    # ---- calls

    def callee(self, i):
        c = i.arg[0]
        if rtype(c) == RCon and self.fn.con[rval(c)].type == CAddr and self.fn.con[rval(c)].i == 0:
            s = str_(self.fn.con[rval(c)].sym.id)
            return s[1:-1] if s.startswith('"') else s
        raise PtxError('%s: calls through function pointers are not supported in GPU code' % self.fn.name)

    def call(self, i, args):
        name = self.callee(i)
        if name == '__cimo_gpu_barrier':
            self.w('bar.sync 0')
            return
        if name == '__cimo_gpu_asm':
            self.asm(i, args)
            return
        if name == '__cimo_gpu_asmout':
            n = rsval(args[1].arg[0]) if rtype(args[1].arg[0]) == RInt else self.fn.con[rval(args[1].arg[0])].i
            reg, k = self.asmouts[n]
            if rtype(i.to) == RTmp:
                d = self.reg(i.to)
                if k == 'h':
                    self.w('cvt.u32.u16 %s, %s' % (d, reg))
                else:
                    self.w('mov%s %s, %s' % (RTYPE[i.cls], d, reg))
            return
        self.calls.add(name)
        area = self.areas[id(i)]
        lines = ['{']
        plist = []
        copies = iter(area['copies'])
        n = 0
        if 'sret' in area:
            lines.append('.param .b64 param%d' % n)
            x = self.xk(Kl)
            self.w('add.s64 %s, %%rdfp, %d' % (x, area['sret']))
            lines.append('st.param.b64 [param%d], %s' % (n, x))
            plist.append('param%d' % n)
            n += 1
        for a in args:
            if a.op == D.Oargv:
                raise PtxError('%s: variadic calls are not supported in GPU code (%s)' % (self.fn.name, name))
            if a.op == D.Oarge:
                raise PtxError('%s: env arguments are not supported in GPU code' % self.fn.name)
            if a.op == D.Oargc:
                off = next(copies)
                t = D.typ[rval(a.arg[0])]
                x = self.xk(Kl)
                self.w('add.s64 %s, %%rdfp, %d' % (x, off))
                self.copy(x, self.rv(a.arg[1], Kl), t.size, '', '')
                lines.append('.param .b64 param%d' % n)
                lines.append('st.param.b64 [param%d], %s' % (n, x))
            else:
                k = a.cls
                lines.append('.param %s param%d' % (PARAMTY[k], n))
                lines.append('st.param%s [param%d], %s' % (PARAMTY[k], n, self.rv(a.arg[0], k)))
            plist.append('param%d' % n)
            n += 1
        hasret = rtype(i.to) == RTmp and 'sret' not in area
        if hasret:
            lines.append('.param %s retval0' % PARAMTY[i.cls])
            lines.append('call.uni (retval0), %s, (%s)' % (mangle(name), ', '.join(plist)))
            lines.append('ld.param%s %s, [retval0]' % (PARAMTY[i.cls], self.reg(i.to)))
        else:
            lines.append('call.uni %s, (%s)' % (mangle(name), ', '.join(plist)))
            if rtype(i.to) == RTmp:
                self.w('add.s64 %s, %%rdfp, %d' % (self.reg(i.to), area['sret']))
        self.out.append('\t{')
        for l in lines[1:]:
            self.out.append('\t' + l + ';')
        self.out.append('\t}')
        self.mod.calltypes.setdefault(name, (i.cls if hasret else None, [a.cls if a.op == D.Oarg else Kl
                                                                          for a in args]))

    def asm(self, i, args):
        n = rsval(args[0].arg[0]) if rtype(args[0].arg[0]) == RInt else self.fn.con[rval(args[0].arg[0])].i
        template, specs = self.mod.asms[n]
        vals = iter(args[1:])
        regs = []
        self.asmouts = {}
        for idx, (l, role, extra) in enumerate(specs):
            if role in ('o', 'b'):
                r = self.asmreg(l)
                if role == 'b':
                    self.asmmove(r, l, next(vals).arg[0])
                self.asmouts[idx] = (r, l)
                regs.append(r)
            elif role == 't':
                r = regs[extra]
                self.asmmove(r, l, next(vals).arg[0])
                regs.append(r)
            elif l == 'n':
                regs.append(str(s64(extra)))
            else:
                r = self.asmreg(l)
                self.asmmove(r, l, next(vals).arg[0])
                regs.append(r)
        self.out.append('\t// asm')
        self.out.append('\t' + _substitute(template, regs, self.fn.name))
        self.features.add(template)

    def asmreg(self, l):
        return self.x({'h': '%hx', 'r': '%rx', 'l': '%rdx', 'f': '%fx', 'd': '%fdx'}[l])

    def asmmove(self, reg, l, r):
        if l == 'h':
            self.w('cvt.u16.u32 %s, %s' % (reg, self.rv(r, Kw)))
        else:
            k = {'r': Kw, 'l': Kl, 'f': Ks, 'd': Kd}[l]
            self.w('mov%s %s, %s' % (RTYPE[k], reg, self.val(r, k)))


def _substitute(template, regs, fname):
    """gcc's operand syntax: %N is operand N, %% a percent sign; %[name] is not supported (the operands' names are
    gone)."""
    out = []
    i = 0
    while i < len(template):
        c = template[i]
        if c == '%' and i + 1 < len(template):
            nx = template[i + 1]
            if nx == '%':
                out.append('%')
                i += 2
                continue
            if nx.isdigit():
                j = i + 1
                while j < len(template) and template[j].isdigit():
                    j += 1
                n = int(template[i + 1:j])
                if n >= len(regs):
                    raise PtxError('%s: asm operand %%%d out of range (%d operands)' % (fname, n, len(regs)))
                out.append(regs[n])
                i = j
                continue
            if nx == '=':  # a number unique to this asm instance, as gcc's %=
                out.append(str(id(regs) & 0xffffff))
                i += 2
                continue
        out.append(c)
        i += 1
    return ''.join(out)


def emitfn(fn, mod):
    mod.funcs.append(_Gen(fn, mod).run())


class DataBuilder:
    """Collects QBE's data stream into the module's variables: name -> (align, bytes or size, relocs)."""

    def __init__(self, mod):
        self.mod = mod
        self.cur = None

    def __call__(self, d):
        if d.type == D.DStart:
            name = d.name[1:-1] if d.name.startswith('"') else d.name
            self.cur = [mangle(name), max(d.lnk.align, 1), bytearray(), [], 0]
            return
        cur = self.cur
        if d.type == D.DEnd:
            if cur[4]:
                cur[2] += bytes(cur[4])
            self.mod.datas.append((cur[0], cur[1], bytes(cur[2]), cur[3]))
            self.cur = None
            return
        if d.type == D.DZ:
            cur[4] += s64(d.num)
            return
        if cur[4]:
            cur[2] += bytes(cur[4])
            cur[4] = 0
        size = {D.DB: 1, D.DH: 2, D.DW: 4, D.DL: 8}[d.type]
        if d.isstr:
            from ....assembler import _string
            cur[2] += _string(d.str, 0)[0]
        elif d.isref:
            name = d.ref_name[1:-1] if d.ref_name.startswith('"') else d.ref_name
            if size != 8:
                raise PtxError('ptx: a %d-byte address in data' % size)
            cur[3].append((len(cur[2]), mangle(name), s64(d.ref_off)))
            cur[2] += bytes(8)
        else:
            cur[2] += (d.num & ((1 << (8 * size)) - 1)).to_bytes(size, 'little')


class Module:
    """One PTX module: the backend's output object (the `outf` of cimo.backend.main)."""

    def __init__(self, info):
        info = info or {}
        self.kernels = set(mangle(n) for n in info.get('#gpukernels', ()))
        self.shared = set(mangle(n) for n in info.get('#gpushared', ()))
        self.asms = info.get('#gpuasm', [])
        sigs = info.get('#signatures', {})
        self.ptrpars = {mangle(n): [t.endswith('*') for t in s[1]] for n, s in sigs.items()}
        self.funcs = []
        self.datas = []
        self.calltypes = {}
        self.databuilder = DataBuilder(self)
        self._defined = None

    def iskernel(self, name):
        return mangle(name) in self.kernels

    def pointer_params(self, name):
        return self.ptrpars.get(mangle(name))

    def defined(self):
        if self._defined is None or len(self._defined) != len(self.datas):
            self._defined = {n for n, *_ in self.datas}
        return self._defined

    def space_of(self, name):
        if name in self.shared:
            return 's'
        if name in self.defined():
            return 'g'
        return None

    def isfunc(self, name):
        return any(f.name == name for f in self.funcs)

    def text(self, target, version):
        """The module's PTX: functions reachable from the kernels, the data they use."""
        byname = {f.name: f for f in self.funcs}
        kernels = [f for f in self.funcs if f.entry]
        if not kernels:
            raise PtxError('no __gpu_kernel function in the source')
        # reachable functions, rejecting recursion and calls of what is not there
        order, state = [], {}

        def visit(f, path):
            st = state.get(f.name)
            if st == 1:
                raise PtxError('recursion is not supported in GPU code (%s)' % ' -> '.join(path + [f.name]))
            if st == 2:
                return
            state[f.name] = 1
            for c in sorted(f.calls):
                g = byname.get(mangle(c))
                if g is None:
                    raise PtxError('%s is not available in GPU code (called from %s)' % (c, f.name))
                if g.entry:
                    raise PtxError('a GPU kernel cannot be called from GPU code (%s calls %s)' % (f.name, c))
                visit(g, path + [f.name])
            state[f.name] = 2
            order.append(f)

        for k in kernels:
            visit(k, [])
        used = set()
        for f in order:
            used.update(_symbols(f.text))
        out = ['//', '// Generated by cimo', '//', '', '.version %s' % version, '.target %s' % target,
               '.address_size 64', '']
        for f in order:
            if not f.entry:
                out.append(f.decl)
        out.append('')
        for name in sorted(self.shared):
            if name in used and name not in self.defined():  # extern: the dynamic shared memory
                out.append('.extern .shared .align 16 .b8 %s[];' % name)
        datas = [d for d in self.datas if d[0] in used]
        more = True
        while more:  # data that data refers to
            more = False
            names = {d[0] for d in datas}
            for d in self.datas:
                if d[0] not in names and any(r[1] == d[0] for x in datas for r in x[3]):
                    datas.append(d)
                    more = True
        for name, align, init, relocs in datas:
            out.append(_variable(name, align, init, relocs, name in self.shared))
        out.append('')
        for f in order:
            out.append(f.text)
        return '\n'.join(out)

    def features(self):
        s = set()
        for f in self.funcs:
            s |= f.features
        return s


def _symbols(text):
    import re
    return set(re.findall(r'[A-Za-z_$][A-Za-z0-9_$]*', text))


def _variable(name, align, init, relocs, shared):
    if shared:
        if any(init):
            raise PtxError('__gpu_shared object %s cannot have an initializer' % name)
        return '.shared .align %d .b8 %s[%d];' % (max(align, 1), name, max(len(init), 1))
    if relocs:
        if len(init) % 8 or any(o % 8 for o, _, _ in relocs):
            raise PtxError('ptx: addresses in data must be 8-byte aligned (%s)' % name)
        items = []
        rel = {o: (s, a) for o, s, a in relocs}
        for o in range(0, len(init), 8):
            if o in rel:
                s, a = rel[o]
                items.append('generic(%s)+%d' % (s, a) if a else 'generic(%s)' % s)
            else:
                items.append(str(int.from_bytes(init[o:o + 8], 'little')))
        return '.global .align %d .u64 %s[%d] = {%s};' % (max(align, 8), name, len(init) // 8, ', '.join(items))
    if not any(init):
        return '.global .align %d .b8 %s[%d];' % (max(align, 1), name, max(len(init), 1))
    return '.global .align %d .b8 %s[%d] = {%s};' % (max(align, 1), name, len(init), ', '.join(str(b) for b in init))
