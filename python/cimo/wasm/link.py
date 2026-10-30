"""Linking wasm objects (cimo.backend.wasm.obj) into one Emscripten side module.

Like the native link (cimo.linker): the module and the runtime members it needs, transitively,
merged into one module that shares Pyodide's linear memory, table and C stack. The module
imports `env.memory`, `env.__indirect_function_table`, `env.__stack_pointer`,
`env.__memory_base` and `env.__table_base` (supplied by Emscripten's dynamic loader, which
also allocates its data and table slots from the `dylink.0` section) and the `__cimo_js_*`
functions of cimo/wasm/glue.c. Data addresses inside data are filled in at load time by the
module's `__wasm_apply_data_relocs`, which the loader calls.

A direct call whose site type differs from the callee's only in the result (a C call that
ignores the value, or a declaration without one) goes through a generated thunk.
"""

from ..backend.wasm.obj import (I32, I64, F32, F64, CALL, ADDR, ADDR32, TYPE, PTR64, PTR32,
                                sigkey, unsig, uleb, sleb, uleb5, sleb5)

G_SP, G_MEM, G_TAB = 0, 1, 2

# runtime hook variables (function pointers that Python sets natively) and the glue functions
# a wasm module initializes them with
HOOKS = {'__cimo_hook_write': '__cimo_wasm_write', '__cimo_hook_grow': '__cimo_wasm_grow',
         '__cimo_hook_release': '__cimo_wasm_release'}
# functions Python calls, exported when present
HELPERS = ('__cimo_flush', '__cimo_heap_reset', '__cimo_ctx')


class LinkError(Exception):
    pass


def _name(s):
    b = s.encode()
    return uleb(len(b)) + b


def _vec(items):
    return uleb(len(items)) + b''.join(items)


def _section(sid, payload):
    return bytes([sid]) + uleb(len(payload)) + payload


def resolve(objs, provide):
    """objs: the module's objects; provide(name) -> an object defining name, or None. Returns
    the full list of objects (runtime members appended) and the names left undefined."""
    objs = list(objs)
    defined = set()
    for o in objs:
        defined |= {x.name for x in o.funcs + o.datas if x.glob}
    seen = set()
    i = 0
    undefined = set()
    while i < len(objs):
        o = objs[i]
        i += 1
        local = {x.name for x in o.funcs + o.datas if not x.glob}
        refs = set()
        for f in o.funcs:
            refs |= {s for _, k, s, _ in f.relocs if k != TYPE}
        for d in o.datas:
            refs |= {s for _, _, s, _ in d.relocs}
            if d.name in HOOKS:
                refs.add(HOOKS[d.name])
        for s in refs - local - defined:
            if s in seen:
                continue
            seen.add(s)
            m = provide(s)
            if m is None:
                undefined.add(s)
                continue
            if any(m is x for x in objs):
                continue
            objs.append(m)
            defined |= {x.name for x in m.funcs + m.datas if x.glob}
    return objs, undefined - defined


def link(objs, exports, imports_ok=lambda name: name.startswith('__cimo_js_')):
    """Objects to module bytes. exports: the function names to export (besides HELPERS that
    are present, __wasm_apply_data_relocs and the stack pointer accessors)."""
    # symbols: (kind, item, obj index); locals are per object
    glob, local = {}, [dict() for _ in objs]
    for oi, o in enumerate(objs):
        for f in o.funcs:
            tab = glob if f.glob else local[oi]
            if f.name in tab:
                raise LinkError('duplicate definition of %s' % f.name)
            tab[f.name] = ('f', f, oi)
        for d in o.datas:
            tab = glob if d.glob else local[oi]
            old = tab.get(d.name)
            if old is not None:
                if old[0] == 'd' and (old[1].common or d.common):
                    if d.common:  # keep the real definition, or the larger common block
                        if old[1].common and d.size > old[1].size:
                            tab[d.name] = ('d', d, oi)
                        continue
                    tab[d.name] = ('d', d, oi)
                    continue
                raise LinkError('duplicate definition of %s' % d.name)
            tab[d.name] = ('d', d, oi)

    def lookup(oi, name):
        return local[oi].get(name) or glob.get(name)

    # data layout
    datas = []
    for oi, o in enumerate(objs):
        for d in o.datas:
            if lookup(oi, d.name)[1] is d:
                datas.append((oi, d))
    off = 0
    daddr = {}
    maxalign = 16
    for oi, d in datas:
        off = (off + d.align - 1) & -d.align
        daddr[id(d)] = off
        off += max(d.size, 1)
        maxalign = max(maxalign, d.align)
    memsize = (off + 15) & -16

    # functions: imports first, then definitions, thunks and synthetic helpers
    funcs = [(oi, f) for oi, o in enumerate(objs) for f in o.funcs]
    imports = {}  # name -> signature key
    for oi, f in funcs:
        for s, key in f.calls.items():
            if lookup(oi, s) is None:
                if not imports_ok(s):
                    raise LinkError('undefined symbol: %s' % s)
                if imports.setdefault(s, key) != key:
                    raise LinkError('conflicting types for %s' % s)
        for _, k, s, _ in f.relocs:
            if k in (ADDR, ADDR32) and lookup(oi, s) is None:
                raise LinkError('undefined symbol: %s' % s)
    for oi, d in datas:
        for _, _, s, _ in d.relocs:
            if lookup(oi, s) is None:
                raise LinkError('undefined symbol: %s' % s)

    types, typelist = {}, []

    def typeidx(key):
        if key not in types:
            types[key] = len(typelist)
            typelist.append(key)
        return types[key]

    fidx = {}
    imp = list(imports)
    for n, s in enumerate(imp):
        fidx[('import', s)] = n
    nimp = len(imp)
    for n, (oi, f) in enumerate(funcs):
        fidx[id(f)] = nimp + n
    bodies = [f.body for _, f in funcs]
    ftypes = [sigkey(f.params, f.results) for _, f in funcs]
    thunks = {}

    def thunk(target, tkey, site):
        """A function of type site that calls target (type tkey), adapting the result."""
        k = (target, site)
        if k not in thunks:
            p, r = unsig(site)
            tp, tr = unsig(tkey)
            if p != tp or len(r) > 1 or len(tr) > 1:
                raise LinkError('conflicting types for a call (%s vs %s)' % (site, tkey))
            code = bytearray(b'\x00')
            for i in range(len(p)):
                code += b'\x20' + uleb(i)
            code += b'\x10' + uleb(target)
            if tr and not r:
                code.append(0x1A)
            elif r and not tr:
                code += {I32: b'\x41\x00', I64: b'\x42\x00', F32: b'\x43' + bytes(4),
                         F64: b'\x44' + bytes(8)}[r[0]]
            elif r != tr:
                raise LinkError('conflicting types for a call (%s vs %s)' % (site, tkey))
            code.append(0x0B)
            thunks[k] = nimp + len(bodies)
            bodies.append(bytes(code))
            ftypes.append(site)
        return thunks[k]

    def ftarget(oi, s):
        y = lookup(oi, s)
        if y is None:
            return fidx[('import', s)], imports[s]
        if y[0] != 'f':
            raise LinkError('%s is not a function' % s)
        return fidx[id(y[1])], sigkey(y[1].params, y[1].results)

    # the table: every function whose address is taken
    table, tslot = [], {}

    def slot(fi):
        if fi not in tslot:
            tslot[fi] = len(table)
            table.append(fi)
        return tslot[fi]

    def symval(oi, s, addend):
        """(global, value) of a symbol's address."""
        y = lookup(oi, s)
        if y[0] == 'd':
            return G_MEM, daddr[id(y[1])] + addend
        if addend:
            raise LinkError('offset from a function address: %s' % s)
        return G_TAB, slot(fidx[id(y[1])])

    # patch code
    for n, (oi, f) in enumerate(funcs):
        body = bytearray(f.body)
        for o, k, s, a in f.relocs:
            if k == CALL:
                fi, key = ftarget(oi, s)
                site = f.calls.get(s, key)
                if site != key:
                    fi = thunk(fi, key, site)
                body[o:o + 5] = uleb5(fi)
            elif k == TYPE:
                body[o:o + 5] = uleb5(typeidx(s))
            else:
                g, v = symval(oi, s, a)
                body[o + 1] = g
                p = o + (4 if k == ADDR else 3)
                body[p:p + 5] = sleb5(v)
        bodies[n] = bytes(body)

    # data image and load-time relocations
    image = bytearray()
    relocs = []  # (data offset, kind, global, value)
    for oi, d in datas:
        base = daddr[id(d)]
        if d.init:
            if len(image) < base + len(d.init):
                image.extend(bytes(base + len(d.init) - len(image)))
            image[base:base + len(d.init)] = d.init
        for o, k, s, a in d.relocs:
            g, v = symval(oi, s, a)
            relocs.append((base + o, k, g, v))
        if d.name in HOOKS and d.glob:
            g, v = symval(oi, HOOKS[d.name], 0)
            relocs.append((base, PTR64, g, v))

    # synthetic functions
    synth = {}
    code = bytearray(b'\x00')
    for o, k, g, v in relocs:
        code += b'\x23' + bytes([G_MEM]) + b'\x41' + sleb(o) + b'\x6A'
        if k == PTR64:
            code += b'\x23' + bytes([g]) + b'\xAD\x42' + sleb(v) + b'\x7C\x37\x03\x00'
        else:
            code += b'\x23' + bytes([g]) + b'\x41' + sleb(v) + b'\x6A\x36\x02\x00'
    code.append(0x0B)
    for name, body, key in (('__wasm_apply_data_relocs', bytes(code), ':'),
                            ('__cimo_get_sp', b'\x00\x23\x00\x0B', ':i'),
                            ('__cimo_set_sp', b'\x00\x20\x00\x24\x00\x0B', 'i:')):
        synth[name] = nimp + len(bodies)
        bodies.append(body)
        ftypes.append(key)

    # exports
    exp = []
    seen = set()
    for name in list(exports) + [h for h in HELPERS if h in glob and glob[h][0] == 'f']:
        if name in seen:
            continue
        seen.add(name)
        y = glob.get(name)
        if y is None or y[0] != 'f':
            raise LinkError('no function %s to export' % name)
        exp.append((name, fidx[id(y[1])]))
    exp += sorted(synth.items())

    # encode
    dylink = _name('dylink.0') + bytes([1]) + _vec_raw(
        uleb(memsize) + uleb(maxalign.bit_length() - 1) + uleb(len(table)) + uleb(0))
    tsec = _vec([b'\x60' + _vec([bytes([t]) for t in p]) + _vec([bytes([t]) for t in r])
                 for p, r in (unsig(k) for k in _typelist(typelist, ftypes, imports, typeidx))])
    isec = [
        _name('env') + _name('memory') + b'\x02\x00\x00',
        _name('env') + _name('__indirect_function_table') + b'\x01\x70\x00\x00',
        _name('env') + _name('__stack_pointer') + b'\x03\x7f\x01',
        _name('env') + _name('__memory_base') + b'\x03\x7f\x00',
        _name('env') + _name('__table_base') + b'\x03\x7f\x00',
    ] + [_name('env') + _name(s) + b'\x00' + uleb(typeidx(imports[s])) for s in imp]
    fsec = _vec([uleb(typeidx(k)) for k in ftypes])
    esec = _vec([_name(n) + b'\x00' + uleb(i) for n, i in exp])
    out = bytearray(b'\0asm\x01\0\0\0')
    out += _section(0, dylink)
    out += _section(1, tsec)
    out += _section(2, _vec(isec))
    out += _section(3, fsec)
    out += _section(7, esec)
    if table:
        out += _section(9, _vec([b'\x00\x23' + bytes([G_TAB]) + b'\x0B'
                                 + _vec([uleb(fi) for fi in table])]))
    out += _section(10, _vec([uleb(len(b)) + b for b in bodies]))
    if image:
        out += _section(11, _vec([b'\x00\x23' + bytes([G_MEM]) + b'\x0B' + uleb(len(image))
                                  + bytes(image)]))
    # function names, for stack traces
    names = [(fidx[('import', n)], n) for n in imp] + \
        [(fidx[id(f)], f.name) for _, f in funcs] + \
        [(fi, 'thunk:%s' % ('?')) for fi in thunks.values()] + [(i, n) for n, i in synth.items()]
    names.sort()
    out += _section(0, _name('name') + bytes([1]) + _vec_raw(
        _vec([uleb(i) + _name(n) for i, n in names])))
    return bytes(out)


def _vec_raw(payload):
    return uleb(len(payload)) + payload


def _typelist(typelist, ftypes, imports, typeidx):
    """Intern every type used, then return the final list."""
    for k in ftypes:
        typeidx(k)
    for k in imports.values():
        typeidx(k)
    return typelist
