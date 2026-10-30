"""The wasm backend's object format: functions and data with symbolic relocations, linked by
cimo.wasm.link into one Emscripten side module.

Pointers are 64-bit in the IR (cimo compiles every target as LP64); wasm32 addresses are their
low 32 bits. An object is plain data and serializes to JSON for the module cache.

Code relocations (offsets into a function's body bytes):
  CALL  sym            a 5-byte padded uLEB function index (`call`)
  ADDR  sym, addend    `global.get G; i64.extend_i32_u; i64.const N; i64.add` with G a 1-byte
                       global index and N a 5-byte padded sLEB: G = __memory_base and N = the
                       data offset for a data symbol, G = __table_base and N = the table slot
                       for a function (its address)
  ADDR32 sym, addend   the same as an i32: `global.get G; i32.const N; i32.add`
  TYPE  key            a 5-byte padded uLEB type index (`call_indirect`); key is a signature
Data relocations: PTR64 / PTR32 sym, addend at a data offset (applied at load time by the
module's __wasm_apply_data_relocs, since bases are only known then).
"""

import json

I32, I64, F32, F64 = 0x7F, 0x7E, 0x7D, 0x7C
VT = {0: I32, 1: I64, 2: F32, 3: F64}  # QBE class -> value type
VTNAME = {I32: 'i', I64: 'l', F32: 'f', F64: 'd'}

CALL, ADDR, ADDR32, TYPE = 'call', 'addr', 'addr32', 'type'
PTR64, PTR32 = 'p64', 'p32'


def sigkey(params, results):
    """'il:d' for (i32, i64) -> f64."""
    return ''.join(VTNAME[t] for t in params) + ':' + ''.join(VTNAME[t] for t in results)


def unsig(key):
    inv = {v: k for k, v in VTNAME.items()}
    p, _, r = key.partition(':')
    return [inv[c] for c in p], [inv[c] for c in r]


class Func:
    __slots__ = ('name', 'glob', 'params', 'results', 'body', 'relocs', 'calls')

    def __init__(self, name, glob, params, results, body, relocs, calls=None):
        self.name, self.glob = name, glob
        self.params, self.results = params, results
        self.body = body          # bytes: local declarations, code, final `end`
        self.relocs = relocs      # [(offset, kind, sym or key, addend)]
        self.calls = calls or {}  # sym -> signature key at its direct call sites


class Data:
    __slots__ = ('name', 'glob', 'align', 'size', 'init', 'relocs', 'common')

    def __init__(self, name, glob, align, size, init, relocs, common=False):
        self.name, self.glob, self.align, self.size = name, glob, align, size
        self.init = init          # bytes, at most size (the rest is zero)
        self.relocs = relocs      # [(offset, PTR64 or PTR32, sym, addend)]
        self.common = common


class Object:
    def __init__(self):
        self.funcs = []
        self.datas = []

    def defined(self):
        return {f.name for f in self.funcs} | {d.name for d in self.datas}

    def dumps(self):
        return json.dumps({
            'f': [[f.name, f.glob, ''.join(VTNAME[t] for t in f.params),
                   ''.join(VTNAME[t] for t in f.results), f.body.decode('latin-1'), f.relocs,
                   f.calls] for f in self.funcs],
            'd': [[d.name, d.glob, d.align, d.size, d.init.decode('latin-1'), d.relocs,
                   d.common] for d in self.datas]}, separators=(',', ':'))

    @staticmethod
    def loads(text):
        j = json.loads(text)
        o = Object()
        inv = {v: k for k, v in VTNAME.items()}
        for name, glob, p, r, body, relocs, calls in j['f']:
            o.funcs.append(Func(name, glob, [inv[c] for c in p], [inv[c] for c in r],
                                body.encode('latin-1'), [tuple(x) for x in relocs], calls))
        for name, glob, align, size, init, relocs, common in j['d']:
            o.datas.append(Data(name, glob, align, size, init.encode('latin-1'),
                                [tuple(x) for x in relocs], common))
        return o


# LEB128

def uleb(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def sleb(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if (n == 0 and not b & 0x40) or (n == -1 and b & 0x40):
            out.append(b)
            return bytes(out)
        out.append(b | 0x80)


def uleb5(n):
    assert 0 <= n < 1 << 35
    return bytes([(n >> (7 * i) & 0x7F) | (0x80 if i < 4 else 0) for i in range(5)])


def sleb5(n):
    assert -(1 << 34) <= n < 1 << 34
    return bytes([(n >> (7 * i) & 0x7F) | (0x80 if i < 4 else 0) for i in range(5)])
