"""Inside Pyodide: build a WASM module in pure Python, instantiate it against Pyodide's own
linear memory, and call it on a Python-owned buffer (zero-copy)."""
import ctypes
import struct

import js
import pyodide_js
from pyodide.ffi import to_js


def leb(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n == 0:
            out.append(b)
            return bytes(out)
        out.append(b | 0x80)


def name(s):
    return leb(len(s)) + s.encode()


def section(sid, payload):
    return bytes([sid]) + leb(len(payload)) + payload


I32, I64 = 0x7F, 0x7E
# long long sum(const long long *p, int n)
body = bytes([
    0x01, 0x01, I64,               # 1 local: s:i64
    0x02, 0x40,                    # block
    0x03, 0x40,                    #  loop
    0x20, 0x01, 0x45, 0x0D, 0x01,  #   br_if 1 (n == 0)
    0x20, 0x02, 0x20, 0x00, 0x29, 0x03, 0x00, 0x7C, 0x21, 0x02,  # s += i64.load(p)
    0x20, 0x00, 0x41, 0x08, 0x6A, 0x21, 0x00,  # p += 8
    0x20, 0x01, 0x41, 0x01, 0x6B, 0x21, 0x01,  # n -= 1
    0x0C, 0x00,                    #   br 0
    0x0B, 0x0B,                    #  end loop, end block
    0x20, 0x02, 0x0B,              # return s
])
module = (
    b"\0asm" + struct.pack("<I", 1)
    # dylink.0 custom section (mem/table size+align = 0) makes this an Emscripten side module
    + section(0, name("dylink.0") + b"\x01\x04\x00\x00\x00\x00")
    + section(1, b"\x01" + bytes([0x60, 2, I32, I32, 1, I64]))
    + section(2, b"\x01" + name("env") + name("memory") + b"\x02\x00\x01")
    + section(3, b"\x01\x00")
    + section(7, b"\x01" + name("sum") + b"\x00\x00")
    + section(10, b"\x01" + leb(len(body)) + body)
)

# Pyodide doesn't expose its WebAssembly.Memory; load as a side module so the
# Emscripten dynamic linker supplies env.memory (Pyodide's own linear memory).
flags = to_js({"loadAsync": False, "global": False, "nodelete": True, "allowUndefined": True},
              dict_converter=js.Object.fromEntries)
exports = pyodide_js._module.loadWebAssemblyModule(to_js(module), flags, "ppcc_test.so", to_js({}, dict_converter=js.Object.fromEntries))

data = (ctypes.c_longlong * 1000)(*range(1000))
ptr = ctypes.addressof(data)
got = int(exports.sum(ptr, 1000))
print("sum via generated wasm:", got, "ok" if got == sum(range(1000)) else "WRONG")
