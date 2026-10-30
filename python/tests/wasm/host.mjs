// A stand-in for Pyodide's dynamic loader, for testing cimo's wasm modules under Node:
// one linear memory, a table, a C stack, and the side module loaded at a memory base.
// Protocol: JSON lines on stdin, one reply per line on stdout.
//   {"load": "<base64 module>"}                      -> {"ok": true, "exports": [...]}
//   {"call": name, "args": [...], "types": "lw.."}  -> {"ret": value, "out": "..."} or {"err": ...}
//   {"alloc": nbytes}                                -> {"ptr": address}
//   {"write": ptr, "b64": data} / {"read": ptr, "n": n} -> {"ok": true} / {"b64": data}
import readline from 'node:readline';

const memory = new WebAssembly.Memory({ initial: 4096 });  // 256 MB
const table = new WebAssembly.Table({ initial: 1, element: 'anyfunc' });
const STACK_TOP = 1 << 20;            // C stack: 64 KB .. 1 MB, grows down
const sp = new WebAssembly.Global({ value: 'i32', mutable: true }, STACK_TOP);
let brk = 2 << 20;                    // bump allocator for data, heap chunks and buffers
const alloc = (n, a = 16) => { brk = (brk + a - 1) & -a; const p = brk; brk += n; return p; };
let out = '';
let exports = null;
const u8 = () => new Uint8Array(memory.buffer);
const cstr = p => { if (!p) return null; const m = u8(); let e = p; while (m[e]) e++; return Buffer.from(m.subarray(p, e)).toString('latin1'); };

class Fail extends Error {}

function load(bytes) {
  const mod = new WebAssembly.Module(bytes);
  const sec = WebAssembly.Module.customSections(mod, 'dylink.0')[0];
  const d = new Uint8Array(sec);
  let i = 0;
  const leb = () => { let r = 0, s = 0, b; do { b = d[i++]; r |= (b & 0x7f) << s; s += 7; } while (b & 0x80); return r >>> 0; };
  if (d[i++] !== 1) throw new Error('no mem info');
  leb();
  const memSize = leb(), memAlign = leb(), tableSize = leb();
  const memBase = memSize ? alloc(memSize, 1 << memAlign) : 0;
  const tableBase = table.length;
  if (tableSize) table.grow(tableSize);
  const env = {
    memory, __indirect_function_table: table, __stack_pointer: sp,
    __memory_base: new WebAssembly.Global({ value: 'i32' }, memBase),
    __table_base: new WebAssembly.Global({ value: 'i32' }, tableBase),
    __cimo_js_write: (p, n) => { out += Buffer.from(u8().subarray(p, p + n)).toString('latin1'); },
    __cimo_js_grow: (n, kind) => alloc(n),
    __cimo_js_release: p => {},
    __cimo_js_fail: (code, expr, file, line, func) => {
      throw new Fail(JSON.stringify({ code, expr: cstr(expr), file: cstr(file), line, func: cstr(func) }));
    },
  };
  const inst = new WebAssembly.Instance(mod, { env });
  exports = inst.exports;
  exports.__wasm_apply_data_relocs();
  return Object.keys(exports);
}

const rl = readline.createInterface({ input: process.stdin });
for await (const line of rl) {
  const m = JSON.parse(line);
  let r;
  try {
    if (m.load) r = { ok: true, exports: load(Buffer.from(m.load, 'base64')) };
    else if (m.alloc !== undefined) r = { ptr: alloc(m.alloc) };
    else if (m.write !== undefined) { u8().set(Buffer.from(m.b64, 'base64'), m.write); r = { ok: true }; }
    else if (m.read !== undefined) r = { b64: Buffer.from(u8().subarray(m.read, m.read + m.n)).toString('base64') };
    else if (m.call) {
      const args = m.args.map((a, k) => m.types[k] === 'l' ? BigInt(a) : typeof a === 'string' ? Number(a) : a);
      out = '';
      const s0 = sp.value;
      try {
        let v = exports[m.call](...args);
        if (exports.__cimo_flush) exports.__cimo_flush();
        r = { ret: typeof v === 'bigint' ? v.toString() : v, out };
      } catch (e) {
        sp.value = s0;
        r = { err: e instanceof Fail ? JSON.parse(e.message) : String(e.stack || e).split('\n').slice(0, 12).join('\n'), out };
      }
    }
  } catch (e) { r = { err: String(e && e.stack || e) }; }
  process.stdout.write(JSON.stringify(r, (k, v) => typeof v === 'number' && !Number.isFinite(v) ? String(v) : v) + '\n');
}
