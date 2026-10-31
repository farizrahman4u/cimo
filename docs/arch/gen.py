"""Generate docs/arch/arch.json: one kernel through every stage of cimo, as the website's
Architecture section shows it. Real compiler output; rerun when the compiler changes:

    python3 docs/arch/gen.py          (from the repository root, on arm64)
"""

import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'docs', 'nn'))

import cimo  # noqa: E402
from cimo import driver as d  # noqa: E402
import cimo.wasm as cw  # noqa: E402
import wat  # noqa: E402

SRC = r'''void saxpy(int n, float a, const float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}
'''


def ir_after_loop_optimizer(ssa, params):
    """QBE IR after the passes that run before instruction selection, including cimo's loop
    optimizer (the same sequence as cimo.backend.main up to the middle end)."""
    from cimo.backend import defs as D
    from cimo.backend.main import _Driver, _targets
    from cimo.backend.parse import parse, printfn
    from cimo.backend import cfg, ssa as S, mem, alias, load, gvn, gcm, ext
    from cimo import middle
    D.T = _targets()[0]
    out = io.StringIO()

    def func(fn):
        D.T.abi0(fn)
        cfg.fillcfg(fn); S.filluse(fn); mem.promote(fn); S.filluse(fn); S.ssa(fn)
        S.filluse(fn); alias.fillalias(fn); load.loadopt(fn); S.filluse(fn)
        alias.fillalias(fn); mem.coalesce(fn); S.filluse(fn); cfg.filldom(fn)
        gvn.gvn(fn); cfg.fillcfg(fn); cfg.simplcfg(fn); S.filluse(fn); cfg.filldom(fn)
        gcm.gcm(fn); S.filluse(fn)
        cfg.filldom(fn)
        if middle.run(fn):
            gvn.gvn(fn); cfg.fillcfg(fn); cfg.simplcfg(fn); S.filluse(fn); cfg.filldom(fn)
            gcm.gcm(fn); S.filluse(fn)
        printfn(fn, out)

    with middle.options(enabled=True, params=params), ext.options(fma=True, math=True):
        drv = _Driver(io.StringIO(), 0)
        parse(ssa.encode('latin-1'), '<cimo>', drv.dbgfile, drv.data, func)
    return out.getvalue()


def asm_with_bytes(text, name):
    """The function's arm64 assembly, each instruction with the 4 bytes cimo encodes it to."""
    from cimo import assembler
    obj = assembler.assemble(text, 'elf')
    sec = obj.section('.text')
    start = obj.symbols[name].value
    lines = text.splitlines()
    i = lines.index(name + ':')
    out, pc = [], start
    for line in lines[i:]:
        s = line.strip()
        if s.startswith('/* end function'):
            break
        if not s or s.startswith('.') and not s.endswith(':') or s.startswith('/*'):
            continue
        if s.endswith(':'):
            out.append(s)
            continue
        word = int.from_bytes(bytes(sec.data[pc:pc + 4]), 'little')
        out.append('%06x  %08x   %s' % (pc - start, word, re.sub(r'\s+', ' ', s, count=1)))
        pc += 4
    return '\n'.join(out), pc - start


def asm_text(text, name):
    """The function's assembly as QBE's backend prints it (directives dropped)."""
    lines = text.splitlines()
    i = lines.index(name + ':')
    out = []
    for line in lines[i:]:
        s = line.strip()
        if s.startswith('/* end function'):
            break
        if not s or s.startswith('/*') or (s.startswith('.') and not s.endswith(':')):
            continue
        out.append(s if s.endswith(':') else '    ' + re.sub(r'\s+', ' ', s, count=1))
    return '\n'.join(out)


def main():
    pp = d._preprocess(SRC)
    ssa, params = d._frontend(pp, '<string>', fma=d.FMA, builtins=True, diagnostics=True,
                              prune=True)
    opt = ir_after_loop_optimizer(ssa, params)
    asm = d._qbe(ssa, params, stkchk=set())
    listing, nbytes = asm_with_bytes(asm, 'saxpy')
    mod, _ = cw.build(SRC)
    watext, _, _ = wat.disassemble(mod, {'saxpy'})
    k = cimo.compile(SRC)
    loader = '''>>> import cimo, numpy as np
>>> k = cimo.compile(open("saxpy.c").read())
>>> k.saxpy                       # %d bytes of arm64, mapped read+execute
<cimo kernel saxpy>
>>> x = np.ones(1000, np.float32); y = np.arange(1000, dtype=np.float32)
>>> k.saxpy(len(x), 2.0, x, y)    # a direct call through ctypes; y is updated in place
>>> y[:4]
array([2., 3., 4., 5.], dtype=float32)

# what happened: the encoded bytes were linked with the runtime parts they use,
# written to an anonymous mmap (RW), switched to RX with mprotect (W^X), the
# instruction cache flushed, and each function wrapped as a ctypes callable.
# In the browser the same module is WebAssembly, loaded into Pyodide's memory.''' % nbytes
    n_ir = len(ssa.splitlines())
    stages = [
        dict(id='c', title='C source', origin='your code', lang='c', text=SRC.rstrip('\n'),
             desc='A kernel as a string or a .c file. No headers needed for this one.',
             stat='%d lines of C' % len(SRC.splitlines())),
        dict(id='ir', title='Front end', origin='from cproc', lang='qbe', text=ssa.rstrip(),
             desc='cproc, ported to Python, parses and type-checks C11 and emits QBE IR: '
                  'SSA-style instructions over 32/64-bit ints and floats.',
             stat='%d lines of IR' % n_ir),
        dict(id='opt', title='Loop optimizer', origin="cimo's own", lang='qbe', text=opt.rstrip(),
             desc='Locals become registers, x[i] and y[i] become pointers that step forward, '
                  'and the loop is unrolled by 4 with a remainder loop, behind a trip-count '
                  'check. a * x[i] + y[i] is one fused multiply-add.',
             stat='%d lines, unrolled x4' % len(opt.splitlines())),
        dict(id='asm', title='Backend', origin='from QBE', lang='asm', text=asm_text(asm, 'saxpy'),
             desc='QBE, ported to Python and extended with arm64 addressing modes, fused '
                  'multiply-add and load/store pairing, selects instructions and allocates '
                  'registers.',
             stat='%d instructions' % (nbytes // 4)),
        dict(id='enc', title='Encoder', origin="cimo's own", lang='hex', text=listing,
             desc='cimo encodes machine code itself, with no assembler or linker: offset, the '
                  '4 bytes of each arm64 instruction, and the instruction. The same compiler '
                  'emits WebAssembly for the browser.',
             stat='%d bytes of arm64' % nbytes,
             alt=dict(label='WebAssembly', lang='wasm', text=watext.rstrip(),
                      stat='%.1f KB module' % (len(mod) / 1024))),
        dict(id='run', title='Executable memory', origin='in your process', lang='py',
             text=loader,
             desc='The code is mapped into the running Python process and called on numpy '
                  'buffers in place. Compiled modules are cached by content hash.',
             stat='called in place'),
    ]
    with open(os.path.join(HERE, 'arch.json'), 'w') as f:
        json.dump({'kernel': 'saxpy', 'stages': stages}, f, indent=1)
    for s in stages:
        print('%-18s %s' % (s['title'], s['stat']))


if __name__ == '__main__':
    main()
