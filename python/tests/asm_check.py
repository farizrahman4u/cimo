"""Compare cimo.asm against GNU as: assemble the same text with both and diff the sections,
relocations and symbols. Used by test_asm.py; also a script:

    python tests/asm_check.py [files.s/.ssa ...]     # default: QBE's tests + kernels.ssa

Relocation targets are normalized to (section, offset) for local symbols, since GNU as rewrites
references to local symbols as section symbol + addend.
"""

import glob
import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
REPO = os.path.dirname(PYROOT)
sys.path.insert(0, PYROOT)

from cimo import asm  # noqa: E402

AS = os.environ.get('AS', 'as')

RELOC_NAMES = {257: 'ABS64', 258: 'ABS32', 259: 'ABS16', 275: 'ADR_PREL_PG_HI21',
               277: 'ADD_ABS_LO12_NC', 278: 'LDST8_ABS_LO12_NC', 280: 'CONDBR19', 282: 'JUMP26',
               283: 'CALL26', 284: 'LDST16_ABS_LO12_NC', 285: 'LDST32_ABS_LO12_NC',
               286: 'LDST64_ABS_LO12_NC', 299: 'LDST128_ABS_LO12_NC', 311: 'ADR_GOT_PAGE',
               312: 'LD64_GOT_LO12_NC'}

SKIP_PREFIX = ('.note', '.debug', '.comment', '.rela', '.symtab', '.strtab', '.shstrtab')


def corpus():
    files = sorted(glob.glob(os.path.expanduser('~/code/qbe/test/*.ssa')))
    k = os.path.join(REPO, 'scratch/qbe_eval/build/kernels.ssa')
    if os.path.exists(k):
        files.append(k)
    return files


def have_gas():
    """GNU as for aarch64 ELF available?"""
    if not shutil.which(AS):
        return False
    try:
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.run([AS, '-o', os.path.join(d, 'a.o'), '-'], input=b'ret\n',
                               capture_output=True)
            if p.returncode != 0:
                return False
            with open(os.path.join(d, 'a.o'), 'rb') as f:
                h = f.read(20)
            return h[:4] == b'\x7fELF' and struct.unpack_from('<H', h, 18)[0] == 183
    except OSError:
        return False


def gas(text):
    with tempfile.TemporaryDirectory() as d:
        s = os.path.join(d, 'a.s')
        o = os.path.join(d, 'a.o')
        with open(s, 'w') as f:
            f.write(text)
        p = subprocess.run([AS, '-o', o, s], capture_output=True)
        if p.returncode != 0:
            raise RuntimeError('as failed: ' + p.stderr.decode(errors='replace'))
        with open(o, 'rb') as f:
            return f.read()


def parse_elf(data):
    """ELF64 LE relocatable -> normalized (sections, relocs, globals)."""
    shoff, = struct.unpack_from('<Q', data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', data, 0x3a)
    shdrs = []
    for i in range(shnum):
        shdrs.append(struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * shentsize))
    strtab_off = shdrs[shstrndx][4]

    def cstr(off):
        return data[off:data.index(b'\0', off)].decode()

    names = [cstr(strtab_off + h[0]) for h in shdrs]
    syms = []
    for i, h in enumerate(shdrs):
        if h[1] == 2:  # SHT_SYMTAB
            stroff = shdrs[h[6]][4]
            for j in range(h[5] // 24):
                nm, info, other, shndx, value, size = struct.unpack_from('<IBBHQQ', data,
                                                                         h[4] + j * 24)
                syms.append((cstr(stroff + nm), info >> 4, info & 15, shndx, value, size))
    sections = {}
    for i, h in enumerate(shdrs):
        name, typ, size, align = names[i], h[1], h[5], h[8]
        if i == 0 or name.startswith(SKIP_PREFIX) or size == 0 or typ in (2, 3, 4, 9):
            continue
        if typ == 8:  # NOBITS
            sections[name] = (('nobits', size), align)
        else:
            sections[name] = (data[h[4]:h[4] + size], align)
    relocs = {}
    for i, h in enumerate(shdrs):
        if h[1] != 4:  # SHT_RELA
            continue
        target = names[h[7]]
        if target.startswith(SKIP_PREFIX):
            continue
        out = []
        for j in range(h[5] // 24):
            off, info, add = struct.unpack_from('<QQq', data, h[4] + j * 24)
            nm, bind, typ, shndx, value, _ = syms[info >> 32]
            t = RELOC_NAMES.get(info & 0xffffffff, info & 0xffffffff)
            if typ == 3:  # STT_SECTION
                tgt = ('sec', names[shndx], add)
            elif bind == 0 and 0 < shndx < 0xff00:
                tgt = ('sec', names[shndx], value + add)
            else:
                tgt = ('sym', nm, add)
            out.append((off, t, tgt))
        relocs[target] = sorted(out)
    globs = {}
    for nm, bind, typ, shndx, value, size in syms:
        if bind == 0:
            continue
        if shndx == 0:
            globs[nm] = ('undef',)
        elif shndx == 0xfff2:  # SHN_COMMON
            globs[nm] = ('common', size, value)
        else:
            globs[nm] = ('def', names[shndx], value)
    return sections, relocs, globs


def normalize(obj, secmap=None):
    """cimo.asm.Object -> the same shape as parse_elf. secmap renames sections (Mach-O)."""
    secmap = secmap or {}

    def sn(s):
        return secmap.get(s.name, s.name)

    sections = {}
    relocs = {}
    for s in obj.sections:
        if s.name.startswith(SKIP_PREFIX) or s.size == 0:
            continue
        sections[sn(s)] = ((('nobits', s.size) if s.nobits else bytes(s.data)), s.align)
        out = []
        for r in s.relocs:
            y = obj.symbols[r.sym]
            if y.defined and not y.globl:
                tgt = ('sec', sn(y.section), y.value + r.addend)
            else:
                tgt = ('sym', r.sym, r.addend)
            out.append((r.offset, r.type, tgt))
        if out:
            relocs[sn(s)] = sorted(out)
    globs = {}
    for y in obj.symbols.values():
        if y.common:
            globs[y.name] = ('common', y.common[0], y.common[1])
        elif not y.defined:
            globs[y.name] = ('undef',)
        elif y.globl:
            globs[y.name] = ('def', sn(y.section), y.value)
    return sections, relocs, globs


def diff(a, b):
    """Describe differences between two normalized objects (a = GNU as, b = cimo)."""
    out = []
    sa, ra, ga = a
    sb, rb, gb = b
    for n in sorted(set(sa) | set(sb)):
        if n not in sa or n not in sb:
            out.append('section %s only in %s' % (n, 'as' if n in sa else 'cimo'))
            continue
        (da, ala), (db, alb) = sa[n], sb[n]
        if ala != alb:
            out.append('section %s alignment: as %d, cimo %d' % (n, ala, alb))
        if da != db:
            if isinstance(da, tuple) or isinstance(db, tuple) or len(da) != len(db):
                out.append('section %s: as %r bytes, cimo %r' % (
                    n, da if isinstance(da, tuple) else len(da),
                    db if isinstance(db, tuple) else len(db)))
            if not isinstance(da, tuple) and not isinstance(db, tuple):
                for i in range(0, min(len(da), len(db)), 4):
                    if da[i:i + 4] != db[i:i + 4]:
                        out.append('section %s offset %#x: as %s, cimo %s' % (
                            n, i, da[i:i + 4].hex(), db[i:i + 4].hex()))
                        break
    for n in sorted(set(ra) | set(rb)):
        x, y = ra.get(n, []), rb.get(n, [])
        if x != y:
            extra = [r for r in x if r not in y][:3]
            missing = [r for r in y if r not in x][:3]
            out.append('relocations in %s: only as %s; only cimo %s' % (n, extra, missing))
    if ga != gb:
        for k in sorted(set(ga) | set(gb)):
            if ga.get(k) != gb.get(k):
                out.append('symbol %s: as %s, cimo %s' % (k, ga.get(k), gb.get(k)))
    return out


def compare(text, syntax='elf'):
    """Assemble `text` with GNU as and cimo.asm; return a list of differences."""
    return diff(parse_elf(gas(text)), normalize(asm.assemble(text, syntax)))


def qbe_asm(path, target='arm64'):
    """QBE IR file -> assembly text with the Python QBE port (fresh process: QBE keeps global
    state such as label counters)."""
    env = dict(os.environ, PYTHONPATH=PYROOT)
    p = subprocess.run([sys.executable, '-m', 'cimo.qbe', '-t', target, path],
                       capture_output=True, env=env)
    if p.returncode != 0:
        raise RuntimeError('qbe failed on %s: %s' % (path, p.stderr.decode(errors='replace')))
    return p.stdout.decode('latin-1')


# Mach-O syntax -> ELF syntax, to check the arm64_apple path against GNU as

MACHO_SECTIONS = {
    '.text': '.text', '.data': '.data', '.bss': '.bss',
    '__TEXT,__literal4,4byte_literals': '.rodata.cst4',
    '__TEXT,__literal8,8byte_literals': '.rodata.cst8',
    '__TEXT,__literal16,16byte_literals': '.rodata.cst16',
}


def macho_to_elf(text):
    import re
    t = text
    for k, v in MACHO_SECTIONS.items():
        if k.startswith('__'):
            t = t.replace('.section %s\n' % k, '.section %s,"a"\n' % v)
    t = re.sub(r'(\S+)@gotpageoff\]', r'#:got_lo12:\1]', t)
    t = re.sub(r'(\S+)@gotpage\b', r':got:\1', t)
    t = re.sub(r'(\S+)@pageoff\b', r'#:lo12:\1', t)
    t = re.sub(r'(\S+)@page\b', r'\1', t)
    # Mach-O .comm alignment is log2
    t = re.sub(r'^\.comm (.*),(\d+),(\d+)$', lambda m: '.comm %s,%s,%d' % (
        m.group(1), m.group(2), 1 << int(m.group(3))), t, flags=re.M)
    return t


def compare_macho(text):
    return diff(parse_elf(gas(macho_to_elf(text))),
                normalize(asm.assemble(text, 'macho'), MACHO_SECTIONS))


def main():
    files = sys.argv[1:] or corpus()
    bad = 0
    for f in files:
        for target in ('arm64', 'arm64_apple'):
            text = open(f).read() if f.endswith('.s') else qbe_asm(f, target)
            try:
                d = compare(text) if target == 'arm64' else compare_macho(text)
            except asm.AsmError as e:
                d = ['cimo.asm: %s' % e]
            if d:
                bad += 1
                print('%s (%s):\n  %s' % (os.path.basename(f), target, '\n  '.join(d[:8])))
            if f.endswith('.s'):
                break
    print('%d of %d differ' % (bad, len(files)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
