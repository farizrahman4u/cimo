"""Port of QBE emit.c: data emission, linkage, fp constant stash, debug info."""

import math

from . import defs as D
from .defs import DStart, DEnd, DB, DH, DW, DL, DZ, M64, U32, s64, s32, bitsd, bitsf
from .util import die, err, intern

SecText, SecData, SecBss = range(3)

_sec = [
    [".text", ".data", ".bss"],
    [".abort \"unreachable\"", ".section .tdata,\"awT\"", ".section .tbss,\"awT\""],
]


def emitlnk(n, l, s, f):
    T = D.T
    pfx = "" if n[0] == '"' else T.assym
    sfx = ""
    if T.apple and l.thread:
        l.sec = "__DATA"
        l.secf = "__thread_data,thread_local_regular"
        sfx = "$tlv$init"
        f.write(".section __DATA,__thread_vars,"
                "thread_local_variables\n")
        f.write("%s%s:\n" % (pfx, n))
        f.write("\t.quad __tlv_bootstrap\n"
                "\t.quad 0\n"
                "\t.quad %s%s%s\n\n" % (pfx, n, sfx))
    if l.sec:
        f.write(".section %s" % l.sec)
        if l.secf:
            f.write(",%s" % l.secf)
    else:
        f.write(_sec[l.thread != 0][s])
    f.write("\n")
    if l.align:
        f.write(".balign %d\n" % l.align)
    if l.export:
        f.write(".globl %s%s\n" % (pfx, n))
    f.write("%s%s%s:\n" % (pfx, n, sfx))


def emitfnlnk(n, l, f):
    emitlnk(n, l, SecText, f)


_di = {
    DB: ("\t.byte", 0xff),
    DH: ("\t.short", 0xffff),
    DW: ("\t.int", 0xffffffff),
    DL: ("\t.quad", M64),
}
_zero = 0


def emitdat(d, f):
    global _zero
    T = D.T
    if d.type == DStart:
        _zero = 0
    elif d.type == DEnd:
        if d.lnk.common:
            if _zero == -1:
                die("invalid common data definition")
            p = "" if d.name[0] == '"' else T.assym
            f.write(".comm %s%s,%d" % (p, d.name, _zero))
            if d.lnk.align:
                f.write(",%d" % d.lnk.align)
            f.write("\n")
        elif _zero != -1:
            emitlnk(d.name, d.lnk, SecBss, f)
            f.write("\t.fill %d,1,0\n" % _zero)
    elif d.type == DZ:
        if _zero != -1:
            _zero = s64(_zero + d.num)
        else:
            f.write("\t.fill %d,1,0\n" % s64(d.num))
    else:
        if _zero != -1:
            emitlnk(d.name, d.lnk, SecData, f)
            if _zero > 0:
                f.write("\t.fill %d,1,0\n" % _zero)
            _zero = -1
        if d.isstr:
            if d.type != DB:
                err("strings only supported for 'b' currently")
            f.write("\t.ascii %s\n" % d.str)
        elif d.isref:
            p = "" if d.ref_name[0] == '"' else T.assym
            f.write("%s %s%s%+d\n" % (_di[d.type][0], p, d.ref_name, s64(d.ref_off)))
        else:
            decl, mask = _di[d.type]
            f.write("%s %d\n" % (decl, s64(d.num & mask)))


class Asmbits:
    __slots__ = ('n', 'size', 'link')

    def __init__(self, n, size):
        self.n = n
        self.size = size
        self.link = None


_stash = None


def stashbits(n, size):
    """n is the 64-bit pattern (signed or unsigned int); compared as uint64."""
    global _stash
    assert size == 4 or size == 8 or size == 16
    n &= M64
    i = 0
    b = _stash
    prev = None
    while b:
        if size <= b.size and b.n == n:
            return i
        prev = b
        b = b.link
        i += 1
    b = Asmbits(n, size)
    if prev is None:
        _stash = b
    else:
        prev.link = b
    return i


def _fmtf(x):
    """printf %f (glibc)."""
    if math.isnan(x):
        return "-nan" if math.copysign(1.0, x) < 0 else "nan"
    if math.isinf(x):
        return "-inf" if x < 0 else "inf"
    return "%f" % x


def emitfin(f, sec):
    global _stash
    T = D.T
    if not _stash:
        return
    f.write("/* floating point constants */\n")
    for lg in (4, 3, 2):
        b = _stash
        i = 0
        while b:
            if b.size == (1 << lg):
                f.write(".section %s\n"
                        ".p2align %d\n"
                        "%sfp%d:" % (sec[lg - 2], lg, T.asloc, i))
                if lg == 4:
                    f.write("\n\t.quad %d"
                            "\n\t.quad 0\n\n" % s64(b.n))
                elif lg == 3:
                    f.write("\n\t.quad %d"
                            " /* %s */\n\n" % (s64(b.n), _fmtf(bitsd(b.n))))
                elif lg == 2:
                    f.write("\n\t.int %d"
                            " /* %s */\n\n" % (s32(b.n), _fmtf(bitsf(b.n))))
            b = b.link
            i += 1
    _stash = None


def elf_emitfin(f):
    sec = [".rodata", ".rodata", ".rodata"]
    emitfin(f, sec)
    f.write(".section .note.GNU-stack,\"\",@progbits\n")


def elf_emitfnfin(fn, f):
    f.write(".type %s, @function\n" % fn)
    f.write(".size %s, .-%s\n" % (fn, fn))


def macho_emitfin(f):
    sec = [
        "__TEXT,__literal4,4byte_literals",
        "__TEXT,__literal8,8byte_literals",
        "__TEXT,__literal16,16byte_literals",
    ]
    emitfin(f, sec)


def pe_emitfin(f):
    sec = [".rodata", ".rodata", ".rodata"]
    emitfin(f, sec)


_file = []
_nfile = 0
_curfile = 0


def emitdbgfile(fn, f):
    global _nfile, _curfile
    id = intern(fn)
    for n in range(_nfile):
        if _file[n] == id:
            # gas requires positive file numbers
            _curfile = n + 1
            return
    _nfile += 1
    _file.append(id)
    _curfile = _nfile
    f.write(".file %u %s\n" % (_curfile, fn))


def emitdbgloc(line, col, f):
    line &= U32
    col &= U32
    if col != 0:
        f.write("\t.loc %u %u %u\n" % (_curfile, line, col))
    else:
        f.write("\t.loc %u %u\n" % (_curfile, line))
