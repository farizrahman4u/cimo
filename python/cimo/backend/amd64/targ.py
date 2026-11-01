"""Port of QBE amd64/targ.c."""

from .. import defs as D
from ..defs import Target, BIT, NOp
from .regs import *  # noqa: F401,F403
from .regs import RAX, XMM0, NGPR, NFPR, NGPS_SYSV, NGPS_WIN, NFPS, RBP, RSP


class Amd64Op:
    __slots__ = ('nmem', 'zflag', 'lflag')

    def __init__(self, nmem=0, zflag=0, lflag=0):
        self.nmem = nmem
        self.zflag = zflag
        self.lflag = lflag


# the X(nmem, zflag, lflag) column of QBE ops.h; ops not listed (cimo's own among them) are
# X(0, 0, 0)
_X = """
add 2 1 0;sub 2 1 0;neg 1 1 0;mul 2 0 0;and 2 1 0;or 2 1 0;xor 2 1 0;sar 1 1 0;shr 1 1 0;
shl 1 1 0;ceqw 0 1 0;cnew 0 1 0;csgew 0 1 0;csgtw 0 1 0;cslew 0 1 0;csltw 0 1 0;cugew 0 1 0;
cugtw 0 1 0;culew 0 1 0;cultw 0 1 0;ceql 0 1 0;cnel 0 1 0;csgel 0 1 0;csgtl 0 1 0;
cslel 0 1 0;csltl 0 1 0;cugel 0 1 0;cugtl 0 1 0;culel 0 1 0;cultl 0 1 0;ceqs 0 1 0;
cges 0 1 0;cgts 0 1 0;cles 0 1 0;clts 0 1 0;cnes 0 1 0;cos 0 1 0;cuos 0 1 0;ceqd 0 1 0;
cged 0 1 0;cgtd 0 1 0;cled 0 1 0;cltd 0 1 0;cned 0 1 0;cod 0 1 0;cuod 0 1 0;storeb 0 0 1;
storeh 0 0 1;storew 0 0 1;storel 0 0 1;stores 0 0 1;stored 0 0 1;loadsb 0 0 1;loadub 0 0 1;
loadsh 0 0 1;loaduh 0 0 1;loadsw 0 0 1;loaduw 0 0 1;load 0 0 1;extsb 0 0 1;extub 0 0 1;
extsh 0 0 1;extuh 0 0 1;extsw 0 0 1;extuw 0 0 1;exts 0 0 1;truncd 0 0 1;stosi 0 0 1;
stoui 0 0 1;dtosi 0 0 1;dtoui 0 0 1;swtof 0 0 1;uwtof 0 0 1;sltof 0 0 1;ultof 0 0 1;
cast 0 0 1;copy 0 0 1;dbgloc 0 0 1;nop 0 0 1;addr 0 0 1;blit0 0 1 0;blit1 0 1 0;swap 1 0 0;
xidiv 1 0 0;xdiv 1 0 0;xcmp 1 1 0;xtest 1 1 0;flagieq 0 0 1;flagine 0 0 1;flagisge 0 0 1;
flagisgt 0 0 1;flagisle 0 0 1;flagislt 0 0 1;flagiuge 0 0 1;flagiugt 0 0 1;flagiule 0 0 1;
flagiult 0 0 1;flagfeq 0 0 1;flagfge 0 0 1;flagfgt 0 0 1;flagfle 0 0 1;flagflt 0 0 1;
flagfne 0 0 1;flagfo 0 0 1;flagfuo 0 0 1
"""

amd64_op = [Amd64Op() for _ in range(NOp)]
for _e in _X.replace('\n', '').split(';'):
    _n, _m, _z, _l = _e.split()
    amd64_op[getattr(D, 'O' + _n)] = Amd64Op(int(_m), int(_z), int(_l))
del _e, _n, _m, _z, _l


def amd64_memargs(op):
    return amd64_op[op].nmem


def _sysv():
    from . import sysv
    return sysv


def _winabi():
    from . import winabi
    return winabi


def _isel():
    from . import isel
    return isel


def _emit():
    from . import emit
    return emit


def _gemit():
    from .. import emit
    return emit


def _elimsb(fn):
    from ..abi import elimsb
    elimsb(fn)


def _common(**kw):
    return Target(
        gpr0=RAX,
        ngpr=NGPR,
        fpr0=XMM0,
        nfpr=NFPR,
        rglob=BIT(RBP) | BIT(RSP),
        nrglob=2,
        memargs=amd64_memargs,
        abi0=_elimsb,
        isel=lambda fn: _isel().amd64_isel(fn),
        cansel=1,
        **kw)


def _sysv_common(**kw):
    return _common(
        abi1=lambda fn: _sysv().amd64_sysv_abi(fn),
        rsave=_sysv().amd64_sysv_rsave,
        nrsave=[NGPS_SYSV, NFPS],
        retregs=lambda r, p: _sysv().amd64_sysv_retregs(r, p),
        argregs=lambda r, p: _sysv().amd64_sysv_argregs(r, p),
        emitfn=lambda fn, f: _emit().amd64_sysv_emitfn(fn, f),
        **kw)


T_amd64_sysv = _sysv_common(
    name="amd64_sysv",
    emitfin=lambda f: _gemit().elf_emitfin(f),
    asloc=".L",
)

T_amd64_apple = _sysv_common(
    name="amd64_apple",
    apple=1,
    emitfin=lambda f: _gemit().macho_emitfin(f),
    asloc="L",
    assym="_",
)

T_amd64_win = _common(
    name="amd64_win",
    windows=1,
    emitfin=lambda f: _gemit().pe_emitfin(f),
    asloc="L",
    abi1=lambda fn: _winabi().amd64_winabi_abi(fn),
    rsave=_winabi().amd64_winabi_rsave,
    nrsave=[NGPS_WIN, NFPS],
    retregs=lambda r, p: _winabi().amd64_winabi_retregs(r, p),
    argregs=lambda r, p: _winabi().amd64_winabi_argregs(r, p),
    emitfn=lambda fn, f: _emit().amd64_winabi_emitfn(fn, f),
)
