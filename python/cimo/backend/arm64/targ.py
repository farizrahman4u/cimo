"""Port of QBE arm64/targ.c."""

from ..defs import Target, BIT
from .regs import *  # noqa: F401,F403
from . import regs as _r

arm64_rsave = [
    R0, R1, R2, R3, R4, R5, R6, R7,
    R8, R9, R10, R11, R12, R13, R14, R15,
    IP0, IP1, R18, LR,
    V0, V1, V2, V3, V4, V5, V6, V7,
    V16, V17, V18, V19, V20, V21, V22, V23,
    V24, V25, V26, V27, V28, V29, V30,
    -1
]
arm64_rclob = [
    R19, R20, R21, R22, R23, R24, R25, R26,
    R27, R28,
    V8, V9, V10, V11, V12, V13, V14, V15,
    -1
]

RGLOB = BIT(FP) | BIT(SP) | BIT(IP1) | BIT(R18)


def arm64_memargs(op):
    return 0


def _abi():
    from . import abi
    return abi


def _isel():
    from . import isel
    return isel


def _emit():
    from . import emit
    return emit


def _gemit():
    from .. import emit
    return emit


def _common(**kw):
    return Target(
        gpr0=R0,
        ngpr=NGPR,
        fpr0=V0,
        nfpr=NFPR,
        rglob=RGLOB,
        nrglob=4,
        rsave=arm64_rsave,
        nrsave=[NGPS, NFPS],
        retregs=lambda r, p: _abi().arm64_retregs(r, p),
        argregs=lambda r, p: _abi().arm64_argregs(r, p),
        memargs=arm64_memargs,
        isel=lambda fn: _isel().arm64_isel(fn),
        abi1=lambda fn: _abi().arm64_abi(fn),
        emitfn=lambda fn, f: _emit().arm64_emitfn(fn, f),
        cansel=0,
        **kw)


def _elimsb(fn):
    from ..abi import elimsb
    elimsb(fn)


T_arm64 = _common(
    name="arm64",
    abi0=_elimsb,
    emitfin=lambda f: _gemit().elf_emitfin(f),
    asloc=".L",
)

# cimo extension: Windows arm64 (AAPCS64 with Windows' variadic convention, x18 untouched,
# stack probes; ELF syntax, since cimo assembles and links in memory)
T_arm64_win = _common(
    name="arm64_win",
    windows=1,
    abi0=_elimsb,
    emitfin=lambda f: _gemit().elf_emitfin(f),
    asloc=".L",
)

T_arm64_apple = _common(
    name="arm64_apple",
    apple=1,
    abi0=lambda fn: _abi().apple_extsb(fn),
    emitfin=lambda f: _gemit().macho_emitfin(f),
    asloc="L",
    assym="_",
)

assert (RGLOB & (BIT(R8 + 1) - 1)) == 0
assert len(arm64_rsave) == NGPS + NFPS + 1 and len(arm64_rclob) == NCLR + 1
