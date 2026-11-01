"""Port of QBE amd64/all.h (registers)."""

from ..defs import RXX, Tmp0

_gp = 'RAX RCX RDX RSI RDI R8 R9 R10 R11 RBX R12 R13 R14 R15 RBP RSP'.split()
_fp = ['XMM%d' % i for i in range(16)]
for _i, _r in enumerate(_gp + _fp):
    globals()[_r] = RXX + 1 + _i
del _i, _r

NFPR = XMM14 - XMM0 + 1  # reserve XMM15
NGPR = RSP - RAX + 1
NFPS = NFPR

NGPS_SYSV = R11 - RAX + 1
NCLR_SYSV = R15 - RBX + 1

NGPS_WIN = R11 - RAX + 1 - 2  # -2 for RSI/RDI
NCLR_WIN = R15 - RBX + 1 + 2  # +2 for RSI/RDI

assert XMM15 < Tmp0
