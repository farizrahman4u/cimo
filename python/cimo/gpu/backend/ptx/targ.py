"""The PTX target: QBE's target-independent passes, then cimo.gpu.backend.ptx.emit (no register allocation or
instruction selection; the driver's PTX compiler does both)."""

from ....backend.defs import Target


def _elimsb(fn):
    from ....backend.abi import elimsb
    elimsb(fn)


def _emitfn(fn, mod):
    from .emit import emitfn
    emitfn(fn, mod)


T_ptx = Target(
    name="ptx",
    gpr0=0, ngpr=0, fpr0=0, nfpr=0, rglob=0, nrglob=0, rsave=[-1], nrsave=[0, 0],
    memargs=lambda op: 0,
    abi0=_elimsb,
    emitfn=_emitfn,
    emitfin=lambda mod: None,
    cansel=0,
)
