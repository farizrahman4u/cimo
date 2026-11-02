"""Port of QBE main.c: the qbe command line.

python -m cimo.backend [-d flags] [-o file] [-t target] {file.ssa, -}
"""

import sys

from . import defs as D
from .arm64.regs import IP1 as _IP1
from .parse import parse, printfn


def _targets():
    from .arm64 import targ
    return [targ.T_arm64, targ.T_arm64_apple, targ.T_arm64_win]


class _Lazy:
    """A pass module, imported at its first use."""

    def __init__(self, name):
        self.name = name

    def __getattr__(self, a):
        import importlib
        return getattr(importlib.import_module('.' + self.name, __package__), a)


class _Driver:
    def __init__(self, outf, dbg):
        self.outf = outf
        self.dbg = dbg
        self.pending = []  # cimo (ext.inline): the module's functions, compiled by finish()
        self.datarefs = set()  # cimo (ext.inline): symbols the module's data refers to
        self.dataitems = {}  # cimo (ext.inline): static data's initial bytes, by name (inline.recorddata)

    def data(self, d):
        from .emit import emitdat
        if d.isref:
            self.datarefs.add(d.ref_name)
        from . import ext as _x
        if _x.inline:
            from . import inline
            inline.recorddata(self.dataitems, d)
        if self.dbg:
            return
        if D.T.name == 'wasm':  # cimo: the wasm target collects an object, not assembly
            if not hasattr(self.outf, 'databuilder'):
                from .wasm.emit import DataBuilder
                self.outf.databuilder = DataBuilder(self.outf)
            self.outf.databuilder(d)
            return
        emitdat(d, self.outf)
        if d.type == D.DEnd:
            self.outf.write("/* end data */\n\n")

    def func(self, fn):
        from . import ext
        if ext.inline:  # cimo: all of the module first, so callees defined later inline too
            self.pending.append(fn)
            return
        self.compile(fn)

    def finish(self):
        """cimo (ext.inline): inline small functions into their callers, callees first, then
        compile the functions in their order in the module."""
        from . import ext, inline
        fns, self.pending = self.pending, []
        if not fns:
            return
        inline.constglobals(fns, self.dataitems, self.datarefs)
        live = inline.expand_all(fns, ext.inline, self.datarefs)
        inline.ipo(live, self.datarefs)
        for fn in live:
            self.compile(fn)

    @staticmethod
    def _cleanup(fn, cfg, ssa, gvn, gcm):
        ssa.ssacheck(fn)
        gvn.gvn(fn)
        cfg.fillcfg(fn)
        cfg.simplcfg(fn)
        ssa.filluse(fn)
        cfg.filldom(fn)
        gcm.gcm(fn)
        ssa.filluse(fn)
        ssa.ssacheck(fn)

    @staticmethod
    def _cleanup_fill(fn, cfg, ssa):
        ssa.filluse(fn)
        cfg.filldom(fn)
        ssa.ssacheck(fn)

    def compile(self, fn):
        T = D.T
        err = sys.stderr
        if self.dbg:
            err.write("**** Function %s ****" % fn.name)
        if D.debug['P']:
            err.write("\n> After parsing:\n")
            printfn(fn, err)
        (cfg, ssa, mem, alias, load, copy, gvn, gcm, ifopt, simpl, live, spill,
         rega) = (_Lazy(m) for m in ('cfg', 'ssa', 'mem', 'alias', 'load', 'copy', 'gvn',
                                     'gcm', 'ifopt', 'simpl', 'live', 'spill', 'rega'))
        T.abi0(fn)
        cfg.fillcfg(fn)
        ssa.filluse(fn)
        from . import ext as _x
        if _x.cse:
            from . import sroa
            if sroa.sroa(fn):
                ssa.filluse(fn)
        mem.promote(fn)
        ssa.filluse(fn)
        ssa.ssa(fn)
        ssa.filluse(fn)
        ssa.ssacheck(fn)
        alias.fillalias(fn)
        load.loadopt(fn)
        ssa.filluse(fn)
        alias.fillalias(fn)
        mem.coalesce(fn)
        ssa.filluse(fn)
        cfg.filldom(fn)
        ssa.ssacheck(fn)
        gvn.gvn(fn)
        cfg.fillcfg(fn)
        cfg.simplcfg(fn)
        ssa.filluse(fn)
        cfg.filldom(fn)
        gcm.gcm(fn)
        ssa.filluse(fn)
        ssa.ssacheck(fn)
        from . import ext, cse
        if ext.cse:  # cimo: gvn made equal addresses one temporary: their loads can merge
            loads = cse.nload(fn)
            cfg.filldom(fn)
            from . import scalar, thread
            if scalar.sccp(fn):
                cfg.fillcfg(fn)
                self._cleanup_fill(fn, cfg, ssa)
            if scalar.adce(fn):
                self._cleanup_fill(fn, cfg, ssa)
            threaded = thread.run(fn)
            merged = cse.domcse(fn)
            merged = cse.reassoc(fn) or merged
            merged = cse.ranges(fn) or merged
            ssa.filluse(fn)
            if T.name.startswith('arm64'):
                merged = cse.divconst(fn) or merged
                merged = cse.divrem(fn) or merged
                merged = cse.bytepatterns(fn) or merged
            ssa.filluse(fn)
            alias.fillalias(fn)
            load.loadopt(fn)
            ssa.filluse(fn)
            ssa.ssacheck(fn)
            from . import memopt
            cfg.fillcfg(fn)
            ssa.filluse(fn)
            moved = memopt.hoist(fn)
            moved = memopt.sink(fn) or moved
            moved = memopt.dse(fn) or moved
            moved = memopt.merge(fn) or moved
            cfg.fillloop(fn)
            ssa.filluse(fn)
            moved = memopt.pre(fn) or moved
            if moved:
                ssa.filluse(fn)
                ssa.ssacheck(fn)
            if merged or threaded or moved or cse.nload(fn) != loads:
                cfg.filldom(fn)
                self._cleanup(fn, cfg, ssa, gvn, gcm)
        from .. import middle
        if middle.enabled:
            # cimo's loop passes (off by default: stock QBE output otherwise), then QBE's
            # cleanup passes again over the new code
            cfg.filldom(fn)
            if middle.run(fn):
                self._cleanup(fn, cfg, ssa, gvn, gcm)
                if ext.cse and T.name.startswith('arm64') and cse.bytepatterns(fn):  # reads made by loops
                    self._cleanup(fn, cfg, ssa, gvn, gcm)
        if ext.cse and T.name.startswith('arm64'):
            cfg.fillloop(fn)
            if cse.hoistcons(fn):
                ssa.filluse(fn)
        if T.name == 'wasm':  # cimo: no machine lowering; the engine allocates registers
            D.unvolatile(fn)
            T.emitfn(fn, self.outf)
            return
        if T.cansel or (ext.isel and T.name.startswith('arm64')):
            ifopt.ifconvert(fn)
            cfg.fillcfg(fn)
            ssa.filluse(fn)
            cfg.filldom(fn)
            ssa.ssacheck(fn)
        T.abi1(fn)
        simpl.simpl(fn)
        cfg.fillcfg(fn)
        ssa.filluse(fn)
        # cimo: volatile accesses are plain ones from here on (ext.volatile)
        D.unvolatile(fn, D.TMP(_IP1) if T.name.startswith('arm64') else None)
        T.isel(fn)
        cfg.fillcfg(fn)
        live.filllive(fn)
        cfg.fillloop(fn)
        spill.fillcost(fn)
        spill.spill(fn)
        rega.rega(fn)
        cfg.fillcfg(fn)
        cfg.simpljmp(fn)
        cfg.fillcfg(fn)
        assert fn.rpo[0] is fn.start
        n = 0
        while True:
            if n == fn.nblk - 1:
                fn.rpo[n].link = None
                break
            else:
                fn.rpo[n].link = fn.rpo[n + 1]
            n += 1
        if not self.dbg:
            T.emitfn(fn, self.outf)
            self.outf.write("/* end function %s */\n\n" % fn.name)
        else:
            err.write("\n")

    def dbgfile(self, fn):
        from .emit import emitdbgfile
        if D.T.name == 'wasm':
            return
        emitdbgfile(fn, self.outf)


_usage_flags = ('-h', '-o file', '-t <target>', '-d <flags>')


def main(argv=None):
    import getopt
    argv = sys.argv[1:] if argv is None else argv
    targets = _targets()
    D.T = targets[0]
    outf = sys.stdout
    dbg = 0
    try:
        opts, args = getopt.getopt(argv, "hd:o:t:")
    except getopt.GetoptError:
        opts, args = [('-?', '')], []
    for c, optarg in opts:
        if c == '-d':
            for ch in optarg:
                if ch.isalpha():
                    D.debug[ch.upper()] = 1
                    dbg = 1
        elif c == '-o':
            if optarg != '-':
                try:
                    outf = open(optarg, 'w')
                except OSError:
                    sys.stderr.write("cannot open '%s'\n" % optarg)
                    return 1
        elif c == '-t':
            if optarg == '?':
                print(D.T.name)
                return 0
            for t in targets:
                if t.name == optarg:
                    D.T = t
                    break
            else:
                sys.stderr.write("unknown target '%s'\n" % optarg)
                return 1
        else:
            hf = sys.stdout if c == '-h' else sys.stderr
            hf.write("qbe [OPTIONS] {file.ssa, -}\n")
            hf.write("\t%-11s prints this help\n" % "-h")
            hf.write("\t%-11s output to file\n" % "-o file")
            hf.write("\t%-11s generate for a target among:\n" % "-t <target>")
            hf.write("\t%-11s " % "")
            hf.write(", ".join(t.name + (" (default)" if t is targets[0] else "")
                               for t in targets))
            hf.write("\n")
            hf.write("\t%-11s dump debug information\n" % "-d <flags>")
            return int(c != '-h')

    drv = _Driver(outf, dbg)
    files = args or ['-']
    try:
        for f in files:
            if f == '-':
                buf = sys.stdin.buffer.read()
            else:
                try:
                    with open(f, 'rb') as fh:
                        buf = fh.read()
                except OSError:
                    sys.stderr.write("cannot open '%s'\n" % f)
                    return 1
            parse(buf, f, drv.dbgfile, drv.data, drv.func)
        drv.finish()
        if not dbg:
            D.T.emitfin(outf)
    except D.BackendError as e:
        outf.flush()
        sys.stderr.write("%s\n" % e)
        return 1
    finally:
        outf.flush()
    return 0
