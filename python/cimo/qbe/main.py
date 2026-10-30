"""Port of QBE main.c: the qbe command line.

python -m cimo.qbe [-d flags] [-o file] [-t target] {file.ssa, -}
"""

import sys

from . import defs as D
from .parse import parse, printfn


def _targets():
    from .arm64 import targ
    return [targ.T_arm64, targ.T_arm64_apple]


class _Driver:
    def __init__(self, outf, dbg):
        self.outf = outf
        self.dbg = dbg

    def data(self, d):
        from .emit import emitdat
        if self.dbg:
            return
        emitdat(d, self.outf)
        if d.type == D.DEnd:
            self.outf.write("/* end data */\n\n")

    def func(self, fn):
        T = D.T
        err = sys.stderr
        if self.dbg:
            err.write("**** Function %s ****" % fn.name)
        if D.debug['P']:
            err.write("\n> After parsing:\n")
            printfn(fn, err)
        from . import cfg, ssa, mem, alias, load, copy, gvn, gcm, ifopt, simpl, live, spill, rega
        T.abi0(fn)
        cfg.fillcfg(fn)
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
        if T.cansel:
            ifopt.ifconvert(fn)
            cfg.fillcfg(fn)
            ssa.filluse(fn)
            cfg.filldom(fn)
            ssa.ssacheck(fn)
        T.abi1(fn)
        simpl.simpl(fn)
        cfg.fillcfg(fn)
        ssa.filluse(fn)
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
        if not dbg:
            D.T.emitfin(outf)
    except D.QbeError as e:
        outf.flush()
        sys.stderr.write("%s\n" % e)
        return 1
    finally:
        outf.flush()
    return 0
