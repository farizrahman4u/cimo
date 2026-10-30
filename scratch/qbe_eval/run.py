"""Build kernels.c with each compiler config, link against harness.c, run, and compare."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
# third-party clones live next to the cimo repo: ~/code/qbe (branch cimo-arm64-isel), ~/code/cproc
CPROC_DIR = os.environ.get("CIMO_CPROC_DIR", os.path.expanduser("~/code/cproc"))
QBE = os.path.join(os.environ.get("CIMO_QBE_DIR", os.path.expanduser("~/code/qbe")), "qbe")
CPROC_QBE = os.path.join(CPROC_DIR, "cproc-qbe")

GCC_CONFIGS = {
    "gcc-O2": ["-O2"],
    "gcc-O3-novec": ["-O3", "-mcpu=native", "-fno-tree-vectorize"],
    # QBE never fuses mul+add, so this is the like-for-like scalar baseline
    "gcc-O3-novec-nofma": ["-O3", "-mcpu=native", "-fno-tree-vectorize", "-ffp-contract=off"],
    "gcc-O3": ["-O3", "-mcpu=native"],
}


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def build_gcc(name, flags):
    obj = os.path.join(BUILD, f"kernels-{name}.o")
    sh(["gcc", *flags, "-c", os.path.join(HERE, "kernels.c"), "-o", obj])
    return obj


def build_qbe():
    ssa = os.path.join(BUILD, "kernels.ssa")
    asm = os.path.join(BUILD, "kernels-qbe.s")
    obj = os.path.join(BUILD, "kernels-qbe.o")
    # cproc-qbe does not preprocess; the cproc driver adds the defines glibc headers need
    pre = sh([os.path.join(CPROC_DIR, "cproc"), "-E", os.path.join(HERE, "kernels.c")]).stdout
    t0 = time.perf_counter()
    ir = subprocess.run([CPROC_QBE], input=pre, check=True, capture_output=True, text=True).stdout
    t1 = time.perf_counter()
    open(ssa, "w").write(ir)
    sh([QBE, "-o", asm, ssa])
    t2 = time.perf_counter()
    sh(["gcc", "-c", asm, "-o", obj])
    print(f"cproc frontend {1e3 * (t1 - t0):.1f} ms, qbe backend {1e3 * (t2 - t1):.1f} ms", file=sys.stderr)
    return obj


def build_qbe_opt(qbe=QBE, tag="qbe-opt"):
    """cproc's IR with the kernels from gen_opt.py replaced by their hand-optimized IR."""
    import re
    sys.path.insert(0, HERE)
    import gen_opt

    ir = open(os.path.join(BUILD, "kernels.ssa")).read()
    for name in gen_opt.KERNELS:
        ir = re.sub(r"export\nfunction (\w+ )?\$" + name + r"\(.*?\n}\n", "", ir, flags=re.S)
    ir += "\n" + "\n\n".join("\n".join(f()) for f in gen_opt.KERNELS.values()) + "\n"
    ssa = os.path.join(BUILD, f"kernels-{tag}.ssa")
    asm = os.path.join(BUILD, f"kernels-{tag}.s")
    obj = os.path.join(BUILD, f"kernels-{tag}.o")
    open(ssa, "w").write(ir)
    sh([qbe, "-o", asm, ssa])
    sh(["gcc", "-c", asm, "-o", obj])
    return obj


def run(obj, name):
    exe = os.path.join(BUILD, f"bench-{name}")
    sh(["gcc", "-O2", os.path.join(HERE, "harness.c"), obj, "-lm", "-o", exe])
    out = {}
    for line in sh([exe]).stdout.splitlines():
        k, ms, cs = line.split()
        out[k] = (float(ms), float(cs))
    return out


def main():
    os.makedirs(BUILD, exist_ok=True)
    objs = {n: build_gcc(n, f) for n, f in GCC_CONFIGS.items()}
    objs["qbe"] = build_qbe()
    objs["qbe-opt"] = build_qbe_opt()
    results = {}
    for n, o in objs.items():
        print(f"running {n}...", file=sys.stderr)
        results[n] = run(o, n)

    cols = list(objs)
    ref = results["gcc-O3"]
    # ratios are against the like-for-like scalar baseline (no vectorization, no FMA)
    base = "gcc-O3-novec-nofma"
    short = {c: c.replace("gcc-", "").replace("-novec", "nv").replace("-nofma", "nf") for c in cols}
    print(f"{'kernel':<14}" + "".join(f"{short[c]:>11}" for c in cols) + f"{'qbe/base':>10}{'opt/base':>10}  check")
    for k in ref:
        row = f"{k:<14}" + "".join(f"{results[c][k][0]:>9.1f}ms" for c in cols)
        b = results[base][k][0]
        bad = [c for c in cols if abs(results[c][k][1] - ref[k][1]) > 1e-3 * max(1.0, abs(ref[k][1]))]
        print(row + f"{results['qbe'][k][0] / b:>9.2f}x{results['qbe-opt'][k][0] / b:>9.2f}x  "
              + ("ok" if not bad else "DIFF:" + ",".join(bad)))


if __name__ == "__main__":
    main()
