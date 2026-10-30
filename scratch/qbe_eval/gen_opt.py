"""Emit hand-optimized QBE IR for a few kernels: what cimo's loop passes are meant to produce.

Transformations applied by hand: loop rotation, pointer induction variables (strength
reduction), invariant hoisting, 4x unrolling with a scalar remainder loop, and scalar
replacement of neighbouring loads across iterations (stencils). FP evaluation order is kept
exactly as in kernels.c (no reassociation, no FMA).
"""
import itertools

_ids = itertools.count()


def tmp(prefix="t"):
    return f"%{prefix}{next(_ids)}"


def addr(out, ptr, off):
    if off == 0:
        return ptr
    a = tmp("a")
    out.append(f"\t{a} =l add {ptr}, {off}")
    return a


def emit_loop(out, p, pred, ptrs, carried, body, guard, end4, end1, exit_label, unroll=4):
    """Unrolled main loop + scalar remainder loop, both bottom-tested.

    ptrs: list of (name, init_value, stride_bytes); the loop advances each pointer.
    carried: list of (name, init_value); body returns their next values.
    body(env, k) -> new carried dict; env maps pointer names to current SSA values.
    The caller must end its current block (label `pred`) right before this call.
    """
    def run(label, prev_label, prev_vals, n, cmp_end, next_label):
        out.append(f"@{label}")
        cur = {}
        for name, _, _ in ptrs:
            cur[name] = tmp(name)
        for name, _ in carried:
            cur[name] = tmp(name)
        nxt = {}
        lines_at = len(out)
        env = dict(cur)
        for k in range(n):
            env = dict(env, **body(out, env, k))
        for name, _, stride in ptrs:
            nxt[name] = tmp(name)
            out.append(f"\t{nxt[name]} =l add {cur[name]}, {stride * n}")
        for name, _ in carried:
            nxt[name] = env[name]
        phis = [f"\t{cur[v]} ={'l' if v in [q for q, _, _ in ptrs] else 's'} phi @{prev_label} {prev_vals[v]}, @{label} {nxt[v]}"
                for v in cur]
        out[lines_at:lines_at] = phis
        c = tmp("c")
        out.append(f"\t{c} =w cultl {nxt[guard]}, {cmp_end}")
        out.append(f"\tjnz {c}, @{label}, @{next_label}")
        return cur, nxt

    init = {name: v for name, v, _ in ptrs} | dict(carried)
    c = tmp("c")
    out.append(f"\t{c} =w cultl {init[guard]}, {end4}")
    out.append(f"\tjnz {c}, @{p}_l4, @{p}_tc")
    cur4, nxt4 = run(f"{p}_l4", pred, init, unroll, end4, f"{p}_tc")
    # remainder-check block: merge values from "skipped main loop" and "after main loop"
    out.append(f"@{p}_tc")
    tc = {}
    for v in cur4:
        tc[v] = tmp(v)
        kind = "l" if v in [q for q, _, _ in ptrs] else "s"
        out.append(f"\t{tc[v]} ={kind} phi @{pred} {init[v]}, @{p}_l4 {nxt4[v]}")
    c = tmp("c")
    out.append(f"\t{c} =w cultl {tc[guard]}, {end1}")
    out.append(f"\tjnz {c}, @{p}_l1, @{exit_label}")
    run(f"{p}_l1", f"{p}_tc", tc, 1, end1, exit_label)


def saxpy():
    out = ["export function $saxpy(w %n, s %fa, l %x, l %y) {", "@start",
           "\t%ok =w csgtw %n, 0", "\tjnz %ok, @pre, @end", "@pre",
           "\t%nl =l extsw %n", "\t%nb =l mul %nl, 4", "\t%n4 =l and %nl, -4", "\t%n4b =l mul %n4, 4",
           "\t%yend =l add %y, %nb", "\t%yend4 =l add %y, %n4b"]

    def body(o, env, k):
        xv, yv, m, s = tmp("xv"), tmp("yv"), tmp("m"), tmp("s")
        o.append(f"\t{xv} =s loads {addr(o, env['xp'], 4 * k)}")
        ya = addr(o, env["yp"], 4 * k)
        o.append(f"\t{yv} =s loads {ya}")
        o.append(f"\t{m} =s mul %fa, {xv}")
        o.append(f"\t{s} =s add {m}, {yv}")
        o.append(f"\tstores {s}, {ya}")
        return {}

    emit_loop(out, "j", "pre", [("xp", "%x", 4), ("yp", "%y", 4)], [], body, "yp", "%yend4", "%yend", "end")
    out += ["@end", "\tret", "}"]
    return out


def gemm():
    out = ["export function $gemm(w %n, l %a, l %b, l %c) {", "@start",
           "\t%ok =w csgtw %n, 0", "\tjnz %ok, @pre, @end", "@pre",
           "\t%nl =l extsw %n", "\t%rb =l mul %nl, 4", "\t%n4 =l and %nl, -4", "\t%n4b =l mul %n4, 4",
           "\t%cb =l mul %rb, %nl", "\t%cend =l add %c, %cb", "\t%rb2 =l add %rb, %rb",
           "@iloop",
           "\t%crow =l phi @pre %c, @inext %crow2", "\t%arow =l phi @pre %a, @inext %arow2",
           "\t%crowe =l add %crow, %rb", "\t%crowe4 =l add %crow, %n4b", "\t%arowe =l add %arow, %rb",
           # unroll-and-jam: k in pairs, so each c[j] is loaded/stored once per two updates.
           # c = (c + a0*b0[j]) + a1*b1[j] is exactly the sequential C evaluation order.
           "\t%arowe1 =l sub %arowe, 4",
           "\t%has2 =w cultl %arow, %arowe1", "\tjnz %has2, @kloop2, @kchk1",
           "@kloop2",
           "\t%ap =l phi @iloop %arow, @knext2 %ap2", "\t%brow =l phi @iloop %b, @knext2 %brow2",
           "\t%a0 =s loads %ap", "\t%ap1 =l add %ap, 4", "\t%a1 =s loads %ap1",
           "\t%brow1 =l add %brow, %rb"]

    def body2(o, env, k):
        cv, b0, b1, m0, m1, s0, s1 = (tmp(x) for x in ("cv", "b", "b", "m", "m", "s", "s"))
        ca = addr(o, env["cp"], 4 * k)
        o.append(f"\t{cv} =s loads {ca}")
        o.append(f"\t{b0} =s loads {addr(o, env['b0p'], 4 * k)}")
        o.append(f"\t{m0} =s mul %a0, {b0}")
        o.append(f"\t{s0} =s add {cv}, {m0}")
        o.append(f"\t{b1} =s loads {addr(o, env['b1p'], 4 * k)}")
        o.append(f"\t{m1} =s mul %a1, {b1}")
        o.append(f"\t{s1} =s add {s0}, {m1}")
        o.append(f"\tstores {s1}, {ca}")
        return {}

    emit_loop(out, "j2", "kloop2", [("cp", "%crow", 4), ("b0p", "%brow", 4), ("b1p", "%brow1", 4)], [],
              body2, "cp", "%crowe4", "%crowe", "knext2")
    out += ["@knext2", "\t%ap2 =l add %ap, 8", "\t%brow2 =l add %brow, %rb2",
            "\t%ck2 =w cultl %ap2, %arowe1", "\tjnz %ck2, @kloop2, @kchk1",
            # odd n: one k left
            "@kchk1",
            "\t%apr =l phi @iloop %arow, @knext2 %ap2", "\t%browr =l phi @iloop %b, @knext2 %brow2",
            "\t%ck1 =w cultl %apr, %arowe", "\tjnz %ck1, @kone, @inext",
            "@kone", "\t%aik =s loads %apr"]

    def body1(o, env, k):
        cv, bv, m, s = tmp("cv"), tmp("bv"), tmp("m"), tmp("s")
        ca = addr(o, env["cp"], 4 * k)
        o.append(f"\t{cv} =s loads {ca}")
        o.append(f"\t{bv} =s loads {addr(o, env['bp'], 4 * k)}")
        o.append(f"\t{m} =s mul %aik, {bv}")
        o.append(f"\t{s} =s add {cv}, {m}")
        o.append(f"\tstores {s}, {ca}")
        return {}

    emit_loop(out, "j1", "kone", [("cp", "%crow", 4), ("bp", "%browr", 4)], [], body1, "cp", "%crowe4", "%crowe", "inext")
    out += ["@inext", "\t%crow2 =l add %crow, %rb", "\t%arow2 =l add %arow, %rb",
            "\t%ci =w cultl %crow2, %cend", "\tjnz %ci, @iloop, @end",
            "@end", "\tret", "}"]
    return out


def jacobi2d():
    # rows i = 1..n-2, cols j = 1..n-2; rotate src[j-1], src[j] across iterations
    out = ["export function $jacobi2d(w %n, l %src, l %dst) {", "@start",
           "\t%ok =w csgtw %n, 2", "\tjnz %ok, @pre, @end", "@pre",
           "\t%nl =l extsw %n", "\t%rb =l mul %nl, 4", "\t%m =l sub %nl, 2", "\t%m4 =l and %m, -4",
           "\t%m4b =l mul %m4, 4", "\t%mb =l mul %m, 4",
           "\t%m1 =l sub %nl, 1", "\t%last =l mul %rb, %m1", "\t%midend =l add %src, %last", "\t%mid0 =l add %src, %rb",
           "\t%drow0 =l add %dst, %rb",
           "@iloop",
           "\t%mid =l phi @pre %mid0, @inext %mid2", "\t%drow =l phi @pre %drow0, @inext %drow2",
           "\t%up =l sub %mid, %rb", "\t%dn =l add %mid, %rb",
           "\t%mp0 =l add %mid, 4", "\t%up0 =l add %up, 4", "\t%dn0 =l add %dn, 4", "\t%op0 =l add %drow, 4",
           "\t%prev0 =s loads %mid", "\t%cur0 =s loads %mp0",
           "\t%oe4 =l add %op0, %m4b", "\t%oe =l add %op0, %mb"]

    def body(o, env, k):
        nx, u, d, t1, t2, t3, t4, r = (tmp(s) for s in ("nx", "u", "d", "t", "t", "t", "t", "r"))
        o.append(f"\t{nx} =s loads {addr(o, env['mp'], 4 * k + 4)}")
        o.append(f"\t{u} =s loads {addr(o, env['upp'], 4 * k)}")
        o.append(f"\t{d} =s loads {addr(o, env['dnp'], 4 * k)}")
        o.append(f"\t{t1} =s add {env['cur']}, {env['prev']}")
        o.append(f"\t{t2} =s add {t1}, {nx}")
        o.append(f"\t{t3} =s add {t2}, {u}")
        o.append(f"\t{t4} =s add {t3}, {d}")
        o.append(f"\t{r} =s mul s_0.2, {t4}")
        o.append(f"\tstores {r}, {addr(o, env['op'], 4 * k)}")
        return {"prev": env["cur"], "cur": nx}

    emit_loop(out, "j", "iloop", [("mp", "%mp0", 4), ("upp", "%up0", 4), ("dnp", "%dn0", 4), ("op", "%op0", 4)],
              [("prev", "%prev0"), ("cur", "%cur0")], body, "op", "%oe4", "%oe", "inext")
    out += ["@inext", "\t%mid2 =l add %mid, %rb", "\t%drow2 =l add %drow, %rb",
            "\t%ci =w cultl %mid2, %midend", "\tjnz %ci, @iloop, @end",
            "@end", "\tret", "}"]
    return out


def conv3x3():
    out = ["export function $conv3x3(w %h, w %w, l %img, l %k, l %out) {", "@start",
           "\t%okh =w csgtw %h, 2", "\t%okw =w csgtw %w, 2", "\t%ok =w and %okh, %okw",
           "\tjnz %ok, @pre, @end", "@pre"]
    for i in range(9):
        out.append(f"\t%k{i} =s loads {addr(out, '%k', 4 * i)}")
    out += ["\t%wl =l extsw %w", "\t%hl =l extsw %h", "\t%rb =l mul %wl, 4", "\t%m =l sub %wl, 2",
            "\t%m4 =l and %m, -4", "\t%m4b =l mul %m4, 4", "\t%mb =l mul %m, 4",
            "\t%hm =l sub %hl, 2", "\t%lastb =l mul %rb, %hm",
            "\t%r10 =l add %img, %rb", "\t%r1end =l add %img, %lastb", "\t%r1end2 =l add %r1end, %rb",
            "\t%orow0 =l add %out, %rb",
            "@yloop",
            "\t%r1 =l phi @pre %r10, @ynext %r1n", "\t%orow =l phi @pre %orow0, @ynext %orown",
            "\t%r0 =l sub %r1, %rb", "\t%r2 =l add %r1, %rb",
            "\t%p0 =l add %r0, 4", "\t%p1 =l add %r1, 4", "\t%p2 =l add %r2, 4", "\t%op0 =l add %orow, 4"]
    for r in range(3):
        out.append(f"\t%a{r}p0 =s loads %r{r}")
        out.append(f"\t%a{r}c0 =s loads %p{r}")
    out += ["\t%oe4 =l add %op0, %m4b", "\t%oe =l add %op0, %mb"]

    def body(o, env, k):
        s = "s_0.0"
        new = {}
        for r in range(3):
            nx = tmp(f"n{r}_")
            o.append(f"\t{nx} =s loads {addr(o, env[f'q{r}'], 4 * k + 4)}")
            for dx, v in enumerate((env[f"a{r}p"], env[f"a{r}c"], nx)):
                m, s2 = tmp("m"), tmp("s")
                o.append(f"\t{m} =s mul {v}, %k{3 * r + dx}")
                o.append(f"\t{s2} =s add {s}, {m}")
                s = s2
            new[f"a{r}p"], new[f"a{r}c"] = env[f"a{r}c"], nx
        o.append(f"\tstores {s}, {addr(o, env['op'], 4 * k)}")
        return new

    carried = [(f"a{r}{w}", f"%a{r}{w}0") for r in range(3) for w in "pc"]
    emit_loop(out, "x", "yloop", [("q0", "%p0", 4), ("q1", "%p1", 4), ("q2", "%p2", 4), ("op", "%op0", 4)],
              carried, body, "op", "%oe4", "%oe", "ynext")
    out += ["@ynext", "\t%r1n =l add %r1, %rb", "\t%orown =l add %orow, %rb",
            "\t%cy =w cultl %r1n, %r1end2", "\tjnz %cy, @yloop, @end",
            "@end", "\tret", "}"]
    return out


KERNELS = {"saxpy": saxpy, "gemm": gemm, "jacobi2d": jacobi2d, "conv3x3": conv3x3}

if __name__ == "__main__":
    for name, f in KERNELS.items():
        print("\n".join(f()))
        print()
