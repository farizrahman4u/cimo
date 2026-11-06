"""Port of QBE amd64/winabi.c (the Windows x64 calling convention).

The main job of amd64_winabi_abi is to lower generic instructions into the
specific details of how arguments are passed, and parameters are
interpreted for win x64. A useful reference is
https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention .

Some of the major differences from SysV (non-exhaustive):
- only 4 int and 4 float regs are used
- when an int register is assigned a value, its associated float register is
  left unused (and vice versa), i.e. there's only one counter as you assign
  arguments to registers.
- any structs that aren't 1/2/4/8 bytes in size are passed by pointer, not
  by copying them into the stack; the pointer to a copy is treated as a regular
  integer argument.
- when calling a varargs functions, floating point values must be duplicated
  to integer registers.
"""

import sys

from .. import defs as D
from ..defs import (R, RTmp, NIns, Ins, Kw, Kl, KBASE, BIT, TMP, INT, CALL, SLOT, rtype,
                    rval, Ostorel, Oadd, Oload, Ocopy, Ocall, Ocast, Oblit0, Oblit1, Oalloc8,
                    Osalloc, Oarg, Opar, Oargc, Oparc, Oarge, Opare, Oargv, Ovastart, Ovaarg,
                    Jret0, Jretc, Jretw, isret, isarg, ispar)
from ..util import err, die, newtmp, getcon, emit, emiti, idup, icpy
from ..parse import printfn
from .regs import (RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11, RBX, R12, R13, R14, R15, RBP,
                   XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11,
                   XMM12, XMM13, XMM14, NGPS_WIN, NCLR_WIN, NFPS)

(APS_Invalid, APS_Register, APS_InlineOnStack, APS_CopyAndPointerInRegister,
 APS_CopyAndPointerOnStack, APS_VarargsTag, APS_EnvTag) = range(7)


class ArgClass:
    __slots__ = ('type', 'style', 'align', 'size', 'cls', 'ref')

    def __init__(self):  # alloc() / {0}
        self.type = None
        self.style = APS_Invalid
        self.align = 0
        self.size = 0
        self.cls = 0
        self.ref = R


class ExtraAlloc:
    __slots__ = ('instr', 'link')

    def __init__(self, instr, link):
        self.instr = instr
        self.link = link


def ALIGN_DOWN(n, a):
    return n & ~(a - 1)


def ALIGN_UP(n, a):
    return ALIGN_DOWN(n + a - 1, a)


# Number of stack bytes required be reserved for the callee.
SHADOW_SPACE_SIZE = 32

amd64_winabi_rsave = [RCX, RDX, R8, R9, R10, R11, RAX, XMM0,
                      XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8,
                      XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, -1]
amd64_winabi_rclob = [RBX, R12, R13, R14, R15, RSI, RDI, -1]

assert (len(amd64_winabi_rsave) == NGPS_WIN + NFPS + 1
        and len(amd64_winabi_rclob) == NCLR_WIN + 1)

# cimo's amd64_windows target (targ.T_amd64_windows): xmm6-xmm15 are callee-saved, as the
# Windows x64 convention says (QBE's amd64_win treats them as scratch, which corrupts a
# caller's xmm6+ and so Python's); the emitter saves the ones a function uses, and xmm15
# when it uses it as scratch
WIN_XCLOB = [XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14]
amd64_windows_rsave = [RCX, RDX, R8, R9, R10, R11, RAX,
                       XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, -1]
NFPS_WINDOWS = 6

# layout of call's second argument (RCall)
#
# bit 0: rax returned
# bit 1: xmm0 returned
# bits 23: 0
# bits 4567: rcx, rdx, r8, r9 passed
# bits 89ab: xmm0,1,2,3 passed
# bit c: env call (rax passed)
# bits d..1f: 0


def amd64_winabi_retregs(r, p):
    assert rtype(r) == D.RCall
    b = 0
    num_int_returns = rval(r) & 1
    num_float_returns = rval(r) & 2
    if num_int_returns == 1:
        b |= BIT(RAX)
    else:
        b |= BIT(XMM0)
    if p is not None:
        p[0] = num_int_returns
        p[1] = num_float_returns
    return b


def popcnt(b):
    return bin(b & 0xffffffffffffffff).count('1') & 0xff


def amd64_winabi_argregs(r, p):
    assert rtype(r) == D.RCall
    # On SysV, these are counts. Here, a count isn't sufficient, we actually need
    # to know which ones are in use because they're not necessarily contiguous.
    int_passed = (rval(r) >> 4) & 15
    float_passed = (rval(r) >> 8) & 15
    env_param = (rval(r) >> 12) & 1

    b = 0
    b |= BIT(RCX) if int_passed & 1 else 0
    b |= BIT(RDX) if int_passed & 2 else 0
    b |= BIT(R8) if int_passed & 4 else 0
    b |= BIT(R9) if int_passed & 8 else 0
    b |= BIT(XMM0) if float_passed & 1 else 0
    b |= BIT(XMM1) if float_passed & 2 else 0
    b |= BIT(XMM2) if float_passed & 4 else 0
    b |= BIT(XMM3) if float_passed & 8 else 0
    b |= BIT(RAX) if env_param else 0
    if p is not None:
        # TODO: The only place this is used is live.c. I'm not sure what should be
        # returned here wrt to using the same counter for int/float regs on win.
        # For now, try the number of registers in use even though they're not
        # contiguous.
        p[0] = popcnt(int_passed)
        p[1] = popcnt(float_passed)
    return b


class RegisterUsage:
    __slots__ = ('num_regs_passed', 'regs_passed', 'rax_returned', 'xmm0_returned',
                 'num_named_args_passed', 'is_varargs_call', 'has_env')

    def __init__(self):
        # Counter for both int/float as they're counted together.
        self.num_regs_passed = 0
        # Indexed first by 0=int, 1=float, use KBASE(cls); second by register index in
        # calling convention (RCX, RDX, R8, R9 / XMM0..XMM3).
        self.regs_passed = [[False] * 4, [False] * 4]
        self.rax_returned = False
        self.xmm0_returned = False
        # also where va_start will start for varargs functions
        self.num_named_args_passed = 0
        # set when classifying the arguments for a call (not for parameters)
        self.is_varargs_call = False
        self.has_env = False


def register_usage_to_call_arg_value(u):
    rp = u.regs_passed
    return ((int(u.rax_returned) << 0)
            | (int(u.xmm0_returned) << 1)
            | (int(rp[0][0]) << 4)
            | (int(rp[0][1]) << 5)
            | (int(rp[0][2]) << 6)
            | (int(rp[0][3]) << 7)
            | (int(rp[1][0]) << 8)
            | (int(rp[1][1]) << 9)
            | (int(rp[1][2]) << 10)
            | (int(rp[1][3]) << 11)
            | (int(u.has_env) << 12))


def assign_register_or_stack(u, arg, is_float, by_copy):
    """Assigns the argument to a register if there's any left according to the
    calling convention, and updates the regs_passed bools. Otherwise marks the
    value as needing stack space to be passed."""
    if u.num_regs_passed == 4:
        arg.style = APS_CopyAndPointerOnStack if by_copy else APS_InlineOnStack
    else:
        u.regs_passed[is_float][u.num_regs_passed] = True
        u.num_regs_passed += 1
        arg.style = APS_CopyAndPointerInRegister if by_copy else APS_Register
    u.num_named_args_passed += 1


def type_is_by_copy(type):
    # Note that only these sizes are passed by register, even though e.g. a
    # 5 byte struct would "fit", it still is passed by copy-and-pointer.
    return bool(type.isdark or type.size not in (1, 2, 4, 8))


def classify_arguments(u, ins, begin, end, arg_classes, env):
    """ins[begin:end] are the Oarg or Opar instructions; env: a one-element list."""
    for k in range(begin, end):
        instr = ins[k]
        arg = arg_classes[k - begin]
        op = instr.op
        if op == Oarg or op == Opar:
            assign_register_or_stack(u, arg, KBASE(instr.cls), False)
            arg.cls = instr.cls
            arg.align = 3
            arg.size = 8
        elif op == Oargc or op == Oparc:
            type = D.typ[rval(instr.arg[0])]
            by_copy = type_is_by_copy(type)
            assign_register_or_stack(u, arg, False, by_copy)
            arg.cls = Kl
            if not by_copy and type.size <= 4:
                arg.cls = Kw
            arg.align = 3
            arg.size = type.size
        elif op == Oarge:
            env[0] = instr.arg[0]
            arg.style = APS_EnvTag
            u.has_env = True
        elif op == Opare:
            env[0] = instr.to
            arg.style = APS_EnvTag
            u.has_env = True
        elif op == Oargv:
            u.is_varargs_call = True
            arg.style = APS_VarargsTag

    if u.has_env and u.is_varargs_call:
        die("can't use env with varargs")

    # During a varargs call, float arguments have to be duplicated to their
    # associated integer register, so mark them as in-use too.
    if u.is_varargs_call:
        for i in range(4):
            if u.regs_passed[1][i]:
                u.regs_passed[0][i] = True


def is_integer_type(ty):
    assert 0 <= ty < 4, "expecting Kw Kl Ks Kd"
    return KBASE(ty) == 0


def register_for_arg(cls, counter):
    assert counter < 4
    if is_integer_type(cls):
        return TMP(amd64_winabi_rsave[counter])
    else:
        return TMP(XMM0 + counter)


def lower_call(func, block, call_idx, pextra_alloc):
    """block.ins[call_idx] is the Ocall; returns the index of the instruction before its
    arguments. pextra_alloc: a one-element list (ExtraAlloc **)."""
    ins = block.ins
    call_instr = ins[call_idx]
    # Call arguments are instructions. Walk through them to find the end of the
    # call+args that we need to process.
    instr_past_args = call_idx - 1
    while instr_past_args >= 0:
        if not isarg(ins[instr_past_args].op):
            break
        instr_past_args -= 1
    earliest_arg_instr = instr_past_args + 1

    num_args = call_idx - earliest_arg_instr
    arg_classes = [ArgClass() for _ in range(num_args)]

    reg_usage = RegisterUsage()
    ret_arg_class = ArgClass()

    # Ocall's arg[1], if the function returns a non-basic type, is a reference to
    # the type of the return.
    il_has_struct_return = call_instr.arg[1] != R
    is_struct_return = False
    if il_has_struct_return:
        ret_type = D.typ[rval(call_instr.arg[1])]
        is_struct_return = type_is_by_copy(ret_type)
        if is_struct_return:
            assign_register_or_stack(reg_usage, ret_arg_class, False, True)
        ret_arg_class.size = ret_type.size
    env = [R]
    classify_arguments(reg_usage, ins, earliest_arg_instr, call_idx, arg_classes, env)

    # We now know which arguments are on the stack and which are in registers, so
    # we can allocate the correct amount of space to stash the stack-located ones
    # into.
    stack_usage = 0
    for arg in arg_classes:
        # stack_usage only accounts for pushes that are for values that don't have
        # enough registers. Large struct copies are alloca'd separately, and then
        # only have (potentially) 8 bytes to add to stack_usage here.
        if arg.style == APS_InlineOnStack:
            if arg.align > 4:
                err("win abi cannot pass alignments > 16")
            # fix over C QBE (arg.size, 1 to 8 for a small struct): every argument on the
            # stack takes an 8-byte slot
            stack_usage += 8
        elif arg.style == APS_CopyAndPointerOnStack:
            stack_usage += 8
    stack_usage = ALIGN_UP(stack_usage, 16) & 0xffffffff

    # Note that here we're logically 'after' the call (due to emitting
    # instructions in reverse order), so we're doing a negative stack
    # allocation to clean up after the call.
    stack_size_ref = getcon(-(stack_usage + SHADOW_SPACE_SIZE), func)
    emit(Osalloc, Kl, R, stack_size_ref, R)

    return_pad = None
    if is_struct_return:
        ret_pad_ref = newtmp("abi.ret_pad", Kl, func)
        return_pad = ExtraAlloc(
            Ins(Oalloc8, Kl, ret_pad_ref, getcon(ret_arg_class.size, func)), pextra_alloc[0])
        pextra_alloc[0] = return_pad
        reg_usage.rax_returned = True
        emit(Ocopy, call_instr.cls, call_instr.to, TMP(RAX), R)
    else:
        if il_has_struct_return:
            # At the IL level a struct return was specified, but as far as the
            # calling convention is concerned it's not actually by pointer: store the
            # return value into an alloca, since subsequent IL treats it as a pointer.
            return_copy = ExtraAlloc(
                Ins(Oalloc8, Kl, call_instr.to, getcon(8, func)), pextra_alloc[0])
            pextra_alloc[0] = return_copy
            copy = newtmp("abi.copy", Kl, func)
            emit(Ostorel, 0, R, copy, call_instr.to)
            emit(Ocopy, Kl, copy, TMP(RAX), R)
            reg_usage.rax_returned = True
        elif is_integer_type(call_instr.cls):
            # Only a basic type returned from the call, integer.
            emit(Ocopy, call_instr.cls, call_instr.to, TMP(RAX), R)
            reg_usage.rax_returned = True
        else:
            # Basic type, floating point.
            emit(Ocopy, call_instr.cls, call_instr.to, TMP(XMM0), R)
            reg_usage.xmm0_returned = True

    # Emit the actual call instruction.
    emit(Ocall, call_instr.cls, R, call_instr.arg[0],
         CALL(register_usage_to_call_arg_value(reg_usage)))

    if env[0] != R:
        # If there's an env arg to be passed, it gets stashed in RAX.
        emit(Ocopy, Kl, TMP(RAX), env[0], R)

    if reg_usage.is_varargs_call:
        # Any float arguments need to be duplicated to integer registers.
        for index, floatreg, intreg in ((0, XMM0, RCX), (1, XMM1, RDX), (2, XMM2, R8),
                                        (3, XMM3, R9)):
            if reg_usage.regs_passed[1][index]:
                emit(Ocast, Kl, TMP(intreg), TMP(floatreg), R)

    reg_counter = 0
    if is_struct_return:
        first_reg = register_for_arg(Kl, reg_counter)
        reg_counter += 1
        emit(Ocopy, Kl, first_reg, return_pad.instr.to, R)

    # This is where we actually do the load of values into registers or into
    # stack slots.
    arg_stack_slots = newtmp("abi.args", Kl, func)
    slot_offset = SHADOW_SPACE_SIZE
    bigcopies = []
    for k in range(earliest_arg_instr, call_idx):
        instr = ins[k]
        arg = arg_classes[k - earliest_arg_instr]
        st = arg.style
        if st == APS_Register:
            into = register_for_arg(arg.cls, reg_counter)
            reg_counter += 1
            if instr.op == Oargc:
                # a small struct passed by value: the instruction's value is a pointer,
                # but it needs to be loaded into the register.
                emit(Oload, arg.cls, into, instr.arg[1], R)
            else:
                # Otherwise, a normal value passed in a register.
                emit(Ocopy, instr.cls, into, instr.arg[0], R)
        elif st == APS_InlineOnStack:
            slot = newtmp("abi.off", Kl, func)
            if instr.op == Oargc:
                # a small struct, not passed by copy, but the instruction is a
                # pointer: copy it into the stack slot (emitted backwards).
                smalltmp = newtmp("abi.smalltmp", arg.cls, func)
                emit(Ostorel, 0, R, smalltmp, slot)
                emit(Oload, arg.cls, smalltmp, instr.arg[1], R)
            else:
                # Stash the value into the stack slot.
                emit(Ostorel, 0, R, instr.arg[0], slot)
            emit(Oadd, Kl, slot, arg_stack_slots, getcon(slot_offset, func))
            slot_offset += 8  # fix over C QBE (arg.size), as above
        elif st in (APS_CopyAndPointerInRegister, APS_CopyAndPointerOnStack):
            # Alloca a space to copy into, and blit the value from the instr to the
            # copied location.
            copy_ref = newtmp("abi.copy", Kl, func)
            arg_copy = ExtraAlloc(Ins(Oalloc8, Kl, copy_ref, getcon(arg.size, func)),
                                  pextra_alloc[0])
            pextra_alloc[0] = arg_copy
            from .. import ext as _ext
            if _ext.bigcopy is not None and arg.size >= _ext.bigcopy:
                bigcopies.append((copy_ref, instr.arg[1], arg.size))  # memcpy before the arguments, below
            else:
                emit(Oblit1, 0, R, INT(arg.size), R)
                emit(Oblit0, 0, R, instr.arg[1], copy_ref)

            # Now load the pointer into the correct register or stack slot.
            if st == APS_CopyAndPointerInRegister:
                into = register_for_arg(arg.cls, reg_counter)
                reg_counter += 1
                emit(Ocopy, Kl, into, copy_ref, R)
            else:
                slot = newtmp("abi.off", Kl, func)
                emit(Ostorel, 0, R, copy_ref, slot)
                emit(Oadd, Kl, slot, arg_stack_slots, getcon(slot_offset, func))
                slot_offset += 8
        elif st in (APS_EnvTag, APS_VarargsTag):
            # Nothing to do here, see right before the call for reg dupe.
            pass
        else:
            die("unreachable")

    if stack_usage:
        # The last (first in call order) thing we do is allocate the the stack
        # space we're going to fill with temporaries.
        emit(Osalloc, Kl, arg_stack_slots,
             getcon(stack_usage + SHADOW_SPACE_SIZE, func), R)
    else:
        # When there's no usage for temporaries, emit it separately (not storing
        # into a reference) so that it doesn't get removed later for being useless.
        emit(Osalloc, Kl, R, getcon(SHADOW_SPACE_SIZE, func), R)
    for dst, src, size in bigcopies:  # cimo: before the call's own stack and registers
        _memcpycall(func, dst, src, size)

    return instr_past_args


def _memcpycall(func, dst, src, size):
    """cimo: memcpy(dst, src, size) as a lowered call with its shadow space (emitted backwards): an unrolled blit
    of a big struct passed or returned by value took minutes (gcc's pr23135: 128 KB, 8 calls)."""
    from ..arm64.abi import _memcpy
    func.leaf = 0
    emit(Osalloc, Kl, R, getcon(-SHADOW_SPACE_SIZE, func), R)
    emit(Ocopy, Kl, R, TMP(RAX), R)  # spill and rega expect a copy after a call
    emit(Ocall, 0, R, _memcpy(func), CALL(1 | 1 << 4 | 1 << 5 | 1 << 6))  # rax; rcx, rdx, r8
    emit(Ocopy, Kl, TMP(R8), getcon(size, func), R)
    emit(Ocopy, Kl, TMP(RDX), src, R)
    emit(Ocopy, Kl, TMP(RCX), dst, R)
    emit(Osalloc, Kl, R, getcon(SHADOW_SPACE_SIZE, func), R)


def lower_block_return(func, block):
    jmp_type = block.jmp.type

    if not isret(jmp_type) or jmp_type == Jret0:
        return

    # Save the argument, and set the block to be a void return because once it's
    # lowered it's handled by the the register/stack manipulation.
    ret_arg = block.jmp.arg
    block.jmp.type = Jret0

    u = RegisterUsage()

    if jmp_type == Jretc:
        type = D.typ[func.retty]
        if type_is_by_copy(type):
            assert rtype(func.retr) == RTmp
            emit(Ocopy, Kl, TMP(RAX), func.retr, R)
            from .. import ext as _ext
            if _ext.bigcopy is not None and type.size >= _ext.bigcopy:
                _memcpycall(func, func.retr, ret_arg, type.size)
            else:
                emit(Oblit1, 0, R, INT(type.size), R)
                emit(Oblit0, 0, R, ret_arg, func.retr)
        else:
            emit(Oload, Kl, TMP(RAX), ret_arg, R)
        u.rax_returned = True
    else:
        k = jmp_type - Jretw
        if is_integer_type(k):
            emit(Ocopy, k, TMP(RAX), ret_arg, R)
            u.rax_returned = True
        else:
            emit(Ocopy, k, TMP(XMM0), ret_arg, R)
            u.xmm0_returned = True
    block.jmp.arg = CALL(register_usage_to_call_arg_value(u))


def lower_vastart(func, param_reg_usage, valist):
    assert func.vararg
    # In varargs functions the int registers are already dumped to the shadow
    # space, float parameters have been duplicated to the integer registers, and
    # there is always an rbp frame: the ... arguments are just indexed past rbp by
    # the number of named values that were actually passed.
    offset = newtmp("abi.vastart", Kl, func)
    emit(Ostorel, 0, R, offset, valist)

    # *8 for sizeof(u64), +16 because the return address and rbp have been pushed
    # by the time we get to the body of the function.
    emit(Oadd, Kl, offset, TMP(RBP),
         getcon(param_reg_usage.num_named_args_passed * 8 + 16, func))


def lower_vaarg(func, vaarg_instr):
    # va_list is just a void** on winx64, so load the pointer, then load the
    # argument from that pointer, then increment the pointer to the next arg.
    # (All emitted backwards as usual.)
    inc = newtmp("abi.vaarg.inc", Kl, func)
    ptr = newtmp("abi.vaarg.ptr", Kl, func)
    emit(Ostorel, 0, R, inc, vaarg_instr.arg[0])
    emit(Oadd, Kl, inc, ptr, getcon(8, func))
    emit(Oload, vaarg_instr.cls, vaarg_instr.to, ptr, R)
    emit(Oload, Kl, ptr, vaarg_instr.arg[0], R)


def lower_args_for_block(func, block, param_reg_usage, pextra_alloc):
    D.curi = NIns

    lower_block_return(func, block)

    if block.nins:
        # Work backwards through the instructions, either copying them unchanged,
        # or modifying as necessary.
        ins = block.ins
        k = block.nins - 1
        while k >= 0:
            instr = ins[k]
            op = instr.op
            if op == Ocall:
                k = lower_call(func, block, k, pextra_alloc)
            elif op == Ovastart:
                lower_vastart(func, param_reg_usage, instr.arg[0])
                k -= 1
            elif op == Ovaarg:
                lower_vaarg(func, instr)
                k -= 1
            elif op == Oarg or op == Oargc:
                die("unreachable")
            else:
                emiti(instr)
                k -= 1

    # This it the start block, which is processed last. Add any allocas that
    # other blocks needed.
    if block is func.start:
        ea = pextra_alloc[0]
        while ea:
            emiti(ea.instr)
            ea = ea.link

    idup(block, D.curi, NIns - D.curi)


def lower_func_parameters(func):
    """Copy from registers/stack into values."""
    start_block = func.start
    ins = start_block.ins
    end_of_params = 0
    while end_of_params < start_block.nins:
        if not ispar(ins[end_of_params].op):
            break
        end_of_params += 1

    num_params = end_of_params
    arg_classes = [ArgClass() for _ in range(num_params)]
    arg_ret = ArgClass()

    D.curi = NIns

    reg_counter = 0
    u = RegisterUsage()
    if func.retty >= 0:
        by_copy = type_is_by_copy(D.typ[func.retty])
        if by_copy:
            assign_register_or_stack(u, arg_ret, False, by_copy)
            ret_ref = newtmp("abi.ret", Kl, func)
            emit(Ocopy, Kl, ret_ref, TMP(RCX), R)
            func.retr = ret_ref
            reg_counter += 1
    env = [R]
    classify_arguments(u, ins, 0, end_of_params, arg_classes, env)
    func.reg = amd64_winabi_argregs(CALL(register_usage_to_call_arg_value(u)), None)

    # Copy from the registers or stack slots into the named parameters.
    slot_offset = SHADOW_SPACE_SIZE // 4 + 4
    for k in range(end_of_params):
        instr = ins[k]
        arg = arg_classes[k]
        st = arg.style
        if st == APS_Register:
            frm = register_for_arg(arg.cls, reg_counter)
            reg_counter += 1
            # If it's a struct at the IL level, copy the register into an alloca so
            # we have something to point at (same for InlineOnStack).
            if instr.op == Oparc:
                arg.ref = newtmp("abi", Kl, func)
                emit(Ostorel, 0, R, arg.ref, instr.to)
                emit(Ocopy, instr.cls, arg.ref, frm, R)
                emit(Oalloc8, Kl, instr.to, getcon(arg.size, func), R)
            else:
                emit(Ocopy, instr.cls, instr.to, frm, R)
        elif st == APS_InlineOnStack:
            if instr.op == Oparc:
                arg.ref = newtmp("abi", Kl, func)
                emit(Ostorel, 0, R, arg.ref, instr.to)
                emit(Ocopy, instr.cls, arg.ref, SLOT(-slot_offset), R)
                emit(Oalloc8, Kl, instr.to, getcon(arg.size, func), R)
            else:
                # fix over C QBE (a Kl copy, which put a float parameter in an integer
                # register: `addss %ecx, %xmm0`): loaded in its own class, as sysv does
                emit(Oload, instr.cls, instr.to, SLOT(-slot_offset), R)
            slot_offset += 2
        elif st == APS_CopyAndPointerOnStack:
            emit(Oload, Kl, instr.to, SLOT(-slot_offset), R)
            slot_offset += 2
        elif st == APS_CopyAndPointerInRegister:
            # Because this has to be a copy (that we own), it is sufficient to just
            # copy the register to the target.
            frm = register_for_arg(Kl, reg_counter)
            reg_counter += 1
            emit(Ocopy, Kl, instr.to, frm, R)
        elif st == APS_EnvTag:
            pass
        else:
            die("unreachable")

    # If there was an `env`, it was passed in RAX, so copy it into the env ref.
    if env[0] != R:
        emit(Ocopy, Kl, env[0], TMP(RAX), R)

    num_created_instrs = NIns - D.curi
    num_other_after_instrs = start_block.nins - num_params
    new_instrs = []
    icpy(new_instrs, 0, D.insb, D.curi, num_created_instrs)
    icpy(new_instrs, num_created_instrs, ins, end_of_params, num_other_after_instrs)
    start_block.nins = num_other_after_instrs + num_created_instrs
    start_block.ins = new_instrs

    return u


def amd64_winabi_abi(func):
    # The first thing to do is lower incoming parameters to this function.
    param_reg_usage = lower_func_parameters(func)

    # Walk all blocks, and rewrite returns, calls, and varargs handling into their
    # win x64 versions. The entry block is done last, so that it can receive the
    # allocas the other blocks need.
    extra_alloc = [None]
    block = func.start.link
    while block:
        lower_args_for_block(func, block, param_reg_usage, extra_alloc)
        block = block.link
    lower_args_for_block(func, func.start, param_reg_usage, extra_alloc)

    if D.debug['A']:
        sys.stderr.write("\n> After ABI lowering:\n")
        printfn(func, sys.stderr)
