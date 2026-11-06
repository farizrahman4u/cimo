"""x86-64 code generation, loading and entry, checked on any host where it doesn't need to run."""

import os
import re
import subprocess
import sys
import textwrap

import pytest

import cimo
import hosts
from cimo import assembler_amd64, driver, execution, linker, loader
from cimo.frontend import qbe

TARGETS = ('amd64_sysv', 'amd64_windows')


def asm(src, target):
    ssa, params = driver._frontend(src, 't.c', target=target, fma=False, nowrap=True, builtins=True)
    return driver._qbe(ssa, params=params, target=target)


def test_the_assembler_is_part_of_the_compiler_key():
    assert any(p.endswith('assembler_amd64.py') for p in linker._compiler_paths())


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('src', [
    'long a[10]; long f(void) { return a[0x100000000]; }',
    'long a[10]; long f(long i) { return a[i + 0x100000000]; }',
    'static long a[10]; long f(void) { return a[0x100000000]; }',
    'static long a[10]; long *f(void) { return a + 0x100000000; }',
    'static long a[10]; void f(long v) { a[0x100000000] = v; }',
])
def test_an_offset_a_relocation_cannot_hold_goes_in_a_register(src, target):
    text = asm(src, target)
    assert '$34359738368' in text and 'addq $34359738368' not in text
    assert 'a+34359738368' not in text
    assembler_amd64.assemble(text)


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('src, want', [
    ('long a[10]; long f(void) { return a[2]; }', r'movq 16\(%\w+\), %\w+'),
    ('long a[10]; void f(long v) { a[2] = v; }', r'movq %\w+, 16\(%\w+\)'),
])
def test_the_offset_of_a_global_is_a_displacement(src, want, target):
    text = asm(src, target)
    assert re.search(want, text) and 'addq' not in text


@pytest.mark.parametrize('decl, want', [
    ('struct s { long a; double b; }', ['i', 's']),
    ('struct s { double a; long b; }', ['s', 'i']),
    ('struct s { float a, b; int c; }', ['s', 'i']),
    ('struct s { float a; int b; }', ['i']),
    ('union s { float a; int b; }', ['i']),
    ('struct s { double a[2]; }', ['s', 's']),
    ('struct s { int f : 3; double d; }', ['i', 's']),
    ('struct s { _Alignas(16) char c; }', ['i', 'n']),
    ('struct s { long a, b, c; }', None),
    ('struct s { char big[1 << 20]; }', None),
    ('struct __attribute__((packed)) s { char c; long l; }', None),
])
def test_va_arg_classes_are_the_backends(decl, want, monkeypatch):
    seen = []
    real = qbe._sysv_classes
    monkeypatch.setattr(qbe, '_sysv_classes', lambda t: seen.append(real(t)) or seen[-1])
    pp = decl + ''';
    typedef __builtin_va_list va_list;
    struct s2 { int x; };
    void f(int n, ...) { __builtin_va_list ap; __builtin_va_start(ap, n);
        %s s v = __builtin_va_arg(ap, %s s); (void)v; __builtin_va_end(ap); }
    ''' % ((decl.split()[0],) * 2)
    driver._frontend(pp, 't.c', target='amd64_sysv', fma=False, builtins=True)
    assert seen == [want]


def test_a_32_bit_python_is_refused(monkeypatch):
    monkeypatch.setattr(loader, '_64BIT', False)
    with pytest.raises(driver.CompileError, match='64-bit Python'):
        driver._check_machine()


@pytest.mark.parametrize('windows', [False, True])
def test_entry_keeps_calls_and_returns_paired(windows):
    text = execution._glue_amd64(['f'], windows)
    assert 'jmp\t*%r10' not in text
    assert text.count(execution._RDSSP_R10) == 1 and text.count(execution._INCSSP_RCX) == 1
    assembler_amd64.assemble(text)


def test_windows_frames_fit_their_page():
    assert execution.WFRAMES + execution.MAXDEPTH * execution.WFRAMESZ <= 12288


HOST_X86_LINUX = hosts.AMD64 and sys.platform.startswith('linux')


@pytest.mark.skipif(not hosts.AMD64 or sys.platform == 'darwin', reason='runs x86-64 code')
def test_a_global_in_the_image_is_reached_without_the_got():
    m = cimo.compile('long a[4] = {1, 2, 3, 4}; long *p(void) { return a; } long f(void) { return a[2]; }',
                     cache=False)
    assert m.f() == 3


def _shstk():
    try:
        with open('/proc/cpuinfo') as f:
            return re.search(r'^flags.*\buser_shstk\b', f.read(), re.M) is not None
    except OSError:
        return False


# Enables the shadow stack, which can't be returned out of, and exits the process from there:
#   rdi = ok, rsi = deep; ok(41) must be 42; deep(n) unwinds; ok(41) must still be 42.
# Each call goes through a helper that returns, since an entry that leaves the shadow stack
# unbalanced only faults at its caller's next ret.
_SHSTK_CHILD = r'''
import ctypes, mmap, os, sys
import cimo
m = cimo.compile("""
int ok(int x) { return x + 1; }
int deep(int n);
int (*volatile again)(int) = deep;
int deep(int n) { if (n == 0) __builtin_trap(); return again(n - 1) + 1; }
""", cache=False)
assert m.ok(1) == 2  # this thread's context exists from here
def blob():
    head = ('4989fc' '4989f5'      # mov %rdi, %r12; mov %rsi, %r13
            'b89e000000'           # mov $158, %eax      arch_prctl
            'bf01500000'           # mov $0x5001, %edi   ARCH_SHSTK_ENABLE
            'be01000000'           # mov $1, %esi        ARCH_SHSTK_SHSTK
            '0f05' '85c0'          # syscall; test %eax, %eax
            'bf4d000000')          # mov $77, %edi       not enabled: skip
    steps = [('bf29000000', '4c89e0', '83f82a' 'bf01000000'),  # 41, %r12; cmp $42; code 1
             ('bf2c010000', '4c89e8', ''),                     # 300, %r13
             ('bf29000000', '4c89e0', '83f82a' 'bf02000000')]  # 41, %r12; cmp $42; code 2
    tail = 'bf00000000'                                        # mov $0, %edi
    leave = 'b8e7000000' '0f05'                                # exit_group
    helper = '4883ec08' 'ffd0' '4883c408' 'c3'                 # sub $8, %rsp; call *%rax; add; ret
    align = '4883e4f0'                                         # and $-16, %rsp
    size = lambda h: len(h) // 2
    body = [size(a) + size(r) + 5 + size(c) + (2 if c else 0) for a, r, c in steps]
    to_leave = size(align) + sum(body) + size(tail)
    out = head + '75%02x' % to_leave + align
    pos = size(align)
    for (a, r, c), n in zip(steps, body):
        pos += size(a) + size(r) + 5
        rel = to_leave + size(leave) - pos
        out += a + r + 'e8' + rel.to_bytes(4, 'little').hex() + c
        pos += size(c)
        if c:
            pos += 2
            out += '75%02x' % (to_leave - pos)
    return bytes.fromhex(out + tail + leave + helper)


code = blob()
buf = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
buf.write(code)
addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(addr)
sys.stdout.flush()
fn(m.ok._image.address('__cimo_entry_ok'), m.deep._image.address('__cimo_entry_deep'))
'''


@pytest.mark.skipif(not HOST_X86_LINUX or not _shstk(), reason='needs user shadow stacks')
def test_entry_and_unwind_under_a_shadow_stack():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    p = subprocess.run([sys.executable, '-c', textwrap.dedent(_SHSTK_CHILD)], env=env,
                       capture_output=True, text=True, timeout=120)
    if p.returncode == 77:
        pytest.skip('the kernel did not enable the shadow stack')
    assert p.returncode == 0, (p.returncode, p.stderr[-2000:])


def test_the_machine_is_this_process_on_windows():
    if sys.platform != 'win32':
        pytest.skip('Windows')
    assert ('AMD64' in sys.version) == loader._X86 and ('ARM64' in sys.version) == loader._ARM64


def _asm86(src, target='amd64_sysv'):
    old = driver._x86
    driver._x86 = lambda t=None: frozenset()
    try:
        pp = driver._preprocess(src, 'x.c', (), False, (), target=target)
        ssa, params = driver._ir(pp, 'x.c')
        return driver._qbe(ssa, params, target=target)
    finally:
        driver._x86 = old


def test_jnz_does_not_test_a_64_bit_value_s_flags():
    # gcc pr94412: a lane loaded as w from the 64-bit `or` that stored both lanes; jnz tests 32 bits, so the
    # `or`'s flags (of 64) can't stand in for the test
    text = _asm86('typedef unsigned V __attribute__((vector_size(8)));\n'
                  'void bar(V *v, V *w) { *w = -18 / -*v; }\n'
                  'int f(void) { V b = {3, 0x7fffffff}, d; bar(&b, &d); if (d[0] != 0) return 1; return 0; }\n')
    body = text[text.index('f:'):]
    assert not re.search(r'\torq [^\n]*\n(\tmov[^\n]*\n)*\tj(nz|z) ', body), body
    assembler_amd64.assemble(text)


@pytest.mark.parametrize('target', ['amd64_sysv', 'amd64_windows'])
def test_big_struct_by_value_copies_with_memcpy(target):
    # gcc pr23135: a 128 KB struct passed by value 8 times was unrolled into 9.6 MB of moves (2 minutes)
    src = ('typedef struct { char c[131072]; } big;\n'
           'void verify(int a, int b, big x);\n'
           'big g(big *p) { verify(1, 2, *p); verify(3, 4, *p); return *p; }\n')
    import time
    t0 = time.perf_counter()
    text = _asm86(src, target)
    assert time.perf_counter() - t0 < 20
    assert text.count('memcpy') >= 3 and len(text) < 20000, len(text)
    assembler_amd64.assemble(text)
