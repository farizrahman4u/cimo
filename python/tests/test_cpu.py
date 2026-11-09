"""cimo.cpu: this machine's x86-64 features, read with cpuid, and their place in the cache key (arm64's: test_intrinsics.py)."""

import pytest

from cimo import cpu, driver
import hosts


@pytest.mark.skipif(not hosts.AMD64, reason='reads an x86-64 CPU')
def test_detects_a_consistent_set():
    f = cpu.x86_features()
    assert f <= cpu.KNOWN
    assert ('fma' not in f and 'avx2' not in f) or 'avx' in f  # both need the OS to save ymm
    assert not f & cpu._AVX512 or 'avx512f' in f  # zmm and mask registers saved by the OS
    assert not f & cpu._AMX or 'amx-tile' in f


def test_override(monkeypatch):
    monkeypatch.setattr(cpu, '_cached', None)
    monkeypatch.setenv('CIMO_X86', 'baseline')
    assert cpu.x86_features() == frozenset()
    monkeypatch.setattr(cpu, '_cached', None)
    monkeypatch.setenv('CIMO_X86', 'nonsense')
    with pytest.raises(ValueError):
        cpu.x86_features()
    monkeypatch.setattr(cpu, '_cached', None)


def test_only_for_this_machine(monkeypatch):
    """Another target, or arm64, never gets this CPU's features."""
    monkeypatch.setattr(cpu, '_cached', frozenset(('fma',)))
    monkeypatch.setattr(driver, 'TARGET', 'amd64_sysv')
    assert driver._x86() == frozenset(('fma',))
    assert driver._x86('amd64_windows') == frozenset()
    assert driver._x86('arm64') == frozenset()


def test_features_change_the_cache_key(monkeypatch):
    monkeypatch.setattr(driver, 'TARGET', 'amd64_sysv')
    monkeypatch.setattr(driver, '_cpu', lambda t=None: frozenset())
    a = driver._key('int f(void) { return 1; }')
    monkeypatch.setattr(driver, '_cpu', lambda t=None: frozenset(('fma',)))
    b = driver._key('int f(void) { return 1; }')
    assert a != b


def test_xcr0_decides():
    """AVX-512 needs the OS to save the opmask and zmm state (XCR0 bits 5-7), AMX the tile state (17, 18)."""
    have = {'osxsave', 'avx', 'avx2', 'avx512f', 'avx512vl', 'avx512bf16', 'amx-tile', 'amx-bf16'}
    assert cpu._consistent(have, 0x60006 | 0xe0) == frozenset(have - {'osxsave'})
    assert cpu._consistent(have, 0x60006) == frozenset({'avx', 'avx2', 'amx-tile', 'amx-bf16'})
    assert cpu._consistent(have, 0xe6) == frozenset({'avx', 'avx2', 'avx512f', 'avx512vl', 'avx512bf16'})
    assert cpu._consistent(have - {'avx512f'}, 0x600e6) == frozenset({'avx', 'avx2', 'amx-tile', 'amx-bf16'})
