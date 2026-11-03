"""cimo.cpu: this machine's x86-64 features, read with cpuid, and their place in the cache key."""

import pytest

from cimo import cpu, driver
import hosts


@pytest.mark.skipif(not hosts.AMD64, reason='reads an x86-64 CPU')
def test_detects_a_consistent_set():
    f = cpu.x86_features()
    assert f <= cpu.KNOWN
    assert ('fma' not in f and 'avx2' not in f) or 'avx' in f  # both need the OS to save ymm


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
    monkeypatch.setattr(driver, '_x86', lambda t=None: frozenset())
    a = driver._key('int f(void) { return 1; }')
    monkeypatch.setattr(driver, '_x86', lambda t=None: frozenset(('fma',)))
    b = driver._key('int f(void) { return 1; }')
    assert a != b
