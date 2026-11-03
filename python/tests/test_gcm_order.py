"""gcm's order within a block (backend/gcm.py schedblk): an instruction gcm moved there waits for what it reads, and
nothing pinned (a call, a load, a store) is pulled up to it. Found by Csmith once small functions were inlined: gvn
moves the pure instructions of a branch that folded away to the start block, simplcfg merges the block of a call into
it, and QBE's order pulled the call, and then a load, up past stores they read."""

import ctypes

import pytest

import cimo
import hosts

pytestmark = pytest.mark.skipif(not hosts.RUNS, reason='runs C')

SRC = '''#include <stdint.h>
#include <string.h>
static int32_t sm(int32_t si1, int32_t si2) {
  return (((si1 > 0) && (si2 > 0) && (si1 > (INT32_MAX / si2))) || ((si1 > 0) && (si2 <= 0) && (si2 < (INT32_MIN / si1))) ||
          ((si1 <= 0) && (si2 > 0) && (si1 < (INT32_MIN / si2))) || ((si1 <= 0) && (si2 <= 0) && (si1 != 0) && (si2 < (INT32_MAX / si1)))) ? si1 : si1 * si2;
}
static int64_t sml(int64_t si1, int64_t si2) {
  return (((si1 > 0) && (si2 > 0) && (si1 > (INT64_MAX / si2))) || ((si1 > 0) && (si2 <= 0) && (si2 < (INT64_MIN / si1))) ||
          ((si1 <= 0) && (si2 > 0) && (si1 < (INT64_MIN / si2))) || ((si1 <= 0) && (si2 <= 0) && (si1 != 0) && (si2 < (INT64_MAX / si1)))) ? si1 : si1 * si2;
}
int32_t g;
int32_t f(char *buf) { buf[0] = 'a'; buf[1] = 0; return sm((g = 0xD34CAA8EL), (int32_t)strlen(buf) - 3); }
int64_t f2(int32_t *p, char *buf) { *p = 7; buf[0] = 'a'; buf[1] = 'b'; buf[2] = 0; return sml(-5, (int64_t)*p * (int64_t)strlen(buf) - 20); }
'''


def test_a_call_and_a_load_stay_after_the_stores_they_read():
    m = cimo.compile(SRC, cache=False)
    m.f2.restype = ctypes.c_int64
    b = ctypes.create_string_buffer(8)
    assert m.f(b) == ctypes.c_int32(-749950322 * -2).value  # strlen saw the store: 1 - 3
    p = (ctypes.c_int32 * 1)(0)
    assert m.f2(p, b) == -5 * (7 * 2 - 20)  # *p and strlen after their stores: 7 * 2 - 20
