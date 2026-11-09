"""cimo extension: the CPU intrinsics the bundled arm_neon.h and immintrin.h declare.

Each is declared there with an asm label `__cimo_intrin_<name>` (so a program's own function of that name is
never taken for one), and a call of it becomes its instruction (qbe.funcintrinsic). It needs the CPU features
listed here: a call on a CPU without them is a compile error naming them (cpufeat, compile(cpufeat=): this
machine's, cimo.cpu).
"""

PREFIX = '__cimo_intrin_'

# name: (features, kind, IR op). kind 'dot': (accumulator, a, b), each 32-bit lane of the accumulator plus the
# products of its pairs (bf16) or fours (bytes) of a and b, as the q op after a qfma0 of the accumulator
INTRINSICS = {
    'vbfdotq_f32': (('bf16',), 'dot', 'qbfdot'),
    'vdotq_s32': (('dotprod',), 'dot', 'qsdot'),
    'vdotq_u32': (('dotprod',), 'dot', 'qudot'),
    'vusdotq_s32': (('i8mm',), 'dot', 'qusdot'),
    '_mm_dpbf16_ps': (('avx512bf16', 'avx512vl'), 'dot', 'qbfdot'),
    '_mm256_dpbf16_ps': (('avx512bf16', 'avx512vl'), 'dot', 'qbfdot'),
    '_mm512_dpbf16_ps': (('avx512bf16', 'avx512vl'), 'dot', 'qbfdot'),
    '_mm_dpbusd_epi32': (('avx512vnni', 'avx512vl'), 'dot', 'qusdot'),
    '_mm256_dpbusd_epi32': (('avx512vnni', 'avx512vl'), 'dot', 'qusdot'),
    '_mm512_dpbusd_epi32': (('avx512vnni', 'avx512vl'), 'dot', 'qusdot'),
    '_mm_dpbusd_avx_epi32': (('avxvnni',), 'dot', 'qusdot'),
    '_mm256_dpbusd_avx_epi32': (('avxvnni',), 'dot', 'qusdot'),
    # AMX tiles: the tile numbers are constants (TILEARGS), the instructions backend/defs.py's x* ones
    '_tile_loadconfig': (('amx-tile',), 'tilecfg', 'xtilecfg'),
    '_tile_storeconfig': (('amx-tile',), 'tilecfg', 'xtilecfgst'),
    '_tile_release': (('amx-tile',), 'tilerel', 'xtilerel'),
    '_tile_zero': (('amx-tile',), 'tilezero', 'xtilezero'),
    '_tile_loadd': (('amx-tile',), 'tileld', 'xtileld'),
    '_tile_stream_loadd': (('amx-tile',), 'tileld', 'xtileldt'),
    '_tile_stored': (('amx-tile',), 'tilest', 'xtilest'),
    '_tile_dpbf16ps': (('amx-tile', 'amx-bf16'), 'tiledp', 0),
    '_tile_dpbssd': (('amx-tile', 'amx-int8'), 'tiledp', 1),
    '_tile_dpbsud': (('amx-tile', 'amx-int8'), 'tiledp', 2),
    '_tile_dpbusd': (('amx-tile', 'amx-int8'), 'tiledp', 3),
    '_tile_dpbuud': (('amx-tile', 'amx-int8'), 'tiledp', 4),
}
# the arguments that are tile numbers, by kind
TILEARGS = {'tilezero': (0,), 'tileld': (0,), 'tilest': (0,), 'tiledp': (0, 1, 2)}

# the features of the CPU being compiled for (compile(cpufeat=)); empty: no intrinsic is available
cpufeat = frozenset()


def lookup(decl):
    """The INTRINSICS entry of a function declaration, or None."""
    a = decl.asmname
    if not a or not a.startswith(PREFIX) or decl.defined:
        return None
    return INTRINSICS.get(a[len(PREFIX):])


def missing(name):
    """An error message if this CPU lacks what intrinsic `name` needs, else None."""
    need = INTRINSICS[name][0]
    lack = [f for f in need if f not in cpufeat]
    if not lack:
        return None
    return "'%s' needs the CPU feature%s %s, which %s" % (
        name, 's' if len(need) > 1 else '', ' and '.join(need),
        'this CPU lacks' if len(lack) == len(need) else 'this CPU lacks (%s)' % ', '.join(lack))
