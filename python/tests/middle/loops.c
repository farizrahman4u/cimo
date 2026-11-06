/* Loop shapes for the middle end's differential tests (python/tests/test_middle.py).
 * Every function has the same signature; buffers have 256 valid elements below the pointer
 * and 767 above. n is 0..100, k is small (-5..5). Functions whose name ends in _alias
 * have no restrict and are also called with fb/ib pointing into fa's/ia's buffer. */
#include <limits.h>
#include <math.h>

#define ARGS int n, int k, float *fa, float *fb, int *ia, int *ib, double *da, unsigned char *ua
#define RARGS int n, int k, float *restrict fa, float *restrict fb, int *restrict ia, \
	int *restrict ib, double *restrict da, unsigned char *restrict ua

static float g[64];
static int gi[64];

double saxpy_alias(ARGS) { for (int i = 0; i < n; i++) fa[i] = 2.0f * fb[i] + fa[i]; return 0; }
double saxpy_r(RARGS) { for (int i = 0; i < n; i++) fa[i] = 2.0f * fb[i] + fa[i]; return 0; }
double reverse(RARGS) { for (int i = n - 1; i >= 0; i--) fa[i] += fb[i] * i; return 0; }
double step2(RARGS) { for (int i = 0; i < n; i += 2) ia[i] = ib[i + 1] - i; return 0; }
double step3_le(RARGS) { long s = 0; for (int i = k; i <= n; i += 3) s += ia[i] * 3 + i; return s; }
double ne_loop(RARGS) { long s = 0; for (int i = 0; i != n; i++) s += ia[i] ^ i; return s; }
double down_ne(RARGS) { long s = 0; for (int i = n; i != 0; i--) s = s * 3 + ia[i]; return s; }
double unsigned_iv(RARGS) { unsigned s = 0; for (unsigned i = 0; i < (unsigned)n; i++) s += ua[i] * i; return s; }
double uwrap_index(RARGS) { long s = 0; for (int i = 0; i < n; i++) s += ia[(unsigned)k + 16u + (unsigned)i - 16u + 5u]; return s; }
double uwrap_iv(RARGS) { long s = 0; for (unsigned i = UINT_MAX - 5; i != (unsigned)(n % 7); i++) s += ia[i + 6] * 2 + 1; return s; }
double intmax_iv(RARGS) { long s = 0; for (int i = INT_MAX - n; i < INT_MAX; i++) s += ia[i - (INT_MAX - 120)]; return s; }
double intmin_down(RARGS) { long s = 0; for (int i = INT_MIN + n; i > INT_MIN; i--) s += ib[i - (INT_MIN + 1)] - s; return s; }
double long_iv(RARGS) { for (long i = 0; i < n; i++) da[i] = da[i] * 0.5 + i; return 0; }
double long_symstep(RARGS) { int c = 0; long st = (k & 3) + 1; for (long j = st * st; j <= n; j += st) { ua[j] = 1; c++; } return c; }
double nest2(RARGS) { int m = n % 20; for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) fa[i * m + j] = fb[j * m + i] + i - j; return 0; }
double early_exit(RARGS) { long s = 0; for (int i = 0; i < n; i++) { if (ia[i] % 7 == k) break; s += ia[i]; } return s; }
double while_and(RARGS) { int i = 0; float s = 0; while (i < n && fa[i] > 0.1f) { s += fa[i]; i++; } return s + i; }
double do_while(RARGS) { long s = 0; int i = 0; do { s += ia[i]; i++; } while (i < n); return s; }
double do_while2(RARGS) { int i = n; float s = 1; do { s = s * fa[i] + fb[i]; i -= 2; } while (i > k); return s; }
double fred(RARGS) { float s = 0; for (int i = 0; i < n; i++) s += fa[i] * fb[i]; return s; }
double dred(RARGS) { double s = 0; for (int i = 0; i < n; i++) s = s * 0.75 + da[i] * fa[i]; return s; }
double cond_body(RARGS) { long s = 0; for (int i = 0; i < n; i++) { if (ia[i] > 0) s += ia[i]; else s -= 1; } return s; }
double stencil_alias(ARGS) { for (int i = 1; i < n; i++) fa[i] = fb[i - 1] + fb[i] + fb[i + 1]; return 0; }
double stencil_r(RARGS) { for (int i = 1; i < n; i++) fa[i] = fb[i - 1] + fb[i] * 0.5f + fb[i + 1]; return 0; }
double stencil5_r(RARGS) { for (int i = 2; i < n; i++) fa[i] = fb[i - 2] - fb[i + 2] + fb[i - 1] * fb[i] + fb[i + 1]; return 0; }
double stencil_gap(RARGS) { for (int i = 1; i < n; i++) fa[i] = fb[i - 1] + fb[i + 1]; return 0; }
double inv_load_alias(ARGS) { for (int i = 0; i < n; i++) fa[i] = fb[0] * 2.0f + fb[k + 3]; return 0; }
double inv_load_r(RARGS) { for (int i = 0; i < n; i++) fa[i] = fb[0] * 2.0f + fb[k + 3]; return 0; }
double global_arr(RARGS) { int m = n < 64 ? n : 64; for (int i = 0; i < m; i++) g[i] = g[i] * 0.5f + fa[i]; float s = 0; for (int i = 0; i < 64; i++) s += g[i]; return s; }
double global_int(RARGS) { int m = n < 60 ? n : 60; for (int i = 0; i < m; i++) gi[i + 2] = gi[i] + ia[i]; long s = 0; for (int i = 0; i < 64; i++) s += gi[i]; return s; }
double char_same_arr(RARGS) { for (int i = 0; i < n; i++) ua[i] = ua[i + 1] + 1; return 0; }
double short_arr(RARGS) { short *p = (short *)ib; long s = 0; for (int i = 0; i < n; i++) { p[i] = (short)(p[i + 1] * 3); s += p[i]; } return s; }
double iv_after(RARGS) { int i; long s = 0; for (i = 0; i < n; i += 3) s += ia[i]; return s * 1000 + i; }
double iir2(RARGS) { float y1 = 0, y2 = 0; for (int i = 0; i < n; i++) { float v = fa[i] - 0.5f * y1 - 0.25f * y2; fb[i] = v; y2 = y1; y1 = v; } return y1 + y2; }
double const_trip(RARGS) { float s = 0; for (int j = 0; j < 4; j++) s += fa[j] * k; return s; }
double conv_like(RARGS) { int m = n % 30; for (int x = 1; x < m - 1; x++) { float s = 0; for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) s += fa[x + dx + 30 * dy] * fb[(dy + 1) * 3 + dx + 1]; da[x] = s; } return 0; }
double bound_changes(RARGS) { long s = 0; int m = n; for (int i = 0; i < m; i++) { s += ia[i]; if (ia[i] & 1) m--; } return s; }
double iv_changes(RARGS) { long s = 0; for (int i = 0; i < n; i++) { s += ia[i]; i += ia[i] & 1; } return s; }
double two_ivs(RARGS) { for (int i = 0, j = n; i < n; i++, j--) fa[i] = fb[j - 1] * 2.0f - i; return 0; }
double ptr_loop(RARGS) { float *p = fa, *e = fa + n; while (p < e) { *p = *p * 2.0f + 1.0f; p++; } return 0; }
double ptr_ne(RARGS) { long s = 0; for (int *p = ia; p != ia + n; p++) s += *p; return s; }
double ptr_down(RARGS) { long s = 0; for (int *p = ia + n; p > ia; ) s = s * 2 + *--p; return s; }
double float_iv(RARGS) { float s = 0; for (float x = 0; x < n; x += 0.5f) s += x * fa[(int)x]; return s; }
double sym_stride(RARGS) { long s = 0; int st = (k & 3); for (int i = 0; i < n; i++) s += ia[i * st] + ia[i * st + 1]; return s; }
double gemm_like(RARGS) { int m = n % 12; for (int i = 0; i < m; i++) for (int kk = 0; kk < m; kk++) { float a = fa[i * m + kk]; for (int j = 0; j < m; j++) da[i * m + j] += a * fb[kk * m + j]; } return 0; }
double gemm_alias(ARGS) { int m = n % 12; for (int i = 0; i < m; i++) for (int kk = 0; kk < m; kk++) { float a = fa[i * m + kk]; for (int j = 0; j < m; j++) fb[i * m + j] += a * fa[kk * m + j + 1]; } return 0; }
double swap_ptrs(RARGS) { int *p = ia, *q = ib; int m = n % 40; for (int i = 0; i <= m; i++) p[i] = i; for (int r = 1; r <= 5; r++) { q[0] = r; for (int j = 1; j <= m; j++) { int a = p[j - 1] + 1, b = q[j - 1] + 2; q[j] = a < b ? a : b; } int *t = p; p = q; q = t; } return p[m]; }
double int_vs_long(RARGS) { long s = 0; for (int i = 0; i < (long)n; i++) s += ia[i]; return s; }
double udown(RARGS) { long s = 0; for (unsigned i = n; i-- > 0;) s = s * 5 + ib[i]; return s; }
double same_addr(RARGS) { for (int i = 0; i < n; i++) ia[0] += (int)fa[i]; return 0; }
double store_fwd(RARGS) { for (int i = 1; i < n; i++) ia[i] = ia[i - 1] + ib[i]; return 0; }
double load_after_store(RARGS) { for (int i = 0; i < n; i++) { ia[i + 1] = ib[i]; ib[i] = ia[i] * 2; } return 0; }
double mixed_width(RARGS) { long s = 0; for (int i = 0; i < n; i++) { short h = (short)(ia[i] * 1000); unsigned char c = (unsigned char)ib[i]; s += h * c + (long)ia[i] * ib[i]; } return s; }
double neg_index(RARGS) { long s = 0; for (int i = 0; i < n; i++) s += ia[-i] - ib[i - 100]; return s; }
double big_k_bound(RARGS) { long s = 0; for (int i = k; i < n + k; i++) s += ia[i] * i; return s; }
double rgb_like(RARGS) { for (int i = 0; i < n / 3; i++) ua[i + 300] = (unsigned char)((77 * ua[3 * i] + 150 * ua[3 * i + 1] + 29 * ua[3 * i + 2]) >> 8); return 0; }
double hist_like(RARGS) { for (int i = 0; i < 64; i++) ib[i] = 0; for (int i = 0; i < n; i++) ib[ua[i] & 63]++; return 0; }
double div_in_loop(RARGS) { long s = 0; int d = k == 0 ? 3 : k; for (int i = 0; i < n; i++) s += ia[i] / d + ia[i] % d; return s; }
double inv_div(RARGS) { long s = 0; for (int i = 0; i < n; i++) s += ia[i] + 1000 / (k == 0 ? 1 : k); return s; }
double pow2_step(RARGS) { long s = 0; for (int i = 1; i < n; i *= 2) s += ia[i]; return s; }
double dot_d(RARGS) { double s = 0; for (int i = 0; i < n; i++) s += da[i] * da[i + 1]; return s; }
double prefix(RARGS) { float s = 0; for (int i = 0; i < n; i++) { s += fa[i]; fb[i] = s; } return s; }
double cmp_le_neg(RARGS) { long s = 0; for (int i = -n; i <= k; i++) s += ia[i] + i; return s; }
double shl_index(RARGS) { long s = 0; for (int i = 0; i < n; i++) s += ia[i << 1] + ia[(i << 2) - 3 * i]; return s; }
double nested_inner_const(RARGS) { for (int i = 0; i < n; i++) { float s = 0; for (int j = 0; j < 3; j++) s += fb[i + j] * (j + 1); fa[i] = s; } return 0; }
double outer_inv(RARGS) { for (int i = 0; i < n % 25; i++) for (int j = 0; j < n % 25; j++) fa[j] += fb[i] * ib[j]; return 0; }
double jam_reuse(RARGS) { int m = n % 14; for (int i = 0; i < m; i++) for (int kk = 0; kk < m + 1; kk++) { float a = fb[kk]; for (int j = 0; j < m; j++) fa[i * 16 + j] = fa[i * 16 + j] * 0.5f + a * fb[kk * 16 + j + 20]; } return 0; }
double jam_int(RARGS) { int m = n % 13; for (int kk = 0; kk < m; kk++) { int a = ib[kk] & 7; for (int j = 0; j <= m; j++) ia[j] = ia[j] * 3 + a * ib[kk * 13 + j]; } return 0; }
double jam_alias(ARGS) { int m = n % 13; for (int kk = 0; kk < m; kk++) { int a = ib[kk] & 7; for (int j = 0; j <= m; j++) ia[j] = ia[j] * 3 + a * ib[kk * 13 + j]; } return 0; }
double ifconv_minmax(RARGS) { long s = 0; for (int i = 0; i < n; i++) { int a = ia[i], b = ib[i]; int lo = a < b ? a : b; long hi = a < b ? (long)b : (long)a; s += lo * 3 + hi; } return s; }
double fwd_same(RARGS) { for (int i = 0; i < n; i++) { ia[i] += ib[i]; ia[i] *= 3; ia[i] -= i; } return 0; }
double fwd_bytes(RARGS) { for (int i = 0; i < n; i++) { ua[i] = (unsigned char)(ia[i] + 250); ib[i] = ua[i] + (signed char)ua[i]; } return 0; }
/* fields: an address the same every iteration, read and written (kept in a register through the loop), next to
 * accesses that may or may not be the same memory */
double fields_r(RARGS) { for (int i = 0; i < n; i++) { da[0] += da[2] * fa[i]; da[1] = da[0] - fb[i]; } return da[0] + da[1]; }
double fields_alias(ARGS) { for (int i = 0; i < n; i++) { fa[-1] += fb[i]; fa[-2] = fa[-1] * 2.0f + fb[i]; } return fa[-1]; }
double fields_moving(RARGS) { for (int i = 0; i < n; i++) { ia[0] += ia[i + 1]; ia[i + 1] = ia[0] & 1023; } return ia[0]; }
double fields_int(RARGS) { for (int i = 0; i < n; i++) ia[3] = ia[3] * 3 + ib[i]; return ia[3]; }
double fields_bytes(RARGS) { for (int i = 0; i < n; i++) { ia[0] += i; ((unsigned char *)ia)[k & 3] = (unsigned char)i; } return ia[0]; }
double fields_wide(RARGS) { for (int i = 0; i < n; i++) { ia[0] += 1; ia[2] += *(long *)&ia[0] & 255; } return ia[2]; }
double fields_global(RARGS) { for (int i = 0; i < n; i++) { gi[1] += ia[i]; ia[i] = gi[1] & 255; } return gi[1]; }
/* unroll-and-jam by 4 around an inner loop that vectorizes (gemm's register block), 0 to 3 rows left over */
double jam4_r(RARGS) { int m = n % 19; for (int i = 0; i < m; i++) for (int q = 0; q < m; q++) { float a = fb[i * m + q]; for (int j = 0; j < m; j++) fa[i * m + j] += a * fb[256 + q * m + j]; } return 0; }
double jam4_int(RARGS) { int m = n % 19; for (int i = 0; i < m; i++) for (int q = 0; q < m; q++) { int a = ib[i * m + q]; for (int j = 0; j < m; j++) ia[i * m + j] += a * ib[256 + q * m + j] + q; } return 0; }
double jam4_alias(ARGS) { int m = n % 13; for (int q = 0; q < m; q++) { float a = fb[q]; for (int j = 0; j < m; j++) fa[j] += a * fb[j + q]; } return 0; }
/* a vector iteration reading what it stored */
double vfwd_r(RARGS) { for (int j = 0; j < n; j++) { fa[j] = fb[j] * 2.0f; fb[j] = fa[j] + 1.0f; fa[j] = fa[j] * fb[j]; } return 0; }
double vfwd_alias(ARGS) { for (int j = 0; j < n; j++) { fa[j] = fb[j] * 2.0f; fb[j] = fa[j] + 1.0f; fa[j] = fa[j] * fb[j]; } return 0; }
/* tiling of k in an i, k, j nest: sizes past one block (16), and nests it must leave alone */
double tile_r(RARGS) { int m = n % 20; const float *a = fb, *b = fb + m * m; for (int i = 0; i < m; i++) for (int q = 0; q < m; q++) { float s = a[i * m + q]; for (int j = 0; j < m; j++) fa[i * m + j] += s * b[q * m + j]; } return 0; }
double tile_long(RARGS) { long m = n % 20; for (long i = 0; i < m; i++) for (long q = 1; q < m + k % 3; q++) { double s = da[i * m + q % m] * 0.5; for (long j = 0; j < m; j++) fa[i * m + j] += (float)s * fb[q * m + j]; } return 0; }
double tile_2d(RARGS) { float (*c)[16] = (float (*)[16])fa; int m = n % 17; for (int i = 0; i < m; i++) for (long q = 0; q < n; q++) for (int j = 0; j < 16; j++) c[i][j] += fb[q % 40 * 16 + j] * (float)q; return 0; }
double tile_alias(ARGS) { int m = n % 13; for (long i = 0; i < m; i++) for (long q = 0; q < m; q++) for (long j = 0; j < m; j++) fa[i * m + j] += fb[q * m + j]; return 0; }
double tile_kstore(RARGS) { long m = n % 20; for (long i = 0; i < m; i++) for (long q = 0; q < m; q++) for (long j = 0; j < m; j++) fa[i * m + q] += fb[q * m + j]; return 0; }
double tile_rowshort(RARGS) { long m = n % 20; for (long i = 0; i < m; i++) for (long q = 0; q < m; q++) for (long j = 0; j < m + 1; j++) fa[i * m + j] += fb[q * m + j]; return 0; }
double tile_uwrap(RARGS) { long m = n % 9; for (long i = 0; i < m; i++) for (unsigned q = 4294967295u - 40u; q < 4294967295u - (unsigned)(k & 3); q++) for (long j = 0; j < m; j++) fa[i * m + j] += fb[(q - (4294967295u - 40u)) * 2 + j]; return 0; }
/* loops over bytes, in 16-bit lanes when no value can leave [0, 65535] */
double bytes_rgb(RARGS) { const unsigned char *s = (const unsigned char *)fb; for (int i = 0; i < n; i++) ua[i] = (unsigned char)((77 * s[3 * i] + 150 * s[3 * i + 1] + 29 * s[3 * i + 2]) >> 8); return 0; }
double bytes_mix(RARGS) { const unsigned char *s = (const unsigned char *)fb; for (int i = 0; i < n; i++) ua[i] = (unsigned char)(((s[i] * 3 + 7) >> 2) ^ (s[i] & 15) | 1); return 0; }
double bytes_edge(RARGS) { const unsigned char *s = (const unsigned char *)fb; for (int i = 0; i < n; i++) ua[i] = (unsigned char)((s[i] * 257) >> 8); return 0; }
double bytes_wide(RARGS) { const unsigned char *s = (const unsigned char *)fb; for (int i = 0; i < n; i++) ua[i] = (unsigned char)((s[i] * 300) >> 8); return 0; }
double bytes_bgr(RARGS) { const unsigned char *s = (const unsigned char *)fb + 1; for (long i = 0; i < n; i++) ua[i + k + 8] = (unsigned char)(s[3 * i + 2] + (s[3 * i] >> 1) - (s[3 * i + 1] >> 2)); return 0; }
double bytes_two(RARGS) { const unsigned char *s = (const unsigned char *)fb; for (int i = 0; i < n; i++) ua[i] = (unsigned char)(s[3 * i] + s[3 * i + 1]); return 0; }
/* outer-loop vectorization: lanes of i, each running the inner loop over j in order */
double outer_f(RARGS) { float *o = (float *)ia, *o2 = (float *)ib; int m = n % 37; for (long i = 0; i < m; i++) { float s = 1.0f, t = fa[i]; for (long j = 0; j < (n & 7) * 3; j++) { float d = fb[j] - fa[i]; s += d * d / (fa[i] * fa[i] + 1.0f); t = t * 0.5f + d; } o[i] = s; o2[i] = t; } return 0; }
double outer_d(RARGS) { double *o = (double *)ib; int m = n % 29; for (long i = 0; i < m; i++) { double s = 0; for (long j = 0; j < m + k; j++) { double d = (double)fb[j] - da[i]; s += d / sqrt(d * d + 1.0); } o[i] = s; } return 0; }
double outer_tri(RARGS) { float *o = (float *)ia; int m = n % 37; for (long i = 0; i < m; i++) { float s = 0; for (long j = 0; j < i; j++) s += fb[j] * fa[i]; o[i] = s; } return 0; }
double outer_alias(ARGS) { int m = n % 37; for (long i = 0; i < m; i++) { float s = 0; for (long j = 0; j < m; j++) s += fb[j] * fa[i]; fa[i + 1] = s; } return 0; }
/* a store the next iteration loads back, carried in a register */
double carry_r(RARGS) { for (int i = 0; i < n; i++) ia[i + 1] = ia[i] * 3 + ib[i]; return ia[n]; }
double carry_two(RARGS) { for (int i = 0; i < n; i++) { ia[i + 1] = ia[i] + ib[i]; ia[i + 2] = ia[i + 1] ^ 5; } return ia[n]; }
double carry_down(RARGS) { for (int i = n; i > 0; i--) fa[i - 1] = fa[i] * 0.5f + fb[i]; return fa[0]; }
double carry_alias(ARGS) { for (int i = 0; i < n; i++) { fa[i + 1] = fa[i] + 1.0f; fb[i] = 2.0f; } return fa[n]; }
double carry_min(RARGS) { for (int i = 1; i <= n; i++) { int v = ib[i] < ia[i - 1] + 1 ? ib[i] : ia[i - 1] + 1; ia[i] = v; } return ia[n]; }
