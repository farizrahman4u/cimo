/* stdlib.h: strtol family and strtod edge cases, atoi, qsort, bsearch, abs, div. */
#include <stdlib.h>
#include <string.h>
#include "out.h"

static const char *ints[] = {
	"0", "-0", "+0", "42", "  \t\n-17xyz", "0x1f", "0X1F", "0x", "0xg", "017", "08", "-0x8000",
	"9223372036854775807", "9223372036854775808", "-9223372036854775808",
	"-9223372036854775809", "18446744073709551615", "18446744073709551616",
	"99999999999999999999999", "-1", "zz", "  +", "-", "", "   ", "1e5", "0b101", "z1",
	"2147483647", "2147483648", "-2147483649", "4294967296", "0x7fffffffffffffff",
	"0xffffffffffffffffff", "123abc", "\xff" "1",
};
static const int bases[] = { 0, 2, 8, 10, 16, 36 };  /* others are undefined */

static const char *flts[] = {
	"0", "-0", "1", "-1.5", "  +3.25e2x", ".5", "5.", ".", "e5", "1e", "1e+", "1e-2", "1E400",
	"-1e400", "1e-400", "4.9406564584124654e-324", "2.4703282292062327e-324",
	"2.4703282292062328e-324", "2.2250738585072011e-308", "2.2250738585072014e-308",
	"1.7976931348623157e308", "1.7976931348623158e308", "1.797693134862315807e308",
	"0.1", "0.30000000000000004", "123456789012345678901234567890", "9007199254740993",
	"9007199254740992.5", "0x1p0", "0x1.8p1", "-0x.8", "0x1p-1074", "0x1p-1075", "0x1.0000000000001p0",
	"0x1.00000000000008p0", "0x1.00000000000008000001p0", "0x1.00000000000018p0",
	"0x1.000000000000180000000000000000001p0", "0x1.00000000000007ffffffffp0",
	"0x8.00000000000040000001p-3", "0x1.000001p0", "0x1.0000010000000000001p0", "0x", "0xp1", "inf", "-INF",
	"Infinity", "infin", "nan", "NAN(123)", "nan(", "nan()x", "1.5e-310", "  -.0e0", "1e-5000",
	"0.000000000000000000000000000000000000001e39", "3.14159265358979323846264338327950288",
	"1e23", "8.589973e9", "2.2250738585072012e-308", "6.9294956446009195e15",
	"1.00000005960464477539062499", "1.000000059604644775390625", "3.4028235677973366e38",
	"1.1754943508222875e-38", "1.4e-45", "7.0064923216240854e-46", "-0x1.fffffep127",
};

/* NaN payloads from "nan(...)" are implementation-defined */
static void o_df(double d)
{
	if (d != d) o_s("nan");
	else o_f(d);
}

static void o_dff(float d)
{
	if (d != d) o_s("nan");
	else o_ff(d);
}

static int cmpint(const void *a, const void *b)
{
	int x = *(const int *)a, y = *(const int *)b;
	return (x > y) - (x < y);
}

struct big { int key; char pad[300]; };

static int cmpbig(const void *a, const void *b)
{
	return cmpint(&((const struct big *)a)->key, &((const struct big *)b)->key);
}

static int arr[3000];
static struct big bigs[40];

long run(char *out, long cap)
{
	o_init(out, cap);
	char *end;
	for (unsigned i = 0; i < sizeof ints / sizeof *ints; i++)
		for (unsigned j = 0; j < sizeof bases / sizeof *bases; j++) {
			int b = bases[j];
			o_s("strtol ");
			o_d(i); o_c(' '); o_d(b); o_c(' ');
			o_d(strtol(ints[i], &end, b)); o_c(' '); o_d(end - ints[i]); o_c(' ');
			o_x(strtoul(ints[i], &end, b)); o_c(' '); o_d(end - ints[i]); o_c(' ');
			o_d(strtoll(ints[i], &end, b)); o_c(' '); o_d(end - ints[i]); o_c(' ');
			o_x(strtoull(ints[i], &end, b)); o_c(' '); o_d(end - ints[i]);
			if (b == 10) {
				/* atoi/atol: only where the value fits (overflow is undefined) */
				long long v = strtoll(ints[i], 0, 10);
				o_c(' ');
				if (v >= -2147483647 - 1 && v <= 2147483647) o_d(atoi(ints[i]));
				o_c(' ');
				if (strlen(ints[i]) < 19) o_d(atol(ints[i]));
			}
			o_c(' ');
			o_d(strtol(ints[i], 0, b));
			o_c('\n');
		}
	for (unsigned i = 0; i < sizeof flts / sizeof *flts; i++) {
		o_s("strtod ");
		o_d(i); o_c(' ');
		o_df(strtod(flts[i], &end));
		o_c(' ');
		o_d(end - flts[i]);
		o_c(' ');
		o_dff(strtof(flts[i], &end));
		o_c(' ');
		o_d(end - flts[i]);
		o_c(' ');
		o_df(atof(flts[i]));
		o_c('\n');
	}
	/* random decimal strings: round trip of digits near halfway points */
	for (int i = 0; i < 2000; i++) {
		char s[64];
		int n = 1 + o_rand() % 25, k = 0;
		if (o_rand() & 1) s[k++] = '-';
		for (int d = 0; d < n; d++) {
			s[k++] = '0' + o_rand() % 10;
			if (d == 0 && n > 1) s[k++] = '.';
		}
		s[k++] = 'e';
		int e = (int)(o_rand() % 700) - 350;
		if (e < 0) s[k++] = '-', e = -e;
		if (e >= 100) s[k++] = '0' + e / 100;
		if (e >= 10) s[k++] = '0' + e / 10 % 10;
		s[k++] = '0' + e % 10;
		s[k] = 0;
		o_f(strtod(s, 0));
		o_c(' ');
		o_ff(strtof(s, 0));
		o_c(i % 8 == 7 ? '\n' : ' ');
	}
	/* qsort, bsearch */
	for (int t = 0; t < 6; t++) {
		int n = t == 0 ? 0 : t == 1 ? 1 : t == 2 ? 17 : t == 3 ? 1000 : 3000;
		for (int i = 0; i < n; i++)
			arr[i] = t == 5 ? i / 3 : (int)(o_rand() % (t == 4 ? 50 : 100000)) - 500;
		if (t == 5) for (int i = 0; i < n / 2; i++) { int x = arr[i]; arr[i] = arr[n - 1 - i]; arr[n - 1 - i] = x; }
		qsort(arr, n, sizeof *arr, cmpint);
		int ok = 1;
		long h = 0;
		for (int i = 0; i < n; i++) {
			if (i && arr[i - 1] > arr[i]) ok = 0;
			h = h * 31 + arr[i];
		}
		o_s("qsort "); o_d(n); o_c(' '); o_d(ok); o_c(' '); o_d(h);
		for (int k = -600; k < 100600 && n; k += 9973) {
			int *p = bsearch(&k, arr, n, sizeof *arr, cmpint);
			o_c(' ');
			o_d(p ? *p == k : -1);
		}
		o_c('\n');
	}
	for (int i = 0; i < 40; i++) {
		bigs[i].key = (int)(o_rand() % 1000);
		memset(bigs[i].pad, bigs[i].key, sizeof bigs[i].pad);
	}
	qsort(bigs, 40, sizeof *bigs, cmpbig);
	o_s("qsortbig");
	for (int i = 0; i < 40; i++) {
		o_c(' ');
		o_d(bigs[i].key);
		o_d(bigs[i].pad[0] == (char)bigs[i].key && bigs[i].pad[299] == (char)bigs[i].key);
	}
	o_c('\n');
	/* abs, div */
	{
		static const long long v[] = { 0, 1, -1, 7, -7, 2147483647, -2147483647, 100, -100 };
		static const long long w[] = { 1, -1, 3, -3, 7, -7 };
		for (int i = 0; i < 9; i++) {
			o_s("abs "); o_d(abs((int)v[i])); o_c(' '); o_d(labs(v[i])); o_c(' ');
			o_d(llabs(v[i] * 4294967296LL));
			for (int j = 0; j < 6; j++) {
				div_t d = div((int)v[i], (int)w[j]);
				ldiv_t l = ldiv(v[i], w[j]);
				lldiv_t q = lldiv(v[i] * 3, w[j]);
				o_c(' '); o_d(d.quot); o_c(','); o_d(d.rem);
				o_c(' '); o_d(l.quot); o_c(','); o_d(l.rem);
				o_c(' '); o_d(q.quot); o_c(','); o_d(q.rem);
			}
			o_c('\n');
		}
	}
	return o_end();
}
