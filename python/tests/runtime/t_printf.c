/* stdio.h formatting: snprintf over many formats and values (compared with glibc), the return
   value with truncation, and printf/puts/putchar (their output goes to sys.stdout; in this
   file only through snprintf). */
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "out.h"

static char buf[2048];

static void line(const char *fmt, int r)
{
	o_s(fmt);
	o_s(" => ");
	o_d(r);
	o_c(' ');
	o_mem(buf, r >= 0 && r < (int)sizeof buf ? r : (long)strlen(buf));
	o_c('\n');
}

static const char *iflags[] = { "", "-", "+", " ", "#", "0", "-0", "+0", "#0", "- ", "+#", "'" };
static const char *iw[] = { "", "1", "5", "12", "*", ".0", ".3", ".12", "8.3", "*.*", "-6", ".*" };
static const long long ivals[] = { 0, 1, -1, 7, 42, -42, 255, 4096, 65535, 2147483647,
	-2147483647 - 1, 4294967295LL, 9223372036854775807LL, -9223372036854775807LL - 1,
	123456789, -987654321 };
static const double dvals[] = { 0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.5, -2.5, 0.1, 1e-5, 123456.789,
	1e15, 1e16, 1e21, 1e22, 1e100, 1.7976931348623157e308, 2.2250738585072014e-308,
	4.9406564584124654e-324, 3.14159265358979, 0.000123456789, 999.9995, 9.5, 99999.5,
	0.05, 0.15, 0.25, 0.35, 1e-300, 6.02214076e23, 1.0 / 3.0, 2.0 / 3.0 };

long run(char *out, long cap)
{
	o_init(out, cap);
	char f[64];
	/* integers */
	static const char *conv[] = { "d", "i", "u", "x", "X", "o", "ld", "lu", "lx", "lld", "llo",
		"hd", "hu", "hhd", "hhu", "hhx", "zu", "zd", "td", "jd", "ju", "c" };
	for (unsigned c = 0; c < sizeof conv / sizeof *conv; c++)
		for (unsigned fl = 0; fl < sizeof iflags / sizeof *iflags; fl++)
			for (unsigned w = 0; w < sizeof iw / sizeof *iw; w++)
				for (unsigned v = 0; v < sizeof ivals / sizeof *ivals; v += 1 + (c > 2) * 2) {
					if (conv[c][0] == 'c' && (ivals[v] < 32 || ivals[v] > 126)) continue;
					/* the ' flag is locale dependent; glibc prints no grouping in "C" */
					snprintf(f, sizeof f, "%%%s%s%s", iflags[fl], iw[w], conv[c]);
					long long x = ivals[v];
					int r;
					int nstar = !strcmp(iw[w], "*") + !strcmp(iw[w], ".*") + 2 * !strcmp(iw[w], "*.*");
					int isll = conv[c][0] == 'l' || conv[c][0] == 'z' || conv[c][0] == 't' || conv[c][0] == 'j';
					if (nstar == 0)
						r = isll ? snprintf(buf, sizeof buf, f, x) : snprintf(buf, sizeof buf, f, (int)x);
					else if (nstar == 1)
						r = isll ? snprintf(buf, sizeof buf, f, (int)(v % 9) - 3, x)
						         : snprintf(buf, sizeof buf, f, (int)(v % 9) - 3, (int)x);
					else
						r = isll ? snprintf(buf, sizeof buf, f, (int)(v % 7), (int)(v % 5) - 1, x)
						         : snprintf(buf, sizeof buf, f, (int)(v % 7), (int)(v % 5) - 1, (int)x);
					line(f, r);
				}
	/* floating point */
	static const char *fconv[] = { "f", "e", "g", "E", "G", "F" };
	static const char *fw[] = { "", "10", ".0", ".1", ".3", ".10", ".17", ".40", "15.6", "-12.3", "020.5", ".300" };
	for (unsigned c = 0; c < sizeof fconv / sizeof *fconv; c++)
		for (unsigned fl = 0; fl < 10; fl++)
			for (unsigned w = 0; w < sizeof fw / sizeof *fw; w++)
				for (unsigned v = 0; v < sizeof dvals / sizeof *dvals; v++) {
					if (fconv[c][0] == 'f' && dvals[v] > 1e30 && w >= 7) continue;
					snprintf(f, sizeof f, "%%%s%s%s", iflags[fl], fw[w], fconv[c]);
					line(f, snprintf(buf, sizeof buf, f, dvals[v]));
					line(f, snprintf(buf, sizeof buf, f, -dvals[v] * 1.000001));
				}
	{
		double sp[] = { INFINITY, -INFINITY, NAN, -NAN };
		static const char *sf[] = { "%f", "%e", "%g", "%F", "%E", "%G", "%10f", "%-10e|", "%+g",
			"% f", "%010f", "%a", "%A", "%.3a" };
		for (int i = 0; i < 4; i++)
			for (unsigned j = 0; j < sizeof sf / sizeof *sf; j++)
				line(sf[j], snprintf(buf, sizeof buf, sf[j], sp[i]));
	}
	/* hex floats: normal numbers (glibc and musl differ for subnormals) */
	{
		static const char *af[] = { "%a", "%A", "%.0a", "%.1a", "%.3a", "%.13a", "%.20a",
			"%#a", "%#.0a", "%+a", "% a", "%20a", "%-20a|", "%020a" };
		static const double av[] = { 1.0, -1.0, 0.0, -0.0, 0.5, 3.0, 0.1, 1e300, 2.2250738585072014e-308,
			1.0000000000000002, 1.9999999999999998, 0.09375, 255.0, 1.03125, 1.96875 };
		for (unsigned i = 0; i < sizeof af / sizeof *af; i++)
			for (unsigned j = 0; j < sizeof av / sizeof *av; j++)
				line(af[i], snprintf(buf, sizeof buf, af[i], av[j]));
	}
	/* strings, chars, pointers, %%, %n */
	{
		static const char *sfmt[] = { "%s", "%10s", "%-10s|", "%.3s", "%10.2s", "%-.0s|", "%*s",
			"%.*s", "[%5c]", "[%-3c]", "%%", "a%%b%%c", "%5%" };
		static const char *sv[] = { "", "a", "hello", "hello, world", "\xc3\xa9t\xc3\xa9" };
		for (unsigned i = 0; i < sizeof sfmt / sizeof *sfmt; i++)
			for (unsigned j = 0; j < sizeof sv / sizeof *sv; j++) {
				int r;
				if (strchr(sfmt[i], '*')) r = snprintf(buf, sizeof buf, sfmt[i], (int)j * 2 - 1, sv[j]);
				else if (strchr(sfmt[i], 'c')) r = snprintf(buf, sizeof buf, sfmt[i], 'A' + (int)j);
				else r = snprintf(buf, sizeof buf, sfmt[i], sv[j]);
				line(sfmt[i], r);
			}
		int n1 = -1, n2 = -1;
		signed char hn = 0;
		long ln = 0;
		line("%n", snprintf(buf, sizeof buf, "ab%ncd%n%hhn%ln", &n1, &n2, &hn, &ln));
		o_d(n1); o_c(' '); o_d(n2); o_c(' '); o_d(hn); o_c(' '); o_d(ln); o_c('\n');
		line("%p", snprintf(buf, sizeof buf, "%p %p", (void *)0x1234, (void *)-1));
		line("pos", snprintf(buf, sizeof buf, "%2$s %1$d %3$.2f", 7, "x", 2.5));
		line("pos*", snprintf(buf, sizeof buf, "%1$*2$d|", 42, 6));
		line("ls", snprintf(buf, sizeof buf, "[%ls] [%lc] [%5ls]", L"wide", (wchar_t)'Q', L"ab"));
	}
	/* truncation and the return value */
	for (int n = 0; n < 14; n++) {
		memset(buf, '#', 20);
		int r = snprintf(n ? buf : 0, n, "%s=%d", "hello", 12345);
		o_s("trunc "); o_d(n); o_c(' '); o_d(r); o_c(' '); o_mem(buf, 16); o_c('\n');
	}
	memset(buf, '#', 40);
	line("sprintf", sprintf(buf, "%d-%s-%.2f", 5, "five", 5.0));
	/* long output */
	{
		static char big[5000];
		int r = snprintf(big, sizeof big, "%3000d|%s", 1, "end");
		o_s("long "); o_d(r); o_c(' '); o_d(strlen(big)); o_c(' '); o_s(big + 2995); o_c('\n');
	}
	return o_end();
}
