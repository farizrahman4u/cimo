/* cimo: <inttypes.h> for LP64 arm64 (Linux, Apple and Windows). */
#ifndef __CIMO_INTTYPES_H
#define __CIMO_INTTYPES_H

#include <stdint.h>

typedef struct { intmax_t quot, rem; } imaxdiv_t;

intmax_t imaxabs(intmax_t);
imaxdiv_t imaxdiv(intmax_t, intmax_t);
intmax_t strtoimax(const char *restrict, char **restrict, int);
uintmax_t strtoumax(const char *restrict, char **restrict, int);

#ifdef __APPLE__
#define __CIMO_PRI64 "ll"
#define __CIMO_PRIFAST ""
#elif defined(_WIN32)
#define __CIMO_PRI64 "ll"
#define __CIMO_PRIFAST "l"
#else
#define __CIMO_PRI64 "l"
#define __CIMO_PRIFAST "l"
#endif

#define PRId8 "d"
#define PRId16 "d"
#define PRId32 "d"
#define PRId64 __CIMO_PRI64 "d"
#define PRIdLEAST8 "d"
#define PRIdLEAST16 "d"
#define PRIdLEAST32 "d"
#define PRIdLEAST64 __CIMO_PRI64 "d"
#define PRIdFAST8 "d"
#define PRIdFAST16 __CIMO_PRIFAST "d"
#define PRIdFAST32 __CIMO_PRIFAST "d"
#define PRIdFAST64 __CIMO_PRI64 "d"
#define PRIdMAX "ld"
#define PRIdPTR "ld"

#define PRIi8 "i"
#define PRIi16 "i"
#define PRIi32 "i"
#define PRIi64 __CIMO_PRI64 "i"
#define PRIiLEAST8 "i"
#define PRIiLEAST16 "i"
#define PRIiLEAST32 "i"
#define PRIiLEAST64 __CIMO_PRI64 "i"
#define PRIiFAST8 "i"
#define PRIiFAST16 __CIMO_PRIFAST "i"
#define PRIiFAST32 __CIMO_PRIFAST "i"
#define PRIiFAST64 __CIMO_PRI64 "i"
#define PRIiMAX "li"
#define PRIiPTR "li"

#define PRIo8 "o"
#define PRIo16 "o"
#define PRIo32 "o"
#define PRIo64 __CIMO_PRI64 "o"
#define PRIoLEAST8 "o"
#define PRIoLEAST16 "o"
#define PRIoLEAST32 "o"
#define PRIoLEAST64 __CIMO_PRI64 "o"
#define PRIoFAST8 "o"
#define PRIoFAST16 __CIMO_PRIFAST "o"
#define PRIoFAST32 __CIMO_PRIFAST "o"
#define PRIoFAST64 __CIMO_PRI64 "o"
#define PRIoMAX "lo"
#define PRIoPTR "lo"

#define PRIu8 "u"
#define PRIu16 "u"
#define PRIu32 "u"
#define PRIu64 __CIMO_PRI64 "u"
#define PRIuLEAST8 "u"
#define PRIuLEAST16 "u"
#define PRIuLEAST32 "u"
#define PRIuLEAST64 __CIMO_PRI64 "u"
#define PRIuFAST8 "u"
#define PRIuFAST16 __CIMO_PRIFAST "u"
#define PRIuFAST32 __CIMO_PRIFAST "u"
#define PRIuFAST64 __CIMO_PRI64 "u"
#define PRIuMAX "lu"
#define PRIuPTR "lu"

#define PRIx8 "x"
#define PRIx16 "x"
#define PRIx32 "x"
#define PRIx64 __CIMO_PRI64 "x"
#define PRIxLEAST8 "x"
#define PRIxLEAST16 "x"
#define PRIxLEAST32 "x"
#define PRIxLEAST64 __CIMO_PRI64 "x"
#define PRIxFAST8 "x"
#define PRIxFAST16 __CIMO_PRIFAST "x"
#define PRIxFAST32 __CIMO_PRIFAST "x"
#define PRIxFAST64 __CIMO_PRI64 "x"
#define PRIxMAX "lx"
#define PRIxPTR "lx"

#define PRIX8 "X"
#define PRIX16 "X"
#define PRIX32 "X"
#define PRIX64 __CIMO_PRI64 "X"
#define PRIXLEAST8 "X"
#define PRIXLEAST16 "X"
#define PRIXLEAST32 "X"
#define PRIXLEAST64 __CIMO_PRI64 "X"
#define PRIXFAST8 "X"
#define PRIXFAST16 __CIMO_PRIFAST "X"
#define PRIXFAST32 __CIMO_PRIFAST "X"
#define PRIXFAST64 __CIMO_PRI64 "X"
#define PRIXMAX "lX"
#define PRIXPTR "lX"

#endif
