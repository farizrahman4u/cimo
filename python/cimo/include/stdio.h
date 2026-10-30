/* cimo: <stdio.h>, formatted output only (C11 7.21.6), provided by cimo's runtime. printf,
   puts and putchar write to Python's sys.stdout (buffered; flushed at a newline and when the
   kernel returns). There are no files: for FILE, fopen, scanf etc. link the host libc
   (cimo.compile(..., link=["libc"], system_headers=True)). */
#ifndef __CIMO_STDIO_H
#define __CIMO_STDIO_H

#ifndef __CIMO_SIZE_T
#define __CIMO_SIZE_T
typedef unsigned long size_t;
#endif
#undef NULL
#define NULL ((void *)0)
#define EOF (-1)

int printf(const char *restrict format, ...);
int vprintf(const char *restrict format, __builtin_va_list arg);
int puts(const char *s);
int putchar(int c);
int snprintf(char *restrict s, size_t n, const char *restrict format, ...);
int vsnprintf(char *restrict s, size_t n, const char *restrict format, __builtin_va_list arg);
int sprintf(char *restrict s, const char *restrict format, ...);
int vsprintf(char *restrict s, const char *restrict format, __builtin_va_list arg);

#endif
