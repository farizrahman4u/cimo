/* cimo: <stdarg.h> (C11 7.16), using cproc's built-ins. */
#ifndef __CIMO_STDARG_H
#define __CIMO_STDARG_H

typedef __builtin_va_list va_list;

#define va_start(ap, parmN) __builtin_va_start(ap, parmN)
#define va_arg(ap, type) __builtin_va_arg(ap, type)
#define va_copy(dest, src) __builtin_va_copy(dest, src)
#define va_end(ap) __builtin_va_end(ap)

#endif
