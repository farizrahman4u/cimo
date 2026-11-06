/* cimo: <setjmp.h> (C11 7.13): setjmp and longjmp are in every module's glue (cimo.execution). A function that
   calls setjmp keeps all its variables in memory and is never inlined, so a variable not modified between
   setjmp and longjmp has its value when setjmp returns again. */
#ifndef __CIMO_SETJMP_H
#define __CIMO_SETJMP_H

typedef long long jmp_buf[32];

int setjmp(jmp_buf);
_Noreturn void longjmp(jmp_buf, int);
int _setjmp(jmp_buf);
_Noreturn void _longjmp(jmp_buf, int);
#define setjmp setjmp

#endif
