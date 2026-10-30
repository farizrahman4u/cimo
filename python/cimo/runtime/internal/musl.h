/* cimo runtime: what musl's internal headers provide that the imported sources expect. */
#ifndef __CIMO_RT_MUSL_H
#define __CIMO_RT_MUSL_H

/* symbol visibility and aliases: cimo links the runtime statically into each module, so
   there is nothing to hide, and the aliased names (drem, lgamma_r, ...) are not provided */
#define hidden
#define weak_alias(old, new) extern int __cimo_no_alias

/* Built-in math: cimo compiles calls to these functions to single instructions (see
   cimo.frontend's builtins), so the runtime's own definitions, needed only for indirect
   calls, are written as calls to the function itself. gcc (used to build the reference
   copy in the tests) spells them __builtin_*. */
#ifndef __GNUC__
#define __builtin_sqrt sqrt
#define __builtin_sqrtf sqrtf
#define __builtin_fabs fabs
#define __builtin_fabsf fabsf
#define __builtin_fmin fmin
#define __builtin_fminf fminf
#define __builtin_fmax fmax
#define __builtin_fmaxf fmaxf
#define __builtin_floor floor
#define __builtin_floorf floorf
#define __builtin_ceil ceil
#define __builtin_ceilf ceilf
#define __builtin_trunc trunc
#define __builtin_truncf truncf
#define __builtin_round round
#define __builtin_roundf roundf
#define __builtin_fma fma
#define __builtin_fmaf fmaf
#endif

#endif
