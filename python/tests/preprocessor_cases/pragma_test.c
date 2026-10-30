#define N 4
#pragma omp parallel for
for (;;) ;
#pragma GCC optimize("O3")
#pragma STDC FP_CONTRACT ON
_Pragma("omp simd") int x;
#define DO_PRAGMA(x) _Pragma (#x)
DO_PRAGMA(omp barrier) int y;
int z;
