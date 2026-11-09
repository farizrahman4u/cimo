/* C (m x n, float) = A (m x k, bf16, row major) B, with B given transposed (Bt: n x k, bf16, row major): each warp
   computes a 16 x 8 tile with mma.sync m16n8k16, k a multiple of 16; a block of 4 warps covers 16 x 32 */
#include <cimo/gpu.h>

__gpu_kernel void gemm_mma(const unsigned short *A, const unsigned short *Bt, float *C, int m, int n, int k)
{
	int lane = gpu_thread_id(0) & 31, warp = gpu_thread_id(0) >> 5;
	int g = lane >> 2, t = lane & 3;
	int row = gpu_block_id(1) * 16, col = (gpu_block_id(0) * 4 + warp) * 8;
	const unsigned *a0p = (const unsigned *)(A + (row + g) * k + t * 2);
	const unsigned *a1p = (const unsigned *)(A + (row + g + 8) * k + t * 2);
	const unsigned *bp = (const unsigned *)(Bt + (col + g) * k + t * 2);
	float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
	for (int kk = 0; kk < k / 2; kk += 8) {
		unsigned a0 = a0p[kk], a1 = a1p[kk], a2 = a0p[kk + 4], a3 = a1p[kk + 4];
		unsigned b0 = bp[kk], b1 = bp[kk + 4];
		__gpu_asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
		          "{%0,%1,%2,%3};"
		          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
		          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
	}
	C[(row + g) * n + col + t * 2] = d0;
	C[(row + g) * n + col + t * 2 + 1] = d1;
	C[(row + g + 8) * n + col + t * 2] = d2;
	C[(row + g + 8) * n + col + t * 2 + 1] = d3;
}
