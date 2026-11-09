/* C = gelu(A B + bias): A (m x k bf16, row major), Bt (n x k bf16, row major: B transposed), bias (n floats), C (m x n
   floats). m, n multiples of 64, k a multiple of 32. A block of 4 warps computes a 64 x 64 tile: warp w rows
   16w..16w+15, all 64 columns as 8 mma.sync m16n8k16 tiles. A and B are staged through shared memory 32 columns of k
   at a time, rows padded so the fragment reads spread over the banks. GELU is the tanh form, tanh from expf. */
#include <cimo/gpu.h>
#include <math.h>

#define BM 64
#define BN 64
#define BK 32
#define PAD 8

__gpu_device static inline float gelu(float x)
{
	float u = 0.7978845608f * (x + 0.044715f * x * x * x);
	float t = 1.0f - 2.0f / (expf(2.0f * u) + 1.0f);
	return 0.5f * x * (1.0f + t);
}

__gpu_device static inline unsigned pair(const unsigned short *p)
{
	return *(const unsigned *)p;
}

__gpu_kernel void gemm_gelu(const unsigned short *A, const unsigned short *Bt, const float *bias, float *C,
                            int m, int n, int k)
{
	__gpu_shared unsigned short As[BM][BK + PAD];
	__gpu_shared unsigned short Bs[BN][BK + PAD];
	int tid = gpu_thread_id(0);
	int lane = tid & 31, warp = tid >> 5;
	int g = lane >> 2, t = lane & 3;
	int row0 = gpu_block_id(1) * BM, col0 = gpu_block_id(0) * BN;
	float acc[8][4];
	for (int j = 0; j < 8; j++)
		for (int r = 0; r < 4; r++)
			acc[j][r] = 0.0f;

	for (int k0 = 0; k0 < k; k0 += BK) {
		/* 64 x 32 halves of each = 1024 pairs: 8 per thread */
		for (int l = 0; l < 8; l++) {
			int e = tid + l * 128;
			int r = e / (BK / 2), c = (e % (BK / 2)) * 2;
			*(unsigned *)&As[r][c] = pair(A + (long)(row0 + r) * k + k0 + c);
			*(unsigned *)&Bs[r][c] = pair(Bt + (long)(col0 + r) * k + k0 + c);
		}
		gpu_barrier();
		for (int ks = 0; ks < BK; ks += 16) {
			int ar = warp * 16 + g;
			unsigned a0 = pair(&As[ar][ks + t * 2]);
			unsigned a1 = pair(&As[ar + 8][ks + t * 2]);
			unsigned a2 = pair(&As[ar][ks + t * 2 + 8]);
			unsigned a3 = pair(&As[ar + 8][ks + t * 2 + 8]);
			for (int j = 0; j < 8; j++) {
				unsigned b0 = pair(&Bs[j * 8 + g][ks + t * 2]);
				unsigned b1 = pair(&Bs[j * 8 + g][ks + t * 2 + 8]);
				__gpu_asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
				          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
				          : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
				          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
			}
		}
		gpu_barrier();
	}

	/* the epilogue: the accumulator layout of m16n8 (rows g and g + 8, columns 2t and 2t + 1) */
	int r0 = row0 + warp * 16 + g;
	for (int j = 0; j < 8; j++) {
		int c = col0 + j * 8 + t * 2;
		float b0 = bias[c], b1 = bias[c + 1];
		C[(long)r0 * n + c] = gelu(acc[j][0] + b0);
		C[(long)r0 * n + c + 1] = gelu(acc[j][1] + b1);
		C[(long)(r0 + 8) * n + c] = gelu(acc[j][2] + b0);
		C[(long)(r0 + 8) * n + c + 1] = gelu(acc[j][3] + b1);
	}
}

/* the same product without tensor cores, for a check of the mma path: one thread per output */
__gpu_kernel void gemm_gelu_ref(const unsigned short *A, const unsigned short *Bt, const float *bias, float *C,
                                int m, int n, int k)
{
	int col = gpu_block_id(0) * gpu_block_dim(0) + gpu_thread_id(0);
	int row = gpu_block_id(1);
	if (col >= n)
		return;
	float s = 0.0f;
	for (int i = 0; i < k; i++) {
		union { unsigned u; float f; } a, b;
		a.u = (unsigned)A[(long)row * k + i] << 16;
		b.u = (unsigned)Bt[(long)col * k + i] << 16;
		s += a.f * b.f;
	}
	C[(long)row * n + col] = gelu(s + bias[col]);
}
