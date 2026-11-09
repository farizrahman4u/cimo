/* C = A B, n x n floats, n a multiple of 64: 16 x 16 threads per block, each computing 4 x 4 outputs of a 64 x 64
   tile, through shared memory */
#include <cimo/gpu.h>

#define TS 64
#define TK 16

__gpu_kernel void matmul(const float *A, const float *B, float *C, int n)
{
	__gpu_shared float As[TK][TS];
	__gpu_shared float Bs[TK][TS];
	int tx = gpu_thread_id(0), ty = gpu_thread_id(1);
	int row0 = gpu_block_id(1) * TS, col0 = gpu_block_id(0) * TS;
	float acc[4][4];
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			acc[i][j] = 0;
	int tid = ty * 16 + tx;
	for (int k0 = 0; k0 < n; k0 += TK) {
		for (int l = 0; l < 4; l++) {
			int e = tid + l * 256, r = e / TK, c = e % TK;  /* A tile: 64 rows x 16 */
			As[c][r] = A[(row0 + r) * n + k0 + c];
			int rb = e / TS, cb = e % TS;  /* B tile: 16 rows x 64 */
			Bs[rb][cb] = B[(k0 + rb) * n + col0 + cb];
		}
		gpu_barrier();
		for (int k = 0; k < TK; k++) {
			float a[4], b[4];
			for (int i = 0; i < 4; i++)
				a[i] = As[k][ty * 4 + i];
			for (int j = 0; j < 4; j++)
				b[j] = Bs[k][tx * 4 + j];
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++)
					acc[i][j] += a[i] * b[j];
		}
		gpu_barrier();
	}
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			C[(row0 + ty * 4 + i) * n + col0 + tx * 4 + j] = acc[i][j];
}
