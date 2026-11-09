/* each block one row of n floats (bf16 in, float out): max, then sum of exp, then the quotients, through shared
   memory; 256 threads */
#include <cimo/gpu.h>
#include <math.h>

__gpu_kernel void softmax(const __bf16 *x, float *y, int n)
{
	__gpu_shared float red[256];
	int tid = gpu_thread_id(0);
	const __bf16 *row = x + (long)gpu_block_id(0) * n;
	float *out = y + (long)gpu_block_id(0) * n;
	float m = -INFINITY;
	for (int i = tid; i < n; i += 256)
		m = fmaxf(m, (float)row[i]);
	red[tid] = m;
	gpu_barrier();
	for (int s = 128; s > 0; s >>= 1) {
		if (tid < s)
			red[tid] = fmaxf(red[tid], red[tid + s]);
		gpu_barrier();
	}
	m = red[0];
	gpu_barrier();
	float sum = 0;
	for (int i = tid; i < n; i += 256)
		sum += expf((float)row[i] - m);
	red[tid] = sum;
	gpu_barrier();
	for (int s = 128; s > 0; s >>= 1) {
		if (tid < s)
			red[tid] += red[tid + s];
		gpu_barrier();
	}
	sum = red[0];
	for (int i = tid; i < n; i += 256)
		out[i] = expf((float)row[i] - m) / sum;
}
