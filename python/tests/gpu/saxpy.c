#include <cimo/gpu.h>

__gpu_kernel void saxpy(float a, const float *x, float *y, int n)
{
	int i = gpu_block_id(0) * gpu_block_dim(0) + gpu_thread_id(0);
	if (i < n)
		y[i] = a * x[i] + y[i];
}
