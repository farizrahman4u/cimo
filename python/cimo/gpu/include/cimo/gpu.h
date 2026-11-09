/* GPU kernel C for cimo (cimo.compile(src, device=gpu)): kernels, shared memory, thread ids, barriers and target
   instructions. The same names under nvcc come from a shim header (see docs/gpu.md). */
#ifndef CIMO_GPU_H
#define CIMO_GPU_H

/* an entry point, launched with a grid and block size: void, not static */
#define __gpu_kernel __attribute__((cimo_gpu_kernel))
/* a function called from kernels (nothing for cimo, where all functions of the file can be; __device__ in CUDA) */
#define __gpu_device
/* memory shared by the threads of a block; inside a kernel, one per block (as static); `extern __gpu_shared T a[];`
   is the block's dynamic shared memory, its size given at launch (shared=) */
#define __gpu_shared __attribute__((cimo_gpu_shared))
/* target instructions (PTX), gcc's extended asm with CUDA's constraint letters: h, r, l (16-, 32-, 64-bit integers),
   f, d (float, double), n (an integer constant); "=" an output, "+" both */
#define __gpu_asm __asm__ volatile

int __cimo_gpu_id(int);
void __cimo_gpu_barrier(void);

/* d = 0, 1, 2 (x, y, z), a constant */
#define gpu_thread_id(d) __cimo_gpu_id(0 + (d))
#define gpu_block_dim(d) __cimo_gpu_id(4 + (d))
#define gpu_block_id(d) __cimo_gpu_id(8 + (d))
#define gpu_grid_dim(d) __cimo_gpu_id(12 + (d))
#define gpu_lane_id() __cimo_gpu_id(16)
#define gpu_warp_size() 32
/* every thread of the block waits here; shared and global memory writes before it are seen after it */
#define gpu_barrier() __cimo_gpu_barrier()

#endif
