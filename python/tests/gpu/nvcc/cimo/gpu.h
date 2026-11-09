/* <cimo/gpu.h> for nvcc: the same kernel sources compile as CUDA (tests/test_gpu.py uses nvcc as the oracle). */
#ifndef CIMO_GPU_H
#define CIMO_GPU_H
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define __gpu_kernel extern "C" __global__
#define __gpu_shared __shared__
#define __gpu_device __device__
#define __gpu_asm asm volatile
#define __bf16 __nv_bfloat16
#define _Float16 __half
#define gpu_thread_id(d) ((d) == 0 ? (int)threadIdx.x : (d) == 1 ? (int)threadIdx.y : (int)threadIdx.z)
#define gpu_block_dim(d) ((d) == 0 ? (int)blockDim.x : (d) == 1 ? (int)blockDim.y : (int)blockDim.z)
#define gpu_block_id(d) ((d) == 0 ? (int)blockIdx.x : (d) == 1 ? (int)blockIdx.y : (int)blockIdx.z)
#define gpu_grid_dim(d) ((d) == 0 ? (int)gridDim.x : (d) == 1 ? (int)gridDim.y : (int)gridDim.z)
#define gpu_lane_id() ((int)(threadIdx.x & 31))
#define gpu_warp_size() 32
#define gpu_barrier() __syncthreads()
#endif
