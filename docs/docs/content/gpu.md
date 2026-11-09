# GPU kernels

cimo compiles explicit GPU kernels, written in C, to PTX and loads them through the NVIDIA driver. Nothing else is
needed: no CUDA toolkit, no NVRTC, no compiler on the machine. `libcuda` (`nvcuda.dll` on Windows) is reached by
`dlsym`.

```python
import array
import cimo

gpu = cimo.gpu.device()          # the first CUDA device, through its primary context
k = cimo.compile(r"""
#include <cimo/gpu.h>

__gpu_kernel void saxpy(float a, const float *x, float *y, int n) {
    int i = gpu_block_id(0) * gpu_block_dim(0) + gpu_thread_id(0);
    if (i < n)
        y[i] = a * x[i] + y[i];
}
""", device=gpu)

n = 1 << 20
x = gpu.array(array.array('f', [1.0] * n))   # copies to the device
y = gpu.array(array.array('f', [2.0] * n))
k.saxpy(3.0, x, y, n, grid=n // 256, block=256)  # doesn't wait
gpu.sync()
out = y.to_host(array.array('f', bytes(4 * n)))
```

## Kernel C

`<cimo/gpu.h>` declares what kernels use:

| | |
|---|---|
| `__gpu_kernel` | an entry point, launched from Python; `void`, not `static` |
| `__gpu_shared` | memory shared by a block's threads. Inside a kernel, one per block, as if `static`. `extern __gpu_shared T a[];` is the block's dynamic shared memory, sized at launch (`shared=`) |
| `__gpu_device` | marks a helper function for nvcc (`__device__`); nothing for cimo, where any function in the file can be called from a kernel |
| `gpu_thread_id(d)`, `gpu_block_id(d)`, `gpu_block_dim(d)`, `gpu_grid_dim(d)` | the thread's index in its block, the block's index in the grid, and their sizes, in dimension `d` (0, 1, 2: x, y, z; a constant) |
| `gpu_lane_id()`, `gpu_warp_size()` | the thread's lane in its warp, and the warp size (32) |
| `gpu_barrier()` | every thread of the block waits here; shared and global memory writes before it are seen after it |
| `__gpu_asm(...)` | PTX instructions, in gcc's extended asm form with CUDA's constraint letters |

Types: `float`, `double`, 8- to 64-bit integers, structs, and `_Float16` and `__bf16` in memory (arithmetic on them is
in `float`). Math: `sqrtf`, `fmaf`, `fmaxf`, `fminf`, `fabsf`, `floorf`, `ceilf`, `truncf`, `roundf`, `rintf` and
their `double` versions are single instructions; `expf` is cimo's (within 1 ulp), compiled into the module when a kernel
calls it.

Rejected, with a compile error: recursion, function pointers, calls of functions not in the file (`malloc`, `printf`),
variable-length arrays, variadic functions, and initialized `__gpu_shared` objects.

### Target instructions

`__gpu_asm` is target text with operands: `%0`, `%1`, ... in order of the outputs, then the inputs; `%%` is a `%`.
Constraints: `h`, `r`, `l` for 16-, 32- and 64-bit integers, `f` and `d` for `float` and `double`, `n` for an integer
constant; outputs start with `=`, or `+` for an operand that is read too, and an input may be tied to an output
(`"0"`). A tensor core instruction on bf16:

```c
__gpu_asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
```

### The same file under nvcc

With a header of the same name that maps the names to CUDA's, the file also compiles with `nvcc -x cu`:

```c
/* cimo/gpu.h for nvcc */
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
```

cimo's tests use nvcc this way as the reference for results and speed.

## Python API

| | |
|---|---|
| `cimo.gpu.device(ordinal=0)` | a `Device`: `name`, `sm` (compute capability as one number, `89` is 8.9), `multiprocessors`, `memory`, `shared_per_block` |
| `cimo.gpu.devices()` | how many CUDA devices there are |
| `cimo.gpu.driver_version()`, `cimo.gpu.max_ptx()` | the driver's CUDA version, and the newest PTX it reads |
| `cimo.compile(src, device=gpu)`, `gpu.compile(src, opt=4, version=None)` | a module of the file's kernels, loaded on the device. `opt`: the driver's optimization level; `version`: a PTX version to state, `(major, minor)`, when asm needs a newer one than the architecture's lowest |
| `cimo.gpu.ptx(src, sm=80)` | the PTX text alone, on any machine |
| `gpu.load(image, signatures)` | a module made elsewhere: PTX text or a cubin, with the kernels' parameters as a cimo module's `signatures` gives them |
| `gpu.array(buf)`, `gpu.zeros(n)`, `gpu.empty(n)` | device memory (`DeviceArray`): a copy of a host buffer, or `n` bytes |
| `a.to_host(out=None)`, `a.from_host(buf)`, `a.free()` | copies (they wait for the work before them), and freeing early |
| `gpu.sync()` | wait for every launch and copy so far |
| `gpu.time(fn, repeat=1)` | milliseconds per call of `fn`, between two events on the GPU |

A launch is `module.kernel(*args, grid=, block=, shared=0)`: `grid` and `block` are 1 to 3 sizes, `shared` the bytes
of dynamic shared memory. It doesn't wait. Pointer parameters take a `DeviceArray`, any object with
`__cuda_array_interface__` (CuPy, PyTorch, Numba arrays: no copy), or a device address as an `int`. A host buffer is
taken too: it is copied in and, unless the parameter points to `const`, back, and then the call waits. Scalars take
Python numbers; a `_Float16` or `__bf16` parameter takes a `float` (rounded) or the bits as an `int`; a struct
parameter takes its bytes. A `DeviceArray` has `__cuda_array_interface__` itself.

Errors: the driver rejecting the PTX raises `cimo.CompileError` with its log; a failed launch or a fault in a kernel
raises `cimo.gpu.GpuError` (a fault is reported at the next call that waits, as the driver does). A module's
`load_time` is the driver's compile time in ms, `compile_time` cimo's in seconds, and `ptx` its text; a kernel's
`registers` is what the driver gave it.

PTX text is cached like CPU code (`gpu-<key>.json` in the cache directory), and the driver caches its own compiles.

## How it works

The front end and the target-independent passes are the CPU's. Small loops with a constant trip count are unrolled
completely, branches and asm statements included, so the arrays they index (accumulators, fragments) become registers;
constants are moved out of scaled indices, so every access to an array is one register plus an offset the driver can
merge into wide loads; loop-invariant parts of addresses are summed first, so a loop keeps one register live per
address. Addresses known to point into shared, global or local memory use that space, shared memory with 32-bit
addresses. The PTX states the lowest version that has the device's architecture (`sm_90a` when the asm uses `wgmma` or
other architecture-specific instructions), and the driver's compiler does register allocation and scheduling.

Speed on an NVIDIA L4 against nvcc 12.9 (`-O3`) on the same files, with the same driver: within a few percent on
memory-bound kernels, a shared-memory matrix product and an `mma.sync` loop, and 12% slower on a tiled bf16 GEMM with a
GELU epilogue. A 100-line kernel compiles to PTX in under a second; a launch costs about 4 µs from Python.

Only NVIDIA GPUs for now. Not yet: other GPUs, and `#pragma` parallel loops.
