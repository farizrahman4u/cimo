/* cimo.gpu's host side for NVIDIA GPUs: the CUDA driver API (libcuda, nvcuda.dll), reached by dlsym when cimo links
   this file (cimo.gpu.cuda). Only the driver: no runtime library, no toolkit. Each entry makes the device's primary
   context current on the calling thread first, so buffers made by another library in the process (which uses the
   primary context too) are usable, and any Python thread may call. Every function returns the driver's CUresult. */

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUevent;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long size_t;  /* not long: 32 bits in Windows DLLs, and cimo warns */

CUresult cuInit(unsigned);
CUresult cuDriverGetVersion(int *);
CUresult cuDeviceGetCount(int *);
CUresult cuDeviceGet(CUdevice *, int);
CUresult cuDeviceGetAttribute(int *, int, CUdevice);
CUresult cuDeviceGetName(char *, int, CUdevice);
CUresult cuDeviceTotalMem_v2(size_t *, CUdevice);
CUresult cuDevicePrimaryCtxRetain(CUcontext *, CUdevice);
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice);
CUresult cuCtxSetCurrent(CUcontext);
CUresult cuCtxSynchronize(void);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t);
CUresult cuMemFree_v2(CUdeviceptr);
CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t);
CUresult cuMemcpyDtoD_v2(CUdeviceptr, CUdeviceptr, size_t);
CUresult cuMemsetD8_v2(CUdeviceptr, unsigned char, size_t);
CUresult cuModuleLoadDataEx(CUmodule *, const void *, unsigned, int *, void **);
CUresult cuModuleUnload(CUmodule);
CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *);
CUresult cuFuncSetAttribute(CUfunction, int, int);
CUresult cuFuncGetAttribute(int *, int, CUfunction);
CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream,
                        void **, void **);
CUresult cuGetErrorString(CUresult, const char **);
CUresult cuGetErrorName(CUresult, const char **);
CUresult cuEventCreate(CUevent *, unsigned);
CUresult cuEventRecord(CUevent, CUstream);
CUresult cuEventSynchronize(CUevent);
CUresult cuEventElapsedTime(float *, CUevent, CUevent);
CUresult cuEventDestroy_v2(CUevent);

enum {
	CU_JIT_WALL_TIME = 2, CU_JIT_INFO_LOG_BUFFER = 3, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
	CU_JIT_ERROR_LOG_BUFFER = 5, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6, CU_JIT_OPTIMIZATION_LEVEL = 7,
	CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
	CU_FUNC_ATTRIBUTE_NUM_REGS = 4, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
};

/* the driver's version, and the number of devices (cuInit first) */
CUresult cimo_cuda_init(int *version, int *count)
{
	CUresult r = cuInit(0);
	if (r)
		return r;
	r = cuDriverGetVersion(version);
	if (r)
		return r;
	return cuDeviceGetCount(count);
}

/* device `ordinal`: its handle, primary context, compute capability, multiprocessors, shared memory per block (opt-in
   maximum), total memory and name (out[0..6], name in name[0..namesize)) */
CUresult cimo_cuda_device(int ordinal, long long *out, char *name, int namesize)
{
	CUdevice dev;
	CUcontext ctx;
	int major, minor, sms, smem;
	size_t mem;
	CUresult r = cuDeviceGet(&dev, ordinal);
	if (r)
		return r;
	if ((r = cuDevicePrimaryCtxRetain(&ctx, dev)))
		return r;
	if ((r = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)))
		return r;
	if ((r = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)))
		return r;
	if ((r = cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)))
		return r;
	if (cuDeviceGetAttribute(&smem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev))
		smem = 48 * 1024;
	if ((r = cuDeviceTotalMem_v2(&mem, dev)))
		return r;
	if ((r = cuDeviceGetName(name, namesize, dev)))
		return r;
	out[0] = dev;
	out[1] = (long long)ctx;
	out[2] = major * 10 + minor;
	out[3] = sms;
	out[4] = smem;
	out[5] = (long long)mem;
	return 0;
}

CUresult cimo_cuda_release(int dev)
{
	return cuDevicePrimaryCtxRelease_v2(dev);
}

/* the driver's name and text for a CUresult */
void cimo_cuda_error(CUresult e, const char **name, const char **text)
{
	if (cuGetErrorName(e, name))
		*name = 0;
	if (cuGetErrorString(e, text))
		*text = 0;
}

/* PTX text to a module, at the driver's optimization level `opt` (0 to 4): the error and information logs into
   err and info (NUL-terminated), the driver's compile time in ms into *wall */
CUresult cimo_cuda_load(CUcontext ctx, const char *ptx, int opt, char *err, int errsize, char *info, int infosize,
                        float *wall, CUmodule *mod)
{
	int opts[6] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER,
	                CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_WALL_TIME, CU_JIT_OPTIMIZATION_LEVEL };
	void *vals[6];
	union { void *p; float f; } t;
	CUresult r = cuCtxSetCurrent(ctx);
	if (r)
		return r;
	err[0] = 0;
	info[0] = 0;
	vals[0] = err;
	vals[1] = (void *)(size_t)errsize;
	vals[2] = info;
	vals[3] = (void *)(size_t)infosize;
	vals[4] = 0;
	vals[5] = (void *)(size_t)opt;
	r = cuModuleLoadDataEx(mod, ptx, 6, opts, vals);
	t.p = vals[4];
	*wall = t.f;
	return r;
}

CUresult cimo_cuda_unload(CUcontext ctx, CUmodule mod)
{
	CUresult r = cuCtxSetCurrent(ctx);
	return r ? r : cuModuleUnload(mod);
}

/* a kernel of the module, and how many registers it uses */
CUresult cimo_cuda_function(CUcontext ctx, CUmodule mod, const char *name, CUfunction *f, int *regs)
{
	CUresult r = cuCtxSetCurrent(ctx);
	if (r)
		return r;
	if ((r = cuModuleGetFunction(f, mod, name)))
		return r;
	return cuFuncGetAttribute(regs, CU_FUNC_ATTRIBUTE_NUM_REGS, *f);
}

/* allow `bytes` of dynamic shared memory (over 48 KiB needs it) */
CUresult cimo_cuda_shared(CUcontext ctx, CUfunction f, int bytes)
{
	CUresult r = cuCtxSetCurrent(ctx);
	return r ? r : cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes);
}

/* a launch: the parameters are `size` bytes at args, laid out as the kernel's .param list; it doesn't wait */
CUresult cimo_cuda_launch(CUcontext ctx, CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                          unsigned by, unsigned bz, unsigned shared, CUstream stream, void *args, size_t size)
{
	void *extra[5];
	CUresult r = cuCtxSetCurrent(ctx);
	if (r)
		return r;
	extra[0] = (void *)1;  /* CU_LAUNCH_PARAM_BUFFER_POINTER */
	extra[1] = args;
	extra[2] = (void *)2;  /* CU_LAUNCH_PARAM_BUFFER_SIZE */
	extra[3] = &size;
	extra[4] = 0;  /* CU_LAUNCH_PARAM_END */
	return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, 0, extra);
}

CUresult cimo_cuda_sync(CUcontext ctx)
{
	CUresult r = cuCtxSetCurrent(ctx);
	return r ? r : cuCtxSynchronize();
}

CUresult cimo_cuda_alloc(CUcontext ctx, size_t n, CUdeviceptr *p)
{
	CUresult r = cuCtxSetCurrent(ctx);
	return r ? r : cuMemAlloc_v2(p, n ? n : 1);
}

CUresult cimo_cuda_free(CUcontext ctx, CUdeviceptr p)
{
	CUresult r = cuCtxSetCurrent(ctx);
	return r ? r : cuMemFree_v2(p);
}

/* copies; they wait for the work before them */
CUresult cimo_cuda_htod(CUcontext ctx, CUdeviceptr d, const void *h, size_t n)
{
	CUresult r = cuCtxSetCurrent(ctx);
	return r ? r : cuMemcpyHtoD_v2(d, h, n);
}

CUresult cimo_cuda_dtoh(CUcontext ctx, void *h, CUdeviceptr d, size_t n)
{
	CUresult r = cuCtxSetCurrent(ctx);
	return r ? r : cuMemcpyDtoH_v2(h, d, n);
}

CUresult cimo_cuda_dtod(CUcontext ctx, CUdeviceptr d, CUdeviceptr s, size_t n)
{
	CUresult r = cuCtxSetCurrent(ctx);
	return r ? r : cuMemcpyDtoD_v2(d, s, n);
}

CUresult cimo_cuda_memset(CUcontext ctx, CUdeviceptr d, int v, size_t n)
{
	CUresult r = cuCtxSetCurrent(ctx);
	return r ? r : cuMemsetD8_v2(d, (unsigned char)v, n);
}

/* timing: events recorded on the default stream; elapsed in ms between two (waits for the second) */
CUresult cimo_cuda_event(CUcontext ctx, CUevent *e)
{
	CUresult r = cuCtxSetCurrent(ctx);
	return r ? r : cuEventCreate(e, 0);
}

CUresult cimo_cuda_record(CUcontext ctx, CUevent e)
{
	CUresult r = cuCtxSetCurrent(ctx);
	return r ? r : cuEventRecord(e, 0);
}

CUresult cimo_cuda_elapsed(CUcontext ctx, CUevent a, CUevent b, float *ms)
{
	CUresult r = cuCtxSetCurrent(ctx);
	if (r)
		return r;
	if ((r = cuEventSynchronize(b)))
		return r;
	return cuEventElapsedTime(ms, a, b);
}

CUresult cimo_cuda_event_free(CUcontext ctx, CUevent e)
{
	CUresult r = cuCtxSetCurrent(ctx);
	return r ? r : cuEventDestroy_v2(e);
}
