/* Test-only CUDA interposer: cap reported free memory without allocating it.
   Build with cc -shared -fPIC -I$CUDA_HOME/include -I$CUDA_MATH_ROOT/include
   gpu_capacity_limit.c -ldl.
   Set LD_PRELOAD to the result and OPENMX_TEST_GPU_FREE_MB to the cap on every
   MPI rank. OPENMX_TEST_GPU_FREE_AFTER_SOLVER=1 delays pressure until that
   rank creates its first cuSOLVER handle, leaving earlier integral setup
   alone. Real CUDA kernels and MPI communication remain enabled. */
#define _GNU_SOURCE
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

static atomic_int solver_created;

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle)
{
    __typeof__(&cusolverDnCreate) real_call = dlsym(RTLD_NEXT, "cusolverDnCreate");
    if (!real_call) return CUSOLVER_STATUS_INTERNAL_ERROR;
    cusolverStatus_t status = real_call(handle);
    if (status == CUSOLVER_STATUS_SUCCESS) atomic_store(&solver_created, 1);
    return status;
}

cudaError_t cudaMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
    __typeof__(&cudaMemGetInfo) real_call = dlsym(RTLD_NEXT, "cudaMemGetInfo");
    static atomic_int announced;
    const char *value = getenv("OPENMX_TEST_GPU_FREE_MB");
    const char *after = getenv("OPENMX_TEST_GPU_FREE_AFTER_SOLVER");
    cudaError_t status;

    if (!real_call) return cudaErrorUnknown;
    status = real_call(free_bytes, total_bytes);
    if (status == cudaSuccess && free_bytes && value && *value &&
        (!(after && *after == '1') || atomic_load(&solver_created))) {
        char *end;
        unsigned long long mib;
        errno = 0;
        mib = strtoull(value, &end, 10);
        if (*value == '-' || end == value || *end || errno || mib > SIZE_MAX / 1048576u) {
            fprintf(stderr, "Invalid OPENMX_TEST_GPU_FREE_MB=%s\n", value);
            abort();
        }
        const size_t limit = (size_t)mib * 1048576u;
        if (*free_bytes > limit) *free_bytes = limit;
        if (!atomic_exchange(&announced, 1))
            fprintf(stderr, "TEST_GPU_CAPACITY: reported free memory capped at %llu MiB\n", mib);
    }
    return status;
}
