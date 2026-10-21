#define _POSIX_C_SOURCE 200809L
#include "openmx_gpu_error.h"
#include "openmx_gpu_runtime.h"
#include "openmx_gpu_blas.h"
#include "openmx_gpu_solver.h"
#include <mpi.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

_Noreturn void openmx_gpu_fatal(const char *message, const char *file, int line)
{
    int initialized = 0, finalized = 0;
    fprintf(stderr, "<OpenMX_GPU> %s\nFile = %s\nLine = %d\n",
        message ? message : "fatal GPU error", file, line);
    fflush(stderr);
    MPI_Initialized(&initialized);
    if (initialized) MPI_Finalized(&finalized);
    if (initialized && !finalized) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
}

void openmx_gpu_check_pending_error(const char *call, const char *file, int line)
{
    const OpenMXGpuError pending = OpenMXGpuPeekAtLastError();
    if (pending != OPENMX_GPU_SUCCESS) {
        char message[768];
        snprintf(message, sizeof(message), "%s pending error before %s: %s (%d)",
            OPENMX_GPU_RUNTIME_NAME, call, OpenMXGpuGetErrorString(pending), (int)pending);
        openmx_gpu_fatal(message, file, line);
    }
}

void openmx_gpu_retry_failure(OpenMXGpuErrorDomain domain, int status,
    unsigned retry, const char *call, const char *file, int line)
{
    const int allocation_failure =
        domain == OPENMX_GPU_ERROR_RUNTIME ? status == (int)OPENMX_GPU_OUT_OF_MEMORY :
        domain == OPENMX_GPU_ERROR_BLAS ? status == (int)OPENMX_BLAS_ALLOC_FAILED :
        domain == OPENMX_GPU_ERROR_SOLVER && status == (int)OPENMX_SOLVER_ALLOC_FAILED;
    const OpenMXGpuError pending = OpenMXGpuPeekAtLastError();
    /* Entry checked the caller's pending status without clearing it.
       Only an allocation error from our own attempt may now be consumed. */
    if (!allocation_failure || retry >= 8 ||
        (pending != OPENMX_GPU_SUCCESS && pending != OPENMX_GPU_OUT_OF_MEMORY)) {
        char message[768];
        snprintf(message, sizeof(message),
            "%s call failed: %s; domain=%d status=%d attempts=%u pending=%d",
            OPENMX_GPU_RUNTIME_NAME, call, (int)domain, status, retry+1, (int)pending);
        openmx_gpu_fatal(message, file, line);
    }
    if (pending == OPENMX_GPU_OUT_OF_MEMORY) (void)OpenMXGpuGetLastError();
    /* Eight waits: 1,2,4,8,16,32,32,32 ms (127 ms in total).
       Sleep instead of spinning or using process-global random state. */
    const unsigned milliseconds = 1u << (retry < 5 ? retry : 5);
    struct timespec remaining = {0, (long)milliseconds * 1000000L};
    while (nanosleep(&remaining, &remaining) != 0 && errno == EINTR) {}
}
