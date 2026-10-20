#ifndef OPENMX_GPU_ERROR_H
#define OPENMX_GPU_ERROR_H

#ifdef __cplusplus
extern "C" {
#define OPENMX_GPU_NORETURN [[noreturn]]
#else
#define OPENMX_GPU_NORETURN _Noreturn
#endif

/* SDK-independent failure boundary. A new backend supplies its status
   interpretation in the implementation, not in the numerical callers. */
typedef enum {
    OPENMX_GPU_ERROR_RUNTIME,
    OPENMX_GPU_ERROR_BLAS,
    OPENMX_GPU_ERROR_SOLVER
} OpenMXGpuErrorDomain;

OPENMX_GPU_NORETURN void openmx_gpu_fatal(const char *message,
    const char *file, int line);
void openmx_gpu_check_pending_error(const char *call, const char *file, int line);
void openmx_gpu_retry_failure(OpenMXGpuErrorDomain domain, int status,
    unsigned retry, const char *call, const char *file, int line);

#undef OPENMX_GPU_NORETURN
#ifdef __cplusplus
}
#endif

/* Only expression re-evaluation lives in this facade. Classification,
   bounded backoff and process-wide termination are shared functions.
   Distinct domains matter: CUDA runtime, BLAS and solver error numbers
   are not interchangeable. A successful operation has no retry overhead. */
#define OPENMX_GPU_WAIT_DOMAIN(domain, call) do { \
    unsigned openmx_wait_retry = 0; \
    openmx_gpu_check_pending_error(#call, __FILE__, __LINE__); \
    for (;;) { \
        int openmx_wait_status = (int)(call); \
        if (openmx_wait_status == 0) break; \
        openmx_gpu_retry_failure((domain), openmx_wait_status, \
            openmx_wait_retry++, #call, __FILE__, __LINE__); \
    } \
} while (0)
#define OpenMXGpuWait(call) OPENMX_GPU_WAIT_DOMAIN(OPENMX_GPU_ERROR_RUNTIME, call)
#define OpenMXBlasWait(call) OPENMX_GPU_WAIT_DOMAIN(OPENMX_GPU_ERROR_BLAS, call)
#define OpenMXSolverWait(call) OPENMX_GPU_WAIT_DOMAIN(OPENMX_GPU_ERROR_SOLVER, call)

#endif /* OPENMX_GPU_ERROR_H */
