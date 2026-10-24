#ifndef OPENMX_GPU_SOLVER_H
#define OPENMX_GPU_SOLVER_H

#include "openmx_gpu_backend.h"

/* Compile-time aliases preserve the native ABI, status codes and ownership.
   Backend policies (memory accounting, solver retries and offload) stay
   in their implementations. No device is initialized by this header. */
#if OPENMX_GPU_HIP
#include "hipsolver_compat.h"
#define OpenMXSolverHandle hipsolverDnHandle_t
#define OpenMXSolverCreate hipsolverDnCreate
#define OpenMXSolverDestroy hipsolverDnDestroy
#define OpenMXSolverStatus hipsolverStatus_t
#define OpenMXSolverEigMode hipsolverEigMode_t
#define OpenMXSolverGetStream hipsolverDnGetStream
#define OPENMX_SOLVER_EIG_VECTOR HIPSOLVER_EIG_MODE_VECTOR
#define OPENMX_SOLVER_EXECUTION_FAILED HIPSOLVER_STATUS_EXECUTION_FAILED
#define OPENMX_SOLVER_INVALID_VALUE HIPSOLVER_STATUS_INVALID_VALUE
#define OPENMX_SOLVER_SUCCESS HIPSOLVER_STATUS_SUCCESS
#define OPENMX_SOLVER_ALLOC_FAILED HIPSOLVER_STATUS_ALLOC_FAILED
#elif OPENMX_GPU_CUDA || OPENMX_GPU_HOST
#include <cusolverDn.h>
#define OpenMXSolverHandle cusolverDnHandle_t
#define OpenMXSolverCreate cusolverDnCreate
#define OpenMXSolverDestroy cusolverDnDestroy
#define OpenMXSolverStatus cusolverStatus_t
#define OpenMXSolverEigMode cusolverEigMode_t
#define OpenMXSolverGetStream cusolverDnGetStream
#define OPENMX_SOLVER_EIG_VECTOR CUSOLVER_EIG_MODE_VECTOR
#define OPENMX_SOLVER_EXECUTION_FAILED CUSOLVER_STATUS_EXECUTION_FAILED
#define OPENMX_SOLVER_INVALID_VALUE CUSOLVER_STATUS_INVALID_VALUE
#define OPENMX_SOLVER_SUCCESS CUSOLVER_STATUS_SUCCESS
#define OPENMX_SOLVER_ALLOC_FAILED CUSOLVER_STATUS_ALLOC_FAILED
#else
#error "This native adapter supports CUDA/HIP only; use the portable core for other backends"
#endif

#endif /* OPENMX_GPU_SOLVER_H */
