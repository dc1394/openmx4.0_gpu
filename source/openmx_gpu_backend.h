#ifndef OPENMX_GPU_BACKEND_H
#define OPENMX_GPU_BACKEND_H

/* The legacy native application still has CUDA and HIP adapters. The portable
   numerical core owns a separate SDK-free capability contract. Other backend
   identities must never silently select CUDA types through an #else branch. */
#define OPENMX_BACKEND_HOST 0
#define OPENMX_BACKEND_CUDA 1
#define OPENMX_BACKEND_HIP 2
#define OPENMX_BACKEND_SYCL 3
#define OPENMX_BACKEND_METAL 4
#ifndef OPENMX_GPU_BACKEND
# if defined(__HIP_PLATFORM_AMD__)
#  define OPENMX_GPU_BACKEND OPENMX_BACKEND_HIP
# else
/* Compatibility default for existing direct CUDA makefile consumers. */
#  define OPENMX_GPU_BACKEND OPENMX_BACKEND_CUDA
# endif
#endif
#if defined(__HIP_PLATFORM_AMD__) && OPENMX_GPU_BACKEND != OPENMX_BACKEND_HIP
#error "HIP compiler platform conflicts with OPENMX_GPU_BACKEND"
#endif
#define OPENMX_GPU_HIP (OPENMX_GPU_BACKEND == OPENMX_BACKEND_HIP)
#define OPENMX_GPU_CUDA (OPENMX_GPU_BACKEND == OPENMX_BACKEND_CUDA)
#define OPENMX_GPU_HOST (OPENMX_GPU_BACKEND == OPENMX_BACKEND_HOST)
#if OPENMX_GPU_HIP
#define OPENMX_GPU_RUNTIME_NAME "HIP"
#define OPENMX_GPU_BLAS_NAME "hipBLAS"
#define OPENMX_GPU_SOLVER_NAME "hipSOLVER"
#define OPENMX_ELPA_GPU_OPTION "amd-gpu"
#elif OPENMX_GPU_CUDA
#define OPENMX_GPU_RUNTIME_NAME "CUDA"
#define OPENMX_GPU_BLAS_NAME "cuBLAS"
#define OPENMX_GPU_SOLVER_NAME "cuSOLVER"
#define OPENMX_ELPA_GPU_OPTION "nvidia-gpu"
#elif OPENMX_GPU_BACKEND == OPENMX_BACKEND_SYCL
#define OPENMX_GPU_RUNTIME_NAME "SYCL"
#elif OPENMX_GPU_BACKEND == OPENMX_BACKEND_METAL
#define OPENMX_GPU_RUNTIME_NAME "Metal"
#elif OPENMX_GPU_BACKEND == OPENMX_BACKEND_HOST
#define OPENMX_GPU_RUNTIME_NAME "Host"
#define OPENMX_GPU_BLAS_NAME "CPU BLAS"
#define OPENMX_GPU_SOLVER_NAME "CPU ELPA"
#define OPENMX_ELPA_GPU_OPTION "nvidia-gpu"
#else
#error "Unknown OPENMX_GPU_BACKEND"
#endif

#endif
