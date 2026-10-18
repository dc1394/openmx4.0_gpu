#ifndef OPENMX_GPU_BACKEND_H
#define OPENMX_GPU_BACKEND_H

/* The native makefiles select one backend per executable. Keep this header
   free of SDK dependencies so host-only policies can use the same selection. */
#if defined(__HIP_PLATFORM_AMD__)
#define OPENMX_GPU_HIP 1
#define OPENMX_GPU_RUNTIME_NAME "HIP"
#define OPENMX_GPU_BLAS_NAME "hipBLAS"
#define OPENMX_GPU_SOLVER_NAME "hipSOLVER"
#define OPENMX_ELPA_GPU_OPTION "amd-gpu"
#else
#define OPENMX_GPU_HIP 0
#define OPENMX_GPU_RUNTIME_NAME "CUDA"
#define OPENMX_GPU_BLAS_NAME "cuBLAS"
#define OPENMX_GPU_SOLVER_NAME "cuSOLVER"
#define OPENMX_ELPA_GPU_OPTION "nvidia-gpu"
#endif

#endif
