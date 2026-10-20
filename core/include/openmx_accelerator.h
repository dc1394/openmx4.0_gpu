#ifndef OPENMX_ACCELERATOR_H
#define OPENMX_ACCELERATOR_H

/* Public, SDK-free capabilities. An API is not a vendor, a memory model,
   or a performance policy. Unknown capabilities never authorize FP64 work.
   METAL reserves an adapter identity; it does not advertise an implementation. */
typedef enum {
    OPENMX_ACCELERATOR_HOST = 0,
    OPENMX_ACCELERATOR_CUDA,
    OPENMX_ACCELERATOR_HIP,
    OPENMX_ACCELERATOR_SYCL,
    OPENMX_ACCELERATOR_METAL
} OpenMXAcceleratorBackend;

typedef enum {
    OPENMX_FP64_UNKNOWN = 0,
    OPENMX_FP64_UNAVAILABLE,
    OPENMX_FP64_NATIVE,
    OPENMX_FP64_NATIVE_HIGH_THROUGHPUT
} OpenMXFP64Capability;

typedef enum {
    OPENMX_MEMORY_UNKNOWN = 0,
    OPENMX_MEMORY_DISCRETE,
    OPENMX_MEMORY_SHARED
} OpenMXMemoryModel;

typedef struct {
    OpenMXAcceleratorBackend backend;
    OpenMXFP64Capability fp64;
    OpenMXMemoryModel memory;
    /* Zero means unknown. No precision reduction may be inferred from it. */
    int fp32_to_fp64_ratio;
} OpenMXAcceleratorCapabilities;

static inline int openmx_accelerator_can_run_fp64(OpenMXAcceleratorCapabilities caps)
{
    return caps.fp64 == OPENMX_FP64_NATIVE ||
           caps.fp64 == OPENMX_FP64_NATIVE_HIGH_THROUGHPUT;
}

/* Calibrated policy for the existing CUDA worker implementation. HIP, SYCL
   and future Metal adapters need their own evidence before enabling overlap. */
static inline int openmx_accelerator_default_worker_overlap(OpenMXAcceleratorCapabilities caps)
{
    return caps.backend == OPENMX_ACCELERATOR_CUDA &&
           openmx_accelerator_can_run_fp64(caps) && caps.fp32_to_fp64_ratio >= 8;
}

#endif
