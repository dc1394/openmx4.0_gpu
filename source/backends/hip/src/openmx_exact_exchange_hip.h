#ifndef OPENMX_EXACT_EXCHANGE_HIP_H
#define OPENMX_EXACT_EXCHANGE_HIP_H

#include "openmx_exx.h"
#include "openmx_gpu_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

OpenMXGpuError Exact_Exchange_HIP_Move(OpenMXGpuStream stream, int rows, int columns,
    const int *lengths, double alpha, int add, double *target, OpenMXExxDeviceIndex target_index,
    const double *source, OpenMXExxDeviceIndex source_index);

OpenMXGpuError Exact_Exchange_HIP_Add(OpenMXGpuStream stream, int rows, int columns,
    double *target, double *scratch, const double *host);

#ifdef __cplusplus
}
#endif

#endif
