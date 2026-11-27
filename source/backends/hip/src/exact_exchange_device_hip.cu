#include <hip/hip_runtime.h>

#include "openmx_exact_exchange_hip.h"

#include <algorithm>
#include <cstddef>

namespace {
constexpr int threads = 256;

__global__ void move_elements(int rows, int columns, const int *lengths, double alpha, bool add,
    double *target, OpenMXExxDeviceIndex to, const double *source, OpenMXExxDeviceIndex from)
{
    const size_t count = (size_t)rows * columns;
    const size_t step = (size_t)gridDim.x * blockDim.x;
    for (size_t entry = (size_t)blockIdx.x * blockDim.x + threadIdx.x; entry < count; entry += step) {
        const int row = entry / columns, col = entry % columns;
        if (lengths && col >= lengths[row]) continue;
        const long long t = (to.rows ? to.rows[row] : row * to.stride)
                          + (to.columns ? to.columns[col] : col * to.step);
        const long long s = (from.rows ? from.rows[row] : row * from.stride)
                          + (from.columns ? from.columns[col] : col * from.step);
        const double value = alpha * source[s];
        if (add) target[t] += value;
        else target[t] = value;
    }
}
}

extern "C" OpenMXGpuError Exact_Exchange_HIP_Move(OpenMXGpuStream stream, int rows, int columns,
    const int *lengths, double alpha, int add, double *target, OpenMXExxDeviceIndex target_index,
    const double *source, OpenMXExxDeviceIndex source_index)
{
    if (rows == 0 || columns == 0) return OPENMX_GPU_SUCCESS;
    const size_t blocks = ((size_t)rows * columns + threads - 1) / threads;
    move_elements<<<std::min(blocks, size_t{65535}), threads, 0, stream>>>(
        rows, columns, lengths, alpha, add != 0, target, target_index, source, source_index);
    return OpenMXGpuGetLastError();
}

/* The solver lends its transform scratch until it starts the eigensolve. */
extern "C" OpenMXGpuError Exact_Exchange_HIP_Add(OpenMXGpuStream stream, int rows, int columns,
    double *target, double *scratch, const double *host)
{
    const OpenMXGpuError uploaded = OpenMXGpuMemcpyAsync(scratch, host, (size_t)rows * columns * sizeof(double),
                                                        OPENMX_GPU_COPY_HOST_TO_DEVICE, stream);
    if (uploaded != OPENMX_GPU_SUCCESS) return uploaded;
    OpenMXExxDeviceIndex dense{};
    dense.stride = rows;
    dense.step = 1;
    const OpenMXGpuError launched = Exact_Exchange_HIP_Move(stream, columns, rows, nullptr, 1.0, 1,
                                                           target, dense, scratch, dense);
    const OpenMXGpuError finished = OpenMXGpuStreamSynchronize(stream);
    return launched == OPENMX_GPU_SUCCESS ? finished : launched;
}
