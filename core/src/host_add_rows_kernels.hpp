#ifndef OPENMX_HOST_ADD_ROWS_KERNELS_HPP
#define OPENMX_HOST_ADD_ROWS_KERNELS_HPP
#include "openmx_core_host_simd.h"
#include "host_simd_dispatch.hpp"

namespace openmx::core::host {
extern "C" OPENMX_HOST_SIMD_EXPORT const OpenMXHostAddRows* openmx_host_add_rows_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OpenMXHostAddRows,add_rows)
}
#endif
