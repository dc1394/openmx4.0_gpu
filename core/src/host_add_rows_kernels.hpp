#ifndef OPENMX_HOST_ADD_ROWS_KERNELS_HPP
#define OPENMX_HOST_ADD_ROWS_KERNELS_HPP
#include "openmx_core_host_simd.h"
#include "host_simd_dispatch.hpp"

namespace openmx::core::host {
extern "C" OPENMX_HOST_SIMD_EXPORT const OpenMXHostAddRows* openmx_host_add_rows_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OpenMXHostAddRows,add_rows)
extern "C" OPENMX_HOST_SIMD_EXPORT const OpenMXHostSum* openmx_host_sum_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OpenMXHostSum,sum)
extern "C" OPENMX_HOST_SIMD_EXPORT const OpenMXHostDot* openmx_host_dot_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OpenMXHostDot,dot)
extern "C" OPENMX_HOST_SIMD_EXPORT const OpenMXHostUpdateRow* openmx_host_update_row_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OpenMXHostUpdateRow,update_row)
}
#endif
