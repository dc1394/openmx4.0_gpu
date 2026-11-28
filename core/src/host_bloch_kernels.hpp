#ifndef OPENMX_HOST_BLOCH_KERNELS_HPP
#define OPENMX_HOST_BLOCH_KERNELS_HPP

#include "openmx_core_host_simd.h"
#include "host_simd_dispatch.hpp"

namespace openmx::core::host {
extern "C" OPENMX_HOST_SIMD_EXPORT const OpenMXHostBlochAdd* openmx_host_bloch_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OpenMXHostBlochAdd,bloch)
extern "C" OPENMX_HOST_SIMD_EXPORT const OpenMXHostMODot* openmx_host_mo_dot_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OpenMXHostMODot,mo_dot)
extern "C" OPENMX_HOST_SIMD_EXPORT const OpenMXHostWeightSum* openmx_host_weight_sum_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OpenMXHostWeightSum,weight_sum)
}
#endif
