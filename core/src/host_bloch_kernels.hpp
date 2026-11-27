#ifndef OPENMX_HOST_BLOCH_KERNELS_HPP
#define OPENMX_HOST_BLOCH_KERNELS_HPP

#include "openmx_core_host_simd.h"
#include "host_simd_dispatch.hpp"

namespace openmx::core::host {
extern "C" OPENMX_HOST_SIMD_EXPORT const OpenMXHostBlochAdd* openmx_host_bloch_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OpenMXHostBlochAdd,bloch)
}
#endif
