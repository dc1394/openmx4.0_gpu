#ifndef OPENMX_HOST_MD_KERNELS_HPP
#define OPENMX_HOST_MD_KERNELS_HPP

#include "openmx_core_md.h"
#include "host_simd_dispatch.hpp"

namespace openmx::core::host {
extern "C" OPENMX_HOST_SIMD_EXPORT const OpenMXHostMDMove* openmx_host_md_move_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OpenMXHostMDMove,md_move)
extern "C" OPENMX_HOST_SIMD_EXPORT const OpenMXHostMDPeriodic* openmx_host_md_periodic_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OpenMXHostMDPeriodic,md_periodic)
extern "C" OPENMX_HOST_SIMD_EXPORT const OpenMXHostMDNPT* openmx_host_md_npt_kernels(int mode) noexcept;
OPENMX_HOST_SIMD_FAMILY(OpenMXHostMDNPT,md_npt)
}
#endif
