#ifndef OPENMX_HOST_SIMD_RUNTIME_HPP
#define OPENMX_HOST_SIMD_RUNTIME_HPP
#include "host_simd_export.hpp"
// Shared by the private host SIMD library (definition) and the Core's C ABI
// layer (caller). The mode numbering is OPENMX_GRID_SCALAR/AVX2/AVX512/NEON
// for every kernel family.
namespace openmx::core::host {
// Nonzero when the variant was compiled into the library and the processor
// and operating system execute it. Scalar is always usable. Never initializes
// Kokkos or a device.
extern "C" OPENMX_HOST_SIMD_EXPORT int openmx_host_simd_usable(int mode) noexcept;
}
#endif
