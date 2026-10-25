#ifndef OPENMX_HOST_SIMD_ABI_HPP
#define OPENMX_HOST_SIMD_ABI_HPP
// Shared by every translation unit of the private host SIMD library that is
// compiled once per ISA (scalar, AVX2, AVX-512, NEON). The explicit ABIs
// matter: simd<double> follows DefaultExecutionSpace and is scalar when that
// space is CUDA/HIP/SYCL in Kokkos 5.2.2. Do not alter Kokkos's global
// architecture macros, or expose these headers to device translation units.
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__SYCL_DEVICE_ONLY__)
#error "Host SIMD kernels must be compiled as ordinary C++, without device compilation"
#endif
#include <cstdint>
#if defined(OPENMX_HOST_SIMD_AVX512)
#include <Kokkos_SIMD_AVX512.hpp>
#define OPENMX_HOST_SIMD_VARIANT avx512
#elif defined(OPENMX_HOST_SIMD_AVX2)
#include <Kokkos_SIMD_AVX2.hpp>
#define OPENMX_HOST_SIMD_VARIANT avx2
#elif defined(OPENMX_HOST_SIMD_NEON)
#include <Kokkos_SIMD_NEON.hpp>
#define OPENMX_HOST_SIMD_VARIANT neon
#else
#include <Kokkos_SIMD_Scalar.hpp>
#define OPENMX_HOST_SIMD_VARIANT scalar
#endif
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
#error "Host SIMD requires its private SERIAL Kokkos configuration"
#endif
#define OPENMX_HOST_SIMD_CONCAT_(a,b) a##_##b
#define OPENMX_HOST_SIMD_CONCAT(a,b) OPENMX_HOST_SIMD_CONCAT_(a,b)
// The factory this translation unit defines: family_scalar/_avx2/_avx512/_neon.
#define OPENMX_HOST_SIMD_FACTORY(family) OPENMX_HOST_SIMD_CONCAT(family,OPENMX_HOST_SIMD_VARIANT)

namespace openmx::core::host {
namespace simd=Kokkos::Experimental;
#if defined(OPENMX_HOST_SIMD_AVX512)
using HostAbi=simd::simd_abi::avx512_fixed_size<8>;       // double, int64, and int32 indices of doubles
using HostWideAbi=simd::simd_abi::avx512_fixed_size<16>;  // float, int32
#elif defined(OPENMX_HOST_SIMD_AVX2)
using HostAbi=simd::simd_abi::avx2_fixed_size<4>;
using HostWideAbi=simd::simd_abi::avx2_fixed_size<8>;
#elif defined(OPENMX_HOST_SIMD_NEON)
using HostAbi=simd::simd_abi::neon_fixed_size<2>;
using HostWideAbi=simd::simd_abi::neon_fixed_size<4>;
#else
using HostAbi=simd::simd_abi::scalar;
using HostWideAbi=simd::simd_abi::scalar;
#endif
using Vector=simd::basic_simd<double,HostAbi>;
using IndexVector=simd::basic_simd<std::int32_t,HostAbi>;      // gather indices of a Vector
using IntVector=simd::basic_simd<std::int32_t,HostWideAbi>;
using FloatVector=simd::basic_simd<float,HostWideAbi>;
constexpr int lanes=Vector::size();
constexpr int wide_lanes=IntVector::size();
#if defined(OPENMX_HOST_SIMD_AVX512)
static_assert(lanes==8 && wide_lanes==16);
#elif defined(OPENMX_HOST_SIMD_AVX2)
static_assert(lanes==4 && wide_lanes==8);
#elif defined(OPENMX_HOST_SIMD_NEON)
static_assert(lanes==2 && wide_lanes==4);
#else
static_assert(lanes==1 && wide_lanes==1);
#endif
inline Vector load(const double* p) { return simd::simd_unchecked_load<Vector>(p); }
inline void store(Vector v,double* p) { simd::simd_unchecked_store(v,p); }
}
#endif
