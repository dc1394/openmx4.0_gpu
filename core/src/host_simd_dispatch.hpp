#ifndef OPENMX_HOST_SIMD_DISPATCH_HPP
#define OPENMX_HOST_SIMD_DISPATCH_HPP
#include "openmx_core_grid.h"
#include "host_simd_runtime.hpp"
// Run-time selection among the per-ISA factories of one kernel family. The
// translation unit that expands OPENMX_HOST_SIMD_DISPATCHER is compiled with
// the ISA macros of the build, so absent variants dispatch to nullptr instead
// of an undefined symbol. All families use the OPENMX_GRID_* mode enum.
namespace openmx::core::host {
template<class Table> const Table* host_simd_select(int mode,const Table&(*scalar)() noexcept,
    const Table&(*avx2)() noexcept,const Table&(*avx512)() noexcept,
    const Table&(*neon)() noexcept,const Table&(*sve128)() noexcept,
    const Table&(*sve256)() noexcept,const Table&(*sve512)() noexcept) noexcept {
  if(!openmx_host_simd_usable(mode))return nullptr;
  switch(mode) {
    case OPENMX_GRID_SCALAR:return &scalar();
    case OPENMX_GRID_AVX2:return avx2?&avx2():nullptr;
    case OPENMX_GRID_AVX512:return avx512?&avx512():nullptr;
    case OPENMX_GRID_NEON:return neon?&neon():nullptr;
    case OPENMX_GRID_SVE128:return sve128?&sve128():nullptr;
    case OPENMX_GRID_SVE256:return sve256?&sve256():nullptr;
    case OPENMX_GRID_SVE512:return sve512?&sve512():nullptr;
    default:return nullptr;
  }
}
}
#if defined(OPENMX_HOST_SIMD_AVX2)
#define OPENMX_HOST_SIMD_AVX2_FACTORY(family) &family##_avx2
#else
#define OPENMX_HOST_SIMD_AVX2_FACTORY(family) nullptr
#endif
#if defined(OPENMX_HOST_SIMD_AVX512)
#define OPENMX_HOST_SIMD_AVX512_FACTORY(family) &family##_avx512
#else
#define OPENMX_HOST_SIMD_AVX512_FACTORY(family) nullptr
#endif
#if defined(OPENMX_HOST_SIMD_NEON)
#define OPENMX_HOST_SIMD_NEON_FACTORY(family) &family##_neon
#else
#define OPENMX_HOST_SIMD_NEON_FACTORY(family) nullptr
#endif
#if defined(OPENMX_HOST_SIMD_SVE128)
#define OPENMX_HOST_SIMD_SVE128_FACTORY(family) &family##_sve128
#else
#define OPENMX_HOST_SIMD_SVE128_FACTORY(family) nullptr
#endif
#if defined(OPENMX_HOST_SIMD_SVE256)
#define OPENMX_HOST_SIMD_SVE256_FACTORY(family) &family##_sve256
#else
#define OPENMX_HOST_SIMD_SVE256_FACTORY(family) nullptr
#endif
#if defined(OPENMX_HOST_SIMD_SVE512)
#define OPENMX_HOST_SIMD_SVE512_FACTORY(family) &family##_sve512
#else
#define OPENMX_HOST_SIMD_SVE512_FACTORY(family) nullptr
#endif
// Declares the factories of a family (each per-ISA translation unit
// defines one of them through OPENMX_HOST_SIMD_FACTORY).
#define OPENMX_HOST_SIMD_FAMILY(Table,family) \
  const Table& family##_scalar() noexcept; const Table& family##_avx2() noexcept; \
  const Table& family##_avx512() noexcept; const Table& family##_neon() noexcept; \
  const Table& family##_sve128() noexcept; const Table& family##_sve256() noexcept; \
  const Table& family##_sve512() noexcept;
// Defines the exported C dispatcher of a family; expand inside openmx::core::host.
#define OPENMX_HOST_SIMD_DISPATCHER(Table,family) \
  extern "C" const Table* openmx_host_##family##_kernels(int mode) noexcept { \
    return host_simd_select<Table>(mode,&family##_scalar, \
        OPENMX_HOST_SIMD_AVX2_FACTORY(family),OPENMX_HOST_SIMD_AVX512_FACTORY(family), \
        OPENMX_HOST_SIMD_NEON_FACTORY(family),OPENMX_HOST_SIMD_SVE128_FACTORY(family), \
        OPENMX_HOST_SIMD_SVE256_FACTORY(family),OPENMX_HOST_SIMD_SVE512_FACTORY(family)); \
  }
#endif
