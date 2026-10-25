#ifndef OPENMX_HOST_SIMD_DISPATCH_HPP
#define OPENMX_HOST_SIMD_DISPATCH_HPP
#include "openmx_core_grid.h"
#include "host_simd_runtime.hpp"
// Run-time selection among the per-ISA factories of one kernel family. The
// translation unit that expands OPENMX_HOST_SIMD_DISPATCHER is compiled with
// the ISA macros of the build, so absent variants dispatch to nullptr instead
// of an undefined symbol. Mode numbering: OPENMX_GRID_SCALAR/AVX2/AVX512/NEON.
namespace openmx::core::host {
template<class Table> const Table* host_simd_select(int mode,const Table&(*scalar)() noexcept,
    const Table&(*avx2)() noexcept,const Table&(*avx512)() noexcept,
    const Table&(*neon)() noexcept) noexcept {
  if(!openmx_host_simd_usable(mode))return nullptr;
  switch(mode) {
    case OPENMX_GRID_SCALAR:return &scalar();
    case OPENMX_GRID_AVX2:return avx2?&avx2():nullptr;
    case OPENMX_GRID_AVX512:return avx512?&avx512():nullptr;
    case OPENMX_GRID_NEON:return neon?&neon():nullptr;
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
// Declares the four factories of a family (each per-ISA translation unit
// defines one of them through OPENMX_HOST_SIMD_FACTORY).
#define OPENMX_HOST_SIMD_FAMILY(Table,family) \
  const Table& family##_scalar() noexcept; const Table& family##_avx2() noexcept; \
  const Table& family##_avx512() noexcept; const Table& family##_neon() noexcept;
// Defines the exported C dispatcher of a family; expand inside openmx::core::host.
#define OPENMX_HOST_SIMD_DISPATCHER(Table,family) \
  extern "C" const Table* openmx_host_##family##_kernels(int mode) noexcept { \
    return host_simd_select<Table>(mode,&family##_scalar, \
        OPENMX_HOST_SIMD_AVX2_FACTORY(family),OPENMX_HOST_SIMD_AVX512_FACTORY(family), \
        OPENMX_HOST_SIMD_NEON_FACTORY(family)); \
  }
#endif
