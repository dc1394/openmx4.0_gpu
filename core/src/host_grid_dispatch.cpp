#include "host_grid_kernels.hpp"
#include <cstring>
#include <cctype>
#include <initializer_list>

namespace openmx::core::host {
OPENMX_HOST_SIMD_DISPATCHER(GridKernels,grid)
namespace {
bool equal(const char* value,const char* name) noexcept {
  if(!value)return false;
  while(*value && *name && std::tolower(static_cast<unsigned char>(*value))==*name) {
    ++value;++name;
  }
  return !*value && !*name;
}
}
int openmx_host_grid_mode(const char* value) noexcept {
  // Recognize NEON before the legacy first-letter "no" spelling.
  const int requested=equal(value,"neon")?OPENMX_GRID_NEON:
      equal(value,"sve128")?OPENMX_GRID_SVE128:equal(value,"sve256")?OPENMX_GRID_SVE256:
      equal(value,"sve512")?OPENMX_GRID_SVE512:-1;
  if(requested>=0 && openmx_host_grid_kernels(requested))return requested;
  // Retain the legacy environment parser's false/no/0 spellings.
  if (value && !equal(value,"neon") && (value[0]=='0' || value[0]=='n' || value[0]=='N' ||
                value[0]=='f' || value[0]=='F' || !std::strcmp(value,"scalar") ||
                !std::strcmp(value,"SCALAR"))) return OPENMX_GRID_SCALAR;
  const bool cap_avx2=value && (!std::strcmp(value,"avx2") || !std::strcmp(value,"AVX2"));
  for(int mode:{OPENMX_GRID_SVE512,OPENMX_GRID_SVE256,OPENMX_GRID_SVE128})
    if((!cap_avx2 || mode!=OPENMX_GRID_SVE512) && openmx_host_grid_kernels(mode))return mode;
  if (!cap_avx2 && openmx_host_grid_kernels(OPENMX_GRID_AVX512)) return OPENMX_GRID_AVX512;
  if (openmx_host_grid_kernels(OPENMX_GRID_AVX2)) return OPENMX_GRID_AVX2;
  if (openmx_host_grid_kernels(OPENMX_GRID_NEON)) return OPENMX_GRID_NEON;
  return OPENMX_GRID_SCALAR;
}
}
