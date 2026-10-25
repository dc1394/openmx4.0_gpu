#include "openmx_core_host_simd.h"
#include "host_simd_runtime.hpp"
#include <cctype>
#include <string>

namespace {
int widest() {
  for(int mode:{OPENMX_GRID_AVX512,OPENMX_GRID_AVX2,OPENMX_GRID_NEON})
    if(openmx::core::host::openmx_host_simd_usable(mode))return mode;
  return OPENMX_GRID_SCALAR;
}
std::string lowered(const char* value) {
  std::string text;
  for(;value && *value;++value)text.push_back(char(std::tolower(static_cast<unsigned char>(*value))));
  return text;
}
}
extern "C" int openmx_core_host_simd_mode(const char* value) {
  const std::string v=lowered(value);
  if(v.empty() || v=="auto" || v=="1" || v=="on" || v=="yes" || v=="true")return widest();
  if(v=="legacy" || v=="off" || v=="0" || v=="no" || v=="false" || v=="none")return OPENMX_HOST_SIMD_LEGACY;
  if(v=="scalar")return OPENMX_GRID_SCALAR;
  const int requested=v=="avx2"?OPENMX_GRID_AVX2:(v=="avx512" || v=="avx-512")?OPENMX_GRID_AVX512:
      v=="neon"?OPENMX_GRID_NEON:-1;
  if(requested>=0 && openmx::core::host::openmx_host_simd_usable(requested))return requested;
  return widest();
}
extern "C" const char* openmx_core_host_simd_name(int mode) {
  return mode==OPENMX_HOST_SIMD_LEGACY?"legacy":mode==OPENMX_GRID_AVX512?"AVX-512":
      mode==OPENMX_GRID_AVX2?"AVX2":mode==OPENMX_GRID_NEON?"NEON":"scalar";
}
extern "C" int openmx_core_host_simd_usable(int mode) {
  return openmx::core::host::openmx_host_simd_usable(mode);
}
