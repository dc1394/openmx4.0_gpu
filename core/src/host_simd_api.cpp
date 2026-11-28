#include "openmx_core_host_simd.h"
#include "host_add_rows_kernels.hpp"
#include "host_md_kernels.hpp"
#include "host_bloch_kernels.hpp"
#include "host_simd_runtime.hpp"
#include <cctype>
#include <string>

namespace {
int widest() {
  for(int mode:{OPENMX_GRID_SVE512,OPENMX_GRID_SVE256,OPENMX_GRID_SVE128,
               OPENMX_GRID_AVX512,OPENMX_GRID_AVX2,OPENMX_GRID_NEON})
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
      v=="neon"?OPENMX_GRID_NEON:v=="sve128"?OPENMX_GRID_SVE128:
      v=="sve256"?OPENMX_GRID_SVE256:v=="sve512"?OPENMX_GRID_SVE512:-1;
  if(requested>=0 && openmx::core::host::openmx_host_simd_usable(requested))return requested;
  return widest();
}
extern "C" const char* openmx_core_host_simd_name(int mode) {
  return mode==OPENMX_HOST_SIMD_LEGACY?"legacy":mode==OPENMX_GRID_AVX512?"AVX-512":
      mode==OPENMX_GRID_AVX2?"AVX2":mode==OPENMX_GRID_NEON?"NEON":
      mode==OPENMX_GRID_SVE128?"SVE128":mode==OPENMX_GRID_SVE256?"SVE256":
      mode==OPENMX_GRID_SVE512?"SVE512":"scalar";
}
extern "C" int openmx_core_host_simd_usable(int mode) {
  return openmx::core::host::openmx_host_simd_usable(mode);
}
extern "C" OpenMXHostAddRows openmx_core_host_add_rows(int mode) {
  const auto* kernel=openmx::core::host::openmx_host_add_rows_kernels(mode);
  return kernel ? *kernel : nullptr;
}
extern "C" OpenMXHostSum openmx_core_host_sum(int mode) {
  const auto* kernel=openmx::core::host::openmx_host_sum_kernels(mode);
  return kernel ? *kernel : nullptr;
}
extern "C" OpenMXHostDot openmx_core_host_dot(int mode) {
  const auto* kernel=openmx::core::host::openmx_host_dot_kernels(mode);
  return kernel ? *kernel : nullptr;
}
extern "C" OpenMXHostUpdateRow openmx_core_host_update_row(int mode) {
  const auto* kernel=openmx::core::host::openmx_host_update_row_kernels(mode);
  return kernel ? *kernel : nullptr;
}
extern "C" OpenMXHostMDMove openmx_core_host_md_move(int mode) {
  const auto* kernel=openmx::core::host::openmx_host_md_move_kernels(mode);
  return kernel ? *kernel : nullptr;
}
extern "C" OpenMXHostMDPeriodic openmx_core_host_md_periodic(int mode) {
  const auto* kernel=openmx::core::host::openmx_host_md_periodic_kernels(mode);
  return kernel ? *kernel : nullptr;
}
extern "C" OpenMXHostMDNPT openmx_core_host_md_npt(int mode) {
  const auto* kernel=openmx::core::host::openmx_host_md_npt_kernels(mode);
  return kernel ? *kernel : nullptr;
}
extern "C" OpenMXHostBlochAdd openmx_core_host_bloch_add(int mode) {
  const auto* kernel=openmx::core::host::openmx_host_bloch_kernels(mode);
  return kernel ? *kernel : nullptr;
}
extern "C" OpenMXHostMODot openmx_core_host_mo_dot(int mode) {
  const auto* kernel=openmx::core::host::openmx_host_mo_dot_kernels(mode);
  return kernel ? *kernel : nullptr;
}
extern "C" OpenMXHostWeightSum openmx_core_host_weight_sum(int mode) {
  const auto* kernel=openmx::core::host::openmx_host_weight_sum_kernels(mode);
  return kernel ? *kernel : nullptr;
}
