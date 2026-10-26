#ifndef OPENMX_SETPRO_HOST_H
#define OPENMX_SETPRO_HOST_H
/* Allocation of the host SIMD tables of the Set_ProExpn_VNA CPU headers
   (openmx_dsvna_cpu.h, openmx_vna_cpu.h, openmx_hvna_cpu.h). Like those
   headers it relies on the includer's SetPro_gpu_abort(). */
#include <stdlib.h>

static inline void *SetPro_HostAlloc(size_t bytes)
{
  void *pointer=malloc(bytes ? bytes : 1);
  if (pointer==NULL) {
    SetPro_gpu_abort("Set_ProExpn_VNA: host allocation failed for a host SIMD table.");
    abort();
  }
  return pointer;
}
#endif
