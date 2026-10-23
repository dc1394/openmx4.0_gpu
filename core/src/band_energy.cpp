#include <openmx_core_energy.h>
#include "calculation_context.hpp"
#include <cstdio>

extern "C" int openmx_core_band_energies(OpenMXCoreContext* context,
    OpenMXBandEnergies** out,char* error,size_t capacity)
{
  if(!context || !out) {
    if(error && capacity)std::snprintf(error,capacity,"missing band energy calculation/output");
    return 0;
  }
  if(error && capacity)*error='\0';
  *out=&context->band_energies;
  return 1;
}
