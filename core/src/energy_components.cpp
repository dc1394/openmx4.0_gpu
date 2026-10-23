#include <openmx_core_energy.h>
#include "calculation_context.hpp"
#include <cstdio>

extern "C" int openmx_core_energy_components(OpenMXCoreContext* context,
    OpenMXEnergyComponents** out,char* error,size_t capacity)
{
  if(!context || !out) {
    if(error && capacity)std::snprintf(error,capacity,"missing energy component calculation/output");
    return 0;
  }
  if(error && capacity)*error='\0';
  *out=&context->energy_components;
  return 1;
}
