#include <openmx_core_energy.h>
#include "calculation_context.hpp"
#include <cstdio>

extern "C" int openmx_core_thermodynamic_energies(OpenMXCoreContext* context,
    OpenMXThermodynamicEnergies** out,char* error,size_t capacity)
{
  if(!context || !out) {
    if(error && capacity)std::snprintf(error,capacity,"missing thermodynamic energy calculation/output");
    return 0;
  }
  if(error && capacity)*error='\0';
  *out=&context->thermodynamic_energies;
  return 1;
}
