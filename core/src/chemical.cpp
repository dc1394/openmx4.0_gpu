#include <openmx_core_chemical.h>
#include "calculation_context.hpp"
#include <cstdio>

extern "C" int openmx_core_chemical_potentials(OpenMXCoreContext* context,
    OpenMXChemicalPotentials** out,char* error,size_t capacity)
{
  if(!context || !out) {
    if(error && capacity)std::snprintf(error,capacity,"missing chemical-potential calculation/output");
    return 0;
  }
  if(error && capacity)*error='\0';
  *out=&context->chemical;
  return 1;
}
