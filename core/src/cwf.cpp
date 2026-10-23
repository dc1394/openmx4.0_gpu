#include <openmx_core_cwf.h>
#include "calculation_context.hpp"
#include <cstdio>

extern "C" int openmx_core_cwf_grids(OpenMXCoreContext* context,
    OpenMXCWFGrids** out,char* error,size_t capacity)
{
  if(!context || !out) {
    if(error && capacity)std::snprintf(error,capacity,"missing CWF grids calculation/output");
    return 0;
  }
  if(error && capacity)*error='\0';
  *out=&context->cwf_grids;
  return 1;
}
