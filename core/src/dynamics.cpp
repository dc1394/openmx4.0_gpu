#include <openmx_core_dynamics.h>
#include "calculation_context.hpp"
#include <cstdio>

extern "C" int openmx_core_dynamics(OpenMXCoreContext* context,
    OpenMXDynamicsState** out,char* error,size_t capacity)
{
  if(!context || !out) {
    if(error && capacity)std::snprintf(error,capacity,"missing dynamics calculation/output");
    return 0;
  }
  if(error && capacity)*error='\0';
  *out=&context->dynamics;
  return 1;
}
