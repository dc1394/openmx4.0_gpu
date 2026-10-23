#include <openmx_core_mixing.h>
#include "calculation_context.hpp"
#include <cstdio>

extern "C" int openmx_core_mixing_history(OpenMXCoreContext* context,
    OpenMXMixingHistory** out,char* error,size_t capacity)
{
  if(!context || !out) {
    if(error && capacity)std::snprintf(error,capacity,"missing mixing history calculation/output");
    return 0;
  }
  if(error && capacity)*error='\0';
  *out=&context->mixing_history;
  return 1;
}
