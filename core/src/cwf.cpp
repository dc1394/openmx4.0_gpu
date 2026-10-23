#include <openmx_core_cwf.h>
#include "calculation_context.hpp"
#include <cstdio>
#include <exception>

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

extern "C" int openmx_core_cwf_observables(OpenMXCoreContext* context,
    OpenMXCWFObservables** out,char* error,size_t capacity)
{
  if(!context || !out) {
    if(error && capacity)std::snprintf(error,capacity,"missing CWF observables calculation/output");
    return 0;
  }
  if(error && capacity)*error='\0';
  *out=context->cwf_observables.view();return 1;
}

extern "C" int openmx_core_cwf_observables_prepare(OpenMXCoreContext* context,
    size_t spins,size_t orbitals,OpenMXCWFObservables** out,char* error,size_t capacity)
{
  if(error && capacity)*error='\0';
  if(!context || !out) {
    if(error && capacity)std::snprintf(error,capacity,"missing CWF observables calculation/output");
    return 0;
  }
  try {
    context->cwf_observables.prepare(spins,orbitals);
    *out=context->cwf_observables.view();return 1;
  }catch(const std::exception& failure) {
    if(error && capacity)std::snprintf(error,capacity,"%s",failure.what());
  }catch(...) {
    if(error && capacity)std::snprintf(error,capacity,"unexpected CWF observables allocation failure");
  }
  return 0;
}
