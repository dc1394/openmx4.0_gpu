#include "openmx_core_real_harmonics.h"
#include "calculation_context.hpp"
#include <cstdio>
#include <stdexcept>

extern "C" int openmx_core_real_harmonics_initialize(OpenMXCoreContext* context,
    int maximum, OpenMXRealHarmonicData* out, char* error, std::size_t capacity)
{
  if(error && capacity)*error='\0';
  try {
    if(!context || !out || context->implementation.finished())
      throw std::invalid_argument("real harmonics require an unfinished context and output");
    if(context->real_harmonics) {
      if(context->real_harmonics->maximum_degree()!=maximum)
        throw std::invalid_argument("cannot resize initialized real harmonic transform");
    }else {
      auto owner=std::make_unique<openmx::basis::RealHarmonicTransform>(maximum);
      context->real_harmonics=std::move(owner);
    }
    *out=context->real_harmonics->view();
    return 1;
  }catch(const std::exception& exception) {
    if(error && capacity)std::snprintf(error,capacity,"%s",exception.what());
  }catch(...) {
    if(error && capacity)std::snprintf(error,capacity,"unknown real harmonic initialization error");
  }
  return 0;
}

extern "C" int openmx_core_real_harmonics_get(const OpenMXCoreContext* context,
    OpenMXRealHarmonicData* out, char* error, std::size_t capacity)
{
  if(error && capacity)*error='\0';
  if(!context || !out || context->implementation.finished() || !context->real_harmonics) {
    if(error && capacity)std::snprintf(error,capacity,"missing or finished real harmonic transform");
    return 0;
  }
  *out=context->real_harmonics->view();
  return 1;
}
