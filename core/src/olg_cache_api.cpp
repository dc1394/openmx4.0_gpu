#include "calculation_context.hpp"
#include "openmx_core_olg_cache.h"
#include <cstdio>
#include <exception>

namespace {
template<class F> int checked(char* error,size_t capacity,F&& f) noexcept {
  if(error && capacity)*error='\0';
  try {f();return 1;}
  catch(const std::exception& e) {if(error && capacity)std::snprintf(error,capacity,"%s",e.what());}
  catch(...) {if(error && capacity)std::snprintf(error,capacity,"overlap-list cache failed");}
  return 0;
}
void require(bool value,const char* message) {if(!value)throw std::invalid_argument(message);}
}
extern "C" int openmx_core_olg_cache_prepare(OpenMXCoreContext* context,const OpenMXOlgCacheInput* input,
    char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(context && input,"missing overlap-list calculation/input");
    auto owner=std::make_unique<openmx::core::OlgCache>(*input);
    context->olg_cache=std::move(owner);
  });
}
extern "C" int openmx_core_olg_cache_matches(OpenMXCoreContext* context,const OpenMXOlgCacheInput* input,
    int* matches,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(context && input && matches,"missing overlap-list calculation/input/match output");
    openmx::core::OlgCache::validate(*input);
    *matches=context->olg_cache && context->olg_cache->matches(*input);
  });
}
extern "C" int openmx_core_olg_cache_store(OpenMXCoreContext* context,size_t atom,size_t pair,size_t count,
    const int* center,const int* neighbor,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(context && context->olg_cache,"missing overlap-list cache");
    context->olg_cache->store(atom,pair,count,center,neighbor);
  });
}
extern "C" int openmx_core_olg_cache_take_atom(OpenMXCoreContext* context,size_t atom,OpenMXOlgCacheAtom* out,
    char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(context && context->olg_cache && out,"missing overlap-list cache/atom output");
    *out=context->olg_cache->take_atom(atom);
  });
}
extern "C" void openmx_core_olg_cache_release(OpenMXCoreContext* context) {
  if(context)context->olg_cache.reset();
}
