#include "calculation_context.hpp"
#include "openmx_core_atomic.h"
#include <cstdio>

extern "C" int openmx_core_atomic_get(OpenMXCoreContext *context,
    OpenMXAtomicViews *views, char *error, std::size_t capacity)
{
  if (error && capacity) *error='\0';
  if (!context || !views || context->implementation.finished()) {
    if (error && capacity) std::snprintf(error,capacity,"missing or finished atomic calculation context");
    return 0;
  }
  OpenMXAtomicViews result{};
  if (context->species) result.species=context->species->views();
  if (context->pao) context->pao->atomic_views(result);
  if (context->vps) context->vps->atomic_views(result);
  *views=result;
  return 1;
}
