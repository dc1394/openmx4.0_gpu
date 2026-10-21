#include "calculation_context.hpp"
#include "openmx_core_reciprocal.h"
#include <cstdio>

extern "C" int openmx_core_reciprocal_get(OpenMXCoreContext *context,
    OpenMXReciprocalViews *views, char *error, std::size_t capacity)
{
  if (error && capacity) *error='\0';
  if (!context || !views || context->implementation.finished()) {
    if (error && capacity) std::snprintf(error,capacity,"missing or finished reciprocal calculation context");
    return 0;
  }
  OpenMXReciprocalViews result{};
  if (context->pao) context->pao->reciprocal_views(result);
  if (context->vps) context->vps->reciprocal_views(result);
  if (context->products) context->products->reciprocal_views(result);
  *views=result;
  return 1;
}
