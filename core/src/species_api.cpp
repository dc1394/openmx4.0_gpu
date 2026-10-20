#include "calculation_context.hpp"
#include <cstdio>
#include <exception>

extern "C" int openmx_core_species_allocate(OpenMXCoreContext *context,
    const OpenMXSpeciesShape *shape, OpenMXSpeciesViews *views, char *error, std::size_t capacity)
{
  if (error && capacity) *error='\0';
  try {
    if (!context || !shape || !views || context->species || context->implementation.finished())
      throw std::invalid_argument("species allocation requires an unfinished empty context, shape and views");
    if (shape->species <= 0 || shape->angular_channels <= 0 || shape->text_capacity < 2)
      throw std::invalid_argument("invalid species metadata dimensions");
    auto owner=std::make_unique<openmx::core::compat::SpeciesOwner>(openmx::core::SpeciesShape{
      static_cast<std::size_t>(shape->species),static_cast<std::size_t>(shape->angular_channels),
      static_cast<std::size_t>(shape->text_capacity)});
    *views=owner->views();
    context->species=std::move(owner);
    return 1;
  } catch (const std::exception& e) {
    if (error && capacity) std::snprintf(error,capacity,"%s",e.what());
  } catch (...) {
    if (error && capacity) std::snprintf(error,capacity,"unexpected species allocation failure");
  }
  return 0;
}
extern "C" int openmx_core_species_get(OpenMXCoreContext *context,
    OpenMXSpeciesViews *views, char *error, std::size_t capacity)
{
  if (error && capacity) *error='\0';
  if (!context || !views || context->implementation.finished()) {
    if (error && capacity) std::snprintf(error,capacity,"missing or finished species calculation context");
    return 0;
  }
  *views=context->species ? context->species->views() : OpenMXSpeciesViews{};
  return 1;
}
extern "C" void openmx_core_species_release(OpenMXCoreContext *context)
{
  if (context) context->species.reset();
}
