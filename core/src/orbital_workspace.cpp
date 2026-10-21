#include <openmx/core/orbital_workspace.hpp>
#include "openmx_core_orbital_workspace.h"

#include <climits>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>

namespace openmx::core {
namespace {
OrbitalShape validated(OrbitalShape shape)
{
  if (shape.maximum_angular < 0 || shape.maximum_angular > 14 || shape.multiplicity <= 0)
    throw std::invalid_argument("invalid orbital workspace dimensions");
  if (static_cast<std::size_t>(shape.maximum_angular+1)*shape.multiplicity > INT_MAX)
    throw std::length_error("orbital workspace exceeds C indexing range");
  return shape;
}
}
OrbitalWorkspace::OrbitalWorkspace(OrbitalShape shape)
  : OrbitalWorkspace(validated(shape),Validated{}) {}
OrbitalWorkspace::OrbitalWorkspace(OrbitalShape shape, Validated)
  : shape_(shape),
    radial_({shape.derivatives ? 2U : 1U,static_cast<std::size_t>(shape.maximum_angular+1),
             static_cast<std::size_t>(shape.multiplicity)}),
    angular_({shape.derivatives ? 3U : 1U,static_cast<std::size_t>(shape.maximum_angular+1),
              static_cast<std::size_t>(2*(shape.maximum_angular+1)+1)})
{}
OpenMXOrbitalBuffers OrbitalWorkspace::buffers() noexcept
{
  const auto r=radial(),a=angular();
  const auto rn=r.extent(1)*r.extent(2),an=a.extent(1)*a.extent(2);
  return {r.extent(1),r.extent(2),a.extent(2),r.data_handle(),
    shape_.derivatives ? r.data_handle()+rn : nullptr,a.data_handle(),
    shape_.derivatives ? a.data_handle()+an : nullptr,
    shape_.derivatives ? a.data_handle()+2*an : nullptr};
}
} // namespace openmx::core

struct OpenMXOrbitalWorkspace {
  explicit OpenMXOrbitalWorkspace(openmx::core::OrbitalShape shape) : implementation(shape) {}
  openmx::core::OrbitalWorkspace implementation;
};

extern "C" OpenMXOrbitalWorkspace *openmx_core_orbital_create(const OpenMXOrbitalShape *shape,
    OpenMXOrbitalBuffers *buffers, char *error, std::size_t capacity)
{
  if (error && capacity) *error='\0';
  try {
    if (!shape || !buffers || (shape->derivatives != 0 && shape->derivatives != 1))
      throw std::invalid_argument("orbital allocation requires shape, buffers and a derivative flag");
    auto owner=std::make_unique<OpenMXOrbitalWorkspace>(openmx::core::OrbitalShape{
      shape->maximum_angular,shape->multiplicity,shape->derivatives != 0});
    *buffers=owner->implementation.buffers();
    return owner.release();
  } catch (const std::exception& e) {
    if (error && capacity) std::snprintf(error,capacity,"%s",e.what());
  } catch (...) {
    if (error && capacity) std::snprintf(error,capacity,"unexpected orbital allocation failure");
  }
  return nullptr;
}
extern "C" void openmx_core_orbital_destroy(OpenMXOrbitalWorkspace *workspace)
{
  delete workspace;
}
