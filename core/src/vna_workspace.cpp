#include <openmx/core/vna_workspace.hpp>
#include <climits>
#include <stdexcept>

namespace openmx::core {
namespace {
VnaShape validated(VnaShape shape)
{
  if (shape.maximum_angular < 0 || shape.maximum_angular > 14 ||
      shape.multiplicity <= 0 || shape.points <= 0)
    throw std::invalid_argument("invalid neutral-potential workspace shape");
  if (shape.multiplicity > INT_MAX/(shape.maximum_angular+1)/(shape.maximum_angular+1) ||
      shape.points > INT_MAX/(2*shape.maximum_angular+1))
    throw std::length_error("neutral-potential offsets exceed C indexing range");
  return shape;
}
std::size_t orbitals(VnaShape shape)
{
  return static_cast<std::size_t>(shape.multiplicity)*(shape.maximum_angular+1)*(shape.maximum_angular+1);
}
} // namespace

VnaWorkspace::VnaWorkspace(VnaShape shape)
  : shape_(validated(shape)), angular_({4,orbitals(shape_),orbitals(shape_)}),
    two_center_(4,shape_.maximum_angular,shape_.maximum_angular),
    bessel_({2,static_cast<std::size_t>(2*shape_.maximum_angular+1),
               static_cast<std::size_t>(shape_.points)}),
    scratch_({2,static_cast<std::size_t>(2*shape_.maximum_angular+3)})
{}

} // namespace openmx::core
