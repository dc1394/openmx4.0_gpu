#include <openmx/core/vps_storage.hpp>

namespace openmx::core {
namespace {
VpsShape validate(VpsShape shape)
{
  if (!shape.species || (shape.spins != 1 && shape.spins != 2) || !shape.projectors ||
      !shape.radial_points || !shape.reciprocal_points)
    throw std::invalid_argument("invalid VPS storage dimensions");
  if (shape.radial_points > std::numeric_limits<std::size_t>::max()-2 ||
      shape.projectors > std::numeric_limits<std::size_t>::max()-2)
    throw std::length_error("VPS storage padding overflows size_t");
  return shape;
}

NeutralProjectorShape validate(NeutralProjectorShape shape)
{
  if (!shape.species || !shape.angular_channels || !shape.multiplicity ||
      !shape.radial_points || !shape.wavevectors)
    throw std::invalid_argument("invalid neutral-projector storage dimensions");
  if (shape.wavevectors > std::numeric_limits<std::size_t>::max()-2)
    throw std::length_error("neutral-projector padding overflows size_t");
  return shape;
}
} // namespace

VpsStorage::VpsStorage(VpsShape shape) : VpsStorage(validate(shape),Validated{}) {}
VpsStorage::VpsStorage(VpsShape s, Validated)
  : log_radii_({s.species,s.radial_points}), radii_({s.species,s.radial_points}),
    neutral_potential_({s.species,s.radial_points}),
    hartree_potential_({s.species,s.radial_points+2}), partial_core_density_({s.species,s.radial_points+2}),
    projectors_({s.spins,s.species,s.projectors,s.radial_points}),
    energies_({s.spins,s.species,s.projectors}), angular_({s.species,s.projectors}),
    transforms_({s.spins,s.species,s.projectors+2,s.reciprocal_points}) {}

NeutralProjectorStorage::NeutralProjectorStorage(NeutralProjectorShape shape)
  : NeutralProjectorStorage(validate(shape),Validated{}) {}
NeutralProjectorStorage::NeutralProjectorStorage(NeutralProjectorShape s, Validated)
  : projectors_({s.species,s.angular_channels,s.multiplicity,s.radial_points}),
    energies_({s.species,s.angular_channels,s.multiplicity}),
    transforms_({s.species,s.angular_channels,s.multiplicity,s.wavevectors+2}),
    potential_transform_({s.species,s.wavevectors+2}) {}

} // namespace openmx::core
