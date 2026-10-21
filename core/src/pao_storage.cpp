#include <openmx/core/pao_storage.hpp>

namespace openmx::core {
namespace {
PaoShape validate(PaoShape shape)
{
  if (!shape.species || !shape.angular_channels || !shape.multiplicity ||
      !shape.radial_points || !shape.reciprocal_points)
    throw std::invalid_argument("PAO storage dimensions must be positive");
  if (shape.radial_points > std::numeric_limits<std::size_t>::max()-2)
    throw std::length_error("PAO density padding overflows size_t");
  return shape;
}
} // namespace

PaoStorage::PaoStorage(PaoShape shape) : PaoStorage(validate(shape),Validated{}) {}

PaoStorage::PaoStorage(PaoShape s, Validated)
  : log_radii_({s.species,s.radial_points}), radii_({s.species,s.radial_points}),
    density_({s.species,s.radial_points+2}), density_backup_({s.species,s.radial_points+2}),
    orbitals_({s.species,s.angular_channels,s.multiplicity,s.radial_points}),
    transforms_({s.species,s.angular_channels,s.multiplicity,s.reciprocal_points}) {}

} // namespace openmx::core
