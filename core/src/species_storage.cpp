#include <openmx/core/species_storage.hpp>
#include <algorithm>

namespace openmx::core {
namespace {
SpeciesShape validate(SpeciesShape shape) {
  if (!shape.species || !shape.angular_channels || shape.angular_channels > 30 ||
      shape.text_capacity < 2)
    throw std::invalid_argument("invalid species metadata dimensions");
  return shape;
}
}
SpeciesStorage::SpeciesStorage(SpeciesShape shape) : SpeciesStorage(validate(shape),Validated{}) {}
SpeciesStorage::SpeciesStorage(SpeciesShape s, Validated)
  : shape_(s), integers_({static_cast<std::size_t>(SpeciesInteger::count),s.species}),
    reals_({static_cast<std::size_t>(SpeciesReal::count),s.species}),
    labels_({static_cast<std::size_t>(SpeciesLabel::count),s.species,s.text_capacity}),
    basis_({2,s.species,s.angular_channels}) {
  auto mass=reals(SpeciesReal::mass);
  std::fill(mass.begin(),mass.end(),-1.0);
}
std::span<int> SpeciesStorage::integers(SpeciesInteger field) noexcept {
  return {integers_.data()+static_cast<std::size_t>(field)*shape_.species,shape_.species};
}
std::span<double> SpeciesStorage::reals(SpeciesReal field) noexcept {
  return {reals_.data()+static_cast<std::size_t>(field)*shape_.species,shape_.species};
}
Tensor<char,2>::view_type SpeciesStorage::labels(SpeciesLabel field) noexcept {
  return {labels_.data()+static_cast<std::size_t>(field)*shape_.species*shape_.text_capacity,
          Tensor<char,2>::extents_type(shape_.species,shape_.text_capacity)};
}
Tensor<int,2>::view_type SpeciesStorage::basis(bool contracted) noexcept {
  return {basis_.data()+static_cast<std::size_t>(contracted)*shape_.species*shape_.angular_channels,
          Tensor<int,2>::extents_type(shape_.species,shape_.angular_channels)};
}
} // namespace openmx::core
