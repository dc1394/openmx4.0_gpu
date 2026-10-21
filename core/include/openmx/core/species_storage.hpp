#pragma once

#include <openmx/core/tensor.hpp>
#include <span>

namespace openmx::core {

enum class SpeciesInteger {
  basis_lmax, open_core, species_index, relativistic, vps_mesh, pao_mesh,
  projector_components, radial_projectors, pao_lmax, pao_multiplicity,
  atomic_number, orbitals, contracted_orbitals, count
};
enum class SpeciesReal { mass, electrons, cutoff, core_charge, count };
enum class SpeciesLabel { name, basis, basis_file, potential_file, count };

struct SpeciesShape { std::size_t species, angular_channels, text_capacity; };

// One calculation owns all species metadata, including mutable input fields.
// Numeric fields are SoA; labels and basis counts have layout_right mappings.
// Addresses stay fixed while C readers fill the tables and consumers borrow them.
class SpeciesStorage {
 public:
  explicit SpeciesStorage(SpeciesShape shape);
  std::span<int> integers(SpeciesInteger field) noexcept;
  std::span<double> reals(SpeciesReal field) noexcept;
  Tensor<char,2>::view_type labels(SpeciesLabel field) noexcept;
  Tensor<int,2>::view_type basis(bool contracted = false) noexcept;
  SpeciesShape shape() const noexcept { return shape_; }

  SpeciesStorage(const SpeciesStorage&) = delete;
  SpeciesStorage& operator=(const SpeciesStorage&) = delete;
  SpeciesStorage(SpeciesStorage&&) = delete;
  SpeciesStorage& operator=(SpeciesStorage&&) = delete;

 private:
  struct Validated {};
  SpeciesStorage(SpeciesShape shape, Validated);
  SpeciesShape shape_;
  Tensor<int,2> integers_;
  Tensor<double,2> reals_;
  Tensor<char,3> labels_;
  Tensor<int,3> basis_;
};

} // namespace openmx::core
