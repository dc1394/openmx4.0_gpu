#pragma once

#include <openmx/core/species_storage.hpp>
#include "legacy_rows.hpp"
#include "openmx_core_species.h"

namespace openmx::core::compat {
class SpeciesOwner {
 public:
  explicit SpeciesOwner(SpeciesShape shape) : storage_(shape), basis_(storage_.basis()),
    contracted_(storage_.basis(true)), name_(storage_.labels(SpeciesLabel::name)),
    basis_name_(storage_.labels(SpeciesLabel::basis)),
    basis_file_(storage_.labels(SpeciesLabel::basis_file)),
    potential_file_(storage_.labels(SpeciesLabel::potential_file)) {}
  OpenMXSpeciesViews views() noexcept {
    const auto s=storage_.shape();
    OpenMXSpeciesViews v{};
    v.species=s.species;v.angular_channels=s.angular_channels;v.text_capacity=s.text_capacity;
    v.basis_lmax=integer(SpeciesInteger::basis_lmax);v.open_core=integer(SpeciesInteger::open_core);
    v.species_index=integer(SpeciesInteger::species_index);v.relativistic=integer(SpeciesInteger::relativistic);
    v.vps_mesh=integer(SpeciesInteger::vps_mesh);v.pao_mesh=integer(SpeciesInteger::pao_mesh);
    v.projector_components=integer(SpeciesInteger::projector_components);
    v.radial_projectors=integer(SpeciesInteger::radial_projectors);
    v.pao_lmax=integer(SpeciesInteger::pao_lmax);v.pao_multiplicity=integer(SpeciesInteger::pao_multiplicity);
    v.atomic_number=integer(SpeciesInteger::atomic_number);v.orbitals=integer(SpeciesInteger::orbitals);
    v.contracted_orbitals=integer(SpeciesInteger::contracted_orbitals);
    v.mass=real(SpeciesReal::mass);v.electrons=real(SpeciesReal::electrons);
    v.cutoff=real(SpeciesReal::cutoff);v.core_charge=real(SpeciesReal::core_charge);
    v.basis=storage_.basis().data_handle();v.contracted_basis=storage_.basis(true).data_handle();
    v.basis_rows=basis_.data();v.contracted_basis_rows=contracted_.data();
    v.name=name_.data();v.basis_name=basis_name_.data();v.basis_file=basis_file_.data();
    v.potential_file=potential_file_.data();
    return v;
  }
 private:
  int *integer(SpeciesInteger field) noexcept { return storage_.integers(field).data(); }
  double *real(SpeciesReal field) noexcept { return storage_.reals(field).data(); }
  SpeciesStorage storage_;
  Rows2<int> basis_, contracted_;
  Rows2<char> name_, basis_name_, basis_file_, potential_file_;
};
} // namespace openmx::core::compat
