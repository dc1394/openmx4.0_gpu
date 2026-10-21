#pragma once

#include <openmx/core/ragged_matrix.hpp>
#include <array>
#include <memory>

namespace openmx::core {

enum class ElectronicField {
  kinetic, contracted_kinetic, nonlocal, imaginary_nonlocal, contracted_imaginary_nonlocal,
  core_hole, imaginary_core_hole, xmcd, hubbard, imaginary_nonlocal_reference,
  angular_overlap, overlap, core_hole_overlap, contracted_overlap,
  hamiltonian, contracted_hamiltonian, projectors, contracted_projectors, lno_density,
  hamiltonian_history, imaginary_hamiltonian_history, hamiltonian_residual,
  imaginary_hamiltonian_residual, density, guiding_density, imaginary_guiding_density,
  partial_density, density_residual, imaginary_density_residual,
  energy_density, previous_density, imaginary_density, count
};

struct ElectronicOptions {
  int spin = 0, spin_orbit = 0, mixing = 0;
  int nonlocal_components = 3, density_history = 1, hamiltonian_history = 1;
  int scratch_orbitals = 0, scratch_neighbors = 1, scratch_projectors = 0;
  bool contracted = false, core_hole = false, xmcd = false, interactions = false;
  bool lno = false, guiding_density = false, partial_density = false;
  // Preserve the legacy distinction between derivative halos and values.
  bool full_overlap_halo_derivatives = false;
};

// All arrays are borrowed only during construction. Atom zero is explicit
// padding. Maps include that slot; first/second maps include their halo atoms.
// New numerical code receives flat neighbor CSR, never legacy pointer trees.
struct ElectronicTopology {
  std::span<const int> atom_species, orbitals, contracted_orbitals, projector_components;
  std::span<const int> local_to_global, first_to_global, second_to_global;
  std::span<const std::size_t> neighbor_offsets;
  std::span<const int> neighbor_atoms;
};

// A topology rebuild constructs a complete replacement before publication.
// Each enabled family has one zero-initialized numeric allocation. Its CSR
// owns the exact local/halo/scratch shape, so teardown never reads topology.
class ElectronicMatrices {
 public:
  ElectronicMatrices(const ElectronicTopology& topology, ElectronicOptions options);
  ElectronicMatrices(const ElectronicMatrices&) = delete;
  ElectronicMatrices& operator=(const ElectronicMatrices&) = delete;
  RaggedMatrix* get(ElectronicField field) noexcept;
  const RaggedMatrix* get(ElectronicField field) const noexcept;
  std::size_t groups(ElectronicField field) const noexcept;
  std::size_t channels(ElectronicField field) const noexcept;
  std::size_t elements() const noexcept;

 private:
  struct Family { std::unique_ptr<RaggedMatrix> matrix; std::size_t groups{}, channels{}; };
  std::array<Family,static_cast<std::size_t>(ElectronicField::count)> families_;
};

} // namespace openmx::core
