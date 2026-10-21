#pragma once

#include <openmx/core/block_matrix.hpp>
#include "openmx_complex.h"
#include <memory>

namespace openmx::core {

enum class LocalRealField { density, potential, inverse_overlap, count };
enum class LocalComplexField { occupation, potential, count };
enum class LocalTensorField { lno_coefficients, lno_populations, contraction, species_contraction, count };

struct LocalBasisOptions {
  int spin = 0, max_orbitals = 0, max_coefficients = 0;
  bool interactions = false, lno = false, contraction = false, inverse_overlap = false;
};

// Borrowed only while constructing the immutable block layouts. Atom zero is
// padding; neighbor rows include self and, for inverse_overlap, second neighbors.
struct LocalBasisTopology {
  std::span<const int> atom_species, orbitals, contracted_orbitals;
  std::span<const int> local_to_global, first_to_global;
  std::size_t extended_atoms = 0; // includes padding, first and second halos
  std::span<const std::size_t> neighbor_offsets;
  std::span<const int> neighbor_atoms;
};

// Onsite operators, localized-orbital coefficients and contraction coefficients
// have one calculation owner. Matrix blocks and rectangular tensors expose
// mdspan; no destructor needs the original species, atom or neighbor arrays.
class LocalBasisState {
 public:
  LocalBasisState(const LocalBasisTopology&, LocalBasisOptions);
  LocalBasisState(const LocalBasisState&) = delete;
  LocalBasisState& operator=(const LocalBasisState&) = delete;
  AtomBlockMatrix<double>* real(LocalRealField) noexcept;
  AtomBlockMatrix<dcomplex>* complex(LocalComplexField) noexcept;
  Tensor<double,3>* tensor(LocalTensorField) noexcept;

 private:
  std::array<std::unique_ptr<AtomBlockMatrix<double>>,static_cast<std::size_t>(LocalRealField::count)> real_;
  std::array<std::unique_ptr<AtomBlockMatrix<dcomplex>>,static_cast<std::size_t>(LocalComplexField::count)> complex_;
  std::array<std::unique_ptr<Tensor<double,3>>,static_cast<std::size_t>(LocalTensorField::count)> tensors_;
};

} // namespace openmx::core
