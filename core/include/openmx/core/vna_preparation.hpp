#pragma once

#include <openmx/core/tensor.hpp>
#include <span>

namespace openmx::core {

struct SpeciesBasisView {
  int maximum_angular;
  std::span<const int> counts;
};

// Immutable snapshot of a ragged radial basis. Empty angular groups own no
// numeric rows; species and angular-group order match the input exactly.
class RadialBasisLayout {
 public:
  RadialBasisLayout(int maximum_angular, int multiplicity, std::span<const SpeciesBasisView> species);
  std::span<const int> maximum_angular() const noexcept { return maximum_angular_; }
  auto counts() const noexcept { return counts_.view(); }
  std::span<const std::size_t> offsets() const noexcept { return offsets_; }
  std::span<const int> orbitals() const noexcept { return orbitals_; }
  std::size_t radial_count() const noexcept { return offsets_.back(); }
 private:
  std::vector<int> maximum_angular_, orbitals_;
  Tensor<int,2> counts_;
  std::vector<std::size_t> offsets_;
};

struct VnaPreparationShape {
  int maximum_angular, multiplicity, projector_angular, projector_multiplicity, points;
};

// Invocation-local preparation shared by CPU and GPU adapters. No global
// inputs or GPU runtime are retained. Weights use [channel, radial row, point].
class VnaPreparation {
 public:
  VnaPreparation(VnaPreparationShape shape, std::span<const SpeciesBasisView> species);
  VnaPreparation(const VnaPreparation&) = delete;
  VnaPreparation& operator=(const VnaPreparation&) = delete;
  const VnaPreparationShape& shape() const noexcept { return shape_; }
  const RadialBasisLayout& basis() const noexcept { return basis_; }
  int projector_count() const noexcept { return projector_count_; }
  int projector_components() const noexcept { return projector_components_; }
  auto projectors() const noexcept { return projectors_.view(); }
  auto weights() noexcept { return weights_.view(); }
  auto weights() const noexcept { return weights_.view(); }
 private:
  VnaPreparationShape shape_;
  RadialBasisLayout basis_;
  int projector_count_, projector_components_;
  Tensor<int,2> projectors_;
  Tensor<double,3> weights_;
};

} // namespace openmx::core
