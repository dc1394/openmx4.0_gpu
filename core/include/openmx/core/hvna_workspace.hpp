#pragma once

#include <openmx/core/vna_preparation.hpp>

namespace openmx::core {

// Shared CPU/GPU snapshot of the projector groups and species basis sizes.
// Neither the preparation nor the legacy energy tables are retained.
class HvnaProjectors {
 public:
  HvnaProjectors(const VnaPreparation& preparation, std::span<const double> energies);
  int components() const noexcept { return components_; }
  int groups() const noexcept { return groups_; }
  std::span<const int> orbitals() const noexcept { return orbitals_; }
  std::span<int> lengths() noexcept { return lengths_; }
  std::span<const int> lengths() const noexcept { return lengths_; }
  auto energies() noexcept { return energies_.view(); }
  auto energies() const noexcept { return energies_.view(); }
 private:
  int components_, groups_;
  std::vector<int> orbitals_, lengths_;
  Tensor<double,2> energies_;
};

// One reusable packed contraction matrix per CPU worker. Only capacity is
// copied from the projector snapshot; no borrowed pointers are retained.
class HvnaWorkspace {
 public:
  explicit HvnaWorkspace(const HvnaProjectors& projectors);
  HvnaWorkspace(const HvnaWorkspace&) = delete;
  HvnaWorkspace& operator=(const HvnaWorkspace&) = delete;
  auto matrix() noexcept { return matrix_.view(); }
  auto matrix() const noexcept { return matrix_.view(); }
 private:
  Tensor<double,2> matrix_;
};

} // namespace openmx::core
