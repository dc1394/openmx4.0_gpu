#pragma once
#include <openmx/core/tensor.hpp>
#include <span>

namespace openmx::core {
struct AtomSiteView {
  int species;
  std::array<double,3> position;
  std::span<const int> neighbors, images; // Global atoms are one-based.
};

// Immutable snapshot of a rank's ordered atom/neighbor tasks. Input views may
// be destroyed after construction. Forward and mirrored displacements retain
// their original arithmetic order, including cancellation and signed zeros.
class AtomPairPlan {
 public:
  AtomPairPlan(std::span<const AtomSiteView> atoms,std::span<const int> local_to_global,
      std::span<const std::array<double,3>> translations,int species);
  auto local() const noexcept { return std::span<const int>(local_); }
  auto neighbor() const noexcept { return std::span<const int>(neighbor_); }
  auto central_global() const noexcept { return std::span<const int>(central_global_); }
  auto central_species() const noexcept { return std::span<const int>(central_species_); }
  auto neighbor_species() const noexcept { return std::span<const int>(neighbor_species_); }
  auto offsets() const noexcept { return std::span<const std::size_t>(offsets_); }
  auto displacements() const noexcept { return displacements_.view(); }
 private:
  std::vector<int> local_,neighbor_,central_global_,central_species_,neighbor_species_;
  std::vector<std::size_t> offsets_;
  Tensor<double,3> displacements_;
};
} // namespace openmx::core
