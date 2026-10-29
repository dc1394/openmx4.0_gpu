#pragma once

#include <openmx/core/tensor.hpp>
#include <openmx/basis/pair_bessel.hpp>

namespace openmx::core {

struct PairBesselTable {
  Tensor<double,3>::const_view_type values;
  int angular;
};
using PairBesselStatistics = openmx::basis::PairBesselStatistics;

// Compatibility view for existing overlap/nonlocal workspaces. Storage,
// cache policy and numerical evaluation belong to the independent library.
class PairBesselWorkspace {
 public:
  PairBesselWorkspace(int maximum_angular, std::span<const double> wavevectors,
                      std::size_t cache_limit);
  PairBesselWorkspace(const PairBesselWorkspace&) = delete;
  PairBesselWorkspace& operator=(const PairBesselWorkspace&) = delete;
  PairBesselTable sample(double distance, int angular);
  PairBesselStatistics statistics() const noexcept { return implementation_.statistics(); }
  std::size_t cached_tables() const noexcept { return implementation_.cached_tables(); }
 private:
  openmx::basis::PairBesselWorkspace implementation_;
};

} // namespace openmx::core
