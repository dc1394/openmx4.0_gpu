#pragma once

#include <openmx/core/tensor.hpp>
#include <span>
#include <vector>

namespace openmx::core {

struct PairBesselTable {
  Tensor<double,3>::const_view_type values;
  int angular;
};

struct PairBesselStatistics {
  long hits = 0;
  long misses = 0;
};

class PairBesselWorkspace {
 public:
  PairBesselWorkspace(int maximum_angular, std::span<const double> wavevectors,
                      std::size_t cache_limit);
  PairBesselWorkspace(const PairBesselWorkspace&) = delete;
  PairBesselWorkspace& operator=(const PairBesselWorkspace&) = delete;
  PairBesselTable sample(double distance, int angular);
  PairBesselStatistics statistics() const noexcept { return statistics_; }
  std::size_t cached_tables() const noexcept { return entries_.size(); }

 private:
  struct Entry {
    double distance;
    int angular;
    Tensor<double,3> table;
  };
  void fill(Tensor<double,3>& table, double distance, int angular) const;
  std::vector<double> wavevectors_;
  int maximum_angular_;
  double maximum_wavevector_;
  std::size_t cache_limit_;
  Tensor<double,3> scratch_;
  std::vector<Entry> entries_;
  PairBesselStatistics statistics_;
};

} // namespace openmx::core
