#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace openmx::basis {

// Immutable borrowed table for j_l(k*r) and d j_l(x)/dx at x=k*r. Flattened
// channel rows are [angular order][wavevector], with wavevector contiguous.
class PairBesselTable {
 public:
  int maximum_order() const noexcept { return maximum_order_; }
  std::size_t points() const noexcept { return points_; }
  std::span<const double> values() const noexcept {
    return storage_.first(static_cast<std::size_t>(maximum_order_+1)*points_);
  }
  std::span<const double> derivatives() const noexcept {
    return storage_.subspan(row_capacity_*points_,
                           static_cast<std::size_t>(maximum_order_+1)*points_);
  }
  // For mdspan adapters: [2][row_capacity][points], row-major. Only rows
  // 0..maximum_order in each channel contain this sample's results.
  std::span<const double> storage() const noexcept { return storage_; }
  std::size_t row_capacity() const noexcept { return row_capacity_; }
 private:
  friend class PairBesselWorkspace;
  PairBesselTable(std::span<const double> storage, std::size_t rows,
      std::size_t points, int maximum_order)
    : storage_(storage), row_capacity_(rows), points_(points), maximum_order_(maximum_order) {}
  std::span<const double> storage_;
  std::size_t row_capacity_, points_;
  int maximum_order_;
};

struct PairBesselStatistics {
  long hits = 0;
  long misses = 0;
};

// Copies the finite nonnegative wavevector grid (in bohr^-1 for OpenMX).
// Each caller/thread owns a workspace; mutation is not synchronized. No MPI,
// execution-space initialization or mutable global state is required.
// Cached views survive later samples until destruction. With caching disabled
// or full, a preallocated scratch table is reused; its views survive until the
// next uncached sample. A cache allocation failure permanently stops cache
// growth and uses that scratch, preserving existing cached views.
class PairBesselWorkspace {
 public:
  PairBesselWorkspace(int maximum_order, std::span<const double> wavevectors,
                      std::size_t cache_limit);
  PairBesselWorkspace(const PairBesselWorkspace&) = delete;
  PairBesselWorkspace& operator=(const PairBesselWorkspace&) = delete;
  // Finite nonnegative distance (bohr), and 0 <= order <= maximum_order.
  // Samples use SphericalBesselSample's recurrence and numerical range.
  // Invalid input leaves all existing tables and statistics unchanged.
  PairBesselTable sample(double distance, int order);
  PairBesselStatistics statistics() const noexcept { return statistics_; }
  std::size_t cached_tables() const noexcept { return entries_.size(); }
 private:
  struct Entry {
    double distance;
    int order;
    std::vector<double> table;
  };
  void fill(std::span<double> table, std::size_t rows, double distance, int order) const;
  std::vector<double> wavevectors_;
  int maximum_order_;
  double maximum_wavevector_;
  std::size_t cache_limit_;
  std::vector<double> scratch_;
  std::vector<Entry> entries_;
  PairBesselStatistics statistics_;
};

} // namespace openmx::basis
