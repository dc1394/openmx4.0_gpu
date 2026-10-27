#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace openmx::basis {

// Owns the OpenMX Lebedev preset with 590, 1202 or 2030 unit-sphere points.
// Data is [point][x,y,z,weight], with weights summing to 1 (not 4*pi).
// No application state, MPI, GPU or Kokkos initialization is required.
// Copies are independent. Views expire on destruction, assignment or reset;
// concurrent readers are permitted while the owner is not modified.
class LebedevRule {
 public:
  explicit LebedevRule(int points);
  LebedevRule(const LebedevRule&) = default;
  LebedevRule(LebedevRule&& other) noexcept;
  LebedevRule& operator=(const LebedevRule& other);
  LebedevRule& operator=(LebedevRule&& other) noexcept;

  // Invalid counts or allocation failure preserve the previous rule and views.
  // Reuse within capacity preserves the address and allocates nothing.
  // Repeating the current count is O(1); all other changes refill O(points).
  void reset(int points);
  std::size_t size() const noexcept { return points_; }
  std::size_t capacity() const noexcept { return storage_.size()/4; }
  std::span<const double> xyzw() const noexcept {
    return {storage_.data(),4*points_};
  }
  // Throws out_of_range for an invalid point, including a moved-from rule.
  std::span<const double,4> point(std::size_t index) const;

 private:
  std::vector<double> storage_;
  std::size_t points_=0;
};

} // namespace openmx::basis
