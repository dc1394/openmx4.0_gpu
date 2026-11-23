// Shared helpers of the exx unit tests.  Every case is a function that throws
// std::runtime_error on failure; main() in test_main.cpp selects the case.
#pragma once

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace exx_test {

inline void require(bool ok, const std::string& what)
{
  if (!ok) throw std::runtime_error(what);
}

inline double relative_difference(double value, double reference)
{
  if (value == reference) return 0.0;
  return std::fabs(value-reference)/std::fabs(reference);
}

// Collects the largest deviation of a group of comparisons, prints it, and
// fails when it exceeds the tolerance.  NaN counts as a failure.
class Worst {
 public:
  Worst(std::string name, double tolerance) : name_(std::move(name)), tolerance_(tolerance) {}
  void absolute(double value, double reference) { add(std::fabs(value-reference)); }
  void relative(double value, double reference) { add(relative_difference(value,reference)); }
  void add(double deviation)
  {
    ++count_;
    if (!(deviation <= worst_)) worst_ = deviation; // keeps NaN
  }
  double worst() const { return worst_; }
  long count() const { return count_; }
  void finish() const
  {
    std::printf("%-58s %7ld comparisons, largest deviation %.3e (limit %.1e)\n",
                name_.c_str(),count_,worst_,tolerance_);
    require(count_ > 0,name_+": nothing was compared");
    require(worst_ <= tolerance_,name_+": the deviation exceeds the limit");
  }
 private:
  std::string name_;
  double tolerance_;
  double worst_ = 0.0;
  long count_ = 0;
};

// Deterministic numbers in (-1, 1) without a pattern: the same sequence on
// every machine.
class Numbers {
 public:
  double next()
  {
    state_ = state_*6364136223846793005ULL+1442695040888963407ULL;
    return static_cast<double>(state_>>11)/9007199254740992.0*2.0-1.0;
  }
 private:
  unsigned long long state_ = 0x9E3779B97F4A7C15ULL;
};

// The cases, by file.
void kernel_analytic(const std::string& fixture);
void two_center_gaussian();
void two_center_symmetry();
void two_center_overlap(const std::string& fixture);
void abf_pca(const std::string& fixture);
void three_center_gaussian();
void lri_identity(const std::string& fixture);
void lri_convergence(const std::string& fixture);
void system_neighbours(const std::string& fixture);
void system_blocks(const std::string& fixture);
void dense_backend();
void exchange_fixture(const std::string& fixture);
void exchange_gamma(const std::string& fixture);
void exchange_gradient(const std::string& fixture);
void exchange_backend(const std::string& fixture);
void exchange_system(const std::string& fixture);

} // namespace exx_test
