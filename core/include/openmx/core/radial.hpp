#pragma once

#include <openmx/core/tensor.hpp>
#include <Kokkos_Core.hpp>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace openmx::core {

struct RadialGrid {
  int lmax;
  int intervals;
  int k_count;
  double rmin;
  double radial_step;
  double k_step;
};

struct Orbital {
  int angular;
  int multiplicity;
};

struct RadialSamples {
  RadialGrid grid;
  std::vector<Orbital> orbitals;
  Tensor<double, 2> values; // [orbital, radial point]
};

using Sampler = std::function<double(int angular, int multiplicity, double radius)>;
RadialSamples sample_radial(RadialGrid grid, std::span<const int> multiplicities,
                           const Sampler& sample);

class Context {
 public:
  Context() = default;
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  ~Context() = default;

  // device < 0 selects a host path that also works without a GPU driver.
  Tensor<double, 2> transform(const RadialSamples& samples, int device = -1);
  void finish() noexcept { runtime_.reset(); finished_ = true; }

 private:
  bool start(int device);
  std::unique_ptr<Kokkos::ScopeGuard> runtime_;
  int device_ = -1;
  bool finished_ = false;
};

bool has_accelerator() noexcept;
Tensor<double, 2> transform_host(const RadialSamples& samples);

} // namespace openmx::core
