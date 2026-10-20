#pragma once

#include <openmx/core/tensor.hpp>
#include <openmx/core/product_storage.hpp>
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

enum class RadialWeighting { trapezoid, gauss };

// Trapezoidal integration in a parameter t: radii=r(t), measures=r^2 dr/dt.
// Gauss integration uses measures=r^2*w and does not halve the endpoint nodes.
// Keeping the measure separate preserves the original order of multiplication
// and endpoint weighting for both uniform-r and uniform-sqrt(r) meshes.
struct RadialQuadrature {
  int lmax;
  std::vector<double> radii;
  std::vector<double> measures;
  std::vector<double> wavevectors;
  double step;
  RadialWeighting weighting = RadialWeighting::trapezoid;
};

struct RadialSamples {
  RadialQuadrature grid;
  std::vector<Orbital> orbitals;
  Tensor<double, 2> values; // [orbital, radial point]
  // Empty for ordinary radial transforms; otherwise output rows are products
  // of two sampled orbitals. Keep factors separate to preserve summation order.
  std::vector<ProductChannel> products;
};

RadialQuadrature uniform_quadrature(RadialGrid grid);
RadialQuadrature squared_radius_quadrature(int intervals, double rmin, double rmax,
    double kmin, double kmax, std::span<const double> gauss_nodes);
RadialQuadrature gauss_quadrature(int lmax, double rmin, double rmax,
    double kmin, double kmax, std::span<const double> radial_nodes,
    std::span<const double> radial_weights, std::span<const double> wavevector_nodes);

using Sampler = std::function<double(int angular, int multiplicity, double radius)>;
RadialSamples sample_radial(RadialGrid grid, std::span<const int> multiplicities,
                           const Sampler& sample);

class Context {
 public:
  // Library calls may own calculation storage without owning a Kokkos
  // runtime. This mode supports repeated create/transform/destroy calls.
  explicit Context(bool host_only = false) : host_only_(host_only) {}
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  ~Context() = default;

  // device < 0 selects a host path that also works without a GPU driver.
  Tensor<double, 2> transform(const RadialSamples& samples, int device = -1);
  void finish() noexcept { runtime_.reset(); finished_ = true; }
  bool finished() const noexcept { return finished_; }

 private:
  bool start(int device);
  std::unique_ptr<Kokkos::ScopeGuard> runtime_;
  int device_ = -1;
  bool finished_ = false;
  const bool host_only_;
};

bool has_accelerator() noexcept;
Tensor<double, 2> transform_host(const RadialSamples& samples);

} // namespace openmx::core
