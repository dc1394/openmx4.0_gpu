#pragma once

#include <span>
#include <utility>
#include <vector>

namespace openmx::basis {

// Checked once, borrowed uniform log mesh and N+2 density entries. The physical
// values start at +1; the two padding entries participate in interpolation.
// All rows must outlive the view and remain unchanged during evaluation.
class LogDensityView {
 public:
  LogDensityView(std::span<const double> log_grid, std::span<const double> radii,
                 std::span<const double> padded_density);
  double sample_log(double log_radius) const;
  double sample(double radius) const;
  // Preserves Dr_KumoF's endpoint slope divided by radius beyond the last
  // point. There it is NOT the derivative of sample()'s constant extension.
  double derivative(double radius) const;
  std::span<const double> log_grid() const noexcept { return log_grid_; }
  std::span<const double> radii() const noexcept { return radii_; }
  std::span<const double> padded_density() const noexcept { return values_; }

 private:
  friend class LogDensityTable;
  struct Validated {};
  LogDensityView(std::span<const double> log_grid, std::span<const double> radii,
      std::span<const double> padded_density, Validated)
    : log_grid_(log_grid),radii_(radii),values_(padded_density) {}
  std::span<const double> log_grid_,radii_,values_;
};

// Owns the supplied physical density and mesh, adding the historical linear
// padding at each endpoint. No normalization or PCC correction is implicit.
class LogDensityTable {
 public:
  LogDensityTable(std::span<const double> log_grid, std::span<const double> radii,
                  std::span<const double> physical_density);
  LogDensityTable(const LogDensityTable&) = default;
  LogDensityTable(LogDensityTable&&) noexcept = default;
  LogDensityTable& operator=(const LogDensityTable& other) {
    if(this!=&other){LogDensityTable replacement(other);*this=std::move(replacement);}
    return *this;
  }
  LogDensityTable& operator=(LogDensityTable&&) noexcept = default;
  LogDensityView view() const;
  double sample(double radius) const { return view().sample(radius); }
  double derivative(double radius) const { return view().derivative(radius); }

 private:
  std::vector<double> log_grid_,radii_,values_;
};

// Nodes on [-1,1] and positive weights, borrowed and validated once. The caller
// chooses the quadrature order. The supplied order of nodes is preserved.
class GaussLegendreView {
 public:
  GaussLegendreView(std::span<const double> nodes, std::span<const double> weights);
  std::span<const double> nodes() const noexcept { return nodes_; }
  std::span<const double> weights() const noexcept { return weights_; }
 private:
  std::span<const double> nodes_,weights_;
};

// Spherical density in electrons/bohr^3, radius in bohr, result in Hartree.
// Uses the inherited logarithmic inside/outside Gauss integrals with explicit
// mesh and quadrature. radius must be finite and >0. No allocations, global
// settings or runtime initialization. The interpolation tail/origin policies
// are the same as the C application's KumoF, including endpoint clamping.
double atomic_hartree_potential(const LogDensityView& density, double radius,
                                const GaussLegendreView& quadrature);

} // namespace openmx::basis
