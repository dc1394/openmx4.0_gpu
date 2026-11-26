#include <openmx/exx/abf.hpp>

#include <openmx/basis/radial_table.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace openmx::exx {
namespace {

constexpr int maximum_orbital_angular = 14;
constexpr double smallest_threshold = 1e-12;

// Rotates the columns in place until they are mutually orthogonal in the
// inner product sum_i weight_i x_i y_i (one-sided Jacobi method of Hestenes).
// The squared norms of the columns are then the eigenvalues of the Gram
// matrix of the original columns, and the normalized columns are its
// eigenvectors applied to the original columns.  Unlike a diagonalization of
// the Gram matrix, this keeps the columns orthogonal to rounding relative to
// their own norms, however small the eigenvalue.
void orthogonalize(std::vector<std::vector<double>>& columns, std::span<const double> weight)
{
  const std::size_t n = columns.size(), points = weight.size();
  // The criterion of LAPACK's one-sided Jacobi (dgesvj): sqrt(rows) * unit roundoff.
  const double tolerance = std::sqrt(static_cast<double>(points))*0.5*std::numeric_limits<double>::epsilon();
  constexpr int maximum_sweeps = 60;
  for (int sweep = 0; sweep < maximum_sweeps; ++sweep) {
    bool rotated = false;
    for (std::size_t p = 0; p+1 < n; ++p)
      for (std::size_t q = p+1; q < n; ++q) {
        double* x = columns[p].data();
        double* y = columns[q].data();
        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        for (std::size_t i = 0; i < points; ++i) {
          const double wx = weight[i]*x[i];
          alpha += wx*x[i];
          beta += weight[i]*y[i]*y[i];
          gamma += wx*y[i];
        }
        if (std::fabs(gamma) <= tolerance*std::sqrt(alpha*beta)) continue;
        rotated = true;
        const double zeta = (beta-alpha)/(2.0*gamma);
        const double t = std::copysign(1.0,zeta)/(std::fabs(zeta)+std::sqrt(1.0+zeta*zeta));
        const double c = 1.0/std::sqrt(1.0+t*t), s = c*t;
        for (std::size_t i = 0; i < points; ++i) {
          const double xi = x[i], yi = y[i];
          x[i] = c*xi-s*yi;
          y[i] = s*xi+c*yi;
        }
      }
    if (!rotated) return;
  }
  throw std::runtime_error("the principal component analysis of the PAO products did not converge");
}

// +1 or -1: the factor that makes int r^(L+2) f dr positive.  If that moment
// vanishes to rounding, the next powers of r decide.
double orientation(const RadialGrid& grid, std::span<const double> f, int angular)
{
  const std::size_t last = grid.size()-1;
  for (int power = angular; power <= angular+4; ++power) {
    double moment = 0.0, scale = 0.0;
    for (std::size_t i = 0; i <= last; ++i) {
      double term = std::pow(grid.radius(i),power+2)*f[i];
      if (i == 0 || i == last) term *= 0.5;
      moment += term;
      scale += std::fabs(term);
    }
    if (std::fabs(moment) > 1e-10*scale) return moment > 0.0 ? 1.0 : -1.0;
  }
  return 1.0;
}

} // namespace

std::size_t AuxiliaryBasis::components() const noexcept
{
  std::size_t count = 0;
  for (const RadialFunction& f : functions) count += static_cast<std::size_t>(2*f.angular+1);
  return count;
}

std::vector<RadialFunction> sample_orbitals(const RadialGrid& grid, std::span<const double> mesh,
                                            std::span<const MeshFunction> orbitals)
{
  check(grid);
  std::vector<basis::RadialChannel> channels;
  channels.reserve(orbitals.size());
  for (const MeshFunction& orbital : orbitals) channels.push_back({orbital.angular,orbital.values});
  // The view checks the mesh and the values once.
  const basis::RadialTableView table(basis::RadialKind::pao,mesh,channels);
  std::vector<RadialFunction> result;
  result.reserve(orbitals.size());
  for (std::size_t k = 0; k < orbitals.size(); ++k) {
    RadialFunction f{orbitals[k].angular,std::vector<double>(grid.size())};
    for (std::size_t i = 0; i < grid.size(); ++i) f.values[i] = table.sample(k,grid.radius(i));
    result.push_back(std::move(f));
  }
  return result;
}

AuxiliaryBasis make_auxiliary_basis(const RadialGrid& grid, std::span<const RadialFunction> orbitals,
                                    double threshold)
{
  check(grid);
  if (!(threshold >= smallest_threshold && threshold < 1.0))
    throw std::invalid_argument("the threshold of the principal component analysis must be in [1e-12,1)");
  if (orbitals.empty()) throw std::invalid_argument("the auxiliary basis needs at least one radial function");
  int maximum = 0;
  for (const RadialFunction& f : orbitals) {
    if (f.angular < 0 || f.angular > maximum_orbital_angular)
      throw std::invalid_argument("the angular momentum of an orbital must be in [0,14]");
    if (f.values.size() != grid.size()) throw std::invalid_argument("a radial function does not match its grid");
    for (const double value : f.values)
      if (!std::isfinite(value)) throw std::invalid_argument("a radial function has a value that is not finite");
    maximum = std::max(maximum,f.angular);
  }

  // Weights of the inner product int r^2 a b dr (trapezoid rule of the grid).
  std::vector<double> weight(grid.size());
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double r = grid.radius(i);
    weight[i] = grid.weight(i)*(r*r);
  }

  AuxiliaryBasis basis;
  for (int angular = 0; angular <= 2*maximum; ++angular) {
    // The products that have a component with this angular momentum.
    std::vector<std::vector<double>> columns;
    for (std::size_t i = 0; i < orbitals.size(); ++i)
      for (std::size_t j = i; j < orbitals.size(); ++j) {
        const int li = orbitals[i].angular, lj = orbitals[j].angular;
        if (angular < std::abs(li-lj) || angular > li+lj || (li+lj+angular)%2 != 0) continue;
        std::vector<double> product(grid.size());
        for (std::size_t k = 0; k < grid.size(); ++k) product[k] = orbitals[i].values[k]*orbitals[j].values[k];
        columns.push_back(std::move(product));
      }
    if (columns.empty()) continue;

    orthogonalize(columns,weight);
    std::vector<double> eigenvalue(columns.size());
    for (std::size_t n = 0; n < columns.size(); ++n) {
      double sum = 0.0;
      for (std::size_t k = 0; k < grid.size(); ++k) sum += weight[k]*columns[n][k]*columns[n][k];
      eigenvalue[n] = sum;
    }
    std::vector<std::size_t> order(columns.size());
    std::iota(order.begin(),order.end(),std::size_t{0});
    std::stable_sort(order.begin(),order.end(),
                     [&](std::size_t a, std::size_t b) { return eigenvalue[a] > eigenvalue[b]; });

    AbfChannel channel;
    channel.angular = angular;
    channel.eigenvalues.reserve(columns.size());
    for (const std::size_t n : order) channel.eigenvalues.push_back(eigenvalue[n]);
    const double largest = channel.eigenvalues.front();
    if (!(largest > 0.0)) throw std::invalid_argument("the products of the orbitals vanish on the grid");
    for (const std::size_t n : order) {
      if (!(eigenvalue[n] > threshold*largest)) break;
      RadialFunction f{angular,std::move(columns[n])};
      const double scale = 1.0/std::sqrt(eigenvalue[n]);
      for (double& value : f.values) value *= scale;
      const double sign = orientation(grid,f.values,angular);
      if (sign < 0.0)
        for (double& value : f.values) value = -value;
      basis.functions.push_back(std::move(f));
      ++channel.kept;
    }
    basis.channels.push_back(std::move(channel));
  }
  return basis;
}

} // namespace openmx::exx
