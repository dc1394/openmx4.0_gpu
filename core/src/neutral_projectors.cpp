#include <openmx/core/neutral_projectors.hpp>
#include <climits>
#include <cmath>

namespace openmx::core {
namespace {

double radial_inner_product(std::span<const double> left, std::span<const double> right,
    std::span<const double> radii, const RadialQuadrature& grid)
{
  double sum = 0;
  for (std::size_t i = 0; i < grid.radii.size(); ++i) {
    const double a = sample_radial_row(RadialKind::neutral_projector,0,radii,left,0,grid.radii[i]);
    const double b = sample_radial_row(RadialKind::neutral_projector,0,radii,right,0,grid.radii[i]);
    sum += grid.measures[i]*a*b;
  }
  const double value = grid.step*sum;
  if (!std::isfinite(value)) throw std::overflow_error("non-finite neutral-projector radial norm");
  return value;
}

} // namespace

NeutralProjectors expand_neutral_projectors(const RadialTableView& pao,
    std::span<const double> radii, std::span<const double> potential,
    NeutralProjectorBasis basis, std::span<const double> nodes,
    std::span<const double> weights)
{
  const auto [lmax,multiplicity,pao_lmax,pao_multiplicity,empty_atom] = basis;
  if (lmax < 0 || lmax > 29 || multiplicity <= 0 || multiplicity > INT_MAX/(lmax+1) ||
      pao_lmax < 0 || pao_lmax > 29 || pao_multiplicity <= 0 ||
      pao_multiplicity > INT_MAX/(pao_lmax+1) ||
      pao.channel_count() != static_cast<std::size_t>((pao_lmax+1)*pao_multiplicity))
    throw std::invalid_argument("invalid neutral-projector basis");
  for (std::size_t row = 0; row < pao.channel_count(); ++row)
    if (pao.angular(row) != static_cast<int>(row/pao_multiplicity))
      throw std::invalid_argument("invalid neutral-projector PAO row order");
  const RadialChannel vna{0,potential};
  // Validate VPS coordinates and all potential values before any sampling.
  const RadialTableView validated(RadialKind::neutral_projector,radii,{&vna,1});
  const auto grid = gauss_quadrature(0,radii.front(),radii.back(),0,0,nodes,weights,{});
  const auto size = radii.size();
  const auto angular = static_cast<std::size_t>(lmax+1);
  const auto count = static_cast<std::size_t>(multiplicity);
  NeutralProjectors result{Tensor<double,3>({angular,count,size}),Tensor<double,2>({angular,count})};
  Tensor<double,2> phi({count,size});
  Tensor<double,1> weighted({size}), inverse_norms({count});
  auto functions = phi.view();
  auto projectors = result.values.view();
  auto energies = result.energies.view();
  const auto phi_row = [&](int m) { return std::span<double>{&element(functions,m,0),size}; };
  const auto projector_row = [&](int l,int m) { return std::span<double>{&element(projectors,l,m,0),size}; };
  const std::span<double> phi2(weighted.data(),size);
  double* pe = inverse_norms.data();

  for (int l = 0; l <= lmax; ++l) {
    for (int m = 0; m < multiplicity; ++m) {
      const auto row = phi_row(m);
      for (std::size_t i = 0; i < size; ++i) {
        const double r = radii[i];
        if (l <= pao_lmax && m < pao_multiplicity)
          row[i] = pao.sample(l*pao_multiplicity+m,r);
        else if (l <= pao_lmax)
          row[i] = std::pow(0.1*potential[i]+1.0e-13,static_cast<double>(m))*element(functions,0,i);
        else if (m < pao_multiplicity)
          row[i] = pao.sample(pao_lmax*pao_multiplicity+m,r)*std::pow(r,static_cast<double>(l-pao_lmax));
        else
          row[i] = pao.sample(pao_lmax*pao_multiplicity+pao_multiplicity-1,r)*
            std::pow(0.1*potential[i]+1.0e-13,static_cast<double>(m-pao_multiplicity+1));
      }
    }
    for (int m = 0; m < multiplicity; ++m) {
      const auto row = phi_row(m);
      const double norm = radial_inner_product(row,row,radii,grid);
      const double scale = norm > 1.0e-17 ? 1.0/std::sqrt(norm) : 0.0;
      for (std::size_t i = 0; i < size; ++i) row[i] = scale*row[i];
    }

    // Classical Gram-Schmidt in the <f|VNA|g> metric. Use the original phi[m]
    // for each coefficient, not an incrementally updated residual.
    for (int m = 0; m < multiplicity; ++m) {
      const auto row = projector_row(l,m);
      if (m == 0) {
        for (std::size_t i = 0; i < size; ++i) row[i] = element(functions,0,i);
      } else {
        for (int n = 0; n < m; ++n) {
          for (std::size_t i = 0; i < size; ++i) phi2[i] = element(functions,m,i)*potential[i];
          const auto previous = projector_row(l,n);
          const double coefficient = radial_inner_product(previous,phi2,radii,grid);
          for (std::size_t i = 0; i < size; ++i) row[i] += previous[i]*pe[n]*coefficient;
        }
        for (std::size_t i = 0; i < size; ++i) row[i] = element(functions,m,i)-row[i];
      }
      for (std::size_t i = 0; i < size; ++i) phi2[i] = row[i]*potential[i];
      const double norm = radial_inner_product(row,phi2,radii,grid);
      pe[m] = empty_atom || std::abs(norm) < 1.0e-15 ? 0.0 : 1.0/norm;
    }

    // Retain OpenMX's renormalization convention (multiply by the squared
    // radial norm, then compensate the projector energy by its square).
    for (int m = 0; m < multiplicity; ++m) {
      const auto row = projector_row(l,m);
      const double norm = radial_inner_product(row,row,radii,grid);
      for (std::size_t i = 0; i < size; ++i) row[i] = norm*row[i];
      element(energies,l,m) = empty_atom || std::abs(norm) < 1.0e-15 ? 0.0 : pe[m]/(norm*norm);
    }
    for (int m = 0; m < multiplicity; ++m)
      for (std::size_t i = 0; i < size; ++i)
        element(projectors,l,m,i) = potential[i]*element(projectors,l,m,i);
  }
  for (std::size_t i = 0; i < result.values.size(); ++i)
    if (!std::isfinite(result.values.data()[i])) throw std::invalid_argument("non-finite neutral projector");
  for (std::size_t i = 0; i < result.energies.size(); ++i)
    if (!std::isfinite(result.energies.data()[i])) throw std::invalid_argument("non-finite neutral-projector energy");
  return result;
}

} // namespace openmx::core
