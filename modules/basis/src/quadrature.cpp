#include <openmx/basis/quadrature.hpp>
#include <openmx/basis/quadrature.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace openmx::basis {

GaussLegendreView::GaussLegendreView(std::span<const double> nodes, std::span<const double> weights)
  : nodes_(nodes), weights_(weights)
{
  if (nodes_.empty() || nodes_.size() != weights_.size() || nodes_.size() > INT_MAX)
    throw std::invalid_argument("invalid Gauss-Legendre quadrature shape");
  for (std::size_t i = 0; i < nodes_.size(); ++i)
    if (!std::isfinite(nodes_[i]) || nodes_[i] < -1 || nodes_[i] > 1 ||
        !std::isfinite(weights_[i]) || weights_[i] <= 0)
      throw std::invalid_argument("invalid Gauss-Legendre quadrature node or weight");
}

GaussLegendreRule::GaussLegendreRule(int count)
{
  if (count < 1 || count == INT_MAX)
    throw std::invalid_argument("invalid Gauss-Legendre order");
  nodes_.resize(count);
  weights_.resize(count);
  // Gauss_Legendre2, T. Ozaki (2001). Retain the recurrence and arithmetic
  // order, but write the physical rows directly without the old +1 scratch.
  constexpr double pi = 3.1415926535897932384626;
  constexpr double tolerance = 1.0e-13;
  const int half = (count + 1) / 2;
  for (int i = 1; i <= half; ++i) {
    double z = std::cos(pi * (i - 0.250) / (count + 0.50));
    double derivative = 0;
    bool converged = false;
    for (int iteration = 0; iteration < 100; ++iteration) {
      double p1 = 1.0, p2 = 0.0;
      for (int j = 1; j <= count; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * static_cast<double>(j) - 1.0) * z * p2
              - (static_cast<double>(j) - 1.0) * p3) / static_cast<double>(j);
      }
      derivative = static_cast<double>(count) * (z * p1 - p2) / (z * z - 1.0);
      const double previous = z;
      z = previous - p1 / derivative;
      if (!std::isfinite(z) || !std::isfinite(derivative) || derivative == 0)
        throw std::runtime_error("invalid Gauss-Legendre Newton iterate");
      if (std::abs(z - previous) <= tolerance) { converged = true; break; }
    }
    if (!converged) throw std::runtime_error("Gauss-Legendre iteration did not converge");
    nodes_[i - 1] = 0.0 - 1.0 * z;
    nodes_[count - i] = 0.0 + 1.0 * z;
    weights_[i - 1] = 2.0 * 1.0 / ((1.0 - z * z) * derivative * derivative);
    weights_[count - i] = weights_[i - 1];
  }
  (void)GaussLegendreView(nodes_, weights_);
}

GaussLegendreView GaussLegendreRule::view() const
{
  if (nodes_.empty()) throw std::logic_error("moved-from Gauss-Legendre rule");
  return {nodes_, weights_, GaussLegendreView::Validated{}};
}

} // namespace openmx::basis

extern "C" int openmx_basis_gauss_legendre(int count, double* nodes, double* weights,
    char* error, std::size_t capacity)
{
  if (error && capacity) *error = '\0';
  try {
    if (!nodes || !weights || nodes == weights)
      throw std::invalid_argument("invalid Gauss-Legendre output arrays");
    const openmx::basis::GaussLegendreRule rule(count);
    const auto view = rule.view();
    std::copy(view.nodes().begin(), view.nodes().end(), nodes);
    std::copy(view.weights().begin(), view.weights().end(), weights);
    return 1;
  } catch (const std::exception& exception) {
    if (error && capacity) std::snprintf(error, capacity, "%s", exception.what());
  } catch (...) {
    if (error && capacity) std::snprintf(error, capacity, "unknown quadrature error");
  }
  return 0;
}
