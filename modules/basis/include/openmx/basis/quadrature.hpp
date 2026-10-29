#pragma once

#include <span>
#include <utility>
#include <vector>

namespace openmx::basis {

// Borrowed nodes on [-1,1] and positive finite weights, checked once. The
// caller supplies the quadrature rule; its order is preserved. Both arrays
// must outlive the view and remain unchanged during use.
class GaussLegendreView {
 public:
  GaussLegendreView(std::span<const double> nodes, std::span<const double> weights);
  std::span<const double> nodes() const noexcept { return nodes_; }
  std::span<const double> weights() const noexcept { return weights_; }
 private:
  friend class GaussLegendreRule;
  struct Validated {};
  GaussLegendreView(std::span<const double> nodes, std::span<const double> weights, Validated)
    : nodes_(nodes), weights_(weights) {}
  std::span<const double> nodes_, weights_;
};

// Owns a full Gauss-Legendre rule on [-1,1], in increasing node order.
// Preserves OpenMX's Newton iteration, 1e-13 stopping criterion and weights
// evaluated with the derivative from the last iteration. Construction costs
// O(count^2); obtaining a view allocates nothing and costs O(1).
class GaussLegendreRule {
 public:
  explicit GaussLegendreRule(int count);
  GaussLegendreRule(const GaussLegendreRule&) = default;
  GaussLegendreRule(GaussLegendreRule&&) noexcept = default;
  GaussLegendreRule& operator=(const GaussLegendreRule& other) {
    if (this != &other) { GaussLegendreRule replacement(other); *this = std::move(replacement); }
    return *this;
  }
  GaussLegendreRule& operator=(GaussLegendreRule&&) noexcept = default;
  GaussLegendreView view() const;
 private:
  std::vector<double> nodes_, weights_;
};

} // namespace openmx::basis
