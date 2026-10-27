#include "openmx_core_quadrature.h"
#include <openmx/basis/quadrature.hpp>

#include <cstdio>
#include <stdexcept>

extern "C" int openmx_core_with_gauss_rule(int count, OpenMXGaussRuleConsumer consumer,
    void* user, char* error, std::size_t capacity)
{
  if (error && capacity) *error = '\0';
  try {
    if (!consumer) throw std::invalid_argument("missing quadrature consumer");
    const openmx::basis::GaussLegendreRule owner(count);
    const auto view = owner.view();
    const OpenMXGaussRuleData rule{count, view.nodes().data(), view.weights().data()};
    if (consumer(user, &rule, error, capacity)) return 1;
    if (error && capacity && !*error) std::snprintf(error, capacity, "quadrature consumer failed");
  } catch (const std::exception& exception) {
    if (error && capacity) std::snprintf(error, capacity, "%s", exception.what());
  } catch (...) {
    if (error && capacity) std::snprintf(error, capacity, "unknown quadrature consumer error");
  }
  return 0;
}
