#include "openmx_core_quadrature.h"
#include "calculation_context.hpp"
#include <openmx/basis/quadrature.hpp>

#include <cstdio>
#include <stdexcept>

namespace {
OpenMXGaussRuleData data(const openmx::basis::GaussLegendreRule& owner)
{
  const auto view = owner.view();
  return {static_cast<int>(view.nodes().size()), view.nodes().data(), view.weights().data()};
}
using RuleSlot = std::unique_ptr<openmx::basis::GaussLegendreRule> OpenMXCoreContext::*;

int initialize(OpenMXCoreContext* context, RuleSlot slot,
    int count, OpenMXGaussRuleData* out, char* error, std::size_t capacity)
{
  if (error && capacity) *error = '\0';
  try {
    if (!context || !out || context->implementation.finished())
      throw std::invalid_argument("quadrature requires an unfinished context and output");
    auto& storage = context->*slot;
    if (storage) {
      const auto rule = data(*storage);
      if (count != rule.count)
        throw std::invalid_argument("cannot change the initialized quadrature count");
      *out = rule;
    } else {
      auto owner = std::make_unique<openmx::basis::GaussLegendreRule>(count);
      const auto rule = data(*owner);
      storage = std::move(owner);
      *out = rule;
    }
    return 1;
  } catch (const std::exception& exception) {
    if (error && capacity) std::snprintf(error, capacity, "%s", exception.what());
  } catch (...) {
    if (error && capacity) std::snprintf(error, capacity, "unknown quadrature initialization error");
  }
  return 0;
}

int get(const OpenMXCoreContext* context, RuleSlot slot,
    OpenMXGaussRuleData* out, char* error, std::size_t capacity)
{
  if (error && capacity) *error = '\0';
  if (!context || !out || context->implementation.finished() || !(context->*slot)) {
    if (error && capacity) std::snprintf(error, capacity, "missing or finished quadrature");
    return 0;
  }
  *out = data(*(context->*slot));
  return 1;
}
}

extern "C" int openmx_core_radial_quadrature_initialize(OpenMXCoreContext* context,
    int count, OpenMXGaussRuleData* out, char* error, std::size_t capacity)
{ return initialize(context,&OpenMXCoreContext::radial_quadrature,count,out,error,capacity); }

extern "C" int openmx_core_radial_quadrature_get(const OpenMXCoreContext* context,
    OpenMXGaussRuleData* out, char* error, std::size_t capacity)
{ return get(context,&OpenMXCoreContext::radial_quadrature,out,error,capacity); }

extern "C" int openmx_core_neutral_quadrature_initialize(OpenMXCoreContext* context,
    int count, OpenMXGaussRuleData* out, char* error, std::size_t capacity)
{ return initialize(context,&OpenMXCoreContext::neutral_quadrature,count,out,error,capacity); }

extern "C" int openmx_core_neutral_quadrature_get(const OpenMXCoreContext* context,
    OpenMXGaussRuleData* out, char* error, std::size_t capacity)
{ return get(context,&OpenMXCoreContext::neutral_quadrature,out,error,capacity); }

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
