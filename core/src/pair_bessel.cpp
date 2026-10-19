#include <openmx/core/pair_bessel.hpp>
#include <openmx/core/bessel.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <new>
#include <stdexcept>

namespace openmx::core {
namespace {

std::vector<double> validated_wavevectors(int angular, std::span<const double> wavevectors)
{
  if (angular < 0 || angular > 29 || wavevectors.empty() || wavevectors.size() > INT_MAX)
    throw std::invalid_argument("invalid pair Bessel table dimensions");
  for (double wavevector : wavevectors)
    if (!std::isfinite(wavevector) || wavevector < 0)
      throw std::invalid_argument("invalid pair Bessel wavevector");
  return {wavevectors.begin(),wavevectors.end()};
}

} // namespace

PairBesselWorkspace::PairBesselWorkspace(int maximum_angular,
    std::span<const double> wavevectors, std::size_t cache_limit)
  : wavevectors_(validated_wavevectors(maximum_angular,wavevectors)), maximum_angular_(maximum_angular),
    maximum_wavevector_(0), cache_limit_(cache_limit),
    scratch_({2,static_cast<std::size_t>(maximum_angular+1),wavevectors.size()})
{
  maximum_wavevector_ = *std::max_element(wavevectors_.begin(),wavevectors_.end());
}

void PairBesselWorkspace::fill(Tensor<double,3>& table, double distance, int angular) const
{
  auto output = table.view();
  for (std::size_t point = 0; point < wavevectors_.size(); ++point) {
    double values[32], derivatives[32];
    spherical_bessel(wavevectors_[point]*distance,angular,values,derivatives);
    for (int order = 0; order <= angular; ++order) {
      element(output,0,order,point) = values[order];
      element(output,1,order,point) = derivatives[order];
    }
  }
}

PairBesselTable PairBesselWorkspace::sample(double distance, int angular)
{
  const double maximum_argument = maximum_wavevector_*distance;
  if (angular < 0 || angular > maximum_angular_ || !std::isfinite(distance) || distance < 0 ||
      !std::isfinite(maximum_argument) || maximum_argument > (INT_MAX-angular-11)/1.5)
    throw std::invalid_argument("invalid pair Bessel distance or angular order");
  for (const auto& entry : entries_) {
    if (entry.distance == distance && entry.angular == angular) {
      ++statistics_.hits;
      return {entry.table.view(),angular};
    }
  }
  ++statistics_.misses;
  if (entries_.size() < cache_limit_) {
    try {
      Entry entry{distance,angular,Tensor<double,3>({2,static_cast<std::size_t>(angular+1),wavevectors_.size()})};
      fill(entry.table,distance,angular);
      entries_.push_back(std::move(entry));
      return {std::as_const(entries_.back().table).view(),angular};
    } catch (const std::bad_alloc&) {
      cache_limit_ = entries_.size();
    }
  }
  fill(scratch_,distance,angular);
  return {std::as_const(scratch_).view(),angular};
}

} // namespace openmx::core
