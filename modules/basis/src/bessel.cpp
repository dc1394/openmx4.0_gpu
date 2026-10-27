#include <openmx/basis/bessel.hpp>
#include <openmx/basis/pair_bessel.hpp>
#include <openmx/basis/detail/bessel.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <new>
#include <stdexcept>
#include <utility>

namespace openmx::basis {
namespace {
struct HostMath {
  static double sin(double x) { return std::sin(x); }
  static double cos(double x) { return std::cos(x); }
  static double abs(double x) { return std::abs(x); }
};

void validate_argument(double x, int order)
{
  if (order < 0 || order > 29 || !std::isfinite(x) || x < 0 ||
      x > (INT_MAX-order-11)/1.5)
    throw std::invalid_argument("invalid spherical Bessel argument or order");
}

std::vector<double> validated_wavevectors(int order, std::span<const double> grid)
{
  if (order < 0 || order > 29 || grid.empty() || grid.size() > INT_MAX)
    throw std::invalid_argument("invalid pair Bessel table dimensions");
  for (double k : grid)
    if (!std::isfinite(k) || k < 0)
      throw std::invalid_argument("invalid pair Bessel wavevector");
  return {grid.begin(),grid.end()};
}

std::size_t table_size(std::size_t rows, std::size_t points)
{
  // rows is validated in 1..30 before reaching this helper.
  if (points > std::vector<double>().max_size()/(2*rows))
    throw std::length_error("pair Bessel table exceeds allocator maximum size");
  return 2*rows*points;
}
} // namespace

SphericalBesselSample::SphericalBesselSample(double argument, int maximum_order)
  : maximum_order_(maximum_order)
{
  validate_argument(argument,maximum_order);
  detail::spherical_bessel<HostMath>(argument,maximum_order,values_.data(),derivatives_.data());
}

PairBesselWorkspace::PairBesselWorkspace(int maximum_order,
    std::span<const double> wavevectors, std::size_t cache_limit)
  : wavevectors_(validated_wavevectors(maximum_order,wavevectors)), maximum_order_(maximum_order),
    maximum_wavevector_(*std::max_element(wavevectors_.begin(),wavevectors_.end())),
    cache_limit_(cache_limit), scratch_(table_size(maximum_order+1,wavevectors_.size()))
{}

void PairBesselWorkspace::fill(std::span<double> table, std::size_t rows,
    double distance, int order) const
{
  const auto points=wavevectors_.size();
  for (std::size_t point=0; point<points; ++point) {
    double values[32], derivatives[32];
    detail::spherical_bessel<HostMath>(wavevectors_[point]*distance,order,values,derivatives);
    for (int l=0; l<=order; ++l) {
      table[static_cast<std::size_t>(l)*points+point]=values[l];
      table[(rows+l)*points+point]=derivatives[l];
    }
  }
}

PairBesselTable PairBesselWorkspace::sample(double distance, int order)
{
  const double argument=maximum_wavevector_*distance;
  if (order < 0 || order > maximum_order_ || !std::isfinite(distance) || distance < 0 ||
      !std::isfinite(argument) || argument > (INT_MAX-order-11)/1.5)
    throw std::invalid_argument("invalid pair Bessel distance or angular order");
  const auto points=wavevectors_.size();
  for (const auto& entry : entries_) {
    if (entry.distance==distance && entry.order==order) {
      ++statistics_.hits;
      return {entry.table,static_cast<std::size_t>(order+1),points,order};
    }
  }
  ++statistics_.misses;
  if (entries_.size()<cache_limit_) {
    try {
      Entry entry{distance,order,std::vector<double>(table_size(order+1,points))};
      fill(entry.table,order+1,distance,order);
      entries_.push_back(std::move(entry));
      return {entries_.back().table,static_cast<std::size_t>(order+1),points,order};
    } catch (const std::bad_alloc&) {
      cache_limit_=entries_.size();
    }
  }
  fill(scratch_,maximum_order_+1,distance,order);
  return {scratch_,static_cast<std::size_t>(maximum_order_+1),points,order};
}

} // namespace openmx::basis
