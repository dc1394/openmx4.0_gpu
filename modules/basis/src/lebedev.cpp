#include <openmx/basis/lebedev.hpp>
#include <openmx/basis/lebedev.h>

#include <cstdio>
#include <stdexcept>
#include <utility>

namespace openmx::basis {
namespace {
std::size_t validate_points(int count)
{
  if (count!=590 && count!=1202 && count!=2030)
    throw std::invalid_argument("unsupported Lebedev grid: use 590, 1202 or 2030 points");
  return static_cast<std::size_t>(count);
}

// All dimensions were validated by the caller. Constant stores cannot fail.
void fill_lebedev(std::size_t count, double* xyzw) noexcept
{
#include "lebedev_data.inc"
}
} // namespace

LebedevRule::LebedevRule(int points) { reset(points); }

LebedevRule::LebedevRule(LebedevRule&& other) noexcept
  : storage_(std::move(other.storage_)), points_(std::exchange(other.points_,0))
{}

LebedevRule& LebedevRule::operator=(const LebedevRule& other)
{
  if (this!=&other) { LebedevRule replacement(other); *this=std::move(replacement); }
  return *this;
}

LebedevRule& LebedevRule::operator=(LebedevRule&& other) noexcept
{
  if (this!=&other) {
    storage_=std::move(other.storage_);
    points_=std::exchange(other.points_,0);
  }
  return *this;
}

void LebedevRule::reset(int points)
{
  const auto count=validate_points(points);
  if (count==points_) return;
  if (count>capacity()) {
    std::vector<double> replacement(4*count);
    fill_lebedev(count,replacement.data());
    storage_.swap(replacement);
  } else {
    fill_lebedev(count,storage_.data());
  }
  points_=count;
}

std::span<const double,4> LebedevRule::point(std::size_t index) const
{
  if (index>=points_) throw std::out_of_range("Lebedev point index out of range");
  return std::span<const double,4>(storage_.data()+4*index,4);
}
} // namespace openmx::basis

extern "C" int openmx_basis_lebedev_fill(int count, double* xyzw, std::size_t capacity,
    char* error, std::size_t error_capacity)
{
  if (error && error_capacity) *error='\0';
  try {
    const auto points=openmx::basis::validate_points(count);
    if (!xyzw || capacity<4*points)
      throw std::invalid_argument("invalid Lebedev output array or capacity");
    openmx::basis::fill_lebedev(points,xyzw);
    return 1;
  } catch (const std::exception& exception) {
    if (error && error_capacity) std::snprintf(error,error_capacity,"%s",exception.what());
  } catch (...) {
    if (error && error_capacity) std::snprintf(error,error_capacity,"unknown Lebedev error");
  }
  return 0;
}
