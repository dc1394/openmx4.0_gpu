#include <openmx/basis/radial.h>
#include <openmx/basis/radial_table.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>

namespace {
void message(char* error, std::size_t capacity, const char* text) noexcept
{
  if (!error || !capacity) return;
  const auto count = std::min(capacity-1,std::strlen(text));
  std::memcpy(error,text,count);
  error[count] = '\0';
}

template<class Function>
int checked(char* error, std::size_t capacity, Function&& function) noexcept
{
  try {
    function();
    message(error,capacity,"");
    return 1;
  } catch (const std::exception& exception) {
    message(error,capacity,exception.what());
  } catch (...) {
    message(error,capacity,"unknown basis error");
  }
  return 0;
}

openmx::basis::RadialKind radial_kind(OpenMXBasisRadialKind kind)
{
  using openmx::basis::RadialKind;
  switch (kind) {
    case OPENMX_BASIS_RADIAL_PAO: return RadialKind::pao;
    case OPENMX_BASIS_RADIAL_PROJECTOR: return RadialKind::projector;
    case OPENMX_BASIS_RADIAL_NEUTRAL_POTENTIAL: return RadialKind::neutral_potential;
    case OPENMX_BASIS_RADIAL_NEUTRAL_PROJECTOR: return RadialKind::neutral_projector;
    default: throw std::invalid_argument("unknown radial interpolation kind");
  }
}

std::span<const double> mesh(const double* radii, std::size_t count)
{
  if (!radii || count < 3 || count > INT_MAX)
    throw std::invalid_argument("invalid radial table mesh");
  return {radii,count};
}

void commit(double value, double* result)
{
  if (!std::isfinite(value)) throw std::invalid_argument("non-finite interpolation result");
  *result = value;
}
} // namespace

struct OpenMXBasisRadialTable {
  openmx::basis::RadialTable table;
};

extern "C" int openmx_basis_radial_sample(OpenMXBasisRadialKind kind, int angular,
    std::size_t count, const double* radii, const double* values, double cutoff,
    double radius, double* result, char* error, std::size_t capacity)
{
  return checked(error,capacity,[&] {
    if (!result || !values) throw std::invalid_argument("missing radial row or output");
    const auto coordinates = mesh(radii,count);
    const double value = openmx::basis::sample_radial_row(radial_kind(kind),angular,
        coordinates,{values,count},cutoff,radius);
    commit(value,result);
  });
}

extern "C" int openmx_basis_reciprocal_sample(OpenMXBasisReciprocalKind kind,
    std::size_t count, double step, const double* values, double wavevector,
    double* result, char* error, std::size_t capacity)
{
  return checked(error,capacity,[&] {
    if (!result || !values || !count || count > INT_MAX ||
        (kind != OPENMX_BASIS_RECIPROCAL_PAO && kind != OPENMX_BASIS_RECIPROCAL_PROJECTOR))
      throw std::invalid_argument("invalid reciprocal interpolation arguments");
    const double value = openmx::basis::sample_reciprocal_row(
        kind == OPENMX_BASIS_RECIPROCAL_PAO ? openmx::basis::ReciprocalKind::pao
                                          : openmx::basis::ReciprocalKind::projector,
        {values,count},step,wavevector);
    commit(value,result);
  });
}

extern "C" OpenMXBasisRadialTable* openmx_basis_radial_table_create(
    OpenMXBasisRadialKind kind, std::size_t count, const double* radii,
    std::size_t channel_count, const int* angular, std::size_t row_stride,
    const double* values, double cutoff, char* error, std::size_t capacity)
{
  OpenMXBasisRadialTable* result = nullptr;
  checked(error,capacity,[&] {
    const auto policy = radial_kind(kind);
    const auto coordinates = mesh(radii,count);
    if (channel_count > INT_MAX || row_stride < count ||
        (channel_count && (!angular || !values)))
      throw std::invalid_argument("invalid radial table channel layout");
    // Check the last occupied entry before constructing any borrowed rows.
    const auto max_values = std::numeric_limits<std::ptrdiff_t>::max()/sizeof(double);
    if (channel_count && (count > max_values ||
        channel_count-1 > (max_values-count)/row_stride))
      throw std::length_error("radial table row layout overflows");
    std::vector<openmx::basis::RadialChannel> channels;
    channels.reserve(channel_count);
    for (std::size_t channel = 0; channel < channel_count; ++channel)
      channels.push_back({angular[channel],{values+channel*row_stride,count}});
    result = new OpenMXBasisRadialTable{{policy,coordinates,channels,cutoff}};
  });
  return result;
}

extern "C" int openmx_basis_radial_table_sample(const OpenMXBasisRadialTable* table,
    std::size_t channel, double radius, double* result, char* error, std::size_t capacity)
{
  return checked(error,capacity,[&] {
    if (!table || !result) throw std::invalid_argument("missing radial table or output");
    commit(table->table.sample(channel,radius),result);
  });
}

extern "C" void openmx_basis_radial_table_destroy(OpenMXBasisRadialTable* table)
{
  delete table;
}
