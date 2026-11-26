#include "host_radial_kernels.hpp"
#include "host_simd_abi.hpp"
#include "radial_weight.hpp"
#include <openmx/core/bessel.hpp>
#include <algorithm>
#include <vector>

namespace openmx::core::host {
namespace {

template<bool product>
Vector integrate(const Vector* weights, const double* left, const double* right, int count)
{
  Vector sum(0.0);
  for (int i = 0; i < count; ++i) {
    Vector term = weights[i]*Vector(left[i]);
    if constexpr (product) term *= Vector(right[i]);
    sum += term;
  }
  return sum;
}

void transform(const HostRadialProblem& p)
{
  // Independent wavevectors fill each vector. Each lane keeps the scalar
  // association of the Bessel weight, orbital product and ascending sum.
  const auto points = static_cast<std::size_t>(p.radial_count);
  std::vector<Vector> weights((p.lmax+1)*points);
  for (int first = 0; first < p.k_count;) {
    const int count = std::min(lanes,p.k_count-first);
    using Mask = Vector::mask_type;
    const Mask active([=](auto lane) { return int(lane) < count; });
    for (int i = 0; i < p.radial_count; ++i) {
      alignas(64) double values[30][lanes];
      for (int lane = 0; lane < count; ++lane) {
        double sb[30];
        spherical_bessel_values(p.wavevectors[first+lane]*p.radii[i],p.lmax,sb);
        for (int l = 0; l <= p.lmax; ++l)
          values[l][lane] = radial_weight(sb[l],p.measures[i],i,p.radial_count-1,p.weighting);
      }
      for (int l = 0; l <= p.lmax; ++l)
        weights[l*points+i] = count == lanes ? load(values[l]) : simd::simd_partial_load<Vector>(values[l],active);
    }
    for (std::size_t row = 0; row < p.channel_count; ++row) {
      const auto channel = p.channels[row];
      const double* left = p.radial_values+channel.left*points;
      const Vector* weighted = weights.data()+channel.angular*points;
      const Vector sum = channel.right < 0 ? integrate<false>(weighted,left,nullptr,p.radial_count)
        : integrate<true>(weighted,left,p.radial_values+channel.right*points,p.radial_count);
      double* output = p.result+row*p.k_count+first;
      const Vector result = sum*Vector(p.step);
      if (count == lanes) store(result,output);
      else simd::simd_partial_store(result,output,active);
    }
    first += count;
  }
}

}

const RadialKernels& OPENMX_HOST_SIMD_FACTORY(radial)() noexcept
{
  static constexpr RadialKernels kernels{lanes,transform};
  return kernels;
}

}
