#include <openmx/core/dispersion_workspace.hpp>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>

namespace openmx::core {
DispersionWorkspace::shape_type DispersionWorkspace::dimensions(const OpenMXDispersionShape& in)
{
  if(in.atoms<0 || in.atoms==INT_MAX)
    throw std::invalid_argument("invalid dispersion atom count");
  const auto atoms=std::size_t(in.atoms)+1;
  shape_type shape{atoms,atoms,0,0,0};
  for(std::size_t axis=0;axis<3;++axis) {
    if(in.image_radius[axis]<0 || in.image_radius[axis]>(INT_MAX-1)/2)
      throw std::invalid_argument("invalid dispersion image radius");
    shape[axis+2]=2*std::size_t(in.image_radius[axis])+1;
  }
  // Validate the largest allocation before constructing any tensor. The bound
  // also makes all borrowed element offsets representable by ptrdiff_t.
  std::size_t count=1;
  for(auto n:shape) {
    if(count>std::size_t(PTRDIFF_MAX)/sizeof(double)/n)
      throw std::length_error("dispersion extent product overflows");
    count*=n;
  }
  return shape;
}
DispersionWorkspace::DispersionWorkspace(const OpenMXDispersionShape& in)
    :shape_(dimensions(in)),coordination_({shape_[0]}),
     c6_derivative_({shape_[0],shape_[1]}),energy_derivative_({shape_[0],shape_[1]}),
     coordination_derivative_(shape_) {}

OpenMXDispersionView DispersionWorkspace::view() noexcept
{
  const auto cn=coordination_.view();
  const auto c6=c6_derivative_.view(),ec=energy_derivative_.view();
  const auto derivative=coordination_derivative_.view();
  return {cn.extent(0),{derivative.extent(2),derivative.extent(3),derivative.extent(4)},
      cn.data_handle(),c6.data_handle(),ec.data_handle(),derivative.data_handle()};
}
void DispersionWorkspace::clear() noexcept
{
  const auto zero=[](auto& tensor){std::fill_n(tensor.data(),tensor.size(),0.0);};
  zero(coordination_);zero(c6_derivative_);zero(energy_derivative_);zero(coordination_derivative_);
}
}
