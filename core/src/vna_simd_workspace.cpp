#include <openmx/core/vna_simd_workspace.hpp>
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>

namespace openmx::core {
namespace {
int checked_orders(int points,int orders)
{
  if(points<=0 || orders<=0 || orders>32 || orders%8)
    throw std::invalid_argument("invalid VNA SIMD points or padded orders");
  return orders;
}
DsvnaSimdShape checked(DsvnaSimdShape shape)
{
  if(shape.points<=0 || shape.projectors<0 || shape.l0max<0 || shape.l0max>14 ||
      shape.l1max<0 || shape.l1max>14 || shape.llmax<0 || shape.llmax>28 ||
      shape.radials>INT_MAX || shape.orbitals>INT_MAX || shape.components>INT_MAX)
    throw std::invalid_argument("invalid DSVNA SIMD capacities");
  constexpr auto limit=std::numeric_limits<std::size_t>::max();
  if((shape.orbitals && shape.components>limit/shape.orbitals) ||
      (shape.radials && std::size_t(shape.projectors)>limit/shape.radials))
    throw std::length_error("DSVNA SIMD plane size overflows size_t");
  return shape;
}
template<class T> T* part(T* data,std::size_t size,std::size_t offset) noexcept
{return size?data+offset:nullptr;}
} // namespace

VnaSimdWorkspace::VnaSimdWorkspace(int points,int orders)
  : orders_(checked_orders(points,orders)),arguments_({std::size_t(points)}),
    bessel_({2,std::size_t(points),std::size_t(orders_)}),sums_({2,std::size_t(orders_)}),
    harmonics_({3,std::size_t(orders_),std::size_t(orders_)})
{}
OpenMXVnaSimdBuffers VnaSimdWorkspace::buffers() noexcept
{
  return {orders_,arguments_.data(),bessel_.data(),bessel_.data()+bessel_.size()/2,
      sums_.data(),sums_.data()+orders_,harmonics_.data()};
}

DsvnaSimdWorkspace::DsvnaSimdWorkspace(DsvnaSimdShape shape,bool angular)
  : shape_(checked(shape)),angular_(angular),pad_(shape_.l0max+shape_.l1max),
    cy_stride_(std::size_t(pad_+std::max(2*shape_.llmax+1,shape_.llmax+pad_+8)+8)),
    cy_length_(std::size_t(shape_.llmax+1)*cy_stride_),
    arguments_({std::size_t(shape_.points)}),
    weighted_({shape_.radials,2,std::size_t(shape_.points)}),
    accumulators_({angular_?8u:0u,shape_.orbitals,shape_.components}),
    harmonics_({angular_?6u:0u,cy_length_}),
    coefficients_({angular_?4u:0u,std::size_t(shape_.llmax+1)}),
    sums_({angular_?2u:0u,std::size_t(shape_.llmax+1),shape_.radials,std::size_t(shape_.projectors)})
{}
OpenMXDsvnaSimdBuffers DsvnaSimdWorkspace::buffers() noexcept
{
  auto* coefficients=coefficients_.data();const auto count=coefficients_.size();
  const auto orders=std::size_t(shape_.llmax+1);
  return {pad_,shape_.llmax,cy_stride_,cy_length_,shape_.orbitals*shape_.components,
      shape_.radials*std::size_t(shape_.projectors),2*std::size_t(shape_.points),
      arguments_.data(),part(weighted_.data(),weighted_.size(),0),
      part(harmonics_.data(),harmonics_.size(),0),part(accumulators_.data(),accumulators_.size(),0),
      part(sums_.data(),sums_.size(),0),part(coefficients,count,0),part(coefficients,count,orders),
      part(coefficients,count,2*orders),part(coefficients,count,3*orders)};
}
} // namespace openmx::core
