#include <openmx/core/angular_grid.hpp>
#include <stdexcept>

namespace openmx::core {
std::size_t AngularGrid::validate(int points)
{
  if(points!=590 && points!=1202 && points!=2030)
    throw std::invalid_argument("unsupported Lebedev grid: use 590, 1202 or 2030 points");
  return std::size_t(points);
}
AngularGrid::AngularGrid(int points)
    :points_(validate(points)),storage_({points_,4}),rows_(points_)
{
  for(std::size_t i=0;i<points_;++i)rows_[i]=storage_.data()+4*i;
}
void AngularGrid::rebind(int points)
{
  const auto count=validate(points);
  if(count>capacity())throw std::length_error("angular grid exceeds owned capacity");
  // The metadata capacity always covers numeric capacity; pointer construction
  // cannot throw, so rebinding within capacity does not allocate or fail.
  rows_.resize(count);
  for(std::size_t i=0;i<count;++i)rows_[i]=storage_.data()+4*i;
  points_=count;
}
OverwriteTensor<double,2>::view_type AngularGrid::grid() noexcept
{
  using Tensor=OverwriteTensor<double,2>;
  return {storage_.data(),Tensor::extents_type(std::array<std::size_t,2>{points_,4})};
}
OpenMXSolverRealTensor AngularGrid::view() noexcept
{
  const auto v=grid();return {{v.extent(0),v.extent(1),1,1},v.data_handle()};
}
}
