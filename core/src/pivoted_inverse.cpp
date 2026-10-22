#include <openmx/core/pivoted_inverse.hpp>
#include <climits>

namespace openmx::core {
namespace {
void require(bool valid,const char* message)
{ if(!valid)throw std::invalid_argument(message); }
void check_storage(PivotedInverse::matrix_view view)
{
  const auto rows=view.extent(0),columns=view.extent(1);
  if(columns && rows>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double)/columns)
    throw std::length_error("pivoted inverse matrix size exceeds allocator maximum");
  require(!rows || !columns || view.data_handle(),"missing pivoted inverse matrix storage");
}
}
std::size_t PivotedInverse::checked_capacity(std::size_t capacity)
{
  if(capacity>INT_MAX || capacity+1>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double)/(capacity+1))
    throw std::length_error("pivoted inverse scratch exceeds allocator maximum");
  return capacity;
}
PivotedInverse::PivotedInverse(std::size_t capacity)
    :capacity_(checked_capacity(capacity)),packed_({(capacity_+1)*(capacity_+1)}),
     work_({capacity_+1}),pivots_({capacity_+1}) {}
void PivotedInverse::validate(matrix_view input,matrix_view inverse,std::size_t dimension,
                             const OpenMXRealLuOperations& operations)
{
  require(operations.factor && operations.invert,"missing native LU operations");
  require(dimension<=INT_MAX,"invalid pivoted inverse dimension");
  if(dimension)UnpivotedInverse::validate(input,inverse,dimension);
  else {check_storage(input);check_storage(inverse);}
}
void PivotedInverse::apply(matrix_view input,matrix_view inverse,std::size_t dimension,
                          const OpenMXRealLuOperations& operations)
{
  validate(input,inverse,dimension,operations);
  require(dimension<=capacity_,"pivoted inverse exceeds scratch capacity");
  const int original=static_cast<int>(dimension);
  // The original native wrapper packs in row order. LAPACK sees its transpose
  // as column-major; unpack in the same row order after both library calls.
  const matrix_view packed{packed_.data(),dimension,dimension};
  for(int i=0;i<original;++i)for(int j=0;j<original;++j)
    element(packed,i,j)=element(input,i,j);
  int m=original,n=m,lda=m,lwork=m,info;
  // Preserve the existing caller's call sequence and INFO behavior, including
  // n=0/lda=0 calls. Numerical INFO is not converted into an allocation error.
  operations.factor(operations.context,&m,&n,packed_.data(),&lda,pivots_.data(),&info);
  operations.invert(operations.context,&n,packed_.data(),&lda,pivots_.data(),work_.data(),&lwork,&info);
  for(int i=0;i<original;++i)for(int j=0;j<original;++j)
    element(inverse,i,j)=element(packed,i,j);
}
} // namespace openmx::core
