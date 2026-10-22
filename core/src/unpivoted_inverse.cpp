#include <openmx/core/unpivoted_inverse.hpp>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cmath>

namespace openmx::core {
namespace {
void require(bool valid,const char* message)
{ if(!valid)throw std::invalid_argument(message); }
std::size_t storage_size(UnpivotedInverse::matrix_view matrix)
{
  const auto rows=matrix.extent(0),columns=matrix.extent(1);
  if(columns && rows>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double)/columns)
    throw std::length_error("inverse matrix size exceeds allocator maximum");
  const auto count=rows*columns;
  require(!count || matrix.data_handle(),"missing inverse matrix storage");
  return count;
}
bool overlap(UnpivotedInverse::matrix_view a,std::size_t na,UnpivotedInverse::matrix_view b,std::size_t nb)
{
  if(!na || !nb)return false;
  const auto x=reinterpret_cast<std::uintptr_t>(a.data_handle()),y=reinterpret_cast<std::uintptr_t>(b.data_handle());
  return x<=y ? y-x<na*sizeof(double) : x-y<nb*sizeof(double);
}
}

std::size_t UnpivotedInverse::checked_capacity(std::size_t capacity)
{
  if(capacity>INT_MAX || (capacity && capacity>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double)/capacity))
    throw std::length_error("inverse scratch capacity exceeds allocator maximum");
  return capacity;
}
UnpivotedInverse::UnpivotedInverse(std::size_t capacity)
    :capacity_(checked_capacity(capacity)),x_({capacity_}),y_({capacity_}),backup_({capacity_,capacity_}) {}

void UnpivotedInverse::clear(matrix_view input)
{
  const auto count=storage_size(input);
  if(count)std::fill_n(input.data_handle(),count,0.0);
}

void UnpivotedInverse::validate(matrix_view input,matrix_view inverse,std::size_t dimension)
{
  require(dimension && dimension<=INT_MAX,"invalid inverse dimension");
  require(dimension<=input.extent(0) && dimension<=input.extent(1) &&
      dimension<=inverse.extent(0) && dimension<=inverse.extent(1),"inverse active subspace exceeds matrix dimensions");
  const auto input_size=storage_size(input),output_size=storage_size(inverse);
  require(!overlap(input,input_size,inverse,output_size),"inverse input and output overlap");
}

void UnpivotedInverse::apply(matrix_view input,matrix_view inverse,std::size_t dimension)
{
  validate(input,inverse,dimension);
  require(dimension<=capacity_,"inverse exceeds scratch capacity");
  const int n=static_cast<int>(dimension-1);
  int i,j,k;double w;
  const auto x=x_.view(),y=y_.view();const auto backup=backup_.view();
  // BEGIN legacy numerical loops

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
        element(backup,i,j) = element(input,i,j);
      }
    }

    /****************************************************
                     LU factorization
    ****************************************************/

    for (k=0; k<=n-1; k++){
      w = 1.0/element(input,k,k);
      for (i=k+1; i<=n; i++){
        element(input,i,k) = w*element(input,i,k);
	for (j=k+1; j<=n; j++){
          element(input,i,j) = std::fma(-element(input,i,k),element(input,k,j),element(input,i,j));
	}
      }
    }
    for (k=0; k<=n; k++){

      /****************************************************
                             Ly = b
      ****************************************************/

      for (i=0; i<=n; i++){
	if (i==k)
	  element(y,i) = 1.0;
	else
	  element(y,i) = 0.0;
	for (j=0; j<=i-1; j++){
          element(y,i) = std::fma(-element(input,i,j),element(y,j),element(y,i));
	}
      }

      /****************************************************
                             Ux = y
      ****************************************************/

      for (i=n; 0<=i; i--){
	element(x,i) = element(y,i);
	for (j=n; (i+1)<=j; j--){
          element(x,i) = std::fma(-element(input,i,j),element(x,j),element(x,i));
	}
        element(x,i) = element(x,i)/element(input,i,i);
        element(inverse,i,k) = element(x,i);
      }
    }

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
        element(input,i,j) = element(backup,i,j);
      }
    }

  // END legacy numerical loops
}

} // namespace openmx::core
