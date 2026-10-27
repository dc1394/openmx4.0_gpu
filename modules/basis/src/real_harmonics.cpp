#include <openmx/basis/real_harmonics.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace openmx::basis {
namespace {
std::size_t validate(int maximum)
{
  if(maximum<0 || maximum>64)
    throw std::invalid_argument("real harmonic degree must be in [0,64]");
  const auto width=static_cast<std::size_t>(2*maximum+1);
  return (maximum+1)*width*width;
}

void fill(int maximum,dcomplex* output) noexcept
{
  const std::size_t width=2*maximum+1;
  std::fill_n(output,(maximum+1)*width*width,dcomplex{0.,0.});
  const double c=1.0/std::sqrt(2.0);
  auto set=[&](int l,int row,int column,double r,double i) {
    output[(l*width+row)*width+column]={r,i};
  };
  set(0,0,0,1.,0.);
  for(int l=1;l<=maximum;++l) {
    // The p shell places m=0 last; the d shell places |m|=2 before |m|=1.
    set(l,l==1?2:0,l,1.,0.);
    for(int m=1;m<=l;++m) {
      const int cosine=l==1?0:(l==2?(m==1?3:1):2*m-1);
      const int sine=cosine+1;
      set(l,cosine,l-m,c,0.);
      set(l,cosine,l+m,m%2?-c:c,0.);
      set(l,sine,l-m,0.,c);
      set(l,sine,l+m,0.,m%2?c:-c);
    }
  }
}
} // namespace

RealHarmonicTransform::RealHarmonicTransform(int maximum)
  : storage_(validate(maximum)),maximum_(maximum)
{ fill(maximum,storage_.data()); }

RealHarmonicTransform::RealHarmonicTransform(RealHarmonicTransform&& other) noexcept
  : storage_(std::move(other.storage_)),maximum_(std::exchange(other.maximum_,-1))
{}
RealHarmonicTransform& RealHarmonicTransform::operator=(const RealHarmonicTransform& other)
{
  if(this!=&other) {RealHarmonicTransform copy(other);*this=std::move(copy);}
  return *this;
}
RealHarmonicTransform& RealHarmonicTransform::operator=(RealHarmonicTransform&& other) noexcept
{
  if(this!=&other) {storage_=std::move(other.storage_);maximum_=std::exchange(other.maximum_,-1);}
  return *this;
}
OpenMXRealHarmonicData RealHarmonicTransform::view() const noexcept
{
  return maximum_<0?OpenMXRealHarmonicData{0,0,nullptr}:
      OpenMXRealHarmonicData{static_cast<std::size_t>(maximum_+1),
          static_cast<std::size_t>(2*maximum_+1),storage_.data()};
}
const dcomplex& RealHarmonicTransform::coefficient(int degree,int row,int column) const
{
  if(degree<0 || degree>maximum_ || row<0 || row>2*degree || column<0 || column>2*degree)
    throw std::out_of_range("real harmonic coefficient index out of range");
  const auto width=view().width;
  return storage_[(degree*width+row)*width+column];
}
void RealHarmonicTransform::transform(int degree,
    std::span<const std::complex<double>> input,std::span<std::complex<double>> output) const
{
  if(degree<0 || degree>maximum_)
    throw std::out_of_range("real harmonic transform degree out of range");
  const auto count=static_cast<std::size_t>(2*degree+1);
  if(input.size()!=count || output.size()!=count)
    throw std::invalid_argument("real harmonic vector dimensions differ");
  std::array<std::complex<double>,129> result;
  const auto width=view().width;
  for(std::size_t row=0;row<count;++row) {
    double r=0.,i=0.;
    for(std::size_t col=0;col<count;++col) {
      const auto t=storage_[(degree*width+row)*width+col];
      r+=t.r*input[col].real()-t.i*input[col].imag();
      i+=t.r*input[col].imag()+t.i*input[col].real();
    }
    result[row]={r,i};
  }
  std::copy_n(result.begin(),count,output.begin());
}
} // namespace openmx::basis

extern "C" int openmx_basis_real_harmonics_fill(int maximum,dcomplex* output,
    std::size_t capacity,char* error,std::size_t error_capacity)
{
  if(error && error_capacity)*error='\0';
  try {
    const auto count=openmx::basis::validate(maximum);
    if(!output || capacity<count)
      throw std::invalid_argument("missing or short real harmonic output array");
    openmx::basis::fill(maximum,output);
    return 1;
  }catch(const std::exception& exception) {
    if(error && error_capacity)std::snprintf(error,error_capacity,"%s",exception.what());
  }catch(...) {
    if(error && error_capacity)std::snprintf(error,error_capacity,"unknown real harmonic error");
  }
  return 0;
}
