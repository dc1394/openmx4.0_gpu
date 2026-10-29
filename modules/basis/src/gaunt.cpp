#include <openmx/basis/gaunt.hpp>
#include <openmx/basis/detail/gaunt.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace openmx::basis {
namespace {
std::size_t table_size(int l0, int l1, int ll)
{
  if(l0<0 || l0>38 || l1<0 || l1>38 || ll<0 || ll>38)
    throw std::invalid_argument("Gaunt table degrees must be in [0,38]");
  std::array maxima{l0,l1,ll};std::sort(maxima.begin(),maxima.end());
  const int pair=maxima[0]+maxima[1];
  if(pair+std::min(pair,maxima[2])>38)
    throw std::invalid_argument("Gaunt table contains a triangle with degree sum >38");
  return std::size_t(l0+1)*(l0+1)*(l1+1)*(ll+1)*(2*l1+1);
}
void fill(int l0max,int l1max,int llmax,double* values) noexcept
{
  const OpenMXGauntTableData table{l0max,l1max,llmax,std::size_t(2*l1max+1),values};
  std::fill_n(values,std::size_t(l0max+1)*(l0max+1)*(llmax+1)*(l1max+1)*table.row_stride,0.);
  for(int l0=0;l0<=l0max;++l0)for(int m0=-l0;m0<=l0;++m0)
    for(int ll=0;ll<=llmax;++ll)for(int l1=0;l1<=l1max;++l1)
      for(int m1=-l1;m1<=l1;++m1)
        values[openmx_gaunt_table_row_offset(&table,l0,m0,ll,l1)+l1+m1]=
            openmx_basis_gaunt_kernel(l0,m0,l1,m1,ll,m0-m1);
}
void validate(HarmonicIndex value)
{
  if(value.degree<0 || value.degree>38 || value.order < -value.degree || value.order>value.degree)
    throw std::invalid_argument("invalid Gaunt harmonic index");
}
} // namespace

double gaunt(HarmonicIndex bra,HarmonicIndex left,HarmonicIndex right)
{
  validate(bra);validate(left);validate(right);
  if(!openmx_basis_gaunt_degrees_valid(bra.degree,left.degree,right.degree))
    throw std::invalid_argument("Gaunt degree sum exceeds 38");
  return openmx_basis_gaunt_kernel(bra.degree,bra.order,left.degree,left.order,right.degree,right.order);
}
GauntTable::GauntTable(int l0,int l1,int ll)
  :storage_(table_size(l0,l1,ll)),l0max_(l0),l1max_(l1),llmax_(ll)
{fill(l0,l1,ll,storage_.data());}
GauntTable::GauntTable(GauntTable&& other) noexcept
  :storage_(std::move(other.storage_)),l0max_(std::exchange(other.l0max_,-1)),
   l1max_(std::exchange(other.l1max_,-1)),llmax_(std::exchange(other.llmax_,-1))
{}
GauntTable& GauntTable::operator=(const GauntTable& other)
{
  if(this!=&other){GauntTable copy(other);*this=std::move(copy);}
  return *this;
}
GauntTable& GauntTable::operator=(GauntTable&& other) noexcept
{
  if(this!=&other){storage_=std::move(other.storage_);l0max_=std::exchange(other.l0max_,-1);
    l1max_=std::exchange(other.l1max_,-1);llmax_=std::exchange(other.llmax_,-1);}
  return *this;
}
OpenMXGauntTableData GauntTable::view() const noexcept
{
  return l0max_<0?OpenMXGauntTableData{-1,-1,-1,0,nullptr}:
      OpenMXGauntTableData{l0max_,l1max_,llmax_,std::size_t(2*l1max_+1),storage_.data()};
}
std::span<const double> GauntTable::values() const noexcept
{return l0max_<0?std::span<const double>{}:std::span<const double>{storage_};}
std::span<const double> GauntTable::row(int l0,int m0,int ll,int l1) const
{
  if(l0<0 || l0>l0max_ || m0 < -l0 || m0>l0 || ll<0 || ll>llmax_ || l1<0 || l1>l1max_)
    throw std::out_of_range("Gaunt table row is outside active indices");
  const auto data=view();
  return {storage_.data()+openmx_gaunt_table_row_offset(&data,l0,m0,ll,l1),std::size_t(2*l1+1)};
}
double GauntTable::coefficient(int l0,int m0,int l1,int m1,int ll) const
{
  const auto data=row(l0,m0,ll,l1);
  if(m1 < -l1 || m1>l1)throw std::out_of_range("Gaunt table magnetic index out of range");
  return data[l1+m1];
}
} // namespace openmx::basis

extern "C" int openmx_basis_gaunt(int l,int m,int l1,int m1,int l2,int m2,
    double* output,char* error,std::size_t capacity)
{
  if(error && capacity)*error='\0';
  try {
    if(!output)throw std::invalid_argument("missing Gaunt output");
    const double result=openmx::basis::gaunt({l,m},{l1,m1},{l2,m2});
    *output=result;return 1;
  }catch(const std::exception& exception) {
    if(error && capacity)std::snprintf(error,capacity,"%s",exception.what());
  }catch(...) {
    if(error && capacity)std::snprintf(error,capacity,"unknown Gaunt error");
  }
  return 0;
}
extern "C" int openmx_basis_gaunt_table_fill(int l0,int l1,int ll,double* values,
    std::size_t count,OpenMXGauntTableData* out,char* error,std::size_t capacity)
{
  if(error && capacity)*error='\0';
  try {
    const auto required=openmx::basis::table_size(l0,l1,ll);
    if(!values || !out || count<required)throw std::invalid_argument("missing or short Gaunt table output");
    openmx::basis::fill(l0,l1,ll,values);
    *out={l0,l1,ll,std::size_t(2*l1+1),values};return 1;
  }catch(const std::exception& exception) {
    if(error && capacity)std::snprintf(error,capacity,"%s",exception.what());
  }catch(...) {
    if(error && capacity)std::snprintf(error,capacity,"unknown Gaunt table error");
  }
  return 0;
}
