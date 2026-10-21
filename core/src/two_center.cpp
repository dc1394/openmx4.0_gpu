#include <openmx/core/two_center.hpp>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>

namespace openmx::core {
namespace {
std::size_t magnetic(int l)
{
  if (l<0 || l>14) throw std::invalid_argument("invalid two-center angular capacity");
  return 2*l+1;
}
std::size_t channels(std::size_t n)
{
  if (n!=4 && n!=8) throw std::invalid_argument("two-center channels must be 4 or 8");
  return n;
}
void addressable(std::size_t a,std::size_t b,std::size_t c,std::size_t bytes)
{
  const auto maximum=static_cast<std::size_t>(PTRDIFF_MAX)/bytes;
  if (!a || !b || !c || a>maximum/b || a*b>maximum/c)
    throw std::length_error("two-center extent product exceeds address range");
}
dcomplex multiply(dcomplex a,dcomplex b) noexcept
{ return {a.r*b.r-a.i*b.i,a.r*b.i+a.i*b.r}; }
void accumulate(dcomplex& a,dcomplex b) noexcept { a.r+=b.r;a.i+=b.i; }
template<class View>
void cartesian(View out,std::size_t base,std::size_t i,std::size_t j,
    const dcomplex* sums,const OpenMXTwoCenterGeometry& g,double scale)
{
  element(out,base,i,j)=scale*sums[0].r;
  if (g.onsite) {
    for (std::size_t c=1;c<4;++c) element(out,base+c,i,j)=0.;
    return;
  }
  const double dr=sums[1].r,dt=sums[2].r,dp=sums[3].r;
  if (std::fabs(g.sin_theta)<10e-14) {
    element(out,base+1,i,j)=-scale*(g.sin_theta*g.cos_phi*dr+g.cos_theta*g.cos_phi/g.distance*dt);
    element(out,base+2,i,j)=-scale*(g.sin_theta*g.sin_phi*dr+g.cos_theta*g.sin_phi/g.distance*dt);
  } else {
    element(out,base+1,i,j)=-scale*(g.sin_theta*g.cos_phi*dr+g.cos_theta*g.cos_phi/g.distance*dt
        -g.sin_phi/g.sin_theta/g.distance*dp);
    element(out,base+2,i,j)=-scale*(g.sin_theta*g.sin_phi*dr+g.cos_theta*g.sin_phi/g.distance*dt
        +g.cos_phi/g.sin_theta/g.distance*dp);
  }
  element(out,base+3,i,j)=-scale*(g.cos_theta*dr-g.sin_theta/g.distance*dt);
}

template<std::size_t Channels>
void transform(const OpenMXTwoCenterInput& in,const OpenMXTwoCenterBuffers& b)
{
  using Extents=md::dextents<std::size_t,3>;
  const md::mdspan<const dcomplex,Extents> source(in.angular,Channels,in.source_rows,in.source_columns);
  const md::mdspan<const dcomplex,Extents> conversion(in.conversion,in.conversion_orders,
      in.conversion_width,in.conversion_width);
  const md::mdspan<dcomplex,Extents> left(b.left,Channels,b.rows,b.columns);
  const md::mdspan<double,Extents> output(b.values,Channels,b.rows,b.columns);
  const int l0=in.left_angular,l1=in.right_angular;
  const std::size_t n0=2*l0+1,n1=2*l1+1;
  for (std::size_t i=0;i<n0;++i) for (std::size_t j=0;j<n1;++j) {
    dcomplex sums[Channels]{};
    for (std::size_t k=0;k<n0;++k) {
      auto c=element(conversion,l0,i,k);c.i=-c.i;
      for (std::size_t ch=0;ch<Channels;++ch)
        accumulate(sums[ch],multiply(c,element(source,ch,in.row+k,in.column+j)));
    }
    for (std::size_t ch=0;ch<Channels;++ch) element(left,ch,i,j)=sums[ch];
  }
  for (std::size_t i=0;i<n0;++i) for (std::size_t j=0;j<n1;++j) {
    dcomplex sums[Channels]{};
    dcomplex momenta[3]{};
    for (std::size_t k=0;k<n1;++k) {
      const auto c=element(conversion,l1,j,k);
      if constexpr (Channels==8) {
        double up=0.,down=0.;
        if (k+1<n1) {
          up=std::sqrt(static_cast<double>((2*l1-static_cast<int>(k))*(static_cast<int>(k)+1)));
          auto v=multiply(element(left,0,i,k+1),c);
          accumulate(momenta[0],{0.5*up*v.r,0.5*up*v.i});
        }
        if (k>0) {
          down=std::sqrt(static_cast<double>(static_cast<int>(k)*(2*l1-static_cast<int>(k)+1)));
          auto v=multiply(element(left,0,i,k-1),c);
          accumulate(momenta[0],{0.5*down*v.r,0.5*down*v.i});
        }
        if (k+1<n1) {
          auto v=multiply(element(left,0,i,k+1),c);
          accumulate(momenta[1],{0.5*up*v.i,-0.5*up*v.r});
        }
        if (k>0) {
          auto v=multiply(element(left,0,i,k-1),c);
          accumulate(momenta[1],{-0.5*down*v.i,0.5*down*v.r});
        }
        auto v=multiply(element(left,0,i,k),c);
        accumulate(momenta[2],{static_cast<double>(static_cast<int>(k)-l1)*v.r,
                              static_cast<double>(static_cast<int>(k)-l1)*v.i});
      }
      for (std::size_t ch=0;ch<Channels;++ch)
        accumulate(sums[ch],multiply(element(left,ch,i,k),c));
    }
    cartesian(output,0,i,j,sums,in.geometry,8.);
    if constexpr (Channels==8) {
      cartesian(output,4,i,j,sums+4,in.geometry,4.);
      const md::mdspan<double,Extents> moments(b.momenta,3,b.rows,b.columns);
      for (int c=0;c<3;++c) element(moments,c,i,j)=8.*momenta[c].i;
    }
  }
}
} // namespace

TwoCenterWorkspace::TwoCenterWorkspace(std::size_t n,int l0,int l1)
  : left_({channels(n),magnetic(l0),magnetic(l1)}),
    values_({n,magnetic(l0),magnetic(l1)}),
    momenta_({n==8 ? 3u : 0u,magnetic(l0),magnetic(l1)}) {}

OpenMXTwoCenterBuffers TwoCenterWorkspace::buffers() noexcept
{ return {left_.extent(0),left_.extent(1),left_.extent(2),left_.data(),values_.data(),
          momenta_.size() ? momenta_.data() : nullptr}; }

void transform_two_center(const OpenMXTwoCenterInput& in,const OpenMXTwoCenterBuffers& b)
{
  const auto n0=magnetic(in.left_angular),n1=magnetic(in.right_angular);
  channels(b.channels);
  addressable(b.channels,in.source_rows,in.source_columns,sizeof(dcomplex));
  addressable(b.channels,b.rows,b.columns,sizeof(dcomplex));
  addressable(in.conversion_orders,in.conversion_width,in.conversion_width,sizeof(dcomplex));
  if (!in.angular || !in.conversion || !b.left || !b.values || (b.channels==8 && !b.momenta) ||
      n0>b.rows || n1>b.columns || in.row>in.source_rows || n0>in.source_rows-in.row ||
      in.column>in.source_columns || n1>in.source_columns-in.column ||
      static_cast<std::size_t>(in.left_angular)>=in.conversion_orders ||
      static_cast<std::size_t>(in.right_angular)>=in.conversion_orders ||
      n0>in.conversion_width || n1>in.conversion_width)
    throw std::invalid_argument("invalid two-center source, conversion or output block");
  const auto& g=in.geometry;
  if (!(g.distance>0) || !std::isfinite(g.distance) || !std::isfinite(g.sin_theta) ||
      !std::isfinite(g.cos_theta) || !std::isfinite(g.sin_phi) || !std::isfinite(g.cos_phi) ||
      (g.onsite!=0 && g.onsite!=1))
    throw std::invalid_argument("invalid two-center geometry");
  if (b.channels==8) transform<8>(in,b);else transform<4>(in,b);
}
} // namespace openmx::core

extern "C" int openmx_core_two_center_transform(const OpenMXTwoCenterInput* input,
    const OpenMXTwoCenterBuffers* buffers,char* error,std::size_t capacity)
{
  if (error && capacity) *error='\0';
  try {
    if (!input || !buffers) throw std::invalid_argument("missing two-center input or output views");
    openmx::core::transform_two_center(*input,*buffers);return 1;
  } catch (const std::exception& e) {
    if (error && capacity) std::snprintf(error,capacity,"%s",e.what());
  } catch (...) {
    if (error && capacity) std::snprintf(error,capacity,"unexpected two-center exception");
  }
  return 0;
}
