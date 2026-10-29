#include <openmx/basis/orbital_radial.hpp>
#include <openmx/basis/detail/orbital_radial.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace openmx::basis {
namespace {
void radius_checked(double radius)
{
  if(!std::isfinite(radius) || radius<0)throw std::invalid_argument("invalid orbital radial coordinate");
}
void owner_ready(std::span<const double> radii)
{
  if(radii.size()<5)throw std::logic_error("orbital radial owner is empty or moved from");
}
} // namespace
OrbitalRadialTableView::OrbitalRadialTableView(std::span<const double> radii,std::span<const RadialChannel> channels)
  : radii_(radii),channels_(channels)
{
  if(radii.size()<5 || radii.size()>INT_MAX || channels.size()>std::vector<double>().max_size()/radii.size())
    throw std::invalid_argument("invalid orbital radial table dimensions");
  for(std::size_t i=0;i<radii.size();++i)
    if(!std::isfinite(radii[i]) || radii[i]<0 || (i && radii[i]<=radii[i-1]))
      throw std::invalid_argument("invalid orbital radial mesh");
  for(const auto& channel:channels) {
    if(channel.angular<0 || channel.angular>29 || channel.values.size()!=radii.size())
      throw std::invalid_argument("invalid orbital radial channel");
    for(double value:channel.values)if(!std::isfinite(value))throw std::invalid_argument("nonfinite orbital radial value");
  }
}
int OrbitalRadialTableView::angular(std::size_t channel) const
{
  if(channel>=channels_.size())throw std::out_of_range("orbital radial channel out of range");
  return channels_[channel].angular;
}
OrbitalRadialSample OrbitalRadialTableView::sample(std::size_t channel,double radius) const
{
  const int l=angular(channel);radius_checked(radius);
  const auto plan=openmx_basis_detail_orbital_radial_plan(radii_.data(),radii_.size(),radius);
  return openmx_basis_detail_orbital_radial_apply(&plan,channels_[channel].values.data(),l,1);
}
void OrbitalRadialTableView::evaluate(double radius,std::span<OrbitalRadialSample> out) const
{
  radius_checked(radius);
  if(out.size()<channels_.size())throw std::invalid_argument("short orbital radial output");
  const auto plan=openmx_basis_detail_orbital_radial_plan(radii_.data(),radii_.size(),radius);
  for(std::size_t c=0;c<channels_.size();++c)
    out[c]=openmx_basis_detail_orbital_radial_apply(&plan,channels_[c].values.data(),channels_[c].angular,1);
}
OrbitalRadialTable::OrbitalRadialTable(std::span<const double> radii,std::span<const RadialChannel> channels)
  : OrbitalRadialTable(OrbitalRadialTableView(radii,channels)) {}
OrbitalRadialTable::OrbitalRadialTable(const OrbitalRadialTableView& input)
  : radii_(input.radii_.begin(),input.radii_.end()),values_(input.radii_.size()*input.channels_.size())
{
  angular_.reserve(input.channels_.size());
  for(std::size_t c=0;c<input.channels_.size();++c) {
    angular_.push_back(input.channels_[c].angular);
    std::copy(input.channels_[c].values.begin(),input.channels_[c].values.end(),values_.begin()+c*radii_.size());
  }
}
OrbitalRadialSample OrbitalRadialTable::sample(std::size_t channel,double radius) const
{
  owner_ready(radii_);const int l=angular_.at(channel);radius_checked(radius);
  const auto plan=openmx_basis_detail_orbital_radial_plan(radii_.data(),radii_.size(),radius);
  return openmx_basis_detail_orbital_radial_apply(&plan,values_.data()+channel*radii_.size(),l,1);
}
void OrbitalRadialTable::evaluate(double radius,std::span<OrbitalRadialSample> out) const
{
  owner_ready(radii_);radius_checked(radius);
  if(out.size()<angular_.size())throw std::invalid_argument("short orbital radial output");
  const auto plan=openmx_basis_detail_orbital_radial_plan(radii_.data(),radii_.size(),radius);
  for(std::size_t c=0;c<angular_.size();++c)
    out[c]=openmx_basis_detail_orbital_radial_apply(&plan,values_.data()+c*radii_.size(),angular_[c],1);
}
} // namespace openmx::basis

extern "C" int openmx_orbital_radial_sample(int angular,const double* radii,const double* values,
    std::size_t points,double radius,OpenMXOrbitalRadialSample* out,char* error,std::size_t capacity)
{
  if(error && capacity)*error='\0';
  try {
    if(!radii || !values || !out || points<5 || points>INT_MAX)
      throw std::invalid_argument("missing or invalid orbital radial input/output");
    const openmx::basis::RadialChannel channel{angular,{values,points}};
    const openmx::basis::OrbitalRadialTableView view({radii,points},{&channel,1});
    const auto value=view.sample(0,radius);*out=value;return 1;
  }catch(const std::exception& exception){if(error && capacity)std::snprintf(error,capacity,"%s",exception.what());}
  catch(...){if(error && capacity)std::snprintf(error,capacity,"unknown orbital radial error");}
  return 0;
}
