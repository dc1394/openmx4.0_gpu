#include <openmx/core/vna_host_tables.hpp>
#include <openmx/core/vna_preparation.hpp>
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>

namespace openmx::core {
namespace {
template<std::size_t N>
std::size_t checked_size(const std::array<std::size_t,N>& shape,std::size_t element_size)
{
  const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/element_size;
  std::size_t size=1;
  for(auto extent:shape) {
    if(extent && size>limit/extent)throw std::length_error("VNA packing input extent overflows");
    size*=extent;
  }
  return size;
}
std::size_t projector_stride(VnaProjectorPackingShape s,const OpenMXVnaHostProjectorInput& input)
{
  if(s.species<=0 || s.points<=0 || s.angular<0 || s.angular>14 || s.multiplicity<0 ||
      s.multiplicity>INT_MAX/(s.angular+1) || (input.convert!=0 && input.convert!=1))
    throw std::invalid_argument("invalid VNA projector packing shape");
  const auto count=std::size_t(s.angular+1)*s.multiplicity;
  if(count) {
    const auto& v=input.projectors;
    if(!v.values || v.extent[0]<std::size_t(s.species) || v.extent[1]<std::size_t(s.angular+1) ||
        v.extent[2]<std::size_t(s.multiplicity) || v.extent[3]<std::size_t(s.points))
      throw std::invalid_argument("short VNA projector packing input");
    (void)checked_size<4>({v.extent[0],v.extent[1],v.extent[2],v.extent[3]},sizeof(double));
  }
  if(input.convert) {
    if(!input.conversion || input.width<std::size_t(2*s.angular+1))
      throw std::invalid_argument("short VNA conversion packing input");
    (void)checked_size<3>({std::size_t(s.angular+1),input.width,input.width},sizeof(dcomplex));
  }
  return (count+7)/8*8;
}
int checked_points(int points)
{
  if(points<=0)throw std::invalid_argument("invalid VNA product point count");
  return points;
}
int product_orders(const RadialBasisLayout& basis)
{
  int top=0;for(int l:basis.maximum_angular())top=std::max(top,l);
  return (2*top+1+7)/8*8;
}
std::vector<std::size_t> product_offsets(const RadialBasisLayout& basis,
    std::vector<int>& radials,int& maximum)
{
  std::vector<std::size_t> result(radials.size()+1);
  const auto channels=basis.counts().extent(1);
  for(std::size_t species=0;species<radials.size();++species) {
    const auto count=basis.offsets()[(species+1)*channels]-basis.offsets()[species*channels];
    radials[species]=static_cast<int>(count);maximum=std::max(maximum,radials[species]);
    const auto blocks=checked_size<2>({count,count},sizeof(double));
    if(result[species]>std::numeric_limits<std::size_t>::max()-blocks)
      throw std::length_error("VNA product block offsets overflow");
    result[species+1]=result[species]+blocks;
  }
  return result;
}
} // namespace

VnaHostProjectors::VnaHostProjectors(VnaProjectorPackingShape s,const OpenMXVnaHostProjectorInput& input)
  : stride_(projector_stride(s,input)),width_(input.width),
    projectors_({std::size_t(s.species),std::size_t(s.points),stride_}),
    conversion_({input.convert?2u:0u,std::size_t(s.angular+1),width_,width_})
{
  const auto& source=input.projectors;
  for(int species=0;species<s.species;++species)
    for(int l=0;l<=s.angular;++l)
      for(int radial=0;radial<s.multiplicity;++radial) {
        const double* row=openmx_radial_row4(&source,species,l,radial);
        const auto column=std::size_t(l)*s.multiplicity+radial;
        for(int point=0;point<s.points;++point)
          element(projectors_.view(),species,point,column)=row[point];
      }
  if(input.convert)
    for(int l=0;l<=s.angular;++l)
      for(int row=0;row<=2*l;++row)
        for(int column=0;column<=2*l;++column) {
          const auto value=input.conversion[(std::size_t(l)*width_+row)*width_+column];
          element(conversion_.view(),0,l,column,row)=value.r;
          element(conversion_.view(),1,l,column,row)=value.i;
        }
}
OpenMXVnaHostProjectorData VnaHostProjectors::view() const noexcept
{
  return {stride_,width_,projectors_.size()?projectors_.data():nullptr,
      conversion_.size()?conversion_.data():nullptr,
      conversion_.size()?conversion_.data()+conversion_.size()/2:nullptr};
}

VnaHostProducts::VnaHostProducts(const RadialBasisLayout& basis,int points,const OpenMXProductRadialView& input)
  : points_(checked_points(points)),orders_(product_orders(basis)),
    radials_(basis.maximum_angular().size()),offsets_(product_offsets(basis,radials_,radials_max_)),
    products_({offsets_.back(),std::size_t(points_),std::size_t(orders_)})
{
  if(!offsets_.back())return;
  if(!input.values || !input.layout || input.species<radials_.size() || input.stride<std::size_t(points_))
    throw std::invalid_argument("missing or short VNA product packing input");
  (void)checked_size<2>({input.rows,input.stride},sizeof(double));
  const auto counts=basis.counts();const auto offsets=basis.offsets();const auto channels=counts.extent(1);
  for(std::size_t s=0;s<radials_.size();++s) {
    if(!radials_[s])continue;
    const auto& layout=input.layout[s];
    if(!layout.orbital_offsets || !layout.pair_rows)
      throw std::invalid_argument("missing VNA product source layout");
    (void)checked_size<2>({layout.orbitals,layout.orbitals},sizeof(std::size_t));
    for(int l=0;l<=basis.maximum_angular()[s];++l) {
      const auto count=std::size_t(element(counts,s,l));
      if(count && (layout.orbital_offsets[l]>layout.orbitals || count>layout.orbitals-layout.orbital_offsets[l]))
        throw std::invalid_argument("short VNA product source orbital range");
    }
    for(int l0=0;l0<=basis.maximum_angular()[s];++l0)
      for(int r0=0;r0<element(counts,s,l0);++r0)
        for(int l1=l0;l1<=basis.maximum_angular()[s];++l1)
          for(int r1=0;r1<element(counts,s,l1);++r1) {
            const auto left=offsets[s*channels+l0]-offsets[s*channels]+r0;
            const auto right=offsets[s*channels+l1]-offsets[s*channels]+r1;
            const auto block=offsets_[s]+left*radials_[s]+right;
            const auto pair=(layout.orbital_offsets[l0]+r0)*layout.orbitals+layout.orbital_offsets[l1]+r1;
            const auto start=layout.pair_rows[pair];
            if(layout.first_row>input.rows || start>input.rows-layout.first_row ||
                std::size_t(2*l1+1)>input.rows-layout.first_row-start)
              throw std::invalid_argument("short VNA product source row range");
            for(int l=0;l<=2*l1;++l) {
              const double* row=input.values+(layout.first_row+start+l)*input.stride;
              for(int point=0;point<points_;++point)
                element(products_.view(),block,point,l)=row[point];
            }
          }
  }
}
OpenMXVnaHostProductData VnaHostProducts::view() const noexcept
{
  return {orders_,radials_max_,offsets_.back(),offsets_.data(),radials_.data(),products_.size()?products_.data():nullptr};
}
} // namespace openmx::core
