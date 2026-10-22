#include <openmx/core/projector_force.hpp>
#include <openmx/core/nonlocal_workspace.hpp>
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace openmx::core {
namespace {
void require(bool ok,const char *message){if(!ok)throw std::invalid_argument(message);}
std::size_t extent_product(std::span<const std::size_t> extents,std::size_t element_size)
{
  std::size_t count=1;
  for(auto n:extents){require(!n || count<=std::size_t(PTRDIFF_MAX)/element_size/n,
      "nonlocal projector input extent overflows");count*=n;}
  return count;
}
} // namespace

ProjectorForce::ProjectorForce(const OpenMXAtomicViews& atomic,int spin,int spin_orbit)
    :spin_(spin),spin_orbit_(spin_orbit)
{
  require((spin==0 || spin==1 || spin==3) && (spin_orbit==0 || spin_orbit==1),
      "invalid nonlocal Force spin configuration");
  const auto& s=atomic.species;const auto& e=atomic.projector_energies;
  require(s.species<=std::size_t(INT_MAX),"nonlocal Force species count exceeds C indexing");
  if(!s.species)return;
  require(s.radial_projectors && s.projector_components && s.relativistic &&
      atomic.projector_angular_species>=s.species,"missing nonlocal projector metadata");
  const std::array<std::size_t,2> angular_shape{atomic.projector_angular_species,atomic.projector_angular_stride};
  extent_product(angular_shape,sizeof(int));extent_product(e.extent,sizeof(double));
  std::vector<NonlocalSpecies> input;input.reserve(s.species);
  std::size_t count=0;
  for(std::size_t w=0;w<s.species;++w) {
    const int n=s.radial_projectors[w];
    require(n>=0 && s.projector_components[w]>=0 && (s.relativistic[w]==0 || s.relativistic[w]==1),
        "invalid nonlocal projector species");
    require(!n || (atomic.projector_angular && std::size_t(n)<atomic.projector_angular_stride &&
        e.values && e.extent[0]>=std::size_t(s.relativistic[w]+1) && e.extent[1]>=s.species && e.extent[2]>=std::size_t(n)),
        "nonlocal projector input is too small");
    input.push_back({n?std::span<const int>(openmx_atomic_projector_angular(&atomic,w)+1,n):std::span<const int>{}});
    require(std::size_t(n)<=std::size_t(INT_MAX)-count,"nonlocal projector count overflows");count+=n;
  }
  const NonlocalProjectors layout(14,input);
  channels_.resize(count);species_.resize(s.species);std::size_t first=0;
  for(std::size_t w=0;w<s.species;++w) {
    const auto n=std::size_t(layout.counts()[w]),base=layout.species_offsets()[w];
    std::size_t components=0;
    for(std::size_t j=0;j<n;++j) {
      const int l=layout.angular()[base+j+1];const auto width=std::size_t(2*l+1);
      require(!s.relativistic[w] || l<=3,"relativistic Force supports s/p/d/f projectors");
      auto& channel=channels_[first+j];channel.twice_angular=2*l;channel.offset=layout.offsets()[base+j+1];
      channel.weight[0]=double(l+1)/double(width);channel.weight[1]=double(l)/double(width);
      for(int c=0;c<=s.relativistic[w];++c) {
        channel.energy[c]=openmx_radial_value3(&e,c,w,j);
        require(std::isfinite(channel.energy[c]),"nonfinite nonlocal projector energy");
      }
      components+=width;
    }
    require(components==std::size_t(s.projector_components[w]),"nonlocal projector component count differs from angular layout");
    species_[w]={s.relativistic[w],n,components,n?channels_.data()+first:nullptr};first+=n;
  }
}
OpenMXProjectorForceView ProjectorForce::view() const noexcept
{
  return {spin_,spin_orbit_,species_.size(),species_.data()};
}

namespace {
template<bool Noncollinear,int Angular>
void projector_spinor_channel(const OpenMXForceProjectorChannel& channel,
    const OpenMXProjectorForceRows& in,Tensor<dcomplex,2>::view_type sum)
{
  using Row=Tensor<double,1>::const_view_type;
  const auto a=[&](int c,int axis,std::size_t index){return element(Row(in.derivative[c][axis],in.columns),index);};
  const auto b=[&](int c,std::size_t index){return element(Row(in.value[c],in.columns),index);};
  constexpr int spin=Noncollinear?3:1,l2=2*Angular;
  std::size_t L=channel.offset;
  const double fugou=in.sign,PFp=channel.weight[0],PFm=channel.weight[1];
  const double ene_p=in.reverse_energy?-channel.energy[0]:channel.energy[0];
  const double ene_m=in.reverse_energy?-channel.energy[1]:channel.energy[1];
  using std::sqrt;


    int l3, i;
    double tmpx, tmpy, tmpz;
    double tmp0, tmp1, tmp2;
    double tmp3, tmp4, tmp5, tmp6;
    double deri[4];

    /****************************************************
      off-diagonal contribution to up-dn matrix
      for spin non-collinear
    ****************************************************/

    if (spin == 3) {

        /* p */
        if (l2 == 2) {

            /* real contribution of l+1/2 to off diagonal up-down matrix */
            tmpx = fugou * (ene_p / 3.0 * a(0,0,L) * b(0,L + 2) - ene_p / 3.0 * a(0,0,L + 2) * b(0,L));

            tmpy = fugou * (ene_p / 3.0 * a(0,1,L) * b(0,L + 2) - ene_p / 3.0 * a(0,1,L + 2) * b(0,L));

            tmpz = fugou * (ene_p / 3.0 * a(0,2,L) * b(0,L + 2) - ene_p / 3.0 * a(0,2,L + 2) * b(0,L));

            element(sum,2,0).r += tmpx;
            element(sum,2,1).r += tmpy;
            element(sum,2,2).r += tmpz;

            /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

            tmpx = fugou * (-ene_p / 3.0 * a(0,0,L + 1) * b(0,L + 2) + ene_p / 3.0 * a(0,0,L + 2) * b(0,L + 1));

            tmpy = fugou * (-ene_p / 3.0 * a(0,1,L + 1) * b(0,L + 2) + ene_p / 3.0 * a(0,1,L + 2) * b(0,L + 1));

            tmpz = fugou * (-ene_p / 3.0 * a(0,2,L + 1) * b(0,L + 2) + ene_p / 3.0 * a(0,2,L + 2) * b(0,L + 1));

            element(sum,2,0).i += tmpx;
            element(sum,2,1).i += tmpy;
            element(sum,2,2).i += tmpz;

            /* real contribution of l-1/2 for to diagonal up-down matrix */

            tmpx = fugou * (ene_m / 3.0 * a(1,0,L) * b(1,L + 2) - ene_m / 3.0 * a(1,0,L + 2) * b(1,L));

            tmpy = fugou * (ene_m / 3.0 * a(1,1,L) * b(1,L + 2) - ene_m / 3.0 * a(1,1,L + 2) * b(1,L));

            tmpz = fugou * (ene_m / 3.0 * a(1,2,L) * b(1,L + 2) - ene_m / 3.0 * a(1,2,L + 2) * b(1,L));

            element(sum,2,0).r -= tmpx;
            element(sum,2,1).r -= tmpy;
            element(sum,2,2).r -= tmpz;

            /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

            tmpx = fugou * (-ene_m / 3.0 * a(1,0,L + 1) * b(1,L + 2) + ene_m / 3.0 * a(1,0,L + 2) * b(1,L + 1));

            tmpy = fugou * (-ene_m / 3.0 * a(1,1,L + 1) * b(1,L + 2) + ene_m / 3.0 * a(1,1,L + 2) * b(1,L + 1));

            tmpz = fugou * (-ene_m / 3.0 * a(1,2,L + 1) * b(1,L + 2) + ene_m / 3.0 * a(1,2,L + 2) * b(1,L + 1));

            element(sum,2,0).i -= tmpx;
            element(sum,2,1).i -= tmpy;
            element(sum,2,2).i -= tmpz;
        }

        /* d */
        if (l2 == 4) {

            tmp0 = sqrt(3.0);
            tmp1 = ene_p / 5.0;
            tmp2 = tmp0 * tmp1;

            /* real contribution of l+1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (-tmp2 * a(0,i-1,L) * b(0,L + 3) + tmp2 * a(0,i-1,L + 3) * b(0,L) + tmp1 * a(0,i-1,L + 1) * b(0,L + 3) - tmp1 * a(0,i-1,L + 3) * b(0,L + 1) + tmp1 * a(0,i-1,L + 2) * b(0,L + 4) - tmp1 * a(0,i-1,L + 4) * b(0,L + 2));
            }
            element(sum,2,0).r += deri[1];
            element(sum,2,1).r += deri[2];
            element(sum,2,2).r += deri[3];

            /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (+tmp2 * a(0,i-1,L) * b(0,L + 4) - tmp2 * a(0,i-1,L + 4) * b(0,L) + tmp1 * a(0,i-1,L + 1) * b(0,L + 4) - tmp1 * a(0,i-1,L + 4) * b(0,L + 1) - tmp1 * a(0,i-1,L + 2) * b(0,L + 3) + tmp1 * a(0,i-1,L + 3) * b(0,L + 2));
            }
            element(sum,2,0).i += deri[1];
            element(sum,2,1).i += deri[2];
            element(sum,2,2).i += deri[3];

            /* real contribution of l-1/2 for to diagonal up-down matrix */

            tmp1 = ene_m / 5.0;
            tmp2 = tmp0 * tmp1;

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (-tmp2 * a(1,i-1,L) * b(1,L + 3) + tmp2 * a(1,i-1,L + 3) * b(1,L) + tmp1 * a(1,i-1,L + 1) * b(1,L + 3) - tmp1 * a(1,i-1,L + 3) * b(1,L + 1) + tmp1 * a(1,i-1,L + 2) * b(1,L + 4) - tmp1 * a(1,i-1,L + 4) * b(1,L + 2));
            }
            element(sum,2,0).r -= deri[1];
            element(sum,2,1).r -= deri[2];
            element(sum,2,2).r -= deri[3];

            /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (+tmp2 * a(1,i-1,L) * b(1,L + 4) - tmp2 * a(1,i-1,L + 4) * b(1,L) + tmp1 * a(1,i-1,L + 1) * b(1,L + 4) - tmp1 * a(1,i-1,L + 4) * b(1,L + 1) - tmp1 * a(1,i-1,L + 2) * b(1,L + 3) + tmp1 * a(1,i-1,L + 3) * b(1,L + 2));
            }
            element(sum,2,0).i -= deri[1];
            element(sum,2,1).i -= deri[2];
            element(sum,2,2).i -= deri[3];
        }

        /* f */
        if (l2 == 6) {

            /* real contribution of l+1/2 to off diagonal up-down matrix */

            tmp0 = sqrt(6.0);
            tmp1 = sqrt(3.0 / 2.0);
            tmp2 = sqrt(5.0 / 2.0);

            tmp3 = ene_p / 7.0;
            tmp4 = tmp1 * tmp3; /* sqrt(3.0/2.0) */
            tmp5 = tmp2 * tmp3; /* sqrt(5.0/2.0) */
            tmp6 = tmp0 * tmp3; /* sqrt(6.0)     */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (-tmp6 * a(0,i-1,L) * b(0,L + 1) + tmp6 * a(0,i-1,L + 1) * b(0,L) - tmp5 * a(0,i-1,L + 1) * b(0,L + 3) + tmp5 * a(0,i-1,L + 3) * b(0,L + 1) - tmp5 * a(0,i-1,L + 2) * b(0,L + 4) + tmp5 * a(0,i-1,L + 4) * b(0,L + 2) - tmp4 * a(0,i-1,L + 3) * b(0,L + 5) + tmp4 * a(0,i-1,L + 5) * b(0,L + 3) - tmp4 * a(0,i-1,L + 4) * b(0,L + 6) + tmp4 * a(0,i-1,L + 6) * b(0,L + 4));
            }
            element(sum,2,0).r += deri[1];
            element(sum,2,1).r += deri[2];
            element(sum,2,2).r += deri[3];

            /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (+tmp6 * a(0,i-1,L) * b(0,L + 2) - tmp6 * a(0,i-1,L + 2) * b(0,L) + tmp5 * a(0,i-1,L + 1) * b(0,L + 4) - tmp5 * a(0,i-1,L + 4) * b(0,L + 1) - tmp5 * a(0,i-1,L + 2) * b(0,L + 3) + tmp5 * a(0,i-1,L + 3) * b(0,L + 2) + tmp4 * a(0,i-1,L + 3) * b(0,L + 6) - tmp4 * a(0,i-1,L + 6) * b(0,L + 3) - tmp4 * a(0,i-1,L + 4) * b(0,L + 5) + tmp4 * a(0,i-1,L + 5) * b(0,L + 4));
            }
            element(sum,2,0).i += deri[1];
            element(sum,2,1).i += deri[2];
            element(sum,2,2).i += deri[3];

            /* real contribution of l-1/2 for to diagonal up-down matrix */

            tmp3 = ene_m / 7.0;
            tmp4 = tmp1 * tmp3; /* sqrt(3.0/2.0) */
            tmp5 = tmp2 * tmp3; /* sqrt(5.0/2.0) */
            tmp6 = tmp0 * tmp3; /* sqrt(6.0)     */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (-tmp6 * a(1,i-1,L) * b(1,L + 1) + tmp6 * a(1,i-1,L + 1) * b(1,L) - tmp5 * a(1,i-1,L + 1) * b(1,L + 3) + tmp5 * a(1,i-1,L + 3) * b(1,L + 1) - tmp5 * a(1,i-1,L + 2) * b(1,L + 4) + tmp5 * a(1,i-1,L + 4) * b(1,L + 2) - tmp4 * a(1,i-1,L + 3) * b(1,L + 5) + tmp4 * a(1,i-1,L + 5) * b(1,L + 3) - tmp4 * a(1,i-1,L + 4) * b(1,L + 6) + tmp4 * a(1,i-1,L + 6) * b(1,L + 4));
            }
            element(sum,2,0).r -= deri[1];
            element(sum,2,1).r -= deri[2];
            element(sum,2,2).r -= deri[3];

            /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

            for (i = 1; i <= 3; i++) {
                deri[i] = fugou * (+tmp6 * a(1,i-1,L) * b(1,L + 2) - tmp6 * a(1,i-1,L + 2) * b(1,L) + tmp5 * a(1,i-1,L + 1) * b(1,L + 4) - tmp5 * a(1,i-1,L + 4) * b(1,L + 1) - tmp5 * a(1,i-1,L + 2) * b(1,L + 3) + tmp5 * a(1,i-1,L + 3) * b(1,L + 2) + tmp4 * a(1,i-1,L + 3) * b(1,L + 6) - tmp4 * a(1,i-1,L + 6) * b(1,L + 3) - tmp4 * a(1,i-1,L + 4) * b(1,L + 5) + tmp4 * a(1,i-1,L + 5) * b(1,L + 4));
            }
            element(sum,2,0).i -= deri[1];
            element(sum,2,1).i -= deri[2];
            element(sum,2,2).i -= deri[3];
        }
    }

    /****************************************************
        off-diagonal contribution on up-up and dn-dn
    ****************************************************/

    /* p */
    if (l2 == 2) {

        tmpx = fugou * (ene_p / 3.0 * a(0,0,L) * b(0,L + 1) - ene_p / 3.0 * a(0,0,L + 1) * b(0,L));

        tmpy = fugou * (ene_p / 3.0 * a(0,1,L) * b(0,L + 1) - ene_p / 3.0 * a(0,1,L + 1) * b(0,L));

        tmpz = fugou * (ene_p / 3.0 * a(0,2,L) * b(0,L + 1) - ene_p / 3.0 * a(0,2,L + 1) * b(0,L));

        /* contribution of l+1/2 for up spin */
        element(sum,0,0).i += -tmpx;
        element(sum,0,1).i += -tmpy;
        element(sum,0,2).i += -tmpz;

        /* contribution of l+1/2 for down spin */
        element(sum,1,0).i += tmpx;
        element(sum,1,1).i += tmpy;
        element(sum,1,2).i += tmpz;

        tmpx = fugou * (ene_m / 3.0 * a(1,0,L) * b(1,L + 1) - ene_m / 3.0 * a(1,0,L + 1) * b(1,L));

        tmpy = fugou * (ene_m / 3.0 * a(1,1,L) * b(1,L + 1) - ene_m / 3.0 * a(1,1,L + 1) * b(1,L));

        tmpz = fugou * (ene_m / 3.0 * a(1,2,L) * b(1,L + 1) - ene_m / 3.0 * a(1,2,L + 1) * b(1,L));

        /* contribution of l-1/2 for up spin */
        element(sum,0,0).i += tmpx;
        element(sum,0,1).i += tmpy;
        element(sum,0,2).i += tmpz;

        /* contribution of l+1/2 for down spin */
        element(sum,1,0).i += -tmpx;
        element(sum,1,1).i += -tmpy;
        element(sum,1,2).i += -tmpz;
    }

    /* d */
    else if (l2 == 4) {

        tmpx = fugou * (ene_p * 2.0 / 5.0 * a(0,0,L + 1) * b(0,L + 2) - ene_p * 2.0 / 5.0 * a(0,0,L + 2) * b(0,L + 1) + ene_p * 1.0 / 5.0 * a(0,0,L + 3) * b(0,L + 4) - ene_p * 1.0 / 5.0 * a(0,0,L + 4) * b(0,L + 3));

        tmpy = fugou * (ene_p * 2.0 / 5.0 * a(0,1,L + 1) * b(0,L + 2) - ene_p * 2.0 / 5.0 * a(0,1,L + 2) * b(0,L + 1) + ene_p * 1.0 / 5.0 * a(0,1,L + 3) * b(0,L + 4) - ene_p * 1.0 / 5.0 * a(0,1,L + 4) * b(0,L + 3));

        tmpz = fugou * (ene_p * 2.0 / 5.0 * a(0,2,L + 1) * b(0,L + 2) - ene_p * 2.0 / 5.0 * a(0,2,L + 2) * b(0,L + 1) + ene_p * 1.0 / 5.0 * a(0,2,L + 3) * b(0,L + 4) - ene_p * 1.0 / 5.0 * a(0,2,L + 4) * b(0,L + 3));

        /* contribution of l+1/2 for up spin */
        element(sum,0,0).i += -tmpx;
        element(sum,0,1).i += -tmpy;
        element(sum,0,2).i += -tmpz;

        /* contribution of l+1/2 for down spin */
        element(sum,1,0).i += tmpx;
        element(sum,1,1).i += tmpy;
        element(sum,1,2).i += tmpz;

        tmpx = fugou * (ene_m * 2.0 / 5.0 * a(1,0,L + 1) * b(1,L + 2) - ene_m * 2.0 / 5.0 * a(1,0,L + 2) * b(1,L + 1) + ene_m * 1.0 / 5.0 * a(1,0,L + 3) * b(1,L + 4) - ene_m * 1.0 / 5.0 * a(1,0,L + 4) * b(1,L + 3));

        tmpy = fugou * (ene_m * 2.0 / 5.0 * a(1,1,L + 1) * b(1,L + 2) - ene_m * 2.0 / 5.0 * a(1,1,L + 2) * b(1,L + 1) + ene_m * 1.0 / 5.0 * a(1,1,L + 3) * b(1,L + 4) - ene_m * 1.0 / 5.0 * a(1,1,L + 4) * b(1,L + 3));

        tmpz = fugou * (ene_m * 2.0 / 5.0 * a(1,2,L + 1) * b(1,L + 2) - ene_m * 2.0 / 5.0 * a(1,2,L + 2) * b(1,L + 1) + ene_m * 1.0 / 5.0 * a(1,2,L + 3) * b(1,L + 4) - ene_m * 1.0 / 5.0 * a(1,2,L + 4) * b(1,L + 3));

        /* contribution of l-1/2 for up spin */
        element(sum,0,0).i += tmpx;
        element(sum,0,1).i += tmpy;
        element(sum,0,2).i += tmpz;

        /* contribution of l-1/2 for down spin */
        element(sum,1,0).i += -tmpx;
        element(sum,1,1).i += -tmpy;
        element(sum,1,2).i += -tmpz;

    }

    /* f */
    else if (l2 == 6) {

        tmpx = fugou * (ene_p * 1.0 / 7.0 * a(0,0,L + 1) * b(0,L + 2) - ene_p * 1.0 / 7.0 * a(0,0,L + 2) * b(0,L + 1) + ene_p * 2.0 / 7.0 * a(0,0,L + 3) * b(0,L + 4) - ene_p * 2.0 / 7.0 * a(0,0,L + 4) * b(0,L + 3) + ene_p * 3.0 / 7.0 * a(0,0,L + 5) * b(0,L + 6) - ene_p * 3.0 / 7.0 * a(0,0,L + 6) * b(0,L + 5));

        tmpy = fugou * (ene_p * 1.0 / 7.0 * a(0,1,L + 1) * b(0,L + 2) - ene_p * 1.0 / 7.0 * a(0,1,L + 2) * b(0,L + 1) + ene_p * 2.0 / 7.0 * a(0,1,L + 3) * b(0,L + 4) - ene_p * 2.0 / 7.0 * a(0,1,L + 4) * b(0,L + 3) + ene_p * 3.0 / 7.0 * a(0,1,L + 5) * b(0,L + 6) - ene_p * 3.0 / 7.0 * a(0,1,L + 6) * b(0,L + 5));

        tmpz = fugou * (ene_p * 1.0 / 7.0 * a(0,2,L + 1) * b(0,L + 2) - ene_p * 1.0 / 7.0 * a(0,2,L + 2) * b(0,L + 1) + ene_p * 2.0 / 7.0 * a(0,2,L + 3) * b(0,L + 4) - ene_p * 2.0 / 7.0 * a(0,2,L + 4) * b(0,L + 3) + ene_p * 3.0 / 7.0 * a(0,2,L + 5) * b(0,L + 6) - ene_p * 3.0 / 7.0 * a(0,2,L + 6) * b(0,L + 5));

        /* contribution of l+1/2 for up spin */
        element(sum,0,0).i += -tmpx;
        element(sum,0,1).i += -tmpy;
        element(sum,0,2).i += -tmpz;

        /* contribution of l+1/2 for down spin */
        element(sum,1,0).i += tmpx;
        element(sum,1,1).i += tmpy;
        element(sum,1,2).i += tmpz;

        tmpx = fugou * (ene_m * 1.0 / 7.0 * a(1,0,L + 1) * b(1,L + 2) - ene_m * 1.0 / 7.0 * a(1,0,L + 2) * b(1,L + 1) + ene_m * 2.0 / 7.0 * a(1,0,L + 3) * b(1,L + 4) - ene_m * 2.0 / 7.0 * a(1,0,L + 4) * b(1,L + 3) + ene_m * 3.0 / 7.0 * a(1,0,L + 5) * b(1,L + 6) - ene_m * 3.0 / 7.0 * a(1,0,L + 6) * b(1,L + 5));

        tmpy = fugou * (ene_m * 1.0 / 7.0 * a(1,1,L + 1) * b(1,L + 2) - ene_m * 1.0 / 7.0 * a(1,1,L + 2) * b(1,L + 1) + ene_m * 2.0 / 7.0 * a(1,1,L + 3) * b(1,L + 4) - ene_m * 2.0 / 7.0 * a(1,1,L + 4) * b(1,L + 3) + ene_m * 3.0 / 7.0 * a(1,1,L + 5) * b(1,L + 6) - ene_m * 3.0 / 7.0 * a(1,1,L + 6) * b(1,L + 5));

        tmpz = fugou * (ene_m * 1.0 / 7.0 * a(1,2,L + 1) * b(1,L + 2) - ene_m * 1.0 / 7.0 * a(1,2,L + 2) * b(1,L + 1) + ene_m * 2.0 / 7.0 * a(1,2,L + 3) * b(1,L + 4) - ene_m * 2.0 / 7.0 * a(1,2,L + 4) * b(1,L + 3) + ene_m * 3.0 / 7.0 * a(1,2,L + 5) * b(1,L + 6) - ene_m * 3.0 / 7.0 * a(1,2,L + 6) * b(1,L + 5));

        /* contribution of l-1/2 for up spin */
        element(sum,0,0).i += tmpx;
        element(sum,0,1).i += tmpy;
        element(sum,0,2).i += tmpz;

        /* contribution of l-1/2 for down spin */
        element(sum,1,0).i += -tmpx;
        element(sum,1,1).i += -tmpy;
        element(sum,1,2).i += -tmpz;
    }

    /****************************************************
        diagonal contribution on up-up and dn-dn
    ****************************************************/

    for (l3 = 0; l3 <= l2; l3++) {

        /* VNL for j=l+1/2 */

        tmpx = PFp * ene_p * a(0,0,L) * b(0,L);
        tmpy = PFp * ene_p * a(0,1,L) * b(0,L);
        tmpz = PFp * ene_p * a(0,2,L) * b(0,L);

        element(sum,0,0).r += tmpx;
        element(sum,0,1).r += tmpy;
        element(sum,0,2).r += tmpz;

        element(sum,1,0).r += tmpx;
        element(sum,1,1).r += tmpy;
        element(sum,1,2).r += tmpz;

        /* VNL for j=l-1/2 */

        tmpx = PFm * ene_m * a(1,0,L) * b(1,L);
        tmpy = PFm * ene_m * a(1,1,L) * b(1,L);
        tmpz = PFm * ene_m * a(1,2,L) * b(1,L);

        element(sum,0,0).r += tmpx;
        element(sum,0,1).r += tmpy;
        element(sum,0,2).r += tmpz;

        element(sum,1,0).r += tmpx;
        element(sum,1,1).r += tmpy;
        element(sum,1,2).r += tmpz;

        ++L;
    }

}
template<bool Noncollinear>
void projector_spinor_channel(const OpenMXForceProjectorChannel& channel,
    const OpenMXProjectorForceRows& in,Tensor<dcomplex,2>::view_type sum)
{
  switch(channel.twice_angular) {
    case 0:projector_spinor_channel<Noncollinear,0>(channel,in,sum);break;
    case 2:projector_spinor_channel<Noncollinear,1>(channel,in,sum);break;
    case 4:projector_spinor_channel<Noncollinear,2>(channel,in,sum);break;
    case 6:projector_spinor_channel<Noncollinear,3>(channel,in,sum);break;
  }
}
template<bool ValueFirst,bool Subtract>
void projector_scalar_sum(const OpenMXForceProjectorSpecies& species,const OpenMXProjectorForceTerm& term,
    std::size_t ai,std::size_t bi,std::array<double,3>& sum)
{
  if(!species.count)return;
  using Row=Tensor<double,1>::const_view_type;
  const auto as=term.derivative_rows*term.derivative_stride,bs=term.value_rows*term.value_stride;
  const Row ax(term.derivative[0][0],as),ay(term.derivative[0][1],as),az(term.derivative[0][2],as),v(term.value[0],bs);
  for(std::size_t j=0;j<species.count;++j) {
    const auto& channel=species.channels[j];
    const double energy=term.reverse_energy?-channel.energy[0]:channel.energy[0];
    for(int magnetic=0;magnetic<=channel.twice_angular;++magnetic) {
      const auto p=channel.offset+magnetic;double x,y,z;
      if constexpr(ValueFirst) {
        x=energy*element(v,bi+p)*element(ax,ai+p);
        y=energy*element(v,bi+p)*element(ay,ai+p);
        z=energy*element(v,bi+p)*element(az,ai+p);
      }else {
        x=energy*element(ax,ai+p)*element(v,bi+p);
        y=energy*element(ay,ai+p)*element(v,bi+p);
        z=energy*element(az,ai+p)*element(v,bi+p);
      }
      if constexpr(Subtract){sum[0]-=x;sum[1]-=y;sum[2]-=z;}
      else{sum[0]+=x;sum[1]+=y;sum[2]+=z;}
    }
  }
}
void projector_scalar_block(const OpenMXForceProjectorSpecies& species,const OpenMXProjectorForceBlock& block)
{
  for(std::size_t m=0;m<block.rows;++m)for(std::size_t n=0;n<block.columns;++n) {
    std::array<double,3> sum{};
    for(std::size_t t=0;t<block.term_count;++t) {
      const auto& term=block.terms[t];
      const auto ai=(term.transpose?n:m)*term.derivative_stride,bi=(term.transpose?m:n)*term.value_stride;
      if(term.value_first) {
        if(term.sign==1)projector_scalar_sum<true,false>(species,term,ai,bi,sum);
        else projector_scalar_sum<true,true>(species,term,ai,bi,sum);
      }else {
        if(term.sign==1)projector_scalar_sum<false,false>(species,term,ai,bi,sum);
        else projector_scalar_sum<false,true>(species,term,ai,bi,sum);
      }
    }
    for(int axis=0;axis<3;++axis) {
      const Tensor<dcomplex,3>::view_type out(block.output[axis],3,block.output_rows,block.output_stride);
      if(block.accumulate)for(int spin=0;spin<2;++spin)element(out,spin,m,n).r+=sum[axis];
      else {for(int spin=0;spin<3;++spin)element(out,spin,m,n)={0,0};
        element(out,0,m,n).r=sum[axis];element(out,1,m,n).r=sum[axis];}
    }
  }
}
template<bool Noncollinear>
void projector_block(const OpenMXForceProjectorSpecies& species,const OpenMXProjectorForceBlock& block)
{
  for(std::size_t m=0;m<block.rows;++m)for(std::size_t n=0;n<block.columns;++n) {
    std::array<dcomplex,9> values{};
    const Tensor<dcomplex,2>::view_type sum(values.data(),3,3);
    for(std::size_t t=0;t<block.term_count;++t) {
      const auto& term=block.terms[t];OpenMXProjectorForceRows rows{};
      rows.columns=species.components;rows.sign=term.sign;rows.reverse_energy=term.reverse_energy;
      const auto a=(term.transpose?n:m)*term.derivative_stride,b=(term.transpose?m:n)*term.value_stride;
      for(int c=0;c<2;++c) {
        rows.value[c]=species.components?term.value[c]+b:nullptr;
        for(int axis=0;axis<3;++axis)rows.derivative[c][axis]=species.components?term.derivative[c][axis]+a:nullptr;
      }
      for(std::size_t j=0;j<species.count;++j)projector_spinor_channel<Noncollinear>(species.channels[j],rows,sum);
    }
    for(int axis=0;axis<3;++axis) {
      const Tensor<dcomplex,3>::view_type out(block.output[axis],3,block.output_rows,block.output_stride);
      for(int c=0;c<3;++c) {
        const auto value=element(sum,c,axis);auto& target=element(out,c,m,n);
        if(block.accumulate){target.r+=value.r;target.i+=value.i;}else target=value;
      }
    }
  }
}
} // namespace


void projector_force_trace(const OpenMXProjectorForceView& view,std::size_t species,
    const OpenMXProjectorForceRows& in,std::span<dcomplex,9> output)
{
  require((view.spin==0 || view.spin==1 || view.spin==3) && (view.spin_orbit==0 || view.spin_orbit==1) &&
      species<view.species && view.species_views &&
      std::isfinite(in.sign) && (in.reverse_energy==0 || in.reverse_energy==1),"invalid nonlocal Force trace mode");
  const auto& item=view.species_views[species];
  require(item.relativistic==1 && item.count<=std::size_t(INT_MAX) && item.components<=in.columns &&
      in.columns<=std::size_t(PTRDIFF_MAX)/sizeof(double) && (!item.count || item.channels),
      "invalid relativistic projector trace layout");
  if(item.components)for(int c=0;c<2;++c) {
    require(in.value[c],"missing projector value row");
    for(auto row:in.derivative[c])require(row,"missing projector derivative row");
  }
  std::array<dcomplex,9> values;std::copy(output.begin(),output.end(),values.begin());
  const Tensor<dcomplex,2>::view_type sum(values.data(),3,3);
  std::size_t offset=0;
  for(std::size_t j=0;j<item.count;++j) {
    const auto& channel=item.channels[j];
    require(channel.twice_angular>=0 && channel.twice_angular<=6 && !(channel.twice_angular%2) &&
        channel.offset==offset && offset<=item.components &&
        std::size_t(channel.twice_angular+1)<=item.components-offset,"invalid projector channel extent");
    if(view.spin==3)projector_spinor_channel<true>(channel,in,sum);
    else projector_spinor_channel<false>(channel,in,sum);
    offset+=channel.twice_angular+1;
  }
  require(offset==item.components,"nonlocal projector trace component count differs");
  std::copy(values.begin(),values.end(),output.begin());
}
void projector_force_block(const OpenMXProjectorForceView& view,std::size_t species,
    const OpenMXProjectorForceBlock& in)
{
  require((view.spin==0 || view.spin==1 || view.spin==3) && (view.spin_orbit==0 || view.spin_orbit==1) &&
      species<view.species && view.species_views && in.term_count>=1 && in.term_count<=2 && in.terms &&
      in.rows<=in.output_rows && in.columns<=in.output_stride && (in.accumulate==0 || in.accumulate==1),
      "invalid nonlocal Force block");
  const auto& item=view.species_views[species];
  require((item.relativistic==0 || item.relativistic==1) && item.count<=std::size_t(INT_MAX) && item.components<=std::size_t(INT_MAX) &&
      (!item.count || item.channels),"invalid projector block layout");
  const std::array<std::size_t,3> shape{3,in.output_rows,in.output_stride};extent_product(shape,sizeof(dcomplex));
  std::size_t offset=0;
  for(std::size_t j=0;j<item.count;++j) {
    const auto& channel=item.channels[j];
    require(channel.twice_angular>=0 && channel.twice_angular<=(item.relativistic?6:28) && !(channel.twice_angular%2) &&
        channel.offset==offset && offset<=item.components && std::size_t(channel.twice_angular+1)<=item.components-offset,
        "invalid projector block channel extent");offset+=channel.twice_angular+1;
  }
  require(offset==item.components,"nonlocal projector block component count differs");
  if(!in.rows || !in.columns)return;
  for(auto out:in.output)require(out,"missing projector block output");
  for(std::size_t t=0;t<in.term_count;++t) {
    const auto& term=in.terms[t];
    require(std::isfinite(term.sign) && (term.reverse_energy==0 || term.reverse_energy==1) &&
        (term.transpose==0 || term.transpose==1) && (term.value_first==0 || term.value_first==1) && term.derivative_stride>=item.components && term.value_stride>=item.components &&
        term.derivative_rows>=(term.transpose?in.columns:in.rows) && term.value_rows>=(term.transpose?in.rows:in.columns) &&
        (item.relativistic || term.sign==1 || term.sign==-1),
        "invalid projector block term");
    const std::array<std::size_t,2> a{term.derivative_rows,term.derivative_stride};
    const std::array<std::size_t,2> b{term.value_rows,term.value_stride};
    extent_product(a,sizeof(double));extent_product(b,sizeof(double));
    if(item.components)for(int c=0;c<=item.relativistic;++c) {
      require(term.value[c],"missing projector block value matrix");
      for(auto row:term.derivative[c])require(row,"missing projector block derivative matrix");
    }
  }
  if(!item.relativistic)projector_scalar_block(item,in);
  else if(view.spin==3)projector_block<true>(item,in);else projector_block<false>(item,in);
}
} // namespace openmx::core

extern "C" int openmx_core_projector_force_trace(const OpenMXProjectorForceView *view,std::size_t species,
    const OpenMXProjectorForceRows *input,dcomplex *sum,char *error,std::size_t capacity)
{
  if(error && capacity)*error='\0';
  try {
    if(!view || !input || !sum)throw std::invalid_argument("missing nonlocal Force trace input/output");
    openmx::core::projector_force_trace(*view,species,*input,std::span<dcomplex,9>(sum,9));return 1;
  }catch(const std::exception& e){if(error && capacity)std::snprintf(error,capacity,"%s",e.what());}
  catch(...){if(error && capacity)std::snprintf(error,capacity,"unexpected nonlocal Force trace failure");}
  return 0;
}

extern "C" int openmx_core_projector_force_block(const OpenMXProjectorForceView *view,std::size_t species,
    const OpenMXProjectorForceBlock *input,char *error,std::size_t capacity)
{
  if(error && capacity)*error='\0';
  try {
    if(!view || !input)throw std::invalid_argument("missing nonlocal Force block input");
    openmx::core::projector_force_block(*view,species,*input);return 1;
  }catch(const std::exception& e){if(error && capacity)std::snprintf(error,capacity,"%s",e.what());}
  catch(...){if(error && capacity)std::snprintf(error,capacity,"unexpected nonlocal Force block failure");}
  return 0;
}
