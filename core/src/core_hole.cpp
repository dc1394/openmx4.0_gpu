#include <openmx/core/core_hole.hpp>
#include <climits>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace openmx::core {
namespace {
void require(bool ok,const char *message){if(!ok)throw std::invalid_argument(message);}
}
CoreHole::Shape CoreHole::validate(const OpenMXCoreHoleInput& in)
{
  require(in.atom>0 && (in.spin==0 || in.spin==1 || in.spin==3) &&
      (in.spin_orbit==0 || in.spin_orbit==1) && in.angular>=0 && in.angular<=3 &&
      in.state>=0 && in.state<=2*(2*in.angular+1) && in.maximum_angular>=0 &&
      in.maximum_angular<=14 && in.orbitals>=0 && in.multiplicity && std::isfinite(in.penalty),
      "invalid core-hole configuration");
  std::size_t count=0,offset=0;bool active=false;
  for(int l=0;l<=in.maximum_angular;++l) {
    require(in.multiplicity[l]>=0,"negative core-hole basis multiplicity");
    if(l==in.angular){offset=count;active=in.multiplicity[l]>0;}
    const auto n=std::size_t(2*l+1)*std::size_t(in.multiplicity[l]);
    require(n<=std::size_t(INT_MAX)-count,"core-hole basis size overflows");count+=n;
  }
  require(count==std::size_t(in.orbitals),"core-hole basis shape differs from orbital count");
  return {in.atom,in.spin,in.spin_orbit,in.angular,in.state,count,offset,active,in.penalty};
}
CoreHole::CoreHole(const OpenMXCoreHoleInput& in):CoreHole(validate(in)){}
CoreHole::CoreHole(Shape shape):shape_(shape),collinear_({2,shape.orbitals})
{
  if(!shape.active)return;
  const int width=2*shape.angular+1;
  auto values=collinear_.view();
  if(shape.state==0)for(int s=0;s<2;++s)for(int m=0;m<width;++m)
    element(values,s,shape.offset+m)=shape.penalty;
  else element(values,shape.state<=width?0:1,shape.offset+(shape.state-1)%width)=shape.penalty;
  const int upper_begin[]={OPENMX_CH_D12_P12,OPENMX_CH_D32_P32,OPENMX_CH_D52_P52,OPENMX_CH_D72_P72};
  const int lower_begin[]={0,OPENMX_CH_D12_P12,OPENMX_CH_D32_P32,OPENMX_CH_D52_P52};
  for(int reverse=0;reverse<2;++reverse)for(int state=1;state<=2*width;++state) {
    const int index=state<=width+1?upper_begin[shape.angular]+state-1:
        lower_begin[shape.angular]+state-width-2;
    const double penalty=reverse?-shape.penalty:shape.penalty;
    weights_[reverse*OPENMX_CH_WEIGHTS+index]=shape.angular==0?penalty*(shape.state==0 || shape.state==state):
        penalty/double(width)*(shape.state==0 || shape.state==state);
  }
}
OpenMXCoreHoleView CoreHole::view() const noexcept
{
  const auto v=collinear_.view();const auto p=v.data_handle();
  return {shape_.atom,shape_.spin,shape_.spin_orbit,shape_.angular,shape_.state,shape_.active,
      shape_.orbitals,shape_.offset,shape_.penalty,{p,shape_.orbitals?p+shape_.orbitals:p},weights_.data(),weights_.data()+OPENMX_CH_WEIGHTS};
}

void core_hole_trace(const OpenMXCoreHoleView& hole,const OpenMXCoreHoleTrace& in,std::span<dcomplex,9> output)
{
  require((hole.spin==0 || hole.spin==1 || hole.spin==3) && hole.angular>=0 && hole.angular<=3 &&
      (hole.active==0 || hole.active==1) && hole.orbitals<=std::size_t(INT_MAX) &&
      hole.offset<=hole.orbitals && std::isfinite(in.sign) && (in.reverse_penalty==0 || in.reverse_penalty==1),"invalid core-hole trace mode");
  if(!hole.active)return;
  const std::size_t width=2*hole.angular+1;
  const double *weights=in.reverse_penalty?hole.opposite_weights:hole.weights;
  require(width<=hole.orbitals-hole.offset && hole.offset<=in.columns && width<=in.columns-hole.offset &&
      in.columns<=std::size_t(PTRDIFF_MAX)/sizeof(double) && weights && in.value,
      "invalid core-hole overlap row extent");
  for(auto row:in.derivative)require(row,"missing core-hole derivative row");
  using Row=Tensor<double,1>::const_view_type;
  const auto a=[&](int axis,std::size_t i){return element(Row(in.derivative[axis],in.columns),i);};
  const auto b=[&](std::size_t i){return element(Row(in.value,in.columns),i);};
  const Tensor<dcomplex,2>::view_type sum(output.data(),3,3);
  const std::size_t L=hole.offset;
  const int L2=2*hole.angular;
  const double fugou=in.sign;
  using std::sqrt;
  const double d12_p12=weights[OPENMX_CH_D12_P12];
  const double d12_m12=weights[OPENMX_CH_D12_M12];
  const double d32_p32=weights[OPENMX_CH_D32_P32];
  const double d32_p12=weights[OPENMX_CH_D32_P12];
  const double d32_m12=weights[OPENMX_CH_D32_M12];
  const double d32_m32=weights[OPENMX_CH_D32_M32];
  const double d52_p52=weights[OPENMX_CH_D52_P52];
  const double d52_p32=weights[OPENMX_CH_D52_P32];
  const double d52_p12=weights[OPENMX_CH_D52_P12];
  const double d52_m12=weights[OPENMX_CH_D52_M12];
  const double d52_m32=weights[OPENMX_CH_D52_M32];
  const double d52_m52=weights[OPENMX_CH_D52_M52];
  const double d72_p72=weights[OPENMX_CH_D72_P72];
  const double d72_p52=weights[OPENMX_CH_D72_P52];
  const double d72_p32=weights[OPENMX_CH_D72_P32];
  const double d72_p12=weights[OPENMX_CH_D72_P12];
  const double d72_m12=weights[OPENMX_CH_D72_M12];
  const double d72_m32=weights[OPENMX_CH_D72_M32];
  const double d72_m52=weights[OPENMX_CH_D72_M52];
  const double d72_m72=weights[OPENMX_CH_D72_M72];
  if (hole.spin == 3) {

      /***************
             p
      ***************/

      if (L2 == 2) {

          /* real contribution of l+1/2 to off-diagonal up-down matrix */
          element(sum,2,0).r += fugou * (d32_m12 * a(0,L) * b(L + 2) - d32_p12 * a(0,L + 2) * b(L));

          element(sum,2,1).r += fugou * (d32_m12 * a(1,L) * b(L + 2) - d32_p12 * a(1,L + 2) * b(L));

          element(sum,2,2).r += fugou * (d32_m12 * a(2,L) * b(L + 2) - d32_p12 * a(2,L + 2) * b(L));

          /* imaginary contribution of l+1/2 to off-diagonal up-down matrix */
          element(sum,2,0).i += fugou * (-d32_m12 * a(0,L + 1) * b(L + 2) + d32_p12 * a(0,L + 2) * b(L + 1));

          element(sum,2,0).i += fugou * (-d32_m12 * a(1,L + 1) * b(L + 2) + d32_p12 * a(1,L + 2) * b(L + 1));

          element(sum,2,0).i += fugou * (-d32_m12 * a(2,L + 1) * b(L + 2) + d32_p12 * a(2,L + 2) * b(L + 1));

          /* real contribution of l-1/2 for to off-diagonal up-down matrix */
          element(sum,2,0).r -= fugou * (d12_m12 * a(0,L) * b(L + 2) - d12_p12 * a(0,L + 2) * b(L));

          element(sum,2,1).r -= fugou * (d12_m12 * a(1,L) * b(L + 2) - d12_p12 * a(1,L + 2) * b(L));

          element(sum,2,2).r -= fugou * (d12_m12 * a(2,L) * b(L + 2) - d12_p12 * a(2,L + 2) * b(L));

          /* imaginary contribution of l-1/2 to off-diagonal up-down matrix */
          element(sum,2,0).i -= fugou * (-d12_m12 * a(0,L + 1) * b(L + 2) + d12_p12 * a(0,L + 2) * b(L + 1));

          element(sum,2,1).i -= fugou * (-d12_m12 * a(1,L + 1) * b(L + 2) + d12_p12 * a(1,L + 2) * b(L + 1));

          element(sum,2,2).i -= fugou * (-d12_m12 * a(2,L + 1) * b(L + 2) + d12_p12 * a(2,L + 2) * b(L + 1));
      }

      /***************
             d
      ***************/

      else if (L2 == 4) {

          /* real contribution of l+1/2 to off diagonal up-down matrix */

          element(sum,2,0).r += fugou * (-sqrt(3.0) * d52_p12 * a(0,L) * b(L + 3) + sqrt(3.0) * d52_m12 * a(0,L + 3) * b(L) + d52_m32 * a(0,L + 1) * b(L + 3) - d52_p32 * a(0,L + 3) * b(L + 1) + d52_m32 * a(0,L + 2) * b(L + 4) - d52_p32 * a(0,L + 4) * b(L + 2));

          element(sum,2,1).r += fugou * (-sqrt(3.0) * d52_p12 * a(1,L) * b(L + 3) + sqrt(3.0) * d52_m12 * a(1,L + 3) * b(L) + d52_m32 * a(1,L + 1) * b(L + 3) - d52_p32 * a(1,L + 3) * b(L + 1) + d52_m32 * a(1,L + 2) * b(L + 4) - d52_p32 * a(1,L + 4) * b(L + 2));

          element(sum,2,2).r += fugou * (-sqrt(3.0) * d52_p12 * a(2,L) * b(L + 3) + sqrt(3.0) * d52_m12 * a(2,L + 3) * b(L) + d52_m32 * a(2,L + 1) * b(L + 3) - d52_p32 * a(2,L + 3) * b(L + 1) + d52_m32 * a(2,L + 2) * b(L + 4) - d52_p32 * a(2,L + 4) * b(L + 2));

          /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

          element(sum,2,0).i += fugou * (sqrt(3.0) * d52_p12 * a(0,L) * b(L + 4) - sqrt(3.0) * d52_m12 * a(0,L + 4) * b(L) + d52_m32 * a(0,L + 1) * b(L + 4) - d52_p32 * a(0,L + 4) * b(L + 1) - d52_m32 * a(0,L + 2) * b(L + 3) + d52_p32 * a(0,L + 3) * b(L + 2));

          element(sum,2,1).i += fugou * (sqrt(3.0) * d52_p12 * a(1,L) * b(L + 4) - sqrt(3.0) * d52_m12 * a(1,L + 4) * b(L) + d52_m32 * a(1,L + 1) * b(L + 4) - d52_p32 * a(1,L + 4) * b(L + 1) - d52_m32 * a(1,L + 2) * b(L + 3) + d52_p32 * a(1,L + 3) * b(L + 2));

          element(sum,2,2).i += fugou * (sqrt(3.0) * d52_p12 * a(2,L) * b(L + 4) - sqrt(3.0) * d52_m12 * a(2,L + 4) * b(L) + d52_m32 * a(2,L + 1) * b(L + 4) - d52_p32 * a(2,L + 4) * b(L + 1) - d52_m32 * a(2,L + 2) * b(L + 3) + d52_p32 * a(2,L + 3) * b(L + 2));

          /* real contribution of l-1/2 for to diagonal up-down matrix */

          element(sum,2,0).r -= fugou * (-sqrt(3.0) * d32_p12 * a(0,L) * b(L + 3) + sqrt(3.0) * d32_m12 * a(0,L + 3) * b(L) + d32_m32 * a(0,L + 1) * b(L + 3) - d32_p32 * a(0,L + 3) * b(L + 1) + d32_m32 * a(0,L + 2) * b(L + 4) - d32_p32 * a(0,L + 4) * b(L + 2));

          element(sum,2,1).r -= fugou * (-sqrt(3.0) * d32_p12 * a(1,L) * b(L + 3) + sqrt(3.0) * d32_m12 * a(1,L + 3) * b(L) + d32_m32 * a(1,L + 1) * b(L + 3) - d32_p32 * a(1,L + 3) * b(L + 1) + d32_m32 * a(1,L + 2) * b(L + 4) - d32_p32 * a(1,L + 4) * b(L + 2));

          element(sum,2,2).r -= fugou * (-sqrt(3.0) * d32_p12 * a(2,L) * b(L + 3) + sqrt(3.0) * d32_m12 * a(2,L + 3) * b(L) + d32_m32 * a(2,L + 1) * b(L + 3) - d32_p32 * a(2,L + 3) * b(L + 1) + d32_m32 * a(2,L + 2) * b(L + 4) - d32_p32 * a(2,L + 4) * b(L + 2));

          /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

          element(sum,2,0).i -= fugou * (sqrt(3.0) * d32_p12 * a(0,L) * b(L + 4) - sqrt(3.0) * d32_m12 * a(0,L + 4) * b(L) + d32_m32 * a(0,L + 1) * b(L + 4) - d32_p32 * a(0,L + 4) * b(L + 1) - d32_m32 * a(0,L + 2) * b(L + 3) + d32_p32 * a(0,L + 3) * b(L + 2));

          element(sum,2,1).i -= fugou * (sqrt(3.0) * d32_p12 * a(1,L) * b(L + 4) - sqrt(3.0) * d32_m12 * a(1,L + 4) * b(L) + d32_m32 * a(1,L + 1) * b(L + 4) - d32_p32 * a(1,L + 4) * b(L + 1) - d32_m32 * a(1,L + 2) * b(L + 3) + d32_p32 * a(1,L + 3) * b(L + 2));

          element(sum,2,2).i -= fugou * (sqrt(3.0) * d32_p12 * a(2,L) * b(L + 4) - sqrt(3.0) * d32_m12 * a(2,L + 4) * b(L) + d32_m32 * a(2,L + 1) * b(L + 4) - d32_p32 * a(2,L + 4) * b(L + 1) - d32_m32 * a(2,L + 2) * b(L + 3) + d32_p32 * a(2,L + 3) * b(L + 2));
      }

      /***************
                     f
      ***************/

      else if (L2 == 6) {

          /* real contribution of l+1/2 to off diagonal up-down matrix */

          element(sum,2,0).r += fugou * (-sqrt(6.0) * d72_p12 * a(0,L) * b(L + 1) + sqrt(6.0) * d72_m12 * a(0,L + 1) * b(L) - sqrt(2.5) * d72_p32 * a(0,L + 1) * b(L + 3) + sqrt(2.5) * d72_m32 * a(0,L + 3) * b(L + 1) - sqrt(2.5) * d72_p32 * a(0,L + 2) * b(L + 4) + sqrt(2.5) * d72_m32 * a(0,L + 4) * b(L + 2) - sqrt(1.5) * d72_p52 * a(0,L + 3) * b(L + 5) + sqrt(1.5) * d72_m52 * a(0,L + 5) * b(L + 3) - sqrt(1.5) * d72_p52 * a(0,L + 4) * b(L + 6) + sqrt(1.5) * d72_m52 * a(0,L + 6) * b(L + 4));

          element(sum,2,1).r += fugou * (-sqrt(6.0) * d72_p12 * a(1,L) * b(L + 1) + sqrt(6.0) * d72_m12 * a(1,L + 1) * b(L) - sqrt(2.5) * d72_p32 * a(1,L + 1) * b(L + 3) + sqrt(2.5) * d72_m32 * a(1,L + 3) * b(L + 1) - sqrt(2.5) * d72_p32 * a(1,L + 2) * b(L + 4) + sqrt(2.5) * d72_m32 * a(1,L + 4) * b(L + 2) - sqrt(1.5) * d72_p52 * a(1,L + 3) * b(L + 5) + sqrt(1.5) * d72_m52 * a(1,L + 5) * b(L + 3) - sqrt(1.5) * d72_p52 * a(1,L + 4) * b(L + 6) + sqrt(1.5) * d72_m52 * a(1,L + 6) * b(L + 4));

          element(sum,2,2).r += fugou * (-sqrt(6.0) * d72_p12 * a(2,L) * b(L + 1) + sqrt(6.0) * d72_m12 * a(2,L + 1) * b(L) - sqrt(2.5) * d72_p32 * a(2,L + 1) * b(L + 3) + sqrt(2.5) * d72_m32 * a(2,L + 3) * b(L + 1) - sqrt(2.5) * d72_p32 * a(2,L + 2) * b(L + 4) + sqrt(2.5) * d72_m32 * a(2,L + 4) * b(L + 2) - sqrt(1.5) * d72_p52 * a(2,L + 3) * b(L + 5) + sqrt(1.5) * d72_m52 * a(2,L + 5) * b(L + 3) - sqrt(1.5) * d72_p52 * a(2,L + 4) * b(L + 6) + sqrt(1.5) * d72_m52 * a(2,L + 6) * b(L + 4));

          /* imaginary contribution of l+1/2 to off diagonal up-down matrix */

          element(sum,2,0).i += fugou * (sqrt(6.0) * d72_p12 * a(0,L) * b(L + 2) - sqrt(6.0) * d72_m12 * a(0,L + 2) * b(L) + sqrt(2.5) * d72_p32 * a(0,L + 1) * b(L + 4) - sqrt(2.5) * d72_m32 * a(0,L + 4) * b(L + 1) - sqrt(2.5) * d72_p32 * a(0,L + 2) * b(L + 3) + sqrt(2.5) * d72_m32 * a(0,L + 3) * b(L + 2) + sqrt(1.5) * d72_p52 * a(0,L + 3) * b(L + 6) - sqrt(1.5) * d72_m52 * a(0,L + 6) * b(L + 3) - sqrt(1.5) * d72_p52 * a(0,L + 4) * b(L + 5) + sqrt(1.5) * d72_m52 * a(0,L + 5) * b(L + 4));

          element(sum,2,1).i += fugou * (sqrt(6.0) * d72_p12 * a(1,L) * b(L + 2) - sqrt(6.0) * d72_m12 * a(1,L + 2) * b(L) + sqrt(2.5) * d72_p32 * a(1,L + 1) * b(L + 4) - sqrt(2.5) * d72_m32 * a(1,L + 4) * b(L + 1) - sqrt(2.5) * d72_p32 * a(1,L + 2) * b(L + 3) + sqrt(2.5) * d72_m32 * a(1,L + 3) * b(L + 2) + sqrt(1.5) * d72_p52 * a(1,L + 3) * b(L + 6) - sqrt(1.5) * d72_m52 * a(1,L + 6) * b(L + 3) - sqrt(1.5) * d72_p52 * a(1,L + 4) * b(L + 5) + sqrt(1.5) * d72_m52 * a(1,L + 5) * b(L + 4));

          element(sum,2,2).i += fugou * (sqrt(6.0) * d72_p12 * a(2,L) * b(L + 2) - sqrt(6.0) * d72_m12 * a(2,L + 2) * b(L) + sqrt(2.5) * d72_p32 * a(2,L + 1) * b(L + 4) - sqrt(2.5) * d72_m32 * a(2,L + 4) * b(L + 1) - sqrt(2.5) * d72_p32 * a(2,L + 2) * b(L + 3) + sqrt(2.5) * d72_m32 * a(2,L + 3) * b(L + 2) + sqrt(1.5) * d72_p52 * a(2,L + 3) * b(L + 6) - sqrt(1.5) * d72_m52 * a(2,L + 6) * b(L + 3) - sqrt(1.5) * d72_p52 * a(2,L + 4) * b(L + 5) + sqrt(1.5) * d72_m52 * a(2,L + 5) * b(L + 4));

          /* real contribution of l-1/2 for to off-diagonal up-down matrix */

          element(sum,2,0).r -= fugou * (-sqrt(6.0) * d52_p12 * a(0,L) * b(L + 1) + sqrt(6.0) * d52_m12 * a(0,L + 1) * b(L) - sqrt(2.5) * d52_p32 * a(0,L + 1) * b(L + 3) + sqrt(2.5) * d52_m32 * a(0,L + 3) * b(L + 1) - sqrt(2.5) * d52_p32 * a(0,L + 2) * b(L + 4) + sqrt(2.5) * d52_m32 * a(0,L + 4) * b(L + 2) - sqrt(1.5) * d52_p52 * a(0,L + 3) * b(L + 5) + sqrt(1.5) * d52_m52 * a(0,L + 5) * b(L + 3) - sqrt(1.5) * d52_p52 * a(0,L + 4) * b(L + 6) + sqrt(1.5) * d52_m52 * a(0,L + 6) * b(L + 4));

          element(sum,2,1).r -= fugou * (-sqrt(6.0) * d52_p12 * a(1,L) * b(L + 1) + sqrt(6.0) * d52_m12 * a(1,L + 1) * b(L) - sqrt(2.5) * d52_p32 * a(1,L + 1) * b(L + 3) + sqrt(2.5) * d52_m32 * a(1,L + 3) * b(L + 1) - sqrt(2.5) * d52_p32 * a(1,L + 2) * b(L + 4) + sqrt(2.5) * d52_m32 * a(1,L + 4) * b(L + 2) - sqrt(1.5) * d52_p52 * a(1,L + 3) * b(L + 5) + sqrt(1.5) * d52_m52 * a(1,L + 5) * b(L + 3) - sqrt(1.5) * d52_p52 * a(1,L + 4) * b(L + 6) + sqrt(1.5) * d52_m52 * a(1,L + 6) * b(L + 4));

          element(sum,2,2).r -= fugou * (-sqrt(6.0) * d52_p12 * a(2,L) * b(L + 1) + sqrt(6.0) * d52_m12 * a(2,L + 1) * b(L) - sqrt(2.5) * d52_p32 * a(2,L + 1) * b(L + 3) + sqrt(2.5) * d52_m32 * a(2,L + 3) * b(L + 1) - sqrt(2.5) * d52_p32 * a(2,L + 2) * b(L + 4) + sqrt(2.5) * d52_m32 * a(2,L + 4) * b(L + 2) - sqrt(1.5) * d52_p52 * a(2,L + 3) * b(L + 5) + sqrt(1.5) * d52_m52 * a(2,L + 5) * b(L + 3) - sqrt(1.5) * d52_p52 * a(2,L + 4) * b(L + 6) + sqrt(1.5) * d52_m52 * a(2,L + 6) * b(L + 4));

          /* imaginary contribution of l-1/2 to off diagonal up-down matrix */

          element(sum,2,0).i -= fugou * (sqrt(6.0) * d52_p12 * a(0,L) * b(L + 2) - sqrt(6.0) * d52_m12 * a(0,L + 2) * b(L) + sqrt(2.5) * d52_p32 * a(0,L + 1) * b(L + 4) - sqrt(2.5) * d52_m32 * a(0,L + 4) * b(L + 1) - sqrt(2.5) * d52_p32 * a(0,L + 2) * b(L + 3) + sqrt(2.5) * d52_m32 * a(0,L + 3) * b(L + 2) + sqrt(1.5) * d52_p52 * a(0,L + 3) * b(L + 6) - sqrt(1.5) * d52_m52 * a(0,L + 6) * b(L + 3) - sqrt(1.5) * d52_p52 * a(0,L + 4) * b(L + 5) + sqrt(1.5) * d52_m52 * a(0,L + 5) * b(L + 4));

          element(sum,2,1).i -= fugou * (sqrt(6.0) * d52_p12 * a(1,L) * b(L + 2) - sqrt(6.0) * d52_m12 * a(1,L + 2) * b(L) + sqrt(2.5) * d52_p32 * a(1,L + 1) * b(L + 4) - sqrt(2.5) * d52_m32 * a(1,L + 4) * b(L + 1) - sqrt(2.5) * d52_p32 * a(1,L + 2) * b(L + 3) + sqrt(2.5) * d52_m32 * a(1,L + 3) * b(L + 2) + sqrt(1.5) * d52_p52 * a(1,L + 3) * b(L + 6) - sqrt(1.5) * d52_m52 * a(1,L + 6) * b(L + 3) - sqrt(1.5) * d52_p52 * a(1,L + 4) * b(L + 5) + sqrt(1.5) * d52_m52 * a(1,L + 5) * b(L + 4));

          element(sum,2,2).i -= fugou * (sqrt(6.0) * d52_p12 * a(2,L) * b(L + 2) - sqrt(6.0) * d52_m12 * a(2,L + 2) * b(L) + sqrt(2.5) * d52_p32 * a(2,L + 1) * b(L + 4) - sqrt(2.5) * d52_m32 * a(2,L + 4) * b(L + 1) - sqrt(2.5) * d52_p32 * a(2,L + 2) * b(L + 3) + sqrt(2.5) * d52_m32 * a(2,L + 3) * b(L + 2) + sqrt(1.5) * d52_p52 * a(2,L + 3) * b(L + 6) - sqrt(1.5) * d52_m52 * a(2,L + 6) * b(L + 3) - sqrt(1.5) * d52_p52 * a(2,L + 4) * b(L + 5) + sqrt(1.5) * d52_m52 * a(2,L + 5) * b(L + 4));
      }

  } /* if (hole.spin==3) */

  /****************************************************
      off-diagonal contribution on up-up and dn-dn
  ****************************************************/

  /* p */

  if (L2 == 2) {

      /* contribution of l+1/2 for up spin */

      element(sum,0,0).i += 0.5 * fugou * (+(d32_m12 - 3.0 * d32_p32) * a(0,L) * b(L + 1) + (-d32_m12 + 3.0 * d32_p32) * a(0,L + 1) * b(L));

      element(sum,0,1).i += 0.5 * fugou * (+(d32_m12 - 3.0 * d32_p32) * a(1,L) * b(L + 1) + (-d32_m12 + 3.0 * d32_p32) * a(1,L + 1) * b(L));

      element(sum,0,2).i += 0.5 * fugou * (+(d32_m12 - 3.0 * d32_p32) * a(2,L) * b(L + 1) + (-d32_m12 + 3.0 * d32_p32) * a(2,L + 1) * b(L));

      /* contribution of l+1/2 for down spin */

      element(sum,1,0).i += 0.5 * fugou * (+(-d32_p12 + 3.0 * d32_m32) * a(0,L) * b(L + 1) + (+d32_p12 - 3.0 * d32_m32) * a(0,L + 1) * b(L));

      element(sum,1,1).i += 0.5 * fugou * (+(-d32_p12 + 3.0 * d32_m32) * a(1,L) * b(L + 1) + (+d32_p12 - 3.0 * d32_m32) * a(1,L + 1) * b(L));

      element(sum,1,2).i += 0.5 * fugou * (+(-d32_p12 + 3.0 * d32_m32) * a(2,L) * b(L + 1) + (+d32_p12 - 3.0 * d32_m32) * a(2,L + 1) * b(L));

      /* contribution of l-1/2 for up spin */

      element(sum,0,0).i += fugou * (d12_m12 * a(0,L) * b(L + 1) - d12_m12 * a(0,L + 1) * b(L));

      element(sum,0,1).i += fugou * (d12_m12 * a(1,L) * b(L + 1) - d12_m12 * a(1,L + 1) * b(L));

      element(sum,0,2).i += fugou * (d12_m12 * a(2,L) * b(L + 1) - d12_m12 * a(2,L + 1) * b(L));

      /* contribution of l-1/2 for down spin */

      element(sum,1,0).i += fugou * (-d12_p12 * a(0,L) * b(L + 1) + d12_p12 * a(0,L + 1) * b(L));

      element(sum,1,1).i += fugou * (-d12_p12 * a(1,L) * b(L + 1) + d12_p12 * a(1,L + 1) * b(L));

      element(sum,1,2).i += fugou * (-d12_p12 * a(2,L) * b(L + 1) + d12_p12 * a(2,L + 1) * b(L));
  }

  /* d */

  else if (L2 == 4) {

      /* contribution of l+1/2 for up spin */

      element(sum,0,0).i += 0.5 * fugou * (+(d52_m32 - 5.0 * d52_p52) * a(0,L + 1) * b(L + 2) + (-d52_m32 + 5.0 * d52_p52) * a(0,L + 2) * b(L + 1) + (2.0 * d52_m12 - 4.0 * d52_p32) * a(0,L + 3) * b(L + 4) + (-2.0 * d52_m12 + 4.0 * d52_p32) * a(0,L + 4) * b(L + 3));

      element(sum,0,1).i += 0.5 * fugou * (+(d52_m32 - 5.0 * d52_p52) * a(1,L + 1) * b(L + 2) + (-d52_m32 + 5.0 * d52_p52) * a(1,L + 2) * b(L + 1) + (2.0 * d52_m12 - 4.0 * d52_p32) * a(1,L + 3) * b(L + 4) + (-2.0 * d52_m12 + 4.0 * d52_p32) * a(1,L + 4) * b(L + 3));

      element(sum,0,2).i += 0.5 * fugou * (+(d52_m32 - 5.0 * d52_p52) * a(2,L + 1) * b(L + 2) + (-d52_m32 + 5.0 * d52_p52) * a(2,L + 2) * b(L + 1) + (2.0 * d52_m12 - 4.0 * d52_p32) * a(2,L + 3) * b(L + 4) + (-2.0 * d52_m12 + 4.0 * d52_p32) * a(2,L + 4) * b(L + 3));

      /* contribution of l+1/2 for down spin */

      element(sum,1,0).i += 0.5 * fugou * (-(d52_p32 - 5.0 * d52_m52) * a(0,L + 1) * b(L + 2) - (-d52_p32 + 5.0 * d52_m52) * a(0,L + 2) * b(L + 1) - (2.0 * d52_p12 - 4.0 * d52_m32) * a(0,L + 3) * b(L + 4) - (-2.0 * d52_p12 + 4.0 * d52_m32) * a(0,L + 4) * b(L + 3));

      element(sum,1,1).i += 0.5 * fugou * (-(d52_p32 - 5.0 * d52_m52) * a(1,L + 1) * b(L + 2) - (-d52_p32 + 5.0 * d52_m52) * a(1,L + 2) * b(L + 1) - (2.0 * d52_p12 - 4.0 * d52_m32) * a(1,L + 3) * b(L + 4) - (-2.0 * d52_p12 + 4.0 * d52_m32) * a(1,L + 4) * b(L + 3));

      element(sum,1,2).i += 0.5 * fugou * (-(d52_p32 - 5.0 * d52_m52) * a(2,L + 1) * b(L + 2) - (-d52_p32 + 5.0 * d52_m52) * a(2,L + 2) * b(L + 1) - (2.0 * d52_p12 - 4.0 * d52_m32) * a(2,L + 3) * b(L + 4) - (-2.0 * d52_p12 + 4.0 * d52_m32) * a(2,L + 4) * b(L + 3));

      /* contribution of l-1/2 for up spin */

      element(sum,0,0).i += 0.5 * fugou * ((4.0 * d32_m32) * a(0,L + 1) * b(L + 2) + (-4.0 * d32_m32) * a(0,L + 2) * b(L + 1) + (3.0 * d32_m12 - d32_p32) * a(0,L + 3) * b(L + 4) + (-3.0 * d32_m12 + d32_p32) * a(0,L + 4) * b(L + 3));

      element(sum,0,1).i += 0.5 * fugou * ((4.0 * d32_m32) * a(1,L + 1) * b(L + 2) + (-4.0 * d32_m32) * a(1,L + 2) * b(L + 1) + (3.0 * d32_m12 - d32_p32) * a(1,L + 3) * b(L + 4) + (-3.0 * d32_m12 + d32_p32) * a(1,L + 4) * b(L + 3));

      element(sum,0,2).i += 0.5 * fugou * ((4.0 * d32_m32) * a(2,L + 1) * b(L + 2) + (-4.0 * d32_m32) * a(2,L + 2) * b(L + 1) + (3.0 * d32_m12 - d32_p32) * a(2,L + 3) * b(L + 4) + (-3.0 * d32_m12 + d32_p32) * a(2,L + 4) * b(L + 3));

      /* contribution of l-1/2 for down spin */

      element(sum,1,0).i += 0.5 * fugou * ((-4.0 * d32_p32) * a(0,L + 1) * b(L + 2) + (+4.0 * d32_p32) * a(0,L + 2) * b(L + 1) - (3.0 * d32_p12 - d32_m32) * a(0,L + 3) * b(L + 4) - (-3.0 * d32_p12 + d32_m32) * a(0,L + 4) * b(L + 3));

      element(sum,1,1).i += 0.5 * fugou * ((-4.0 * d32_p32) * a(1,L + 1) * b(L + 2) + (+4.0 * d32_p32) * a(1,L + 2) * b(L + 1) - (3.0 * d32_p12 - d32_m32) * a(1,L + 3) * b(L + 4) - (-3.0 * d32_p12 + d32_m32) * a(1,L + 4) * b(L + 3));

      element(sum,1,2).i += 0.5 * fugou * ((-4.0 * d32_p32) * a(2,L + 1) * b(L + 2) + (+4.0 * d32_p32) * a(2,L + 2) * b(L + 1) - (3.0 * d32_p12 - d32_m32) * a(2,L + 3) * b(L + 4) - (-3.0 * d32_p12 + d32_m32) * a(2,L + 4) * b(L + 3));
  }

  /* f */

  else if (L2 == 6) {

      /* contribution of l+1/2 for up spin */

      element(sum,0,0).i += 0.5 * fugou * ((3.0 * d72_m12 - 5.0 * d72_p32) * a(0,L + 1) * b(L + 2) + (-3.0 * d72_m12 + 5.0 * d72_p32) * a(0,L + 2) * b(L + 1) + (2.0 * d72_m32 - 6.0 * d72_p52) * a(0,L + 3) * b(L + 4) + (-2.0 * d72_m32 + 6.0 * d72_p52) * a(0,L + 4) * b(L + 3) + (1.0 * d72_m52 - 7.0 * d72_p72) * a(0,L + 5) * b(L + 6) + (-1.0 * d72_m52 + 7.0 * d72_p72) * a(0,L + 6) * b(L + 5));

      element(sum,0,1).i += 0.5 * fugou * ((3.0 * d72_m12 - 5.0 * d72_p32) * a(1,L + 1) * b(L + 2) + (-3.0 * d72_m12 + 5.0 * d72_p32) * a(1,L + 2) * b(L + 1) + (2.0 * d72_m32 - 6.0 * d72_p52) * a(1,L + 3) * b(L + 4) + (-2.0 * d72_m32 + 6.0 * d72_p52) * a(1,L + 4) * b(L + 3) + (1.0 * d72_m52 - 7.0 * d72_p72) * a(1,L + 5) * b(L + 6) + (-1.0 * d72_m52 + 7.0 * d72_p72) * a(1,L + 6) * b(L + 5));

      element(sum,0,2).i += 0.5 * fugou * ((3.0 * d72_m12 - 5.0 * d72_p32) * a(2,L + 1) * b(L + 2) + (-3.0 * d72_m12 + 5.0 * d72_p32) * a(2,L + 2) * b(L + 1) + (2.0 * d72_m32 - 6.0 * d72_p52) * a(2,L + 3) * b(L + 4) + (-2.0 * d72_m32 + 6.0 * d72_p52) * a(2,L + 4) * b(L + 3) + (1.0 * d72_m52 - 7.0 * d72_p72) * a(2,L + 5) * b(L + 6) + (-1.0 * d72_m52 + 7.0 * d72_p72) * a(2,L + 6) * b(L + 5));

      /* contribution of l+1/2 for down spin */

      element(sum,1,0).i += 0.5 * fugou * (-(3.0 * d72_p12 - 5.0 * d72_m32) * a(0,L + 1) * b(L + 2) - (-3.0 * d72_p12 + 5.0 * d72_m32) * a(0,L + 2) * b(L + 1) - (2.0 * d72_p32 - 6.0 * d72_m52) * a(0,L + 3) * b(L + 4) - (-2.0 * d72_p32 + 6.0 * d72_m52) * a(0,L + 4) * b(L + 3) - (1.0 * d72_p52 - 7.0 * d72_m72) * a(0,L + 5) * b(L + 6) - (-1.0 * d72_p52 + 7.0 * d72_m72) * a(0,L + 6) * b(L + 5));

      element(sum,1,1).i += 0.5 * fugou * (-(3.0 * d72_p12 - 5.0 * d72_m32) * a(1,L + 1) * b(L + 2) - (-3.0 * d72_p12 + 5.0 * d72_m32) * a(1,L + 2) * b(L + 1) - (2.0 * d72_p32 - 6.0 * d72_m52) * a(1,L + 3) * b(L + 4) - (-2.0 * d72_p32 + 6.0 * d72_m52) * a(1,L + 4) * b(L + 3) - (1.0 * d72_p52 - 7.0 * d72_m72) * a(1,L + 5) * b(L + 6) - (-1.0 * d72_p52 + 7.0 * d72_m72) * a(1,L + 6) * b(L + 5));

      element(sum,1,2).i += 0.5 * fugou * (-(3.0 * d72_p12 - 5.0 * d72_m32) * a(2,L + 1) * b(L + 2) - (-3.0 * d72_p12 + 5.0 * d72_m32) * a(2,L + 2) * b(L + 1) - (2.0 * d72_p32 - 6.0 * d72_m52) * a(2,L + 3) * b(L + 4) - (-2.0 * d72_p32 + 6.0 * d72_m52) * a(2,L + 4) * b(L + 3) - (1.0 * d72_p52 - 7.0 * d72_m72) * a(2,L + 5) * b(L + 6) - (-1.0 * d72_p52 + 7.0 * d72_m72) * a(2,L + 6) * b(L + 5));

      /* contribution of l-1/2 for up spin */

      element(sum,0,0).i += 0.5 * fugou * ((4.0 * d52_m12 - 2.0 * d52_p32) * a(0,L + 1) * b(L + 2) + (-4.0 * d52_m12 + 2.0 * d52_p32) * a(0,L + 2) * b(L + 1) + (5.0 * d52_m32 - 1.0 * d52_p52) * a(0,L + 3) * b(L + 4) + (-5.0 * d52_m32 + 1.0 * d52_p52) * a(0,L + 4) * b(L + 3) + (+6.0 * d52_m52) * a(0,L + 5) * b(L + 6) + (-6.0 * d52_m52) * a(0,L + 6) * b(L + 5));

      element(sum,0,1).i += 0.5 * fugou * ((4.0 * d52_m12 - 2.0 * d52_p32) * a(1,L + 1) * b(L + 2) + (-4.0 * d52_m12 + 2.0 * d52_p32) * a(1,L + 2) * b(L + 1) + (5.0 * d52_m32 - 1.0 * d52_p52) * a(1,L + 3) * b(L + 4) + (-5.0 * d52_m32 + 1.0 * d52_p52) * a(1,L + 4) * b(L + 3) + (+6.0 * d52_m52) * a(1,L + 5) * b(L + 6) + (-6.0 * d52_m52) * a(1,L + 6) * b(L + 5));

      element(sum,0,2).i += 0.5 * fugou * ((4.0 * d52_m12 - 2.0 * d52_p32) * a(2,L + 1) * b(L + 2) + (-4.0 * d52_m12 + 2.0 * d52_p32) * a(2,L + 2) * b(L + 1) + (5.0 * d52_m32 - 1.0 * d52_p52) * a(2,L + 3) * b(L + 4) + (-5.0 * d52_m32 + 1.0 * d52_p52) * a(2,L + 4) * b(L + 3) + (+6.0 * d52_m52) * a(2,L + 5) * b(L + 6) + (-6.0 * d52_m52) * a(2,L + 6) * b(L + 5));

      /* contribution of l-1/2 for down spin */

      element(sum,1,0).i += 0.5 * fugou * (-(4.0 * d52_p12 - 2.0 * d52_m32) * a(0,L + 1) * b(L + 2) - (-4.0 * d52_p12 + 2.0 * d52_m32) * a(0,L + 2) * b(L + 1) - (5.0 * d52_p32 - 1.0 * d52_m52) * a(0,L + 3) * b(L + 4) - (-5.0 * d52_p32 + 1.0 * d52_m52) * a(0,L + 4) * b(L + 3) - (+6.0 * d52_p52) * a(0,L + 5) * b(L + 6) - (-6.0 * d52_p52) * a(0,L + 6) * b(L + 5));

      element(sum,1,1).i += 0.5 * fugou * (-(4.0 * d52_p12 - 2.0 * d52_m32) * a(1,L + 1) * b(L + 2) - (-4.0 * d52_p12 + 2.0 * d52_m32) * a(1,L + 2) * b(L + 1) - (5.0 * d52_p32 - 1.0 * d52_m52) * a(1,L + 3) * b(L + 4) - (-5.0 * d52_p32 + 1.0 * d52_m52) * a(1,L + 4) * b(L + 3) - (+6.0 * d52_p52) * a(1,L + 5) * b(L + 6) - (-6.0 * d52_p52) * a(1,L + 6) * b(L + 5));

      element(sum,1,2).i += 0.5 * fugou * (-(4.0 * d52_p12 - 2.0 * d52_m32) * a(2,L + 1) * b(L + 2) - (-4.0 * d52_p12 + 2.0 * d52_m32) * a(2,L + 2) * b(L + 1) - (5.0 * d52_p32 - 1.0 * d52_m52) * a(2,L + 3) * b(L + 4) - (-5.0 * d52_p32 + 1.0 * d52_m52) * a(2,L + 4) * b(L + 3) - (+6.0 * d52_p52) * a(2,L + 5) * b(L + 6) - (-6.0 * d52_p52) * a(2,L + 6) * b(L + 5));
  }

  /****************************************************
       diagonal contribution on up-up and dn-dn
  ****************************************************/

  /* s */

  if (L2 == 0) {

      /* VNL for j=l+1/2 */

      element(sum,0,0).r += d12_p12 * a(0,L) * b(L);
      element(sum,0,1).r += d12_p12 * a(1,L) * b(L);
      element(sum,0,2).r += d12_p12 * a(2,L) * b(L);

      element(sum,1,0).r += d12_m12 * a(0,L) * b(L);
      element(sum,1,1).r += d12_m12 * a(1,L) * b(L);
      element(sum,1,2).r += d12_m12 * a(2,L) * b(L);

      /* note that VNL for j=l-1/2 is zero */
  }

  /* p */

  else if (L2 == 2) {

      /* VNL for j=l+1/2 */
      element(sum,0,0).r += 0.5 * (d32_m12 + 3.0 * d32_p32) * a(0,L) * b(L);
      element(sum,0,0).r += 0.5 * (d32_m12 + 3.0 * d32_p32) * a(0,L + 1) * b(L + 1);
      element(sum,0,0).r += 0.5 * (4.0 * d32_p12) * a(0,L + 2) * b(L + 2);

      element(sum,0,1).r += 0.5 * (d32_m12 + 3.0 * d32_p32) * a(1,L) * b(L);
      element(sum,0,1).r += 0.5 * (d32_m12 + 3.0 * d32_p32) * a(1,L + 1) * b(L + 1);
      element(sum,0,1).r += 0.5 * (4.0 * d32_p12) * a(1,L + 2) * b(L + 2);

      element(sum,0,2).r += 0.5 * (d32_m12 + 3.0 * d32_p32) * a(2,L) * b(L);
      element(sum,0,2).r += 0.5 * (d32_m12 + 3.0 * d32_p32) * a(2,L + 1) * b(L + 1);
      element(sum,0,2).r += 0.5 * (4.0 * d32_p12) * a(2,L + 2) * b(L + 2);

      element(sum,1,0).r += 0.5 * (d32_p12 + 3.0 * d32_m32) * a(0,L) * b(L);
      element(sum,1,0).r += 0.5 * (d32_p12 + 3.0 * d32_m32) * a(0,L + 1) * b(L + 1);
      element(sum,1,0).r += 0.5 * (4.0 * d32_m12) * a(0,L + 2) * b(L + 2);

      element(sum,1,1).r += 0.5 * (d32_p12 + 3.0 * d32_m32) * a(1,L) * b(L);
      element(sum,1,1).r += 0.5 * (d32_p12 + 3.0 * d32_m32) * a(1,L + 1) * b(L + 1);
      element(sum,1,1).r += 0.5 * (4.0 * d32_m12) * a(1,L + 2) * b(L + 2);

      element(sum,1,2).r += 0.5 * (d32_p12 + 3.0 * d32_m32) * a(2,L) * b(L);
      element(sum,1,2).r += 0.5 * (d32_p12 + 3.0 * d32_m32) * a(2,L + 1) * b(L + 1);
      element(sum,1,2).r += 0.5 * (4.0 * d32_m12) * a(2,L + 2) * b(L + 2);

      /* VNL for j=l-1/2 */
      element(sum,0,0).r += d12_m12 * a(0,L) * b(L);
      element(sum,0,0).r += d12_m12 * a(0,L + 1) * b(L + 1);
      element(sum,0,0).r += d12_p12 * a(0,L + 2) * b(L + 2);

      element(sum,0,1).r += d12_m12 * a(1,L) * b(L);
      element(sum,0,1).r += d12_m12 * a(1,L + 1) * b(L + 1);
      element(sum,0,1).r += d12_p12 * a(1,L + 2) * b(L + 2);

      element(sum,0,2).r += d12_m12 * a(2,L) * b(L);
      element(sum,0,2).r += d12_m12 * a(2,L + 1) * b(L + 1);
      element(sum,0,2).r += d12_p12 * a(2,L + 2) * b(L + 2);

      element(sum,1,0).r += d12_p12 * a(0,L) * b(L);
      element(sum,1,0).r += d12_p12 * a(0,L + 1) * b(L + 1);
      element(sum,1,0).r += d12_m12 * a(0,L + 2) * b(L + 2);

      element(sum,1,1).r += d12_p12 * a(1,L) * b(L);
      element(sum,1,1).r += d12_p12 * a(1,L + 1) * b(L + 1);
      element(sum,1,1).r += d12_m12 * a(1,L + 2) * b(L + 2);

      element(sum,1,2).r += d12_p12 * a(2,L) * b(L);
      element(sum,1,2).r += d12_p12 * a(2,L + 1) * b(L + 1);
      element(sum,1,2).r += d12_m12 * a(2,L + 2) * b(L + 2);
  }

  /* d */

  else if (L2 == 4) {

      /* VNL for j=l+1/2 */
      element(sum,0,0).r += 0.5 * (6.0 * d52_p12) * a(0,L) * b(L);
      element(sum,0,0).r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * a(0,L + 1) * b(L + 1);
      element(sum,0,0).r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * a(0,L + 2) * b(L + 2);
      element(sum,0,0).r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * a(0,L + 3) * b(L + 3);
      element(sum,0,0).r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * a(0,L + 4) * b(L + 4);

      element(sum,0,1).r += 0.5 * (6.0 * d52_p12) * a(1,L) * b(L);
      element(sum,0,1).r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * a(1,L + 1) * b(L + 1);
      element(sum,0,1).r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * a(1,L + 2) * b(L + 2);
      element(sum,0,1).r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * a(1,L + 3) * b(L + 3);
      element(sum,0,1).r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * a(1,L + 4) * b(L + 4);

      element(sum,0,2).r += 0.5 * (6.0 * d52_p12) * a(2,L) * b(L);
      element(sum,0,2).r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * a(2,L + 1) * b(L + 1);
      element(sum,0,2).r += 0.5 * (1.0 * d52_m32 + 5.0 * d52_p52) * a(2,L + 2) * b(L + 2);
      element(sum,0,2).r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * a(2,L + 3) * b(L + 3);
      element(sum,0,2).r += 0.5 * (2.0 * d52_m12 + 4.0 * d52_p32) * a(2,L + 4) * b(L + 4);

      element(sum,1,0).r += 0.5 * (6.0 * d52_m12) * a(0,L) * b(L);
      element(sum,1,0).r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * a(0,L + 1) * b(L + 1);
      element(sum,1,0).r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * a(0,L + 2) * b(L + 2);
      element(sum,1,0).r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * a(0,L + 3) * b(L + 3);
      element(sum,1,0).r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * a(0,L + 4) * b(L + 4);

      element(sum,1,1).r += 0.5 * (6.0 * d52_m12) * a(1,L) * b(L);
      element(sum,1,1).r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * a(1,L + 1) * b(L + 1);
      element(sum,1,1).r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * a(1,L + 2) * b(L + 2);
      element(sum,1,1).r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * a(1,L + 3) * b(L + 3);
      element(sum,1,1).r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * a(1,L + 4) * b(L + 4);

      element(sum,1,2).r += 0.5 * (6.0 * d52_m12) * a(2,L) * b(L);
      element(sum,1,2).r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * a(2,L + 1) * b(L + 1);
      element(sum,1,2).r += 0.5 * (1.0 * d52_p32 + 5.0 * d52_m52) * a(2,L + 2) * b(L + 2);
      element(sum,1,2).r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * a(2,L + 3) * b(L + 3);
      element(sum,1,2).r += 0.5 * (2.0 * d52_p12 + 4.0 * d52_m32) * a(2,L + 4) * b(L + 4);

      /* VNL for j=l-1/2 */
      element(sum,0,0).r += 0.5 * (4.0 * d32_p12) * a(0,L) * b(L);
      element(sum,0,0).r += 0.5 * (4.0 * d32_m32) * a(0,L + 1) * b(L + 1);
      element(sum,0,0).r += 0.5 * (4.0 * d32_m32) * a(0,L + 2) * b(L + 2);
      element(sum,0,0).r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * a(0,L + 3) * b(L + 3);
      element(sum,0,0).r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * a(0,L + 4) * b(L + 4);

      element(sum,0,1).r += 0.5 * (4.0 * d32_p12) * a(1,L) * b(L);
      element(sum,0,1).r += 0.5 * (4.0 * d32_m32) * a(1,L + 1) * b(L + 1);
      element(sum,0,1).r += 0.5 * (4.0 * d32_m32) * a(1,L + 2) * b(L + 2);
      element(sum,0,1).r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * a(1,L + 3) * b(L + 3);
      element(sum,0,1).r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * a(1,L + 4) * b(L + 4);

      element(sum,0,2).r += 0.5 * (4.0 * d32_p12) * a(2,L) * b(L);
      element(sum,0,2).r += 0.5 * (4.0 * d32_m32) * a(2,L + 1) * b(L + 1);
      element(sum,0,2).r += 0.5 * (4.0 * d32_m32) * a(2,L + 2) * b(L + 2);
      element(sum,0,2).r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * a(2,L + 3) * b(L + 3);
      element(sum,0,2).r += 0.5 * (3.0 * d32_m12 + 1.0 * d32_p32) * a(2,L + 4) * b(L + 4);

      element(sum,1,0).r += 0.5 * (4.0 * d32_m12) * a(0,L) * b(L);
      element(sum,1,0).r += 0.5 * (4.0 * d32_p32) * a(0,L + 1) * b(L + 1);
      element(sum,1,0).r += 0.5 * (4.0 * d32_p32) * a(0,L + 2) * b(L + 2);
      element(sum,1,0).r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * a(0,L + 3) * b(L + 3);
      element(sum,1,0).r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * a(0,L + 4) * b(L + 4);

      element(sum,1,1).r += 0.5 * (4.0 * d32_m12) * a(1,L) * b(L);
      element(sum,1,1).r += 0.5 * (4.0 * d32_p32) * a(1,L + 1) * b(L + 1);
      element(sum,1,1).r += 0.5 * (4.0 * d32_p32) * a(1,L + 2) * b(L + 2);
      element(sum,1,1).r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * a(1,L + 3) * b(L + 3);
      element(sum,1,1).r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * a(1,L + 4) * b(L + 4);

      element(sum,1,2).r += 0.5 * (4.0 * d32_m12) * a(2,L) * b(L);
      element(sum,1,2).r += 0.5 * (4.0 * d32_p32) * a(2,L + 1) * b(L + 1);
      element(sum,1,2).r += 0.5 * (4.0 * d32_p32) * a(2,L + 2) * b(L + 2);
      element(sum,1,2).r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * a(2,L + 3) * b(L + 3);
      element(sum,1,2).r += 0.5 * (3.0 * d32_p12 + 1.0 * d32_m32) * a(2,L + 4) * b(L + 4);
  }

  /* f */

  else if (L2 == 6) {

      /* VNL for j=l+1/2 */
      element(sum,0,0).r += 0.5 * (8.0 * d72_p12) * a(0,L) * b(L);
      element(sum,0,0).r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * a(0,L + 1) * b(L + 1);
      element(sum,0,0).r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * a(0,L + 2) * b(L + 2);
      element(sum,0,0).r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * a(0,L + 3) * b(L + 3);
      element(sum,0,0).r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * a(0,L + 4) * b(L + 4);
      element(sum,0,0).r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * a(0,L + 5) * b(L + 5);
      element(sum,0,0).r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * a(0,L + 6) * b(L + 6);

      element(sum,0,1).r += 0.5 * (8.0 * d72_p12) * a(1,L) * b(L);
      element(sum,0,1).r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * a(1,L + 1) * b(L + 1);
      element(sum,0,1).r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * a(1,L + 2) * b(L + 2);
      element(sum,0,1).r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * a(1,L + 3) * b(L + 3);
      element(sum,0,1).r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * a(1,L + 4) * b(L + 4);
      element(sum,0,1).r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * a(1,L + 5) * b(L + 5);
      element(sum,0,1).r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * a(1,L + 6) * b(L + 6);

      element(sum,0,2).r += 0.5 * (8.0 * d72_p12) * a(2,L) * b(L);
      element(sum,0,2).r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * a(2,L + 1) * b(L + 1);
      element(sum,0,2).r += 0.5 * (3.0 * d72_m12 + 5.0 * d72_p32) * a(2,L + 2) * b(L + 2);
      element(sum,0,2).r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * a(2,L + 3) * b(L + 3);
      element(sum,0,2).r += 0.5 * (2.0 * d72_m32 + 6.0 * d72_p52) * a(2,L + 4) * b(L + 4);
      element(sum,0,2).r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * a(2,L + 5) * b(L + 5);
      element(sum,0,2).r += 0.5 * (1.0 * d72_m52 + 7.0 * d72_p72) * a(2,L + 6) * b(L + 6);

      element(sum,1,0).r += 0.5 * (8.0 * d72_m12) * a(0,L) * b(L);
      element(sum,1,0).r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * a(0,L + 1) * b(L + 1);
      element(sum,1,0).r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * a(0,L + 2) * b(L + 2);
      element(sum,1,0).r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * a(0,L + 3) * b(L + 3);
      element(sum,1,0).r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * a(0,L + 4) * b(L + 4);
      element(sum,1,0).r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * a(0,L + 5) * b(L + 5);
      element(sum,1,0).r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * a(0,L + 6) * b(L + 6);

      element(sum,1,1).r += 0.5 * (8.0 * d72_m12) * a(1,L) * b(L);
      element(sum,1,1).r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * a(1,L + 1) * b(L + 1);
      element(sum,1,1).r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * a(1,L + 2) * b(L + 2);
      element(sum,1,1).r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * a(1,L + 3) * b(L + 3);
      element(sum,1,1).r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * a(1,L + 4) * b(L + 4);
      element(sum,1,1).r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * a(1,L + 5) * b(L + 5);
      element(sum,1,1).r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * a(1,L + 6) * b(L + 6);

      element(sum,1,2).r += 0.5 * (8.0 * d72_m12) * a(2,L) * b(L);
      element(sum,1,2).r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * a(2,L + 1) * b(L + 1);
      element(sum,1,2).r += 0.5 * (3.0 * d72_p12 + 5.0 * d72_m32) * a(2,L + 2) * b(L + 2);
      element(sum,1,2).r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * a(2,L + 3) * b(L + 3);
      element(sum,1,2).r += 0.5 * (2.0 * d72_p32 + 6.0 * d72_m52) * a(2,L + 4) * b(L + 4);
      element(sum,1,2).r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * a(2,L + 5) * b(L + 5);
      element(sum,1,2).r += 0.5 * (1.0 * d72_p52 + 7.0 * d72_m72) * a(2,L + 6) * b(L + 6);

      /* VNL for j=l-1/2 */
      element(sum,0,0).r += 0.5 * (6.0 * d52_p12) * a(0,L) * b(L);
      element(sum,0,0).r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * a(0,L + 1) * b(L + 1);
      element(sum,0,0).r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * a(0,L + 2) * b(L + 2);
      element(sum,0,0).r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * a(0,L + 3) * b(L + 3);
      element(sum,0,0).r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * a(0,L + 4) * b(L + 4);
      element(sum,0,0).r += 0.5 * (6.0 * d52_m52) * a(0,L + 5) * b(L + 5);
      element(sum,0,0).r += 0.5 * (6.0 * d52_m52) * a(0,L + 6) * b(L + 6);

      element(sum,0,1).r += 0.5 * (6.0 * d52_p12) * a(1,L) * b(L);
      element(sum,0,1).r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * a(1,L + 1) * b(L + 1);
      element(sum,0,1).r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * a(1,L + 2) * b(L + 2);
      element(sum,0,1).r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * a(1,L + 3) * b(L + 3);
      element(sum,0,1).r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * a(1,L + 4) * b(L + 4);
      element(sum,0,1).r += 0.5 * (6.0 * d52_m52) * a(1,L + 5) * b(L + 5);
      element(sum,0,1).r += 0.5 * (6.0 * d52_m52) * a(1,L + 6) * b(L + 6);

      element(sum,0,2).r += 0.5 * (6.0 * d52_p12) * a(2,L) * b(L);
      element(sum,0,2).r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * a(2,L + 1) * b(L + 1);
      element(sum,0,2).r += 0.5 * (4.0 * d52_m12 + 2.0 * d52_p32) * a(2,L + 2) * b(L + 2);
      element(sum,0,2).r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * a(2,L + 3) * b(L + 3);
      element(sum,0,2).r += 0.5 * (5.0 * d52_m32 + 1.0 * d52_p52) * a(2,L + 4) * b(L + 4);
      element(sum,0,2).r += 0.5 * (6.0 * d52_m52) * a(2,L + 5) * b(L + 5);
      element(sum,0,2).r += 0.5 * (6.0 * d52_m52) * a(2,L + 6) * b(L + 6);

      element(sum,1,0).r += 0.5 * (6.0 * d52_m12) * a(0,L) * b(L);
      element(sum,1,0).r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * a(0,L + 1) * b(L + 1);
      element(sum,1,0).r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * a(0,L + 2) * b(L + 2);
      element(sum,1,0).r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * a(0,L + 3) * b(L + 3);
      element(sum,1,0).r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * a(0,L + 4) * b(L + 4);
      element(sum,1,0).r += 0.5 * (6.0 * d52_p52) * a(0,L + 5) * b(L + 5);
      element(sum,1,0).r += 0.5 * (6.0 * d52_p52) * a(0,L + 6) * b(L + 6);

      element(sum,1,1).r += 0.5 * (6.0 * d52_m12) * a(1,L) * b(L);
      element(sum,1,1).r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * a(1,L + 1) * b(L + 1);
      element(sum,1,1).r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * a(1,L + 2) * b(L + 2);
      element(sum,1,1).r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * a(1,L + 3) * b(L + 3);
      element(sum,1,1).r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * a(1,L + 4) * b(L + 4);
      element(sum,1,1).r += 0.5 * (6.0 * d52_p52) * a(1,L + 5) * b(L + 5);
      element(sum,1,1).r += 0.5 * (6.0 * d52_p52) * a(1,L + 6) * b(L + 6);

      element(sum,1,2).r += 0.5 * (6.0 * d52_m12) * a(2,L) * b(L);
      element(sum,1,2).r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * a(2,L + 1) * b(L + 1);
      element(sum,1,2).r += 0.5 * (4.0 * d52_p12 + 2.0 * d52_m32) * a(2,L + 2) * b(L + 2);
      element(sum,1,2).r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * a(2,L + 3) * b(L + 3);
      element(sum,1,2).r += 0.5 * (5.0 * d52_p32 + 1.0 * d52_m52) * a(2,L + 4) * b(L + 4);
      element(sum,1,2).r += 0.5 * (6.0 * d52_p52) * a(2,L + 5) * b(L + 5);
      element(sum,1,2).r += 0.5 * (6.0 * d52_p52) * a(2,L + 6) * b(L + 6);
  }
}
} // namespace openmx::core

extern "C" int openmx_core_core_hole_trace(const OpenMXCoreHoleView *hole,const OpenMXCoreHoleTrace *input,
    dcomplex *sum,char *error,std::size_t capacity)
{
  if(error && capacity)*error='\0';
  try {
    if(!hole || !input || !sum)throw std::invalid_argument("missing core-hole trace input/output");
    openmx::core::core_hole_trace(*hole,*input,std::span<dcomplex,9>(sum,9));return 1;
  } catch(const std::exception& e){if(error && capacity)std::snprintf(error,capacity,"%s",e.what());}
  catch(...){if(error && capacity)std::snprintf(error,capacity,"unexpected core-hole trace failure");}
  return 0;
}
