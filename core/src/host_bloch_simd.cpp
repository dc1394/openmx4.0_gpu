#include "host_bloch_kernels.hpp"
#include "host_simd_abi.hpp"

namespace openmx::core::host {
namespace {
template<bool Complex>
void add_block(double* dst, int ld, const double* real, const double* imag,
               int rows, int cols, double cosine, double sine)
{
  const Vector real_phase([=](auto i) { return int(i)%2 == 0 ? cosine : sine; });
  const Vector imag_phase([=](auto i) { return int(i)%2 == 0 ? -sine : cosine; });
  for (int j = 0; j < cols; ++j) {
    double* col = dst + 2*size_t(j)*ld;
    const double* r = real + size_t(j)*rows;
    const double* im = Complex ? imag + size_t(j)*rows : nullptr;
    int i = 0;
    if constexpr (lanes > 1) {
      for (; i <= rows-lanes/2; i += lanes/2) {
        const Vector rv([&](auto lane) { return r[i+int(lane)/2]; });
        Vector delta = rv*real_phase;
        if constexpr (Complex) {
          const Vector iv([&](auto lane) { return im[i+int(lane)/2]; });
          delta = delta + iv*imag_phase;
        }
        store(load(col+2*i) + delta,col+2*i);
      }
    }
    for (; i < rows; ++i) {
      double re = r[i]*cosine, in = r[i]*sine;
      if constexpr (Complex) {
        re -= im[i]*sine;
        in += im[i]*cosine;
      }
      col[2*i] += re;
      col[2*i+1] += in;
    }
  }
}

void add_bloch(double* dst, int ld, const double* real, const double* imag,
               int rows, int cols, double cosine, double sine)
{
  if (imag) add_block<true>(dst,ld,real,imag,rows,cols,cosine,sine);
  else add_block<false>(dst,ld,real,imag,rows,cols,cosine,sine);
}

template<bool Complex>
void mo_dot(int count, const float* ao, const double* real, const double* imag, double* out)
{
  Vector r(0.0),im(0.0);
  int i=0;
  for (; i<=count-lanes; i+=lanes) {
    const Vector p(simd::simd_unchecked_load<NarrowFloatVector>(ao+i));
    r=r+load(real+i)*p;
    if constexpr (Complex) im=im+load(imag+i)*p;
  }
  double re=0.0,in=0.0;
  for (int lane=0; lane<lanes; ++lane) {
    re+=r[lane];
    if constexpr (Complex) in+=im[lane];
  }
  for (; i<count; ++i) {
    re+=real[i]*double(ao[i]);
    if constexpr (Complex) in+=imag[i]*double(ao[i]);
  }
  out[0]=re;
  out[1]=in;
}

void contract_mo(int count, const float* ao, const double* real, const double* imag, double* out)
{
  if (imag) mo_dot<true>(count,ao,real,imag,out);
  else mo_dot<false>(count,ao,real,nullptr,out);
}
}

const OpenMXHostBlochAdd& OPENMX_HOST_SIMD_FACTORY(bloch)() noexcept
{
  static constexpr OpenMXHostBlochAdd kernel = add_bloch;
  return kernel;
}

const OpenMXHostMODot& OPENMX_HOST_SIMD_FACTORY(mo_dot)() noexcept
{
  static constexpr OpenMXHostMODot kernel=contract_mo;
  return kernel;
}
}
