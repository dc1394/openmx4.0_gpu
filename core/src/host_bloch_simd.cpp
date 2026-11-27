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
}

const OpenMXHostBlochAdd& OPENMX_HOST_SIMD_FACTORY(bloch)() noexcept
{
  static constexpr OpenMXHostBlochAdd kernel = add_bloch;
  return kernel;
}
}
