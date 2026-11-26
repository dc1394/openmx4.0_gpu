#include "host_add_rows_kernels.hpp"
#include "host_simd_abi.hpp"

namespace openmx::core::host {
namespace {
template<int Sign, class Value>
Value update(Value dst, Value src, Value factor)
{
  if constexpr (Sign == 1) return dst+src;
  if constexpr (Sign == -1) return dst-src;
  return dst+factor*src;
}

template<int Sign>
void update_rows(double* dst, size_t dst_stride, const double* src, size_t src_stride,
                 size_t rows, size_t columns, double factor)
{
  const Vector scale(factor);
  const size_t full = columns/lanes*lanes;
#if defined(OPENMX_HOST_SIMD_AVX512)
  const int remaining = static_cast<int>(columns-full);
  const Vector::mask_type active([remaining](auto lane) { return int(lane)<remaining; });
#endif
  for (size_t i = 0; i < rows; ++i) {
    double* target = dst+i*dst_stride;
    const double* source = src+i*src_stride;
    for (size_t j = 0; j < full; j += lanes)
      store(update<Sign>(load(target+j),load(source+j),scale),target+j);
#if defined(OPENMX_HOST_SIMD_AVX512)
    if (remaining != 0) {
      const Vector x = simd::simd_partial_load<Vector>(target+full,active);
      const Vector y = simd::simd_partial_load<Vector>(source+full,active);
      simd::simd_partial_store(update<Sign>(x,y,scale),target+full,active);
    }
#else
    for (size_t j = full; j < columns; ++j) target[j] = update<Sign>(target[j],source[j],factor);
#endif
  }
}

void add_rows(double* dst, size_t dst_stride, const double* src, size_t src_stride,
              size_t rows, size_t columns, double factor)
{
  if (rows == 0 || columns == 0) return;
  if (factor == 1.0) update_rows<1>(dst,dst_stride,src,src_stride,rows,columns,factor);
  else if (factor == -1.0) update_rows<-1>(dst,dst_stride,src,src_stride,rows,columns,factor);
  else update_rows<0>(dst,dst_stride,src,src_stride,rows,columns,factor);
}
}

const OpenMXHostAddRows& OPENMX_HOST_SIMD_FACTORY(add_rows)() noexcept
{
  static constexpr OpenMXHostAddRows kernel = add_rows;
  return kernel;
}
}
