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

void sum(double* dst, const double* left, const double* right, size_t count)
{
  const size_t full = count/lanes*lanes;
  for (size_t j = 0; j < full; j += lanes)
    store(load(left+j)+load(right+j),dst+j);
  for (size_t j = full; j < count; ++j) dst[j] = left[j]+right[j];
}

double dot(const double* left, const double* right, size_t count)
{
  Vector sum0(0.0), sum1(0.0);
  const size_t full = count/(2*lanes)*(2*lanes);
  for (size_t j = 0; j < full; j += 2*lanes) {
    sum0 += load(left+j)*load(right+j);
    sum1 += load(left+j+lanes)*load(right+j+lanes);
  }
  const Vector sums = sum0+sum1;
  double sum = 0.0;
  for (int lane = 0; lane < lanes; ++lane) sum += sums[lane];
  for (size_t j = full; j < count; ++j) sum += left[j]*right[j];
  return sum;
}

void update_row(double* row, const double* left, const double* right, size_t count,
                double left_scale, double right_scale)
{
  const Vector a(left_scale), b(right_scale);
  const size_t full = count/lanes*lanes;
  for (size_t j = 0; j < full; j += lanes)
    store(load(row+j)+(a*load(left+j)+b*load(right+j)),row+j);
  for (size_t j = full; j < count; ++j)
    row[j] += left_scale*left[j]+right_scale*right[j];
}
}

const OpenMXHostAddRows& OPENMX_HOST_SIMD_FACTORY(add_rows)() noexcept
{
  static constexpr OpenMXHostAddRows kernel = add_rows;
  return kernel;
}

const OpenMXHostSum& OPENMX_HOST_SIMD_FACTORY(sum)() noexcept
{
  static constexpr OpenMXHostSum kernel = sum;
  return kernel;
}

const OpenMXHostDot& OPENMX_HOST_SIMD_FACTORY(dot)() noexcept
{
  static constexpr OpenMXHostDot kernel = dot;
  return kernel;
}

const OpenMXHostUpdateRow& OPENMX_HOST_SIMD_FACTORY(update_row)() noexcept
{
  static constexpr OpenMXHostUpdateRow kernel = update_row;
  return kernel;
}
}
