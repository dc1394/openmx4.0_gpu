#pragma once
#include <openmx/core/unpivoted_inverse.hpp>
#include <openmx_core_lu_ops.h>

namespace openmx::core {
// Native density/potential mixing's packed, pivoted LAPACK inverse. Only the
// scratch is owned here; each call supplies its exact native library operations.
class PivotedInverse {
 public:
  using matrix_view=UnpivotedInverse::matrix_view;
  explicit PivotedInverse(std::size_t capacity);
  PivotedInverse(const PivotedInverse&)=delete;
  PivotedInverse& operator=(const PivotedInverse&)=delete;
  std::size_t capacity() const noexcept { return capacity_; }
  static void validate(matrix_view input,matrix_view inverse,std::size_t dimension,
                       const OpenMXRealLuOperations&);
  void apply(matrix_view input,matrix_view inverse,std::size_t dimension,
             const OpenMXRealLuOperations&);
 private:
  static std::size_t checked_capacity(std::size_t);
  std::size_t capacity_;
  OverwriteTensor<double,1> packed_,work_;
  OverwriteTensor<int,1> pivots_;
};
} // namespace openmx::core
