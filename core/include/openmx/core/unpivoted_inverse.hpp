#pragma once

#include <openmx/core/overwrite_tensor.hpp>

namespace openmx::core {

// Legacy density mixing's non-pivoted LU. Singular/zero-pivot arithmetic is
// deliberately allowed to produce nonfinite values for the caller's fallback.
// This is separate from the pivoted LAPACK and regularized eigensystem paths.
// Multiply-subtract updates are explicitly fused, matching the native C
// arithmetic even when the rest of the core disables implicit contraction.
class UnpivotedInverse {
 public:
  using matrix_view=Tensor<double,2>::view_type;
  explicit UnpivotedInverse(std::size_t capacity);
  UnpivotedInverse(const UnpivotedInverse&)=delete;
  UnpivotedInverse& operator=(const UnpivotedInverse&)=delete;
  std::size_t capacity() const noexcept { return capacity_; }
  static void validate(matrix_view input,matrix_view inverse,std::size_t dimension);
  void apply(matrix_view input,matrix_view inverse,std::size_t dimension);
  static void clear(matrix_view input);

 private:
  static std::size_t checked_capacity(std::size_t);
  std::size_t capacity_;
  OverwriteTensor<double,1> x_,y_;
  OverwriteTensor<double,2> backup_;
};

} // namespace openmx::core
