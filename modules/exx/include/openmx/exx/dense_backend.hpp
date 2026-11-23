#pragma once

#include <openmx/exx/exx.h>

#include <cstddef>

// The matrix products of the exchange matrix, section 7.2 of the
// implementation guide.  The caller may pass a table with its own product
// (BLAS, cuBLAS, GEMMul8); without one the library uses plain loops.
namespace openmx::exx {

class DenseBackend {
 public:
  // The built-in loops.
  DenseBackend() noexcept = default;
  // The product of a table.  A null pointer, or a table without a function,
  // selects the built-in loops.
  explicit DenseBackend(const OpenMXExxDenseBackend* table) noexcept;

  bool built_in() const noexcept { return table_.dgemm == nullptr; }

  // C = alpha op(A) op(B) + beta C with column-major arrays, as DGEMM of the
  // BLAS: op(A) is m x k, op(B) is k x n, C is m x n; transa and transb are
  // 'N' or 'T'.  Throws std::runtime_error when the table reports a failure,
  // and std::length_error when a size does not fit the int of the table.
  void gemm(char transa, char transb, std::size_t m, std::size_t n, std::size_t k, double alpha,
            const double* a, std::size_t lda, const double* b, std::size_t ldb, double beta,
            double* c, std::size_t ldc) const;

  // The same products for row-major arrays without padding.
  // C (m x n) = alpha A (m x k) B (k x n) + beta C
  void multiply(std::size_t m, std::size_t n, std::size_t k, double alpha, const double* a, const double* b,
                double beta, double* c) const;
  // C (m x n) = alpha A (m x k) B^T, B (n x k), + beta C
  void multiply_transposed(std::size_t m, std::size_t n, std::size_t k, double alpha, const double* a,
                           const double* b, double beta, double* c) const;
  // C (m x n) = alpha A^T B, A (k x m), B (k x n), + beta C
  void transposed_multiply(std::size_t m, std::size_t n, std::size_t k, double alpha, const double* a,
                           const double* b, double beta, double* c) const;

 private:
  OpenMXExxDenseBackend table_{nullptr,nullptr};
};

} // namespace openmx::exx
