#pragma once

#include <openmx/exx/exx.h>

#include <cstddef>

// The matrix products of the exchange matrix, section 7.2 of the
// implementation guide.  The caller may pass a table with its own product
// (BLAS, cuBLAS, GEMMul8); without one the library uses plain loops.  The
// table can also bring a loop that runs the pieces of the work of one atom on
// several threads, and a device on which the build of several atoms runs
// (section 11).
namespace openmx::exx {

class DenseBackend {
 public:
  // The built-in loops.
  DenseBackend() noexcept = default;
  // The product and the parallel loop of a table.  A null pointer, or a table
  // without dgemm, selects the built-in product; a table without
  // parallel_for, or with fewer than two threads, the loop on the calling
  // thread.
  explicit DenseBackend(const OpenMXExxDenseBackend* table, OpenMXExxAddRows add_rows = nullptr) noexcept;

  bool built_in() const noexcept { return table_.dgemm == nullptr; }

  // The device of the table, nullptr without one (exx.h).
  const OpenMXExxDevice* device() const noexcept { return table_.device; }

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

  void add_rows(double* dst, std::size_t dst_stride, const double* src, std::size_t src_stride,
                std::size_t rows, std::size_t columns, double factor) const;
  void add_row(double* dst, const double* src, std::size_t count, double factor) const
  {
    add_rows(dst,count,src,count,1,count,factor);
  }

  // The largest number of pieces into which a caller divides a loop for
  // parallel_for(): the threads of the table, 1 without a parallel loop.  A
  // loop over fewer items has fewer pieces.
  std::size_t pieces() const noexcept;

  // Calls body(index) for index = 0..count-1 and returns when all calls have
  // returned.  With a parallel loop in the table and more than one index the
  // calls may run at the same time on different threads: they must write to
  // different memory, and the products above may be called from all of them.
  // An exception of a call does not stop the other calls; the one of the
  // lowest index is thrown again afterwards.  Throws std::runtime_error when
  // the loop of the table fails or does not call every index once, and
  // std::length_error when count does not fit the int of the table.  Without
  // a parallel loop the calls run in order on the calling thread, and an
  // exception ends the loop at once.  A body must not call parallel_for().
  template <class Body>
  void parallel_for(std::size_t count, Body body) const
  {
    run(count,[](void* callable, std::size_t index) { (*static_cast<Body*>(callable))(index); },&body);
  }

 private:
  void run(std::size_t count, void (*call)(void*, std::size_t), void* callable) const;

  OpenMXExxDenseBackend table_{nullptr,nullptr,nullptr,0,nullptr};
  OpenMXExxAddRows add_rows_ = nullptr;
};

} // namespace openmx::exx
