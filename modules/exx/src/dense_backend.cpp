#include <openmx/exx/dense_backend.hpp>

#include <atomic>
#include <climits>
#include <exception>
#include <stdexcept>
#include <vector>

namespace {

// What the calls of one loop of DenseBackend::parallel_for() share.  Every
// index has its own entry in failure and in called.
struct Loop {
  void (*call)(void*, std::size_t);
  void* callable;
  std::size_t count;
  std::exception_ptr* failure;
  std::atomic<unsigned char>* called;
  std::atomic<bool> wrong;   // an index outside the loop, or one index twice
};

// 0: no transposition, 1: transposition, -1: not a valid letter.  For real
// matrices 'C' means 'T'.
int transposition(char letter)
{
  switch (letter) {
    case 'N': case 'n': return 0;
    case 'T': case 't': case 'C': case 'c': return 1;
    default: return -1;
  }
}

} // namespace

extern "C" {

// One call of a loop, as the loop of a table makes it.  No exception leaves
// the function: the frames of the caller need not be C++.  An index runs
// once; a call that the loop must not make is noted and does nothing.
static void openmx_exx_loop_call(void* argument, int index) noexcept
{
  Loop& loop = *static_cast<Loop*>(argument);
  if (index < 0 || static_cast<std::size_t>(index) >= loop.count) {
    loop.wrong.store(true);
    return;
  }
  const std::size_t i = static_cast<std::size_t>(index);
  if (loop.called[i].exchange(1) != 0) {
    loop.wrong.store(true);
    return;
  }
  try {
    loop.call(loop.callable,i);
  } catch (...) {
    loop.failure[i] = std::current_exception();
  }
}

} // extern "C"

extern "C" int openmx_exx_reference_dgemm(void*, char transa, char transb, int m, int n, int k, double alpha,
                                          const double* a, int lda, const double* b, int ldb, double beta,
                                          double* c, int ldc)
{
  const int ta = transposition(transa), tb = transposition(transb);
  if (ta < 0 || tb < 0 || m < 0 || n < 0 || k < 0) return 1;
  const int rows_a = ta ? k : m, rows_b = tb ? n : k;
  if (lda < (rows_a > 1 ? rows_a : 1) || ldb < (rows_b > 1 ? rows_b : 1) || ldc < (m > 1 ? m : 1)) return 1;
  if (m == 0 || n == 0) return 0;
  if (c == nullptr || (k > 0 && alpha != 0.0 && (a == nullptr || b == nullptr))) return 1;

  const std::size_t M = static_cast<std::size_t>(m), N = static_cast<std::size_t>(n), K = static_cast<std::size_t>(k);
  const std::size_t LA = static_cast<std::size_t>(lda), LB = static_cast<std::size_t>(ldb), LC = static_cast<std::size_t>(ldc);
  // As in the BLAS, C is not read when beta is zero.
  for (std::size_t j = 0; j < N; ++j) {
    double* column = c+j*LC;
    if (beta == 0.0) for (std::size_t i = 0; i < M; ++i) column[i] = 0.0;
    else if (beta != 1.0) for (std::size_t i = 0; i < M; ++i) column[i] *= beta;
  }
  if (alpha == 0.0 || K == 0) return 0;

  if (!ta) {
    // C(:,j) += alpha b_lj A(:,l): the inner loop runs down two columns.
    for (std::size_t j = 0; j < N; ++j) {
      double* column = c+j*LC;
      for (std::size_t l = 0; l < K; ++l) {
        const double factor = alpha*(tb ? b[j+l*LB] : b[l+j*LB]);
        const double* source = a+l*LA;
        for (std::size_t i = 0; i < M; ++i) column[i] += factor*source[i];
      }
    }
  } else if (!tb) {
    // C(i,j) += alpha A(:,i) . B(:,j)
    for (std::size_t j = 0; j < N; ++j) {
      const double* right = b+j*LB;
      for (std::size_t i = 0; i < M; ++i) {
        const double* left = a+i*LA;
        double sum = 0.0;
        for (std::size_t l = 0; l < K; ++l) sum += left[l]*right[l];
        c[i+j*LC] += alpha*sum;
      }
    }
  } else {
    // C(i,j) += alpha sum_l A(l,i) B(j,l)
    for (std::size_t j = 0; j < N; ++j)
      for (std::size_t i = 0; i < M; ++i) {
        const double* left = a+i*LA;
        double sum = 0.0;
        for (std::size_t l = 0; l < K; ++l) sum += left[l]*b[j+l*LB];
        c[i+j*LC] += alpha*sum;
      }
  }
  return 0;
}

namespace openmx::exx {

DenseBackend::DenseBackend(const OpenMXExxDenseBackend* table, OpenMXExxAddRows add_rows) noexcept
  : add_rows_(add_rows)
{
  if (table != nullptr) table_ = *table;
}

void DenseBackend::add_rows(double* dst, std::size_t dst_stride, const double* src, std::size_t src_stride,
                            std::size_t rows, std::size_t columns, double factor) const
{
  if (rows == 0 || columns == 0) return;
  if (add_rows_ != nullptr) {
    add_rows_(dst,dst_stride,src,src_stride,rows,columns,factor);
    return;
  }
  for (std::size_t i = 0; i < rows; ++i) {
    double* target = dst+i*dst_stride;
    const double* source = src+i*src_stride;
    if (factor == 1.0) {
      for (std::size_t j = 0; j < columns; ++j) target[j] += source[j];
    } else if (factor == -1.0) {
      for (std::size_t j = 0; j < columns; ++j) target[j] -= source[j];
    } else {
      for (std::size_t j = 0; j < columns; ++j) target[j] += factor*source[j];
    }
  }
}

std::size_t DenseBackend::pieces() const noexcept
{
  return table_.parallel_for != nullptr && table_.threads > 1 ? static_cast<std::size_t>(table_.threads) : 1;
}

void DenseBackend::run(std::size_t count, void (*call)(void*, std::size_t), void* callable) const
{
  if (count < 2 || pieces() < 2) {
    for (std::size_t index = 0; index < count; ++index) call(callable,index);
    return;
  }
  if (count > static_cast<std::size_t>(INT_MAX))
    throw std::length_error("a loop of the exchange build has too many pieces for the dense backend");
  std::vector<std::exception_ptr> failure(count);
  std::vector<std::atomic<unsigned char>> called(count);
  for (std::atomic<unsigned char>& flag : called) flag.store(0);
  Loop loop{call,callable,count,failure.data(),called.data(),{false}};
  const int status = table_.parallel_for(table_.context,static_cast<int>(count),&openmx_exx_loop_call,&loop);
  for (const std::exception_ptr& thrown : failure)
    if (thrown) std::rethrow_exception(thrown);
  if (status != 0) throw std::runtime_error("the parallel loop of the dense backend failed");
  bool complete = !loop.wrong.load();
  for (const std::atomic<unsigned char>& flag : called) complete = complete && flag.load() == 1;
  if (!complete) throw std::runtime_error("the parallel loop of the dense backend did not make every call once");
}

void DenseBackend::gemm(char transa, char transb, std::size_t m, std::size_t n, std::size_t k, double alpha,
                        const double* a, std::size_t lda, const double* b, std::size_t ldb, double beta,
                        double* c, std::size_t ldc) const
{
  const std::size_t limit = static_cast<std::size_t>(INT_MAX);
  if (m > limit || n > limit || k > limit || lda > limit || ldb > limit || ldc > limit)
    throw std::length_error("a matrix of the exchange build is too large for the dense backend");
  const auto narrow = [](std::size_t value) { return static_cast<int>(value); };
  const int status = table_.dgemm != nullptr
      ? table_.dgemm(table_.context,transa,transb,narrow(m),narrow(n),narrow(k),alpha,a,narrow(lda),b,narrow(ldb),
                     beta,c,narrow(ldc))
      : openmx_exx_reference_dgemm(nullptr,transa,transb,narrow(m),narrow(n),narrow(k),alpha,a,narrow(lda),b,
                                   narrow(ldb),beta,c,narrow(ldc));
  if (status != 0) throw std::runtime_error("the matrix product of the dense backend failed");
}

// A row-major m x n array is the column-major array of its transpose.  With
// C^T = B^T A^T the three row-major products become the calls below.
void DenseBackend::multiply(std::size_t m, std::size_t n, std::size_t k, double alpha, const double* a,
                            const double* b, double beta, double* c) const
{
  if (m == 0 || n == 0) return;
  gemm('N','N',n,m,k,alpha,b,n,a,k > 0 ? k : 1,beta,c,n);
}

void DenseBackend::multiply_transposed(std::size_t m, std::size_t n, std::size_t k, double alpha, const double* a,
                                       const double* b, double beta, double* c) const
{
  if (m == 0 || n == 0) return;
  const std::size_t ld = k > 0 ? k : 1;
  gemm('T','N',n,m,k,alpha,b,ld,a,ld,beta,c,n);
}

void DenseBackend::transposed_multiply(std::size_t m, std::size_t n, std::size_t k, double alpha, const double* a,
                                       const double* b, double beta, double* c) const
{
  if (m == 0 || n == 0) return;
  gemm('N','T',n,m,k,alpha,b,n,a,m,beta,c,n);
}

} // namespace openmx::exx
