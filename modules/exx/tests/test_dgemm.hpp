// A second implementation of the product of the function table
// OpenMXExxDenseBackend, for the tests.  Every element of C is one sum over
// the inner index, taken from the last term to the first; the library's own
// loops add the terms in the opposite order, so the two differ by rounding.
#pragma once

#include <cstddef>

namespace exx_test {

// context points to a long that counts the calls, or is null.
inline int counting_dgemm(void* context, char transa, char transb, int m, int n, int k, double alpha,
                          const double* a, int lda, const double* b, int ldb, double beta, double* c, int ldc)
{
  if (context != nullptr) ++*static_cast<long*>(context);
  const bool ta = transa == 'T' || transa == 't', tb = transb == 'T' || transb == 't';
  if ((!ta && transa != 'N' && transa != 'n') || (!tb && transb != 'N' && transb != 'n')) return 1;
  if (m < 0 || n < 0 || k < 0 || lda < 1 || ldb < 1 || ldc < 1) return 1;
  const std::size_t M = static_cast<std::size_t>(m), N = static_cast<std::size_t>(n), K = static_cast<std::size_t>(k);
  const std::size_t LA = static_cast<std::size_t>(lda), LB = static_cast<std::size_t>(ldb), LC = static_cast<std::size_t>(ldc);
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      double sum = 0.0;
      for (std::size_t l = K; l-- > 0;) sum += (ta ? a[l+i*LA] : a[i+l*LA])*(tb ? b[j+l*LB] : b[l+j*LB]);
      double& target = c[i+j*LC];
      target = alpha*sum+(beta == 0.0 ? 0.0 : beta*target);
    }
  return 0;
}

// A product that always reports a failure.
inline int failing_dgemm(void*, char, char, int, int, int, double, const double*, int, const double*, int, double,
                         double*, int)
{
  return 7;
}

} // namespace exx_test
