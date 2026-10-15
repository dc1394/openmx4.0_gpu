/**********************************************************************
  elpa_cosma_memory.cpp

  Host memory of COSMA's ScaLAPACK-API multiplication (cosma::pxgemm)
  for the transform choice of elpa_cosma_bridge.c.  pxgemm builds a
  strategy from the problem, the grid (adapted to the block-cyclic layout
  when the blocks tile the matrices exactly) and COSMA_CPU_MAX_MEMORY,
  takes the strategy's buffers from a memory pool and moves the matrices
  between the two layouts through COSTA's send and receive buffers.  The
  pool and the COSTA buffers stay allocated for the next call of the same
  scalar type, so on a node every rank of the grid keeps them.

  Built with the C++ compiler of the COSMA stack (GPUSOLVER2_HOST_CXX);
  the bridge calls it on one rank and broadcasts the result.
***********************************************************************/

#include <cosma/cosma_pxgemm.hpp>
#include <cosma/environment_variables.hpp>
#include <cosma/scalapack.hpp>
#include <cosma/strategy.hpp>

#include <algorithm>
#include <cctype>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

extern "C" void Cblacs_gridinfo(int ictxt, int *nprow, int *npcol, int *myrow, int *mycol);

namespace {

template <typename T>
double cosma_host_bytes(char transa, char transb, int m, int n, int k,
                        const int *desca, const int *descb, const int *descc, int ictxt)
{
    int procrows = 0, proccols = 0, myrow = 0, mycol = 0;
    Cblacs_gridinfo(ictxt, &procrows, &proccols, &myrow, &mycol);
    const int P = procrows * proccols;
    if (P < 1) return -1.0;

    /* the limit as cosma::get_cpu_max_memory<T>() reads it */
    long long limit = std::numeric_limits<long long>::max();
    if (const char *v = std::getenv("COSMA_CPU_MAX_MEMORY")) {
        limit = std::atoll(v) * 1024LL * 1024LL / (long long)sizeof(T);
    }

    /* the prefix that pxgemm adds when it adapts its strategy to the grid */
    std::vector<int> divisors;
    std::string dimensions, step_type;
    if (P > 1 && cosma::get_adapt_strategy()) {
        cosma::scalapack::global_matrix_size dim_a(desca), dim_b(descb), dim_c(descc);
        cosma::scalapack::block_size blk_a(desca), blk_b(descb), blk_c(descc);
        const char order =
            cosma::scalapack::rank_ordering(ictxt, P) == costa::scalapack::ordering::column_major
                ? 'C' : 'R';
        cosma::adapt_strategy_to_block_cyclic_grid(divisors, dimensions, step_type,
                                                   m, n, k, P,
                                                   dim_a, dim_b, dim_c,
                                                   blk_a, blk_b, blk_c,
                                                   1, 1, 1, 1, 1, 1,
                                                   (char)std::toupper(transa),
                                                   (char)std::toupper(transb),
                                                   procrows, proccols, order);
    }

    /* a strategy that does not fit the limit throws after printing its
       diagnosis to std::cout; the bridge reports the consequence itself */
    long long used;
    {
        std::streambuf *out = std::cout.rdbuf(nullptr);
        try {
            cosma::Strategy strategy(m, n, k, P, divisors, dimensions, step_type, limit);
            used = strategy.memory_used;
        }
        catch (...) {
            used = -1;
        }
        std::cout.rdbuf(out);
        std::cout.clear();
    }
    if (used < 0) return -1.0;

    /* resident pool: the strategy's buffers (the pool reserves an
       amortization margin on top, which stays untouched); COSTA: a send
       and a receive buffer, each as large as the bigger phase (A and B
       into COSMA's layout, then C back), with 10% for the block-cyclic
       and COSMA layouts' imbalance */
    const double elem = (double)sizeof(T);
    const double ab = ((double)m * k + (double)k * n) / P;
    const double c = (double)m * n / P;
    return (double)used * elem + 2.0 * 1.1 * std::max(ab, c) * elem;
}

} // namespace

/* Host bytes one rank keeps after cosma::pxgemm computes the m x n x k
   product with these descriptors on the grid of ictxt; -1 when COSMA
   cannot build a strategy within COSMA_CPU_MAX_MEMORY (pxgemm would
   throw).  Local to this rank: Cblacs grid queries only. */
extern "C" double openmx_gs2_cosma_host_bytes(char transa, char transb, int m, int n, int k,
                                              const int *desca, const int *descb,
                                              const int *descc, int ictxt, int is_complex)
{
    try {
        if (is_complex) {
            return cosma_host_bytes<std::complex<double>>(transa, transb, m, n, k,
                                                          desca, descb, descc, ictxt);
        }
        return cosma_host_bytes<double>(transa, transb, m, n, k, desca, descb, descc, ictxt);
    }
    catch (...) {
        return -1.0;
    }
}
