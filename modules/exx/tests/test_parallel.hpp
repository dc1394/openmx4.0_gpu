// Loops for the entry parallel_for of the function table
// OpenMXExxDenseBackend, and a product that several threads may call, for the
// tests.
#pragma once

#include "test_dgemm.hpp"

#include <atomic>
#include <thread>
#include <vector>

namespace exx_test {

// The context of the functions below, or null: the calls of the product and
// the loops.
struct Calls {
  std::atomic<long> products{0};
  std::atomic<long> loops{0};
};

using Loop = int (*)(void*, int, void (*)(void*, int), void*);

inline void count_loop(void* context)
{
  if (context != nullptr) ++static_cast<Calls*>(context)->loops;
}

// counting_dgemm() with a counter that several threads may raise.
inline int shared_dgemm(void* context, char transa, char transb, int m, int n, int k, double alpha, const double* a,
                        int lda, const double* b, int ldb, double beta, double* c, int ldc)
{
  if (context != nullptr) ++static_cast<Calls*>(context)->products;
  return counting_dgemm(nullptr,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}

// The calls one after the other on the calling thread.
inline int in_order_loop(void* context, int count, void (*body)(void*, int), void* argument)
{
  count_loop(context);
  for (int index = 0; index < count; ++index) body(argument,index);
  return 0;
}

// The same from the last index to the first.
inline int reversed_loop(void* context, int count, void (*body)(void*, int), void* argument)
{
  count_loop(context);
  for (int index = count; index-- > 0;) body(argument,index);
  return 0;
}

// Every call on a thread of its own.
inline int threaded_loop(void* context, int count, void (*body)(void*, int), void* argument)
{
  count_loop(context);
  std::vector<std::thread> threads;
  int status = 0;
  try {
    for (int index = 0; index < count; ++index) threads.emplace_back(body,argument,index);
  } catch (...) {
    status = 1;
  }
  for (std::thread& thread : threads) thread.join();
  return status;
}

// Loops that break the contract: a status that reports a failure, a call
// left out, a call made twice, and indices outside the loop.
inline int failing_loop(void* context, int count, void (*body)(void*, int), void* argument)
{
  in_order_loop(context,count,body,argument);
  return 5;
}

inline int skipping_loop(void* context, int count, void (*body)(void*, int), void* argument)
{
  return in_order_loop(context,count-1,body,argument);
}

inline int repeating_loop(void* context, int count, void (*body)(void*, int), void* argument)
{
  in_order_loop(context,count,body,argument);
  body(argument,0);
  return 0;
}

inline int stray_loop(void* context, int count, void (*body)(void*, int), void* argument)
{
  in_order_loop(context,count,body,argument);
  body(argument,count);
  body(argument,-1);
  return 0;
}

} // namespace exx_test
