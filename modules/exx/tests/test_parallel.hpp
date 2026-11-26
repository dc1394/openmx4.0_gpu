// Loops for the entry parallel_for of the function table
// OpenMXExxDenseBackend, and products that several threads may call, for the
// tests.
#pragma once

#include "test_dgemm.hpp"

#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

namespace exx_test {

// The calls of loops by their count: counts[c] loops were asked for c calls
// of the body.
using Counts = std::map<int,long>;

// The context of the functions below, or null: the calls of the products and
// of the loops.
struct Calls {
  std::atomic<long> products{0};
  std::atomic<long> loops{0};
  // The loops by the count that the library passed.  The library calls a
  // loop from the thread that called the library, one loop at a time and
  // never from a body (exx.h), so the map needs no lock.  enter_loop() holds
  // the library to that.
  Counts counts;
  // The thread that made the context: the tests call the library from it.
  const std::thread::id owner = std::this_thread::get_id();
  // The call of a loop whose products piece_failing_dgemm() lets fail, or -1.
  // It is set before a build and not changed while the build runs.
  int failing = -1;

  void reset()
  {
    products = 0;
    loops = 0;
    counts.clear();
  }
};

using Loop = int (*)(void*, int, void (*)(void*, int), void*);

// The index of the call of a loop that the current thread is running, or -1
// outside the calls of the loops below.
inline int& running_call()
{
  thread_local int index = -1;
  return index;
}

// One call of the body of a loop.
inline void call(void (*body)(void*, int), void* argument, int index)
{
  running_call() = index;
  body(argument,index);
  running_call() = -1;
}

// Notes a call of a loop with its count.  False for a loop that is called
// from the body of a loop, or from another thread than the one that made the
// context, which the library promises not to do: the loops below then make no
// call and report a failure, so that a build with such a loop does not pass.
inline bool enter_loop(void* context, int count)
{
  if (running_call() != -1) return false;
  if (context != nullptr) {
    Calls& calls = *static_cast<Calls*>(context);
    if (std::this_thread::get_id() != calls.owner) return false;
    ++calls.loops;
    ++calls.counts[count];
  }
  return true;
}

// counting_dgemm() with a counter that several threads may raise.
inline int shared_dgemm(void* context, char transa, char transb, int m, int n, int k, double alpha, const double* a,
                        int lda, const double* b, int ldb, double beta, double* c, int ldc)
{
  if (context != nullptr) ++static_cast<Calls*>(context)->products;
  return counting_dgemm(nullptr,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}

// shared_dgemm() that reports a failure for every product of one call of a
// loop: the call "failing" of the context, which must not be null.  The
// products of the other calls, and those outside a loop, are computed.  A
// product knows the call that makes it from running_call(), so the products
// that fail do not depend on the timing of the threads.
inline int piece_failing_dgemm(void* context, char transa, char transb, int m, int n, int k, double alpha,
                               const double* a, int lda, const double* b, int ldb, double beta, double* c, int ldc)
{
  Calls& calls = *static_cast<Calls*>(context);
  ++calls.products;
  if (running_call() >= 0 && running_call() == calls.failing) return 9;
  return counting_dgemm(nullptr,transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}

// The calls one after the other on the calling thread.
inline int in_order_loop(void* context, int count, void (*body)(void*, int), void* argument)
{
  if (!enter_loop(context,count)) return 1;
  for (int index = 0; index < count; ++index) call(body,argument,index);
  return 0;
}

// The same from the last index to the first.
inline int reversed_loop(void* context, int count, void (*body)(void*, int), void* argument)
{
  if (!enter_loop(context,count)) return 1;
  for (int index = count; index-- > 0;) call(body,argument,index);
  return 0;
}

// Every call on a thread of its own.  A thread that runs its call as soon as
// it exists lets a short call end before the next thread begins, and two
// calls that share memory by mistake then never meet.  So the threads wait
// behind a gate until all threads of the loop exist.  They leave the gate
// one after the other, each after the mutex; a thread that is through waits,
// without sleeping, until all are through, and the calls begin together.
//
// The calls overlap only when the threads have cores to run on at the same
// time.  On one core a short call still runs to its end before the next
// thread is scheduled, and the comparisons of the tests then do not see
// memory that two calls share by mistake: that takes two or more free cores,
// or calls that are long enough to be interrupted.  A build with
// ThreadSanitizer (OPENMX_EXX_THREAD_SANITIZER, "ctest -L threads") reports
// such memory without an overlap, on one core too.  The results of a correct
// library do not depend on any of this.
//
// A thread that cannot be created gives the status 1: the calls that have a
// thread still run, and every thread is joined.  "unavailable" is the index
// of a call whose thread counts as one that cannot be created, or -1.
inline int run_on_threads(int count, void (*body)(void*, int), void* argument, int unavailable)
{
  std::mutex mutex;
  std::condition_variable gate;
  int admitted = -1;              // the number of threads once the gate is open; guarded by the mutex
  std::atomic<int> through{0};    // the threads that have left the gate
  std::vector<std::thread> threads;
  int status = 0;
  try {
    for (int index = 0; index < count; ++index) {
      if (index == unavailable)
        throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
      threads.emplace_back([&mutex,&gate,&admitted,&through,body,argument,index] {
        int expected = 0;
        {
          std::unique_lock<std::mutex> lock(mutex);
          gate.wait(lock,[&admitted] { return admitted >= 0; });
          expected = admitted;
        }
        ++through;
        while (through.load() < expected) std::this_thread::yield();
        call(body,argument,index);
      });
    }
  } catch (...) {
    status = 1;
  }
  {
    const std::lock_guard<std::mutex> lock(mutex);
    admitted = static_cast<int>(threads.size());
  }
  gate.notify_all();
  for (std::thread& thread : threads) thread.join();
  return status;
}

inline int threaded_loop(void* context, int count, void (*body)(void*, int), void* argument)
{
  if (!enter_loop(context,count)) return 1;
  return run_on_threads(count,body,argument,-1);
}

// threaded_loop() on a system that cannot create the thread of the last
// call.
inline int starved_loop(void* context, int count, void (*body)(void*, int), void* argument)
{
  if (!enter_loop(context,count)) return 1;
  return run_on_threads(count,body,argument,count-1);
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
  if (!enter_loop(context,count)) return 1;
  for (int index = 0; index+1 < count; ++index) call(body,argument,index);
  return 0;
}

inline int repeating_loop(void* context, int count, void (*body)(void*, int), void* argument)
{
  const int status = in_order_loop(context,count,body,argument);
  if (status == 0) call(body,argument,0);
  return status;
}

inline int stray_loop(void* context, int count, void (*body)(void*, int), void* argument)
{
  const int status = in_order_loop(context,count,body,argument);
  if (status == 0) {
    call(body,argument,count);
    call(body,argument,-1);
  }
  return status;
}

} // namespace exx_test
