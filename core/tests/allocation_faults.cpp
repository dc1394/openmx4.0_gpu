#include "allocation_faults.hpp"
#include <atomic>
#include <cstdlib>
#include <new>

// Compile this test-only interposition as ordinary C++, even for CUDA/HIP
// tests: device allocation operators must retain their native semantics.
namespace {
// Fault injection is controlled by a single test thread. Allocation counting
// also supports independent owners being exercised concurrently.
std::atomic<long> countdown{-1};
std::atomic<long> live{0};
}
namespace openmx::test {
void fail_allocation_after(long successes) noexcept { countdown = successes; }
long remaining_allocations() noexcept { return countdown; }
long live_allocations() noexcept { return live; }
}

void* operator new(std::size_t bytes)
{
  if (countdown == 0) {
    countdown = -1;
    throw std::bad_alloc();
  }
  if (countdown > 0) --countdown;
  if (void* pointer = std::malloc(bytes ? bytes : 1)) {
    ++live;
    return pointer;
  }
  throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept
{
  if (pointer) --live;
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
