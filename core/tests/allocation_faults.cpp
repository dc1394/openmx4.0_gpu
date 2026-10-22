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

namespace {
void* fault_allocate(std::size_t bytes)
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
void fault_free(void* pointer) noexcept
{
  if (pointer) --live;
  std::free(pointer);
}
}

#if defined(__APPLE__)
// Mach-O's two-level namespace (and ASan's own interposers) can leave calls
// from the shared core bound to libc++ rather than executable operator new.
// Use dyld interposition for test-only allocation faults on Darwin, as the
// sanitizer runtime does: compiler-rt/lib/interception/interception.h.
namespace {
#if defined(__cpp_sized_deallocation)
void fault_free_sized(void* pointer, std::size_t) noexcept { fault_free(pointer); }
using FreeSized = void (*)(void*, std::size_t) noexcept;
#endif
using Allocate = void* (*)(std::size_t);
using Free = void (*)(void*) noexcept;
struct Interpose { const void* replacement; const void* original; };
__attribute__((used, section("__DATA,__interpose")))
const Interpose allocation_interposers[] = {
  {reinterpret_cast<const void*>(&fault_allocate),
   reinterpret_cast<const void*>(static_cast<Allocate>(&::operator new))},
  {reinterpret_cast<const void*>(&fault_allocate),
   reinterpret_cast<const void*>(static_cast<Allocate>(&::operator new[]))},
  {reinterpret_cast<const void*>(&fault_free),
   reinterpret_cast<const void*>(static_cast<Free>(&::operator delete))},
  {reinterpret_cast<const void*>(&fault_free),
   reinterpret_cast<const void*>(static_cast<Free>(&::operator delete[]))},
#if defined(__cpp_sized_deallocation)
  {reinterpret_cast<const void*>(&fault_free_sized),
   reinterpret_cast<const void*>(static_cast<FreeSized>(&::operator delete))},
  {reinterpret_cast<const void*>(&fault_free_sized),
   reinterpret_cast<const void*>(static_cast<FreeSized>(&::operator delete[]))},
#endif
};
}
#else
void* operator new(std::size_t bytes) { return fault_allocate(bytes); }
void operator delete(void* pointer) noexcept { fault_free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
// Array allocation has its own interceptor under ASan. Route it explicitly
// through the same counter so overwrite buffers receive the same fault/leak
// coverage as vector storage in every compiler/sanitizer configuration.
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
#endif
