#pragma once

#include <atomic>
#include <cstdlib>
#include <new>

// Included by one translation unit per test executable. Never linked into
// the production library. Allocation is enabled except during explicit probes.
namespace { std::atomic<long> allocations_before_failure{-1}; }
void* operator new(std::size_t bytes)
{
  auto remaining = allocations_before_failure.load();
  if (remaining >= 0) {
    if (remaining == 0) throw std::bad_alloc();
    allocations_before_failure.store(remaining-1);
  }
  if (auto* p = std::malloc(bytes ? bytes : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
