#pragma once

namespace openmx::test {
void fail_allocation_after(long successes) noexcept;
long remaining_allocations() noexcept;
long live_allocations() noexcept;
void fill_allocations_with(int byte) noexcept;
}
