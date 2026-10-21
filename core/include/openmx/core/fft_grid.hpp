#pragma once

#include <array>

namespace openmx::core {

constexpr std::array<int,4> fft_radices() noexcept { return {2,3,5,7}; }

constexpr bool fft_grid_supported(int size) noexcept
{
  if (size < 1) return false;
  for (int radix : fft_radices())
    while (size % radix == 0) size /= radix;
  return size == 1;
}

} // namespace openmx::core
