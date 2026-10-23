#pragma once
#include <openmx_core_mixing.h>

namespace openmx::core {
// Allocation-free algorithms operating on a caller's calculation-owned POD.
void shift_mixing_history(OpenMXMixingHistory&) noexcept;
void adapt_mixing_history(OpenMXMixingHistory&, OpenMXMixingPolicy,
    double minimum, double maximum, const OpenMXMixingScalarOperations&);
} // namespace openmx::core
