#pragma once
#include <openmx/core/local_basis_state.hpp>
#include "openmx_core_local_basis.h"

namespace openmx::core::compat {
class LocalBasisOwner {
 public:
  explicit LocalBasisOwner(const OpenMXLocalBasisInput&);
  OpenMXLocalBasisViews views() noexcept;

 private:
  std::unique_ptr<LocalBasisState> state_;
};
} // namespace openmx::core::compat
