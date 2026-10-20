#pragma once

#include "pao_compat.hpp"
#include "vps_compat.hpp"
#include "product_compat.hpp"
#include "vna_matrix_compat.hpp"
#include <openmx/core/radial.hpp>

// Private definition shared by C ABI modules. No implicit/default context.
// Declaration order keeps the runtime alive until all calculation storage has
// been destroyed, including when an API entry unwinds after a failure.
struct OpenMXCoreContext {
  openmx::core::Context implementation;
  std::unique_ptr<openmx::core::compat::PaoOwner> pao;
  std::unique_ptr<openmx::core::compat::VpsOwner> vps;
  std::unique_ptr<openmx::core::compat::ProductOwner> products;
  std::unique_ptr<openmx::core::compat::VnaMatrixOwner> vna_matrices;
};
