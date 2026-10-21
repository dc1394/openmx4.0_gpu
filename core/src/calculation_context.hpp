#pragma once

#include "pao_compat.hpp"
#include "species_compat.hpp"
#include "vps_compat.hpp"
#include "product_compat.hpp"
#include "vna_matrix_compat.hpp"
#include "electronic_matrix_compat.hpp"
#include "local_basis_compat.hpp"
#include "native_resources.hpp"
#include <openmx/core/orbital_optimization.hpp>
#include <openmx/core/radial.hpp>

// Private definition shared by C ABI modules. No implicit/default context.
// Declaration order keeps the runtime alive until all calculation storage has
// been destroyed, including when an API entry unwinds after a failure.
struct OpenMXCoreContext {
  explicit OpenMXCoreContext(bool host_only = false) : implementation(host_only) {}
  openmx::core::Context implementation;
  std::unique_ptr<openmx::core::compat::SpeciesOwner> species;
  std::unique_ptr<openmx::core::compat::PaoOwner> pao;
  std::unique_ptr<openmx::core::compat::VpsOwner> vps;
  std::unique_ptr<openmx::core::compat::ProductOwner> products;
  std::unique_ptr<openmx::core::compat::VnaMatrixOwner> vna_matrices;
  std::unique_ptr<openmx::core::compat::ElectronicMatrixOwner> electronic;
  std::unique_ptr<openmx::core::compat::LocalBasisOwner> local_basis;
  std::unique_ptr<openmx::core::OrbitalOptimization> orbital_optimization;
  NativeResourceGroups native_resources;
};
