#pragma once

#include "pao_compat.hpp"
#include "species_compat.hpp"
#include "vps_compat.hpp"
#include "product_compat.hpp"
#include "neutral_grid_compat.hpp"
#include "neutral_grid_metadata.hpp"
#include "vna_matrix_compat.hpp"
#include "electronic_matrix_compat.hpp"
#include "local_basis_compat.hpp"
#include "native_resources.hpp"
#include <openmx_core_mixing.h>
#include <openmx_core_energy.h>
#include <openmx_core_dynamics.h>
#include <openmx_core_chemical.h>
#include <openmx_core_cwf.h>
#include "cwf_observables.hpp"
#include "grid_assembly.hpp"
#include "olg_cache.hpp"
#include <openmx/core/orbital_optimization.hpp>
#include <openmx/core/radial.hpp>

// Private definition shared by C ABI modules. No implicit/default context.
// Declaration order keeps the runtime alive until all calculation storage has
// been destroyed, including when an API entry unwinds after a failure.
struct OpenMXCoreContext {
  explicit OpenMXCoreContext(bool host_only = false) : implementation(host_only) {}
  OpenMXCoreContext(const OpenMXCoreContext&)=delete;
  OpenMXCoreContext& operator=(const OpenMXCoreContext&)=delete;
  OpenMXCoreContext(OpenMXCoreContext&&)=delete;
  OpenMXCoreContext& operator=(OpenMXCoreContext&&)=delete;
  openmx::core::Context implementation;
  std::unique_ptr<openmx::core::compat::SpeciesOwner> species;
  std::unique_ptr<openmx::core::compat::PaoOwner> pao;
  std::unique_ptr<openmx::core::compat::VpsOwner> vps;
  std::unique_ptr<openmx::core::compat::ProductOwner> products;
  std::unique_ptr<openmx::core::compat::NeutralGridOwner> neutral_grid;
  std::unique_ptr<openmx::core::compat::NeutralGridMetadataOwner> neutral_grid_metadata;
  std::unique_ptr<openmx::core::compat::VnaMatrixOwner> vna_matrices;
  std::unique_ptr<openmx::core::compat::ElectronicMatrixOwner> electronic;
  std::unique_ptr<openmx::core::compat::LocalBasisOwner> local_basis;
  std::unique_ptr<openmx::core::OrbitalOptimization> orbital_optimization;
  std::unique_ptr<OpenMXGridWorkspace,GridWorkspaceDelete> grid;
  std::unique_ptr<openmx::core::OlgCache> olg_cache;
  // Fixed-size domain state needs no separate heap/group construction. Keep
  // it alive until all native states borrowing it have drained.
  OpenMXMixingHistory mixing_history{};
  OpenMXBandEnergies band_energies{};
  OpenMXEnergyComponents energy_components{};
  OpenMXThermodynamicEnergies thermodynamic_energies{};
  OpenMXDynamicsState dynamics{};
  OpenMXChemicalPotentials chemical{};
  OpenMXCWFGrids cwf_grids{};
  openmx::core::CwfObservables cwf_observables;
  NativeResourceGroups native_resources;
};
