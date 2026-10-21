#pragma once
#include <openmx/core/block_matrix.hpp>
#include <memory>
#include "openmx_core_force.h"

namespace openmx::core {

// Physical topology, padding and halo atoms have explicit, immutable layout.
// Borrowed legacy rows are confined to the C adapter, not numeric ownership.
class ForceMatrices {
 public:
  ForceMatrices(AtomBlockLayout layout,int spin,bool imaginary,std::size_t local_atoms);
  auto& density() noexcept {return density_;}
  const auto& density() const noexcept {return density_;}
  auto* imaginary() noexcept {return imaginary_.get();}
  auto forces() noexcept {return forces_.view();}
 private:
  AtomBlockMatrix<double> density_;
  std::unique_ptr<AtomBlockMatrix<double>> imaginary_;
  Tensor<double,2> forces_;
};

AtomBlockLayout force_layout(const OpenMXForceInput& input);
void force_overlap(const OpenMXElectronicMatrixView& overlap,
    const OpenMXElectronicMatrixView& energy_density,int spin,int atom,int neighbors,
    std::span<double,3> force);
void force_spin_trace(const OpenMXForceSpinTrace& input,std::span<double,3> force);
} // namespace openmx::core
