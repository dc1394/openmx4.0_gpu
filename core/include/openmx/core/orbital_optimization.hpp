#pragma once
#include "openmx_core_orbital_optimization.h"
#include <memory>

namespace openmx::core {
// One DFT invocation owns history, Hessian, immutable maps and reusable scratch.
// Numerical kernels use flat tensor/mdspan views and explicit service callbacks.
class OrbitalOptimization {
 public:
  explicit OrbitalOptimization(const OpenMXOrbOptInput&);
  ~OrbitalOptimization();
  OrbitalOptimization(const OrbitalOptimization&) = delete;
  OrbitalOptimization& operator=(const OrbitalOptimization&) = delete;
  OpenMXOrbOptViews views() noexcept;
  double step(int, double, const OpenMXElectronicViews&, const OpenMXLocalBasisViews&,
              const OpenMXOrbOptCallbacks&);
  double gradient(const OpenMXElectronicViews&, const OpenMXLocalBasisViews&,
                  const OpenMXOrbOptCallbacks&);

 private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};
} // namespace openmx::core
