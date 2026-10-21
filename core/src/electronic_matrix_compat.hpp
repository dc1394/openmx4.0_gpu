#pragma once

#include <openmx/core/electronic_matrices.hpp>
#include "openmx_core_electronic.h"

namespace openmx::core::compat {

// Adapter pointers own no numeric rows. One set of flat pointer tables per
// enabled family bridges old kernels until their consumers use CSR/mdspan.
class ElectronicMatrixOwner {
 public:
  explicit ElectronicMatrixOwner(const OpenMXElectronicMatrixInput& input);
  OpenMXElectronicViews views() noexcept;

 private:
  struct Adapters {
    std::vector<double*****> groups;
    std::vector<double****> components;
    std::vector<double***> atoms;
    std::vector<double**> blocks;
    std::vector<double*> rows;
    double empty{};
    void bind(RaggedMatrix& matrix,std::size_t group_count,std::size_t channels);
  };
  std::unique_ptr<ElectronicMatrices> matrices_;
  std::array<Adapters,static_cast<std::size_t>(ElectronicField::count)> adapters_;
};

} // namespace openmx::core::compat
