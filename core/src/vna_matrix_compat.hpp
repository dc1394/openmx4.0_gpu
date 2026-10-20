#pragma once

#include "block_matrix_compat.hpp"
#include "openmx_core_vna_matrix.h"
#include <memory>

namespace openmx::core::compat {

class VnaMatrixOwner {
 public:
  explicit VnaMatrixOwner(const OpenMXVnaMatrixInput& input);
  OpenMXVnaMatrixViews views() noexcept;

 private:
  std::unique_ptr<BlockMatrixOwner<double>> potential_, one_center_, mirrored_;
  std::unique_ptr<BlockMatrixOwner<float>> projectors_;
  std::unique_ptr<BlockMatrixOwner<double>> contracted_one_center_, contracted_mirrored_;
  std::unique_ptr<BlockMatrixOwner<float>> contracted_projectors_;
};

} // namespace openmx::core::compat
