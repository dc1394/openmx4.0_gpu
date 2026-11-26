#pragma once
#include <openmx/core/overwrite_tensor.hpp>
#include "openmx_core_force_hnl_batch.h"

namespace openmx::core {
/* A single device input holds metadata and density. Host-only atom indices
   and force results live separately. Row metadata is neighbour-indexed in
   case 1 and pair-indexed in case 2. */
class ForceHnlBatch {
 public:
  ForceHnlBatch(int kind, std::size_t items, std::size_t rows, std::size_t density_values);
  ForceHnlBatch(const ForceHnlBatch&)=delete;
  ForceHnlBatch& operator=(const ForceHnlBatch&)=delete;
  OpenMXForceHnlBatchView view() noexcept;
 private:
  int kind_;
  std::size_t items_, rows_, density_values_;
  struct InputLayout {
    InputLayout(std::size_t items,std::size_t rows,std::size_t density);
    std::size_t offsets, tags, density, bytes = 0;
  } layout_;
  OverwriteTensor<std::byte,1> input_;
  OverwriteTensor<int,2> atoms_;
  OverwriteTensor<double,2> force_;
};
} // namespace openmx::core
