#pragma once
#include <openmx/core/overwrite_tensor.hpp>
#include "openmx_core_force_hnl_batch.h"

namespace openmx::core {
/* Six contiguous allocations replace the ten/eleven independent native
   buffers. Row metadata is neighbour-indexed in case 1 and pair-indexed in
   case 2. All aliases are derived from the owner at the callback boundary. */
class ForceHnlBatch {
 public:
  ForceHnlBatch(int kind, std::size_t items, std::size_t rows, std::size_t density_values);
  ForceHnlBatch(const ForceHnlBatch&)=delete;
  ForceHnlBatch& operator=(const ForceHnlBatch&)=delete;
  OpenMXForceHnlBatchView view() noexcept;
 private:
  int kind_;
  std::size_t items_, rows_, density_values_;
  OverwriteTensor<OpenMXForcePairItem,1> item_;
  OverwriteTensor<int,2> atoms_, tags_;
  OverwriteTensor<std::size_t,2> offsets_;
  OverwriteTensor<double,1> density_;
  OverwriteTensor<double,2> force_;
};
} // namespace openmx::core
