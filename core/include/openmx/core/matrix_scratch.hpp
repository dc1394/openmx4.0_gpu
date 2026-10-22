#pragma once

#include <openmx/core/block_matrix.hpp>
#include <openmx/core/overwrite_tensor.hpp>
#include "openmx_core_electronic.h"
#include <vector>

namespace openmx::core {

// Copy one history group's shape, without borrowing metadata or copying values.
// The overwrite allocation preserves native scratch's write-before-read contract.
class MatrixScratch {
 public:
  MatrixScratch(const OpenMXElectronicMatrixView& source, std::size_t group);
  MatrixScratch(const MatrixScratch&)=delete;
  MatrixScratch& operator=(const MatrixScratch&)=delete;
  MatrixScratch(MatrixScratch&&)=delete;
  MatrixScratch& operator=(MatrixScratch&&)=delete;
  OpenMXElectronicMatrixView view() noexcept;
  Tensor<double,2>::view_type block(std::size_t component, std::size_t atom, std::size_t neighbor);

 private:
  struct Shape {
    std::vector<std::size_t> components, atoms;
    std::vector<MatrixBlockShape> blocks;
  };
  static Shape copy_shape(const OpenMXElectronicMatrixView&, std::size_t group);
  explicit MatrixScratch(Shape shape);
  std::vector<std::size_t> components_;
  AtomBlockLayout layout_;
  OverwriteTensor<double,1> values_;
  double empty_{};
};

} // namespace openmx::core
