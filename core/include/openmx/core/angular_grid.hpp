#pragma once
#include <openmx/core/overwrite_tensor.hpp>
#include "openmx_core_angular_grid.h"
#include <vector>

namespace openmx::core {
class AngularGrid {
 public:
  static std::size_t validate(int points);
  explicit AngularGrid(int points);
  std::size_t capacity() const noexcept { return storage_.extent(0); }
  void rebind(int points);
  OverwriteTensor<double,2>::view_type grid() noexcept;
  OpenMXSolverRealTensor view() noexcept;
  double** legacy_rows() noexcept { return rows_.data(); }
  AngularGrid(const AngularGrid&)=delete;
  AngularGrid& operator=(const AngularGrid&)=delete;
 private:
  std::size_t points_;
  OverwriteTensor<double,2> storage_;
  std::vector<double*> rows_;
};
}
