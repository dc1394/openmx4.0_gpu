#pragma once

#include <openmx/core/block_matrix.hpp>

namespace openmx::core {

// Components may have different atom counts, block shapes and scratch slots.
// Three CSR levels describe [component][atom][neighbor][row,column].
// This also covers history/spin families without padding every block to the
// largest basis. All metadata and values belong to this object.
class RaggedMatrix {
 public:
  RaggedMatrix(std::span<const std::size_t> component_offsets,
      std::span<const std::size_t> atom_offsets,
      std::span<const MatrixBlockShape> blocks);
  RaggedMatrix(const RaggedMatrix&) = delete;
  RaggedMatrix& operator=(const RaggedMatrix&) = delete;
  RaggedMatrix(RaggedMatrix&&) = delete;
  RaggedMatrix& operator=(RaggedMatrix&&) = delete;
  std::size_t components() const noexcept { return components_.size()-1; }
  std::size_t atoms() const noexcept { return layout_.atoms(); }
  std::size_t blocks() const noexcept { return layout_.blocks(); }
  std::size_t elements() const noexcept { return values_.size(); }
  auto component_offsets() const noexcept { return std::span<const std::size_t>(components_); }
  const AtomBlockLayout& layout() const noexcept { return layout_; }
  auto values() noexcept { return values_.view(); }
  auto values() const noexcept { return values_.view(); }
  Tensor<double,2>::view_type block(std::size_t component, std::size_t atom, std::size_t neighbor);
  Tensor<double,2>::const_view_type block(std::size_t component, std::size_t atom, std::size_t neighbor) const;

 private:
  std::size_t index(std::size_t component, std::size_t atom, std::size_t neighbor) const;
  std::vector<std::size_t> components_;
  AtomBlockLayout layout_;
  Tensor<double,1> values_;
  double empty_{};
};

} // namespace openmx::core
