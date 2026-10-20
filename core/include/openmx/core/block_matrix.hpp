#pragma once

#include <openmx/core/tensor.hpp>
#include <span>

namespace openmx::core {

struct MatrixBlockShape { int rows, columns; };

// An immutable CSR layout. Atom zero and communication scratch atoms, when
// present, are ordinary entries in the supplied layout, not hidden globals.
class AtomBlockLayout {
 public:
  AtomBlockLayout(std::span<const std::size_t> atom_offsets,
      std::span<const MatrixBlockShape> blocks);
  AtomBlockLayout(const AtomBlockLayout&) = default;
  AtomBlockLayout(AtomBlockLayout&&) noexcept = default;
  AtomBlockLayout& operator=(const AtomBlockLayout&) = delete;
  AtomBlockLayout& operator=(AtomBlockLayout&&) = delete;
  std::size_t atoms() const noexcept { return atoms_.empty() ? 0 : atoms_.size()-1; }
  std::size_t blocks() const noexcept { return offsets_.empty() ? 0 : offsets_.size()-1; }
  std::size_t stride() const noexcept { return offsets_.empty() ? 0 : offsets_.back(); }
  auto atom_offsets() const noexcept { return std::span<const std::size_t>(atoms_); }
  auto block_offsets() const noexcept { return std::span<const std::size_t>(offsets_); }
  auto shapes() const { return shapes_.view(); }
  std::size_t block_index(std::size_t atom, std::size_t neighbor) const;

 private:
  std::vector<std::size_t> atoms_, offsets_;
  Tensor<int,2> shapes_;
};

// One allocation holds every component and every variable-sized block.
// Numeric consumers use mdspan blocks; legacy pointer adapters own no values.
template<class Scalar>
class AtomBlockMatrix {
 public:
  AtomBlockMatrix(AtomBlockLayout layout, std::size_t components)
      : layout_(std::move(layout)), values_({components,layout_.stride()}) {
    if (!components) throw std::invalid_argument("matrix requires a component");
  }
  AtomBlockMatrix(const AtomBlockMatrix&) = delete;
  AtomBlockMatrix& operator=(const AtomBlockMatrix&) = delete;
  AtomBlockMatrix(AtomBlockMatrix&&) = delete;
  AtomBlockMatrix& operator=(AtomBlockMatrix&&) = delete;
  const AtomBlockLayout& layout() const noexcept { return layout_; }
  auto values() { return values_.view(); }
  auto values() const { return values_.view(); }
  auto block(std::size_t component, std::size_t atom, std::size_t neighbor) {
    const auto index=checked_block(component,atom,neighbor);
    return typename Tensor<Scalar,2>::view_type{
        address(component,index),static_cast<std::size_t>(element(layout_.shapes(),index,0)),
        static_cast<std::size_t>(element(layout_.shapes(),index,1))};
  }
  auto block(std::size_t component, std::size_t atom, std::size_t neighbor) const {
    const auto index=checked_block(component,atom,neighbor);
    return typename Tensor<Scalar,2>::const_view_type{
        address(component,index),static_cast<std::size_t>(element(layout_.shapes(),index,0)),
        static_cast<std::size_t>(element(layout_.shapes(),index,1))};
  }

 private:
  std::size_t checked_block(std::size_t component, std::size_t atom, std::size_t neighbor) const {
    if (component>=values_.extent(0)) throw std::out_of_range("matrix component");
    return layout_.block_index(atom,neighbor);
  }
  Scalar* address(std::size_t component, std::size_t index) {
    return values_.size() ? values_.data()+component*layout_.stride()+layout_.block_offsets()[index] : &empty_;
  }
  const Scalar* address(std::size_t component, std::size_t index) const {
    return values_.size() ? values_.data()+component*layout_.stride()+layout_.block_offsets()[index] : &empty_;
  }
  AtomBlockLayout layout_;
  Tensor<Scalar,2> values_;
  Scalar empty_{};
};

} // namespace openmx::core
