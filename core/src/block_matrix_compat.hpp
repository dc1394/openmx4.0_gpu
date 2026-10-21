#pragma once

#include <openmx/core/block_matrix.hpp>

namespace openmx::core::compat {

// A temporary bridge for native kernels and unmigrated C numerical loops.
// Four flat pointer tables replace thousands of individually owned rows.
// Addresses remain stable until this calculation/topology owner is destroyed.
template<class Scalar>
class BlockMatrixRows {
 public:
  explicit BlockMatrixRows(AtomBlockMatrix<Scalar>& matrix)
      : components_(matrix.values().extent(0)) {
    const auto components=components_.size();
    const auto& shape=matrix.layout();
    // Tensor checks the component*element product. The smaller pointer tables
    // also need independent checked products when blocks have zero width.
    atoms_.resize(product(components,shape.atoms()));
    blocks_.resize(product(components,shape.blocks()));
    std::size_t rows=0;
    for (std::size_t b=0;b<shape.blocks();++b) {
      const auto count=static_cast<std::size_t>(element(shape.shapes(),b,0));
      if (count>std::numeric_limits<std::size_t>::max()-rows)
        throw std::length_error("matrix row-pointer count overflow");
      rows+=count;
    }
    rows_.resize(product(components,rows));
    std::size_t r=0;
    for (std::size_t c=0;c<components;++c) {
      components_[c]=shape.atoms() ? atoms_.data()+c*shape.atoms() : nullptr;
      for (std::size_t a=0;a<shape.atoms();++a) {
        const auto first=shape.atom_offsets()[a],last=shape.atom_offsets()[a+1];
        atoms_[c*shape.atoms()+a]=last>first ? blocks_.data()+c*shape.blocks()+first : nullptr;
        for (auto b=first;b<last;++b) {
          const auto block=matrix.block(c,a,b-first);
          const auto height=block.extent(0),width=block.extent(1);
          blocks_[c*shape.blocks()+b]=height ? rows_.data()+r : nullptr;
          for (std::size_t i=0;i<height;++i)
            rows_[r++]=width ? block.data_handle()+i*width : &empty_;
        }
      }
    }
  }
  BlockMatrixRows(const BlockMatrixRows&) = delete;
  BlockMatrixRows& operator=(const BlockMatrixRows&) = delete;
  Scalar***** data() noexcept { return components_.data(); }

 private:
  static std::size_t product(std::size_t a,std::size_t b) {
    if (a && b>std::numeric_limits<std::size_t>::max()/a)
      throw std::length_error("matrix pointer-table count overflow");
    return a*b;
  }
  std::vector<Scalar****> components_;
  std::vector<Scalar***> atoms_;
  std::vector<Scalar**> blocks_;
  std::vector<Scalar*> rows_;
  Scalar empty_{};
};

// Ordinary owners compose the same borrowed adapter as invocation-owned Force
// matrices; every numerical value still belongs to the AtomBlockMatrix.
template<class Scalar>
class BlockMatrixOwner {
 public:
  BlockMatrixOwner(AtomBlockLayout layout,std::size_t components)
      : matrix_(std::move(layout),components),rows_(matrix_) {}
  auto& matrix() noexcept {return matrix_;}
  const auto& matrix() const noexcept {return matrix_;}
  Scalar***** legacy() noexcept {return rows_.data();}
 private:
  AtomBlockMatrix<Scalar> matrix_;
  BlockMatrixRows<Scalar> rows_;
};

} // namespace openmx::core::compat
