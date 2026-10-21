#pragma once

#include <openmx/core/tensor.hpp>
#include <span>

namespace openmx::core {

// Indices into the sampled PAO rows, and the Bessel angular momentum.
struct ProductChannel { int left, right, angular; };

// One species: (l1,m1,l2,m2) identifies a block of (2*l2+1) rows.
// Only l1 <= l2 is stored, in the original restart-file order.
class ProductLayout {
 public:
  explicit ProductLayout(std::span<const int> multiplicities);
  std::span<const int> multiplicities() const noexcept { return multiplicities_; }
  std::span<const ProductChannel> channels() const noexcept { return channels_; }
  std::size_t orbital_count() const noexcept { return orbital_offsets_.empty() ? 0 : orbital_offsets_.back(); }
  std::size_t pair_count() const noexcept { return pairs_; }
  std::span<const std::size_t> orbital_offsets() const noexcept { return orbital_offsets_; }
  std::span<const std::size_t> pair_rows() const noexcept { return rows_; }
  std::size_t row(std::size_t l1, std::size_t m1, std::size_t l2, std::size_t m2) const;

  ProductLayout(const ProductLayout&) = default;
  ProductLayout(ProductLayout&&) noexcept = default;
  ProductLayout& operator=(const ProductLayout&) = delete;
  ProductLayout& operator=(ProductLayout&&) = delete;

 private:
  std::vector<int> multiplicities_;
  std::vector<std::size_t> orbital_offsets_, rows_;
  std::vector<ProductChannel> channels_;
  std::size_t pairs_ = 0;
};

struct ProductSpecies { std::span<const int> multiplicities; };

// All species share one contiguous numeric allocation. Layout metadata and
// species row offsets describe the ragged blocks; no numeric pointer tree.
class ProductStorage {
 public:
  ProductStorage(std::span<const ProductSpecies> species, std::size_t wavevectors);
  const ProductLayout& layout(std::size_t species) const { return layouts_.at(species); }
  std::size_t species_count() const noexcept { return layouts_.size(); }
  std::span<const std::size_t> species_rows() const noexcept { return species_rows_; }
  auto values() { return values_.view(); }
  auto values() const { return values_.view(); }
  Tensor<double,2>::view_type species(std::size_t species);
  Tensor<double,2>::const_view_type species(std::size_t species) const;
  Tensor<double,2>::view_type block(std::size_t species, std::size_t l1,
      std::size_t m1, std::size_t l2, std::size_t m2);
  Tensor<double,2>::const_view_type block(std::size_t species, std::size_t l1,
      std::size_t m1, std::size_t l2, std::size_t m2) const;

  ProductStorage(const ProductStorage&) = delete;
  ProductStorage& operator=(const ProductStorage&) = delete;
  ProductStorage(ProductStorage&&) = delete;
  ProductStorage& operator=(ProductStorage&&) = delete;

 private:
  struct Plan {
    std::vector<ProductLayout> layouts;
    std::vector<std::size_t> offsets;
    std::size_t stride;
  };
  static Plan plan(std::span<const ProductSpecies> species, std::size_t wavevectors);
  explicit ProductStorage(Plan plan);
  std::vector<ProductLayout> layouts_;
  std::vector<std::size_t> species_rows_;
  Tensor<double,2> values_;
};

} // namespace openmx::core
