#include <openmx/core/product_storage.hpp>
#include <climits>

namespace openmx::core {

ProductLayout::ProductLayout(std::span<const int> multiplicities)
{
  // The common Bessel implementation supports l <= 29; products use 2*l.
  if (multiplicities.empty() || multiplicities.size() > 15)
    throw std::invalid_argument("invalid product angular capacity");
  std::size_t orbitals = 0, count = 0;
  for (int n : multiplicities) {
    if (n < 0 || static_cast<std::size_t>(n) > INT_MAX-orbitals)
      throw std::invalid_argument("invalid product multiplicity");
    orbitals += n;
  }
  for (std::size_t l1 = 0; l1 < multiplicities.size(); ++l1)
    for (std::size_t l2 = l1; l2 < multiplicities.size(); ++l2) {
      // Check before multiplying; channel indices also fit the device int ABI.
      const auto n1 = static_cast<std::size_t>(multiplicities[l1]);
      const auto n2 = static_cast<std::size_t>(multiplicities[l2]);
      const auto angular = 2*l2+1;
      if (n1 && n2 > (INT_MAX-count)/angular/n1)
        throw std::length_error("too many orbital-product channels");
      count += n1*n2*angular;
      pairs_ += n1*n2;
    }
  multiplicities_.assign(multiplicities.begin(),multiplicities.end());
  orbital_offsets_.push_back(0);
  for (int n : multiplicities_) orbital_offsets_.push_back(orbital_offsets_.back()+n);
  // At least half the ordered pairs are stored, so count bounds this product.
  if (orbitals && orbitals > rows_.max_size()/orbitals)
    throw std::length_error("product layout exceeds allocator size");
  rows_.assign(orbitals*orbitals,std::numeric_limits<std::size_t>::max());
  channels_.reserve(count);
  for (std::size_t l1 = 0; l1 < multiplicities_.size(); ++l1)
    for (int m1 = 0; m1 < multiplicities_[l1]; ++m1)
      for (std::size_t l2 = l1; l2 < multiplicities_.size(); ++l2)
        for (int m2 = 0; m2 < multiplicities_[l2]; ++m2) {
          const auto left = orbital_offsets_[l1]+m1;
          const auto right = orbital_offsets_[l2]+m2;
          rows_[left*orbitals+right] = channels_.size();
          for (std::size_t l = 0; l <= 2*l2; ++l)
            channels_.push_back({static_cast<int>(left),static_cast<int>(right),static_cast<int>(l)});
        }
}

std::size_t ProductLayout::row(std::size_t l1, std::size_t m1, std::size_t l2, std::size_t m2) const
{
  if (l1 >= multiplicities_.size() || l2 >= multiplicities_.size() || l1 > l2 ||
      m1 >= static_cast<std::size_t>(multiplicities_[l1]) ||
      m2 >= static_cast<std::size_t>(multiplicities_[l2]))
    throw std::out_of_range("inactive or invalid orbital-product block");
  return rows_[(orbital_offsets_[l1]+m1)*orbital_count()+orbital_offsets_[l2]+m2];
}

ProductStorage::Plan ProductStorage::plan(std::span<const ProductSpecies> species, std::size_t wavevectors)
{
  if (species.empty() || !wavevectors)
    throw std::invalid_argument("empty product storage shape");
  if (wavevectors > std::numeric_limits<std::size_t>::max()-2)
    throw std::length_error("product padding overflows size_t");
  Plan result{{},{0},wavevectors+2};
  for (const auto& s : species) {
    result.layouts.emplace_back(s.multiplicities);
    const auto rows = result.layouts.back().channels().size();
    if (rows > std::numeric_limits<std::size_t>::max()-result.offsets.back())
      throw std::length_error("product row count overflows size_t");
    result.offsets.push_back(result.offsets.back()+rows);
  }
  return result;
}

ProductStorage::ProductStorage(std::span<const ProductSpecies> species, std::size_t wavevectors)
  : ProductStorage(plan(species,wavevectors)) {}
ProductStorage::ProductStorage(Plan plan)
  : layouts_(std::move(plan.layouts)), species_rows_(std::move(plan.offsets)),
    values_({species_rows_.back(),plan.stride}) {}

Tensor<double,2>::view_type ProductStorage::species(std::size_t species)
{
  const auto rows = layout(species).channels().size();
  const auto offset = species_rows_[species]*values_.extent(1);
  return Tensor<double,2>::view_type{offset ? values_.data()+offset : values_.data(),rows,values_.extent(1)};
}
Tensor<double,2>::const_view_type ProductStorage::species(std::size_t species) const
{
  const auto rows = layout(species).channels().size();
  const auto offset = species_rows_[species]*values_.extent(1);
  return Tensor<double,2>::const_view_type{offset ? values_.data()+offset : values_.data(),rows,values_.extent(1)};
}
Tensor<double,2>::view_type ProductStorage::block(std::size_t species, std::size_t l1,
    std::size_t m1, std::size_t l2, std::size_t m2)
{
  const auto row = species_rows_.at(species)+layout(species).row(l1,m1,l2,m2);
  return Tensor<double,2>::view_type{values_.data()+row*values_.extent(1),2*l2+1,values_.extent(1)};
}
Tensor<double,2>::const_view_type ProductStorage::block(std::size_t species, std::size_t l1,
    std::size_t m1, std::size_t l2, std::size_t m2) const
{
  const auto row = species_rows_.at(species)+layout(species).row(l1,m1,l2,m2);
  return Tensor<double,2>::const_view_type{values_.data()+row*values_.extent(1),2*l2+1,values_.extent(1)};
}

} // namespace openmx::core
