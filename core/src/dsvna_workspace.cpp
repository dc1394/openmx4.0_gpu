#include <openmx/core/dsvna_workspace.hpp>
#include <algorithm>

namespace openmx::core {
namespace {
std::size_t radial_capacity(const VnaPreparation& preparation)
{
  const auto& basis = preparation.basis();
  const auto offsets = basis.offsets();
  const std::size_t channels = preparation.shape().maximum_angular+1;
  std::size_t capacity = 0;
  for (std::size_t s = 0; s < basis.orbitals().size(); ++s)
    capacity = std::max(capacity,offsets[(s+1)*channels]-offsets[s*channels]);
  return capacity;
}
std::size_t bessel_orders(const VnaPreparation& preparation)
{
  return 2*std::max(preparation.shape().maximum_angular,preparation.shape().projector_angular)+1;
}
} // namespace

DsvnaWorkspace::DsvnaWorkspace(const VnaPreparation& preparation)
  : angular_({4,static_cast<std::size_t>(*std::max_element(
        preparation.basis().orbitals().begin(),preparation.basis().orbitals().end())),
        static_cast<std::size_t>(preparation.projector_components())}),
    conversion_({4,static_cast<std::size_t>(2*preparation.shape().maximum_angular+1),
                   static_cast<std::size_t>(2*preparation.shape().projector_angular+1)}),
    sums_({2,radial_capacity(preparation),static_cast<std::size_t>(preparation.projector_count())}),
    bessel_({2,bessel_orders(preparation),static_cast<std::size_t>(preparation.shape().points)}),
    radial_({2,static_cast<std::size_t>(preparation.shape().points)}),
    scratch_({2,bessel_orders(preparation)+2})
{}

} // namespace openmx::core
