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
DsvnaSimdShape simd_shape(const VnaPreparation& preparation)
{
  const auto& shape=preparation.shape();
  const auto& basis=preparation.basis();
  const int l0=std::max(0,*std::max_element(basis.maximum_angular().begin(),basis.maximum_angular().end()));
  return {shape.points,l0,shape.projector_angular,
      2*std::max(l0,shape.projector_multiplicity?shape.projector_angular:0),preparation.projector_count(),
      radial_capacity(preparation),std::size_t(*std::max_element(basis.orbitals().begin(),basis.orbitals().end())),
      std::size_t(preparation.projector_components())};
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
    scratch_({2,bessel_orders(preparation)+2}),simd_shape_(simd_shape(preparation))
{}

DsvnaSimdWorkspace& DsvnaWorkspace::simd(bool angular)
{
  if(simd_ && simd_->angular()!=angular)
    throw std::invalid_argument("DSVNA SIMD workspace options cannot change");
  if(!simd_)simd_=std::make_unique<DsvnaSimdWorkspace>(simd_shape_,angular);
  return *simd_;
}

} // namespace openmx::core
