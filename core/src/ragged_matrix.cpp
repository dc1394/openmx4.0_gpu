#include <openmx/core/ragged_matrix.hpp>
#include <algorithm>

namespace openmx::core {
namespace {
std::size_t checked_elements(std::span<const std::size_t> components,const AtomBlockLayout& layout)
{
  if (components.size()<2 || components.front()!=0 || components.back()!=layout.atoms() ||
      !std::is_sorted(components.begin(),components.end()))
    throw std::invalid_argument("invalid component-atom CSR offsets");
  return layout.stride();
}
}

RaggedMatrix::RaggedMatrix(std::span<const std::size_t> components,
    std::span<const std::size_t> atoms, std::span<const MatrixBlockShape> blocks)
    : components_(components.begin(),components.end()), layout_(atoms,blocks),
      values_({checked_elements(components_,layout_)}) {}

std::size_t RaggedMatrix::index(std::size_t component, std::size_t atom, std::size_t neighbor) const
{
  if (component>=components() || atom>=components_[component+1]-components_[component])
    throw std::out_of_range("matrix component or atom index");
  return layout_.block_index(components_[component]+atom,neighbor);
}

Tensor<double,2>::view_type RaggedMatrix::block(std::size_t component,
    std::size_t atom, std::size_t neighbor)
{
  const auto b=index(component,atom,neighbor);
  const auto r=static_cast<std::size_t>(element(layout_.shapes(),b,0));
  const auto c=static_cast<std::size_t>(element(layout_.shapes(),b,1));
  return Tensor<double,2>::view_type{
      values_.size() ? values_.data()+layout_.block_offsets()[b] : &empty_,r,c};
}

Tensor<double,2>::const_view_type RaggedMatrix::block(std::size_t component,
    std::size_t atom, std::size_t neighbor) const
{
  const auto b=index(component,atom,neighbor);
  const auto r=static_cast<std::size_t>(element(layout_.shapes(),b,0));
  const auto c=static_cast<std::size_t>(element(layout_.shapes(),b,1));
  return Tensor<double,2>::const_view_type{
      values_.size() ? values_.data()+layout_.block_offsets()[b] : &empty_,r,c};
}

} // namespace openmx::core
