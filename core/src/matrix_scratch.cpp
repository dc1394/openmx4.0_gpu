#include <openmx/core/matrix_scratch.hpp>
#include <algorithm>

namespace openmx::core {
namespace {
void require(bool valid,const char* message)
{ if(!valid)throw std::invalid_argument(message); }
void metadata_size(std::size_t count,std::size_t bytes)
{
  if(count>=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes)
    throw std::length_error("scratch matrix metadata exceeds maximum size");
}
}

MatrixScratch::Shape MatrixScratch::copy_shape(const OpenMXElectronicMatrixView& source,std::size_t group)
{
  require(source.groups && source.channels && group<source.groups,"invalid scratch matrix group");
  require(source.groups<=std::numeric_limits<std::size_t>::max()/source.channels &&
      source.components==source.groups*source.channels,"invalid scratch matrix component count");
  metadata_size(source.components,sizeof(std::size_t));
  metadata_size(source.atoms,sizeof(std::size_t));
  metadata_size(source.blocks,2*sizeof(int));
  require(source.component_offsets && source.atom_offsets && (!source.blocks || source.shapes),
      "missing scratch matrix shape");
  const auto* components=source.component_offsets;
  const auto* atoms=source.atom_offsets;
  require(components[0]==0 && components[source.components]==source.atoms &&
      std::is_sorted(components,components+source.components+1),"invalid scratch component offsets");
  require(atoms[0]==0 && atoms[source.atoms]==source.blocks &&
      std::is_sorted(atoms,atoms+source.atoms+1),"invalid scratch atom offsets");
  const auto first=group*source.channels,last=first+source.channels;
  const auto a0=components[first],a1=components[last],b0=atoms[a0],b1=atoms[a1];
  Shape shape;
  shape.components.reserve(source.channels+1);
  shape.atoms.reserve(a1-a0+1);
  shape.blocks.reserve(b1-b0);
  for(auto c=first;c<=last;++c)shape.components.push_back(components[c]-a0);
  for(auto a=a0;a<=a1;++a)shape.atoms.push_back(atoms[a]-b0);
  for(auto b=b0;b<b1;++b)shape.blocks.push_back({source.shapes[2*b],source.shapes[2*b+1]});
  return shape;
}

MatrixScratch::MatrixScratch(const OpenMXElectronicMatrixView& source,std::size_t group)
    :MatrixScratch(copy_shape(source,group)) {}
MatrixScratch::MatrixScratch(Shape shape)
    :components_(std::move(shape.components)),layout_(shape.atoms,shape.blocks),values_({layout_.stride()}) {}

OpenMXElectronicMatrixView MatrixScratch::view() noexcept
{
  const auto channels=components_.size()-1;
  return {1,channels,channels,layout_.atoms(),layout_.blocks(),values_.size(),components_.data(),
      layout_.atom_offsets().data(),layout_.block_offsets().data(),layout_.shapes().data_handle(),
      values_.size()?values_.data():&empty_};
}

Tensor<double,2>::view_type MatrixScratch::block(std::size_t component,std::size_t atom,std::size_t neighbor)
{
  if(component>=components_.size()-1 || atom>=components_[component+1]-components_[component])
    throw std::out_of_range("scratch matrix component or atom");
  const auto b=layout_.block_index(components_[component]+atom,neighbor);
  return Tensor<double,2>::view_type{values_.size()?values_.data()+layout_.block_offsets()[b]:&empty_,
      static_cast<std::size_t>(element(layout_.shapes(),b,0)),
      static_cast<std::size_t>(element(layout_.shapes(),b,1))};
}

} // namespace openmx::core
