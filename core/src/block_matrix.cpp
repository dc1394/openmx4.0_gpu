#include <openmx/core/block_matrix.hpp>
#include <algorithm>

namespace openmx::core {

AtomBlockLayout::AtomBlockLayout(std::span<const std::size_t> atoms,
    std::span<const MatrixBlockShape> blocks)
    : atoms_(atoms.begin(),atoms.end()), offsets_(blocks.size()+1), shapes_({blocks.size(),2})
{
  if (atoms.empty() || atoms.front()!=0 || atoms.back()!=blocks.size() ||
      !std::is_sorted(atoms.begin(),atoms.end()))
    throw std::invalid_argument("invalid atom-block CSR offsets");
  std::size_t count=0;
  for (std::size_t b=0;b<blocks.size();++b) {
    const auto [rows,columns]=blocks[b];
    if (rows<0 || columns<0) throw std::invalid_argument("negative matrix block dimension");
    const auto r=static_cast<std::size_t>(rows),c=static_cast<std::size_t>(columns);
    if (r && c>std::numeric_limits<std::size_t>::max()/r)
      throw std::length_error("matrix block size overflows size_t");
    const auto size=r*c;
    if (size>std::numeric_limits<std::size_t>::max()-count)
      throw std::length_error("matrix block offsets overflow size_t");
    offsets_[b]=count;
    count+=size;
    element(shapes_.view(),b,0)=rows;
    element(shapes_.view(),b,1)=columns;
  }
  offsets_.back()=count;
}

std::size_t AtomBlockLayout::block_index(std::size_t atom, std::size_t neighbor) const
{
  if (atom>=atoms() || neighbor>=atoms_[atom+1]-atoms_[atom])
    throw std::out_of_range("atom or neighbor block index");
  return atoms_[atom]+neighbor;
}

} // namespace openmx::core
