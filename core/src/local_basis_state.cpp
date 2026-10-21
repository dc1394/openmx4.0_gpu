#include <openmx/core/local_basis_state.hpp>
#include <algorithm>
#include <climits>
#include <numeric>

namespace openmx::core {
namespace {
void validate(const LocalBasisTopology& t, const LocalBasisOptions& o)
{
  if ((o.spin!=0 && o.spin!=1 && o.spin!=3) || o.max_orbitals<0 || o.max_coefficients<0)
    throw std::invalid_argument("invalid local basis options");
  if (t.atom_species.empty() || t.local_to_global.empty() ||
      t.first_to_global.size()<t.local_to_global.size() || t.extended_atoms<t.first_to_global.size() ||
      t.orbitals.size()!=t.contracted_orbitals.size() ||
      t.neighbor_offsets.size()!=t.atom_species.size()+1 || t.neighbor_offsets.front()!=0 ||
      t.neighbor_offsets.back()!=t.neighbor_atoms.size() ||
      !std::is_sorted(t.neighbor_offsets.begin(),t.neighbor_offsets.end()))
    throw std::invalid_argument("invalid local basis topology");
  for (std::size_t s=0;s<t.orbitals.size();++s)
    if (t.orbitals[s]<0 || t.orbitals[s]>o.max_orbitals || t.contracted_orbitals[s]<0 ||
        t.contracted_orbitals[s]>t.orbitals[s])
      throw std::invalid_argument("invalid local basis species dimensions");
}

std::size_t species(const LocalBasisTopology& t, int atom)
{
  if (atom<1 || static_cast<std::size_t>(atom)>=t.atom_species.size())
    throw std::invalid_argument("invalid local basis atom index");
  const int s=t.atom_species[atom];
  if (s<0 || static_cast<std::size_t>(s)>=t.orbitals.size())
    throw std::invalid_argument("invalid local basis species index");
  return static_cast<std::size_t>(s);
}

enum class Shape { square, density, inverse_overlap };
AtomBlockLayout layout(const LocalBasisTopology& t,std::span<const int> map,Shape kind)
{
  std::vector<std::size_t> offsets(map.size()+1);
  std::iota(offsets.begin(),offsets.end(),std::size_t{0});
  std::vector<MatrixBlockShape> blocks(map.size(),{1,1});
  for (std::size_t a=1;a<map.size();++a) {
    const int gc=map[a];
    const auto s=species(t,gc);
    int rows=t.orbitals[s],columns=rows;
    if (kind!=Shape::square) {
      const auto begin=t.neighbor_offsets[gc],end=t.neighbor_offsets[gc+1];
      if (begin==end) throw std::invalid_argument("missing local basis neighbor row");
      if (kind==Shape::density) columns=t.orbitals[species(t,t.neighbor_atoms[begin])];
      else {
        // Legacy DC storage reserves three extra rows, even without contraction.
        rows=3;
        for (auto h=begin;h<end;++h) {
          const int n=t.contracted_orbitals[species(t,t.neighbor_atoms[h])];
          if (n>INT_MAX-rows) throw std::length_error("local overlap dimension overflow");
          rows+=n;
        }
        columns=rows;
      }
    }
    blocks[a]={rows,columns};
  }
  return {offsets,blocks};
}
}

LocalBasisState::LocalBasisState(const LocalBasisTopology& t,LocalBasisOptions o)
{
  validate(t,o);
  const auto spin=static_cast<std::size_t>(o.spin+1);
  if (o.interactions) {
    real_[0]=std::make_unique<AtomBlockMatrix<double>>(layout(t,t.local_to_global,Shape::density),2*spin);
    real_[1]=std::make_unique<AtomBlockMatrix<double>>(layout(t,t.first_to_global,Shape::square),spin);
    if (o.spin==3) {
      complex_[0]=std::make_unique<AtomBlockMatrix<dcomplex>>(layout(t,t.local_to_global,Shape::square),8);
      complex_[1]=std::make_unique<AtomBlockMatrix<dcomplex>>(layout(t,t.first_to_global,Shape::square),4);
    }
  }
  if (o.inverse_overlap)
    real_[2]=std::make_unique<AtomBlockMatrix<double>>(layout(t,t.local_to_global,Shape::inverse_overlap),1);
  const auto n=static_cast<std::size_t>(o.max_orbitals),m=static_cast<std::size_t>(o.max_coefficients);
  if (n && n>std::numeric_limits<std::size_t>::max()/n)
    throw std::length_error("local orbital coefficient extent overflow");
  const auto add=[&](LocalTensorField field,std::array<std::size_t,3> shape) {
    tensors_[static_cast<std::size_t>(field)]=std::make_unique<Tensor<double,3>>(shape);
  };
  if (o.lno) {
    add(LocalTensorField::lno_coefficients,{spin,t.extended_atoms,n*n});
    add(LocalTensorField::lno_populations,{spin,t.extended_atoms,n});
  }
  if (o.contraction) {
    add(LocalTensorField::contraction,{t.first_to_global.size(),n,m});
    add(LocalTensorField::species_contraction,{t.orbitals.size()+1,n,m});
  }
}

AtomBlockMatrix<double>* LocalBasisState::real(LocalRealField field) noexcept
{
  const auto i=static_cast<std::size_t>(field);return i<real_.size() ? real_[i].get() : nullptr;
}
AtomBlockMatrix<dcomplex>* LocalBasisState::complex(LocalComplexField field) noexcept
{
  const auto i=static_cast<std::size_t>(field);return i<complex_.size() ? complex_[i].get() : nullptr;
}
Tensor<double,3>* LocalBasisState::tensor(LocalTensorField field) noexcept
{
  const auto i=static_cast<std::size_t>(field);return i<tensors_.size() ? tensors_[i].get() : nullptr;
}
} // namespace openmx::core
