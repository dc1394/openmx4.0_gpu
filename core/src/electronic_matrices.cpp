#include <openmx/core/electronic_matrices.hpp>
#include <algorithm>
#include <climits>

namespace openmx::core {
namespace {
using Field = ElectronicField;
enum class Kind { ordinary, overlap, core_overlap, projectors, dummy };

void validate(const ElectronicTopology& t,const ElectronicOptions& o)
{
  if ((o.spin!=0 && o.spin!=1 && o.spin!=3) || (o.spin_orbit!=0 && o.spin_orbit!=1) ||
      o.mixing<0 || o.mixing>7 || o.nonlocal_components<1 || o.density_history<1 ||
      o.hamiltonian_history<0 ||
      ((o.mixing==1 || o.mixing==5 || o.mixing==6) && o.hamiltonian_history==0) ||
      o.scratch_orbitals<0 || o.scratch_neighbors<1 ||
      o.scratch_projectors<0)
    throw std::invalid_argument("invalid electronic matrix options");
  if (t.atom_species.empty() || t.local_to_global.empty() ||
      t.first_to_global.size()<t.local_to_global.size() ||
      t.second_to_global.size()<t.first_to_global.size() ||
      t.neighbor_offsets.size()!=t.atom_species.size()+1 || t.neighbor_offsets.front()!=0 ||
      t.neighbor_offsets.back()!=t.neighbor_atoms.size() ||
      !std::is_sorted(t.neighbor_offsets.begin(),t.neighbor_offsets.end()) ||
      t.projector_components.size()!=t.orbitals.size() ||
      (o.contracted && t.contracted_orbitals.size()!=t.orbitals.size()))
    throw std::invalid_argument("invalid electronic matrix topology");
  for (std::size_t s=0;s<t.orbitals.size();++s) {
    if (t.orbitals[s]<0 || t.orbitals[s]>o.scratch_orbitals ||
        t.projector_components[s]<0 || t.projector_components[s]>INT_MAX-2 ||
        t.projector_components[s]+2>o.scratch_projectors ||
        (o.contracted && (t.contracted_orbitals[s]<0 || t.contracted_orbitals[s]>t.orbitals[s])))
      throw std::invalid_argument("invalid electronic species dimensions or scratch capacity");
  }
}

std::size_t species(const ElectronicTopology& t,int atom)
{
  if (atom<1 || static_cast<std::size_t>(atom)>=t.atom_species.size())
    throw std::invalid_argument("invalid electronic matrix atom index");
  const int s=t.atom_species[atom];
  if (s<0 || static_cast<std::size_t>(s)>=t.orbitals.size())
    throw std::invalid_argument("invalid electronic matrix species index");
  return static_cast<std::size_t>(s);
}

std::unique_ptr<RaggedMatrix> make_matrix(const ElectronicTopology& t,const ElectronicOptions& o,
    std::span<const int> map,std::size_t groups,std::size_t channels,bool contracted,Kind kind)
{
  if (groups && channels>std::numeric_limits<std::size_t>::max()/groups)
    throw std::length_error("electronic component count overflow");
  std::vector<std::size_t> components{0},atoms{0};
  components.reserve(groups*channels+1);
  std::vector<MatrixBlockShape> blocks;
  const auto basis=contracted ? t.contracted_orbitals : t.orbitals;
  const bool has_scratch=kind==Kind::overlap || kind==Kind::core_overlap || kind==Kind::projectors;
  const auto local_end=t.local_to_global.size();
  const auto atom_count=kind==Kind::dummy ? 1 : map.size()+has_scratch;
  for (std::size_t c=0;c<groups*channels;++c) {
    for (std::size_t a=0;a<atom_count;++a) {
      if (a==0) blocks.push_back({1,1});
      else if (has_scratch && (a==map.size() || a==local_end)) {
        for (int h=0;h<o.scratch_neighbors;++h)
          blocks.push_back({o.scratch_orbitals,
              kind==Kind::projectors ? o.scratch_projectors : o.scratch_orbitals});
      } else {
        const int gc=map[a];
        const auto s=species(t,gc),first=t.neighbor_offsets[gc],last=t.neighbor_offsets[gc+1];
        if (first==last) throw std::invalid_argument("missing electronic matrix neighbor row");
        if (last-first>static_cast<std::size_t>(o.scratch_neighbors))
          throw std::invalid_argument("electronic neighbor row exceeds scratch capacity");
        const bool reduced=c%channels>0 && a>local_end && !o.full_overlap_halo_derivatives;
        for (auto h=first;h<last;++h) {
          const auto other=species(t,t.neighbor_atoms[h]);
          int rows=basis[s],columns=basis[other];
          if (kind==Kind::projectors) columns=t.projector_components[other]+2;
          if (reduced && kind==Kind::overlap) rows=columns=1;
          if (reduced && kind==Kind::core_overlap) columns=1;
          blocks.push_back({rows,columns});
        }
      }
      atoms.push_back(blocks.size());
    }
    components.push_back(atoms.size()-1);
  }
  return std::make_unique<RaggedMatrix>(components,atoms,blocks);
}
} // namespace

ElectronicMatrices::ElectronicMatrices(const ElectronicTopology& t,ElectronicOptions o)
{
  validate(t,o);
  const auto local=t.local_to_global,first=t.first_to_global,extended=t.second_to_global;
  const auto first_local=first.first(local.size()),second_local=extended.first(local.size());
  const auto spin=static_cast<std::size_t>(o.spin+1),nl=static_cast<std::size_t>(o.nonlocal_components);
  const auto density_history=static_cast<std::size_t>(o.density_history);
  const auto hamiltonian_history=static_cast<std::size_t>(o.hamiltonian_history);
  const auto add=[&](Field field,std::span<const int> map,std::size_t groups,
      std::size_t channels,bool contracted=false,Kind kind=Kind::ordinary) {
    auto& family=families_[static_cast<std::size_t>(field)];
    family.matrix=make_matrix(t,o,map,groups,channels,contracted,kind);
    family.groups=groups;family.channels=channels;
  };
  add(Field::kinetic,local,1,4);
  add(Field::nonlocal,local,1,nl);
  add(Field::angular_overlap,first_local,1,3);
  add(Field::overlap,extended,1,4,false,Kind::overlap);
  add(Field::hamiltonian,extended,1,spin);
  add(Field::projectors,first_local,static_cast<std::size_t>(o.spin_orbit+1),4,false,Kind::projectors);
  add(Field::density,local,density_history,spin);
  add(Field::energy_density,local,1,spin);
  add(Field::previous_density,local,1,spin);
  add(Field::imaginary_density,local,density_history,2);
  if (o.contracted) {
    add(Field::contracted_kinetic,local,1,4,true);
    add(Field::contracted_overlap,extended,1,4,true);
    add(Field::contracted_hamiltonian,extended,1,spin,true);
    add(Field::contracted_projectors,first_local,static_cast<std::size_t>(o.spin_orbit+1),4,true,Kind::projectors);
    if (o.spin_orbit) add(Field::contracted_imaginary_nonlocal,extended,1,nl,true);
  }
  if (o.spin==3) {
    add(Field::imaginary_nonlocal,extended,1,nl);
    add(Field::imaginary_nonlocal_reference,local,1,nl);
  }
  if (o.core_hole) {
    add(Field::core_hole,local,1,nl);
    add(Field::core_hole_overlap,extended,1,4,false,Kind::core_overlap);
    if (o.spin==3) add(Field::imaginary_core_hole,extended,1,nl);
  }
  if (o.xmcd) add(Field::xmcd,local,1,spin);
  if (o.interactions) add(Field::hubbard,local,1,spin);
  if (o.lno) add(Field::lno_density,first,1,spin);
  if (o.mixing==1 || o.mixing==5 || o.mixing==6) {
    add(Field::hamiltonian_history,second_local,hamiltonian_history,spin);
    if (o.spin==3) add(Field::imaginary_hamiltonian_history,second_local,hamiltonian_history,3);
    if (o.mixing==5) {
      add(Field::hamiltonian_residual,second_local,hamiltonian_history,spin);
      if (o.spin==3) add(Field::imaginary_hamiltonian_residual,second_local,hamiltonian_history,3);
    }
  }
  const bool density_mixing=o.mixing==0 || o.mixing==1 || o.mixing==2 || o.mixing==6;
  if (density_mixing) add(Field::density_residual,local,density_history,spin);
  if (density_mixing && o.spin==3 && (o.spin_orbit || o.interactions))
    add(Field::imaginary_density_residual,local,density_history,2);
  else add(Field::imaginary_density_residual,{},density_history,1,false,Kind::dummy);
  if (o.guiding_density) {
    add(Field::guiding_density,local,1,spin);
    if (o.spin==3) add(Field::imaginary_guiding_density,local,1,2);
  }
  if (o.partial_density) add(Field::partial_density,local,1,2);
}

RaggedMatrix* ElectronicMatrices::get(Field field) noexcept
{
  const auto i=static_cast<std::size_t>(field);
  return i<families_.size() ? families_[i].matrix.get() : nullptr;
}
const RaggedMatrix* ElectronicMatrices::get(Field field) const noexcept
{
  const auto i=static_cast<std::size_t>(field);
  return i<families_.size() ? families_[i].matrix.get() : nullptr;
}
std::size_t ElectronicMatrices::groups(Field field) const noexcept
{
  const auto i=static_cast<std::size_t>(field);
  return i<families_.size() ? families_[i].groups : 0;
}
std::size_t ElectronicMatrices::channels(Field field) const noexcept
{
  const auto i=static_cast<std::size_t>(field);
  return i<families_.size() ? families_[i].channels : 0;
}
std::size_t ElectronicMatrices::elements() const noexcept
{
  std::size_t total=0;
  for (const auto& family:families_) if (family.matrix) total+=family.matrix->elements();
  return total;
}

} // namespace openmx::core
