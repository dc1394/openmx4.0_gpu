#include "calculation_context.hpp"
#include "openmx_core_electronic.h"
#include <climits>
#include <cstdio>
#include <exception>

namespace openmx::core::compat {
namespace {
using Field=ElectronicField;
static_assert(static_cast<int>(Field::count)==OPENMX_ELECTRONIC_FIELD_COUNT);
static_assert(static_cast<int>(Field::imaginary_density)==OPENMX_ELECTRONIC_IMAGINARY_DENSITY);

std::unique_ptr<ElectronicMatrices> create(const OpenMXElectronicMatrixInput& in)
{
  if (in.atoms<0 || in.local_atoms<0 || in.local_atoms>in.atoms ||
      in.first_halo_atoms<0 || in.second_halo_atoms<0 || in.species<0 ||
      in.local_atoms>INT_MAX-2-in.first_halo_atoms ||
      in.local_atoms+in.first_halo_atoms>INT_MAX-2-in.second_halo_atoms ||
      (in.species && (!in.orbitals || !in.projector_components ||
          (in.contracted && !in.contracted_orbitals))) ||
      (in.atoms && !in.atom_species))
    throw std::invalid_argument("invalid electronic matrix input dimensions");
  for (int flag:{in.contracted,in.core_hole,in.xmcd,in.interactions,in.lno,
      in.guiding_density,in.partial_density,in.full_overlap_halo_derivatives})
    if (flag!=0 && flag!=1) throw std::invalid_argument("invalid electronic matrix option flag");
  const auto global=static_cast<std::size_t>(in.atoms)+1;
  const auto local=static_cast<std::size_t>(in.local_atoms)+1;
  const auto first=local+static_cast<std::size_t>(in.first_halo_atoms);
  const auto second=first+static_cast<std::size_t>(in.second_halo_atoms);
  std::vector<unsigned char> needed(global);
  const auto map=[&](const int* input,std::size_t count) {
    if (count>1 && !input) throw std::invalid_argument("missing electronic atom map");
    std::vector<int> result(count);
    for (std::size_t a=1;a<count;++a) {
      const int gc=input[a];
      if (gc<1 || gc>in.atoms) throw std::invalid_argument("invalid electronic atom map index");
      result[a]=gc;needed[gc]=1;
    }
    return result;
  };
  const auto local_map=map(in.local_to_global,local);
  const auto first_map=map(in.first_to_global,first);
  const auto second_map=map(in.second_to_global,second);
  std::vector<std::size_t> offsets(global+1);
  std::vector<int> neighbors;
  for (std::size_t gc=0;gc<global;++gc) {
    if (needed[gc]) {
      if (!in.neighbors || !in.neighbor_atoms || !in.neighbor_atoms[gc] ||
          in.neighbors[gc]<0 || in.neighbors[gc]==INT_MAX)
        throw std::invalid_argument("missing or invalid electronic neighbor row");
      for (int h=0;h<=in.neighbors[gc];++h) neighbors.push_back(in.neighbor_atoms[gc][h]);
    }
    offsets[gc+1]=neighbors.size();
  }
  // Slot zero can be uninitialized in the old input arrays. Never read it.
  std::vector<int> atom_species(global);
  for (std::size_t gc=1;gc<global;++gc) atom_species[gc]=in.atom_species[gc];
  const auto species=static_cast<std::size_t>(in.species);
  const ElectronicTopology topology{atom_species,{in.orbitals,species},
      {in.contracted_orbitals,in.contracted ? species : 0},{in.projector_components,species},
      local_map,first_map,second_map,offsets,neighbors};
  ElectronicOptions o;
  o.spin=in.spin;o.spin_orbit=in.spin_orbit;o.mixing=in.mixing;
  o.nonlocal_components=in.nonlocal_components;
  o.density_history=in.density_history;o.hamiltonian_history=in.hamiltonian_history;
  o.scratch_orbitals=in.scratch_orbitals;o.scratch_neighbors=in.scratch_neighbors;
  o.scratch_projectors=in.scratch_projectors;
  o.contracted=in.contracted;o.core_hole=in.core_hole;o.xmcd=in.xmcd;
  o.interactions=in.interactions;o.lno=in.lno;o.guiding_density=in.guiding_density;
  o.partial_density=in.partial_density;o.full_overlap_halo_derivatives=in.full_overlap_halo_derivatives;
  return std::make_unique<ElectronicMatrices>(topology,o);
}
} // namespace

void ElectronicMatrixOwner::Adapters::bind(RaggedMatrix& matrix,
    std::size_t group_count,std::size_t channels)
{
  groups.resize(group_count);components.resize(matrix.components());
  atoms.resize(matrix.atoms());blocks.resize(matrix.blocks());
  const auto& layout=matrix.layout();
  std::size_t count=0;
  for (std::size_t b=0;b<layout.blocks();++b) {
    const auto height=static_cast<std::size_t>(element(layout.shapes(),b,0));
    if (height>std::numeric_limits<std::size_t>::max()-count)
      throw std::length_error("electronic row adapter count overflow");
    count+=height;
  }
  rows.resize(count);
  for (std::size_t g=0;g<group_count;++g) groups[g]=components.data()+g*channels;
  const auto component_offsets=matrix.component_offsets();
  const auto atom_offsets=layout.atom_offsets(),block_offsets=layout.block_offsets();
  for (std::size_t c=0;c<matrix.components();++c)
    components[c]=component_offsets[c+1]>component_offsets[c] ? atoms.data()+component_offsets[c] : nullptr;
  for (std::size_t a=0;a<layout.atoms();++a)
    atoms[a]=atom_offsets[a+1]>atom_offsets[a] ? blocks.data()+atom_offsets[a] : nullptr;
  std::size_t row=0;
  for (std::size_t b=0;b<layout.blocks();++b) {
    const auto height=static_cast<std::size_t>(element(layout.shapes(),b,0));
    const auto width=static_cast<std::size_t>(element(layout.shapes(),b,1));
    blocks[b]=height ? rows.data()+row : nullptr;
    for (std::size_t i=0;i<height;++i)
      rows[row++]=width ? matrix.values().data_handle()+block_offsets[b]+i*width : &empty;
  }
}

ElectronicMatrixOwner::ElectronicMatrixOwner(const OpenMXElectronicMatrixInput& input)
    : matrices_(create(input))
{
  for (std::size_t i=0;i<adapters_.size();++i) {
    const auto field=static_cast<Field>(i);
    if (auto* matrix=matrices_->get(field))
      adapters_[i].bind(*matrix,matrices_->groups(field),matrices_->channels(field));
  }
}

OpenMXElectronicViews ElectronicMatrixOwner::views() noexcept
{
  OpenMXElectronicViews result{};
  for (std::size_t i=0;i<adapters_.size();++i) {
    const auto field=static_cast<Field>(i);
    if (auto* matrix=matrices_->get(field)) {
      const auto& layout=matrix->layout();
      result.fields[i]={matrices_->groups(field),matrices_->channels(field),matrix->components(),
          matrix->atoms(),matrix->blocks(),matrix->elements(),matrix->component_offsets().data(),
          layout.atom_offsets().data(),layout.block_offsets().data(),layout.shapes().data_handle(),
          matrix->values().data_handle()};
    }
  }
  const auto five=[&](Field f) -> double***** {
    auto& a=adapters_[static_cast<std::size_t>(f)];
    return a.components.empty() ? nullptr : a.components.data();
  };
  const auto six=[&](Field f) -> double****** {
    auto& a=adapters_[static_cast<std::size_t>(f)];
    return a.groups.empty() ? nullptr : a.groups.data();
  };
  result.H0=five(Field::kinetic);result.CntH0=five(Field::contracted_kinetic);
  result.HNL=five(Field::nonlocal);result.iHNL=five(Field::imaginary_nonlocal);
  result.iCntHNL=five(Field::contracted_imaginary_nonlocal);
  result.HCH=five(Field::core_hole);result.iHCH=five(Field::imaginary_core_hole);
  result.H_XMCD=five(Field::xmcd);result.H_Hub=five(Field::hubbard);
  result.iHNL0=five(Field::imaginary_nonlocal_reference);result.OLP_L=five(Field::angular_overlap);
  result.OLP=five(Field::overlap);result.OLP_CH=five(Field::core_hole_overlap);
  result.CntOLP=five(Field::contracted_overlap);result.H=five(Field::hamiltonian);
  result.CntH=five(Field::contracted_hamiltonian);
  result.DS_NL=six(Field::projectors);result.CntDS_NL=six(Field::contracted_projectors);
  result.DM0=five(Field::lno_density);result.HisH1=six(Field::hamiltonian_history);
  result.HisH2=six(Field::imaginary_hamiltonian_history);result.ResidualH1=six(Field::hamiltonian_residual);
  result.ResidualH2=six(Field::imaginary_hamiltonian_residual);result.DM=six(Field::density);
  result.ParDM=five(Field::guiding_density);result.iParDM=five(Field::imaginary_guiding_density);
  result.Partial_DM=five(Field::partial_density);result.ResidualDM=six(Field::density_residual);
  result.iResidualDM=six(Field::imaginary_density_residual);result.EDM=five(Field::energy_density);
  result.PDM=five(Field::previous_density);result.iDM=six(Field::imaginary_density);
  return result;
}
} // namespace openmx::core::compat

namespace {
template<class Function>
int checked(char* error,std::size_t capacity,Function&& function) noexcept
{
  if (error && capacity) *error='\0';
  try {function();return 1;}
  catch (const std::exception& e) {if (error && capacity) std::snprintf(error,capacity,"%s",e.what());}
  catch (...) {if (error && capacity) std::snprintf(error,capacity,"unexpected electronic matrix exception");}
  return 0;
}
}
extern "C" int openmx_core_electronic_allocate(OpenMXCoreContext* context,
    const OpenMXElectronicMatrixInput* input,OpenMXElectronicViews* views,char* error,std::size_t capacity)
{
  return checked(error,capacity,[&] {
    if (!context || !input || !views || context->implementation.finished())
      throw std::invalid_argument("missing or finished electronic calculation context");
    auto owner=std::make_unique<openmx::core::compat::ElectronicMatrixOwner>(*input);
    const auto result=owner->views();
    context->electronic=std::move(owner);*views=result;
  });
}
extern "C" int openmx_core_electronic_get(OpenMXCoreContext* context,
    OpenMXElectronicViews* views,char* error,std::size_t capacity)
{
  return checked(error,capacity,[&] {
    if (!context || !views || context->implementation.finished())
      throw std::invalid_argument("missing or finished electronic calculation context");
    *views=context->electronic ? context->electronic->views() : OpenMXElectronicViews{};
  });
}
extern "C" void openmx_core_electronic_release(OpenMXCoreContext* context)
{
  if (context) context->electronic.reset();
}
