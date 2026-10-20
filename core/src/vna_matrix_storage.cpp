#include "calculation_context.hpp"
#include <climits>
#include <cstdio>
#include <exception>

namespace openmx::core::compat {
namespace {
enum class Kind { potential, one_center, mirrored, projectors };

void validate(const OpenMXVnaMatrixInput& in)
{
  if (in.atoms<0 || in.local_atoms<0 || in.local_atoms>in.atoms || in.species<0 ||
      in.local_atoms>INT_MAX-2 || in.scratch_neighbors<1 || in.scratch_orbitals<0 ||
      in.projector_components<0 || (in.contracted!=0 && in.contracted!=1))
    throw std::invalid_argument("invalid VNA matrix dimensions");
  if (in.local_atoms && (!in.species || !in.local_to_global || !in.projector_to_global ||
      !in.atom_species || !in.neighbors || !in.neighbor_atoms || !in.orbitals ||
      (in.contracted && !in.contracted_orbitals)))
    throw std::invalid_argument("missing VNA matrix topology");
  for (int s=0;s<in.species;++s) {
    if (!in.orbitals || in.orbitals[s]<0 || (in.contracted &&
        (!in.contracted_orbitals || in.contracted_orbitals[s]<0 || in.contracted_orbitals[s]>in.orbitals[s])))
      throw std::invalid_argument("invalid VNA orbital count");
    if (in.orbitals[s]>in.scratch_orbitals)
      throw std::invalid_argument("VNA receive orbital capacity too small");
  }
}

int species(const OpenMXVnaMatrixInput& in,int atom)
{
  if (atom<1 || atom>in.atoms) throw std::invalid_argument("invalid VNA matrix atom index");
  const int s=in.atom_species[atom];
  if (s<0 || s>=in.species) throw std::invalid_argument("invalid VNA matrix species index");
  return s;
}

AtomBlockLayout layout(const OpenMXVnaMatrixInput& in,Kind kind,bool contracted)
{
  std::vector<std::size_t> offsets{0};
  std::vector<MatrixBlockShape> blocks;
  const int count=in.local_atoms+1+(kind==Kind::projectors);
  const int *basis=contracted ? in.contracted_orbitals : in.orbitals;
  for (int a=0;a<count;++a) {
    if (a==0) blocks.push_back({1,1});
    else if (a==in.local_atoms+1) {
      for (int h=0;h<in.scratch_neighbors;++h)
        blocks.push_back({in.scratch_orbitals,in.projector_components});
    } else {
      const int gc=(kind==Kind::potential ? in.local_to_global : in.projector_to_global)[a];
      const int s=species(in,gc),neighbors=in.neighbors[gc];
      if (neighbors<0 || neighbors==INT_MAX || !in.neighbor_atoms[gc])
        throw std::invalid_argument("invalid VNA neighbor row");
      if (neighbors>=in.scratch_neighbors)
        throw std::invalid_argument("VNA receive neighbor capacity too small");
      for (int h=0;h<=neighbors;++h) {
        const int other=species(in,in.neighbor_atoms[gc][h]);
        switch (kind) {
          case Kind::potential: blocks.push_back({basis[s],basis[other]}); break;
          case Kind::one_center: blocks.push_back({basis[s],basis[s]}); break;
          case Kind::mirrored: blocks.push_back({basis[other],basis[other]}); break;
          case Kind::projectors: blocks.push_back({basis[s],in.projector_components}); break;
        }
      }
    }
    offsets.push_back(blocks.size());
  }
  return {offsets,blocks};
}

template<class Scalar>
OpenMXAtomBlockLayout export_layout(const BlockMatrixOwner<Scalar>& owner)
{
  const auto& matrix=owner.matrix();
  const auto& shape=matrix.layout();
  const auto values=matrix.values();
  return {shape.atoms(),values.extent(0),shape.blocks(),shape.stride(),
      values.extent(0)*shape.stride(),shape.atom_offsets().data(),
      shape.block_offsets().data(),shape.shapes().data_handle()};
}
} // namespace

VnaMatrixOwner::VnaMatrixOwner(const OpenMXVnaMatrixInput& in)
{
  validate(in);
  potential_=std::make_unique<BlockMatrixOwner<double>>(layout(in,Kind::potential,false),1);
  one_center_=std::make_unique<BlockMatrixOwner<double>>(layout(in,Kind::one_center,false),4);
  mirrored_=std::make_unique<BlockMatrixOwner<double>>(layout(in,Kind::mirrored,false),4);
  projectors_=std::make_unique<BlockMatrixOwner<float>>(layout(in,Kind::projectors,false),4);
  if (in.contracted) {
    contracted_one_center_=std::make_unique<BlockMatrixOwner<double>>(layout(in,Kind::one_center,true),4);
    contracted_mirrored_=std::make_unique<BlockMatrixOwner<double>>(layout(in,Kind::mirrored,true),4);
    contracted_projectors_=std::make_unique<BlockMatrixOwner<float>>(layout(in,Kind::projectors,true),4);
  }
}

OpenMXVnaMatrixViews VnaMatrixOwner::views() noexcept
{
  OpenMXVnaMatrixViews result{};
  const auto doubles=[](auto& ptr) -> OpenMXDoubleBlockMatrix {
    return {export_layout(*ptr),ptr->matrix().values().data_handle()};
  };
  const auto floats=[](auto& ptr) -> OpenMXFloatBlockMatrix {
    return {export_layout(*ptr),ptr->matrix().values().data_handle()};
  };
  result.potential=doubles(potential_);
  result.one_center=doubles(one_center_);
  result.mirrored=doubles(mirrored_);
  result.projectors=floats(projectors_);
  result.HVNA=potential_->legacy()[0];
  result.HVNA2=one_center_->legacy();
  result.HVNA3=mirrored_->legacy();
  result.DS_VNA=projectors_->legacy();
  if (contracted_projectors_) {
    result.contracted_one_center=doubles(contracted_one_center_);
    result.contracted_mirrored=doubles(contracted_mirrored_);
    result.contracted_projectors=floats(contracted_projectors_);
    result.CntHVNA2=contracted_one_center_->legacy();
    result.CntHVNA3=contracted_mirrored_->legacy();
    result.CntDS_VNA=contracted_projectors_->legacy();
  }
  return result;
}
} // namespace openmx::core::compat

namespace {
template<class Function>
int checked(char *error,std::size_t capacity,Function&& function) noexcept
{
  if (error && capacity) *error='\0';
  try { function(); return 1; }
  catch (const std::exception& e) { if (error && capacity) std::snprintf(error,capacity,"%s",e.what()); }
  catch (...) { if (error && capacity) std::snprintf(error,capacity,"unexpected VNA matrix exception"); }
  return 0;
}
}

extern "C" int openmx_core_vna_matrices_allocate(OpenMXCoreContext *context,
    const OpenMXVnaMatrixInput *input,OpenMXVnaMatrixViews *views,char *error,std::size_t capacity)
{
  return checked(error,capacity,[&] {
    if (!context || context->implementation.finished() || !input || !views)
      throw std::invalid_argument("missing or finished VNA calculation context");
    auto owner=std::make_unique<openmx::core::compat::VnaMatrixOwner>(*input);
    const auto result=owner->views();
    context->vna_matrices=std::move(owner);
    *views=result;
  });
}

extern "C" int openmx_core_vna_matrices_get(OpenMXCoreContext *context,
    OpenMXVnaMatrixViews *views,char *error,std::size_t capacity)
{
  return checked(error,capacity,[&] {
    if (!context || !views || context->implementation.finished())
      throw std::invalid_argument("missing or finished VNA calculation context");
    *views=context->vna_matrices ? context->vna_matrices->views() : OpenMXVnaMatrixViews{};
  });
}

extern "C" void openmx_core_vna_matrices_release(OpenMXCoreContext *context)
{
  if (context) context->vna_matrices.reset();
}
