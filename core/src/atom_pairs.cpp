#include <openmx/core/atom_pairs.hpp>
#include "openmx_core_atom_pairs.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <memory>

namespace openmx::core {
namespace {
std::size_t pair_count(std::span<const AtomSiteView> atoms,std::span<const int> local,int species)
{
  if (species<0 || (species==0 && !local.empty()) || local.size()>INT_MAX)
    throw std::invalid_argument("invalid atom-pair species or local count");
  std::size_t count=0;
  for (int global:local) {
    if (global<1 || static_cast<std::size_t>(global)>atoms.size())
      throw std::invalid_argument("atom-pair local map outside global atoms");
    const auto n=atoms[global-1].neighbors.size();
    if (!n || n>INT_MAX || count>static_cast<std::size_t>(INT_MAX)-n)
      throw std::length_error("atom-pair task count exceeds int or lacks self row");
    count+=n;
  }
  return count;
}
}

AtomPairPlan::AtomPairPlan(std::span<const AtomSiteView> atoms,std::span<const int> local,
    std::span<const std::array<double,3>> translations,int species)
  : displacements_({pair_count(atoms,local,species),2,3})
{
  const auto count=displacements_.extent(0);
  local_.reserve(count);neighbor_.reserve(count);central_global_.reserve(count);
  central_species_.reserve(count);neighbor_species_.reserve(count);offsets_.reserve(local.size()+1);
  offsets_.push_back(0);
  for (std::size_t mc=0;mc<local.size();++mc) {
    const int global=local[mc];const auto& central=atoms[global-1];
    if (central.species<0 || central.species>=species || central.images.size()!=central.neighbors.size())
      throw std::invalid_argument("invalid atom-pair central species or image row");
    for (std::size_t h=0;h<central.neighbors.size();++h) {
      const int other=central.neighbors[h],image=central.images[h];
      if (other<1 || static_cast<std::size_t>(other)>atoms.size() || image<0 ||
          static_cast<std::size_t>(image)>=translations.size())
        throw std::invalid_argument("atom-pair neighbor or image outside geometry");
      const auto& partner=atoms[other-1];
      if (partner.species<0 || partner.species>=species)
        throw std::invalid_argument("invalid atom-pair neighbor species");
      const auto index=local_.size();
      local_.push_back(static_cast<int>(mc)+1);neighbor_.push_back(static_cast<int>(h));
      central_global_.push_back(global);central_species_.push_back(central.species);
      neighbor_species_.push_back(partner.species);
      for (int d=0;d<3;++d) {
        const double a=central.position[d],b=partner.position[d],t=translations[image][d];
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(t))
          throw std::invalid_argument("nonfinite atom-pair coordinate");
        element(displacements_.view(),index,0,d)=b+t-a;
        element(displacements_.view(),index,1,d)=a-b-t;
      }
    }
    offsets_.push_back(local_.size());
  }
}
} // namespace openmx::core

struct OpenMXAtomPairPlan {
  OpenMXAtomPairPlan(std::span<const openmx::core::AtomSiteView> atoms,std::span<const int> local,
      std::span<const std::array<double,3>> translations,int species) : implementation(atoms,local,translations,species) {}
  openmx::core::AtomPairPlan implementation;
};

extern "C" OpenMXAtomPairPlan *openmx_core_atom_pairs_create(const OpenMXAtomPairInput *input,
    OpenMXAtomPairBuffers *buffers,char *error,std::size_t capacity)
{
  if (error && capacity) error[0]='\0';
  try {
    if (!input || !buffers || input->atoms<0 || input->local_atoms<0 || input->species<0 || input->translations<0 ||
        input->local_atoms>input->atoms)
      throw std::invalid_argument("invalid atom-pair input dimensions");
    std::vector<openmx::core::AtomSiteView> atoms;
    std::vector<std::array<double,3>> translations;
    std::span<const int> local;
    if (input->local_atoms) {
      if (!input->local_to_global || !input->atom_species || !input->neighbor_counts || !input->neighbors ||
          !input->images || !input->positions || !input->translation_vectors || !input->translations)
        throw std::invalid_argument("null atom-pair input geometry");
      local={input->local_to_global+1,static_cast<std::size_t>(input->local_atoms)};
      atoms.resize(input->atoms);
      // Only local neighbor rows are needed. Every partner's position/species
      // is copied, while unrelated neighbor rows may be absent on this rank.
      for (int g=1;g<=input->atoms;++g) {
        if (!input->positions[g]) throw std::invalid_argument("null atom position row");
        atoms[g-1].species=input->atom_species[g];
        std::copy_n(input->positions[g]+1,3,atoms[g-1].position.begin());
      }
      for (int g:local) {
        if (g<1 || g>input->atoms) throw std::invalid_argument("invalid atom-pair local map");
        const int n=input->neighbor_counts[g];
        if (n<0 || n==INT_MAX || !input->neighbors[g] || !input->images[g])
          throw std::invalid_argument("invalid atom-pair neighbor row");
        atoms[g-1].neighbors={input->neighbors[g],static_cast<std::size_t>(n)+1};
        atoms[g-1].images={input->images[g],static_cast<std::size_t>(n)+1};
      }
      translations.resize(input->translations);
      for (int image=0;image<input->translations;++image) {
        if (!input->translation_vectors[image]) throw std::invalid_argument("null translation vector");
        std::copy_n(input->translation_vectors[image]+1,3,translations[image].begin());
      }
    }
    auto result=std::make_unique<OpenMXAtomPairPlan>(atoms,local,translations,input->species);
    const auto& plan=result->implementation;
    *buffers={static_cast<int>(plan.local().size()),input->local_atoms,plan.local().data(),plan.neighbor().data(),
      plan.central_global().data(),plan.central_species().data(),plan.neighbor_species().data(),
      plan.offsets().data(),plan.displacements().data_handle()};
    return result.release();
  }
  catch (const std::exception& e) { if (error && capacity) std::snprintf(error,capacity,"%s",e.what()); }
  catch (...) { if (error && capacity) std::snprintf(error,capacity,"unknown atom-pair failure"); }
  return nullptr;
}
extern "C" void openmx_core_atom_pairs_destroy(OpenMXAtomPairPlan *plan) { delete plan; }
