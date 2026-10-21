#include "calculation_context.hpp"
#include "local_basis_compat.hpp"
#include <climits>
#include <cstdio>

namespace openmx::core::compat {
namespace {
static_assert(static_cast<int>(LocalRealField::count)==OPENMX_LOCAL_REAL_COUNT);
static_assert(static_cast<int>(LocalComplexField::count)==OPENMX_LOCAL_COMPLEX_COUNT);
static_assert(static_cast<int>(LocalTensorField::count)==OPENMX_LOCAL_TENSOR_COUNT);

std::unique_ptr<LocalBasisState> create(const OpenMXLocalBasisInput& in)
{
  if (in.atoms<0 || in.local_atoms<0 || in.local_atoms>in.atoms || in.species<0 ||
      in.first_halo_atoms<0 || in.second_halo_atoms<0 ||
      in.local_atoms>INT_MAX-2-in.first_halo_atoms ||
      in.local_atoms+in.first_halo_atoms>INT_MAX-2-in.second_halo_atoms ||
      (in.atoms && !in.atom_species) || (in.species && (!in.orbitals || !in.contracted_orbitals)))
    throw std::invalid_argument("invalid local basis input dimensions");
  for (int flag:{in.interactions,in.lno,in.contraction,in.inverse_overlap})
    if (flag!=0 && flag!=1) throw std::invalid_argument("invalid local basis option flag");
  const auto global=static_cast<std::size_t>(in.atoms)+1;
  const auto local=static_cast<std::size_t>(in.local_atoms)+1;
  const auto first=local+static_cast<std::size_t>(in.first_halo_atoms);
  const auto extended=first+static_cast<std::size_t>(in.second_halo_atoms);
  const auto map=[&](const int* p,std::size_t count) {
    if (count>1 && !p) throw std::invalid_argument("missing local basis atom map");
    std::vector<int> result(count);
    for (std::size_t a=1;a<count;++a) {
      if (p[a]<1 || p[a]>in.atoms) throw std::invalid_argument("invalid local basis atom map index");
      result[a]=p[a];
    }
    return result;
  };
  const auto local_map=map(in.local_to_global,local),first_map=map(in.first_to_global,first);
  std::vector<unsigned char> needed(global);
  if (in.interactions || in.inverse_overlap)
    for (std::size_t a=1;a<local;++a) needed[local_map[a]]=1;
  std::vector<std::size_t> offsets(global+1);
  std::vector<int> neighbors;
  for (std::size_t a=0;a<global;++a) {
    if (needed[a]) {
      if (!in.first_neighbors || !in.neighbor_atoms || !in.neighbor_atoms[a] || in.first_neighbors[a]<0)
        throw std::invalid_argument("missing local basis neighbor row");
      int count=1;
      if (in.inverse_overlap) {
        if (!in.second_neighbors || in.second_neighbors[a]<0 ||
            in.first_neighbors[a]>INT_MAX-1-in.second_neighbors[a])
          throw std::invalid_argument("invalid local overlap neighbor count");
        count+=in.first_neighbors[a]+in.second_neighbors[a];
      }
      for (int h=0;h<count;++h) neighbors.push_back(in.neighbor_atoms[a][h]);
    }
    offsets[a+1]=neighbors.size();
  }
  std::vector<int> atom_species(global);
  // Padding slots in old topology arrays need not be initialized.
  for (std::size_t a=1;a<global;++a) atom_species[a]=in.atom_species[a];
  const auto species=static_cast<std::size_t>(in.species);
  const LocalBasisTopology t{atom_species,{in.orbitals,species},{in.contracted_orbitals,species},
      local_map,first_map,extended,offsets,neighbors};
  const LocalBasisOptions o{in.spin,in.max_orbitals,in.max_coefficients,
      bool(in.interactions),bool(in.lno),bool(in.contraction),bool(in.inverse_overlap)};
  return std::make_unique<LocalBasisState>(t,o);
}

template<class T> OpenMXLocalMatrixLayout layout_view(AtomBlockMatrix<T>& m)
{
  const auto& layout=m.layout();const auto values=m.values();
  return {values.extent(0),layout.atoms(),layout.stride(),values.size(),
      layout.block_offsets().data(),layout.shapes().data_handle()};
}
}

template<class T> void LocalBasisOwner::MatrixRows<T>::bind(AtomBlockMatrix<T>& matrix)
{
  const auto& layout=matrix.layout();const auto values=matrix.values();
  const auto count=values.extent(0),n=layout.atoms();
  if (n && count>std::numeric_limits<std::size_t>::max()/n)
    throw std::length_error("local matrix adapter extent overflow");
  components.resize(count);atoms.resize(count*n);
  std::size_t total_rows=0;
  for (std::size_t a=0;a<n;++a) {
    const auto height=static_cast<std::size_t>(element(layout.shapes(),a,0));
    if (height>std::numeric_limits<std::size_t>::max()-total_rows)
      throw std::length_error("local matrix row count overflow");
    total_rows+=height;
  }
  if (count && total_rows>std::numeric_limits<std::size_t>::max()/count)
    throw std::length_error("local matrix row adapter extent overflow");
  rows.resize(count*total_rows);
  std::size_t next=0;
  for (std::size_t c=0;c<count;++c) {
    components[c]=n ? atoms.data()+c*n : nullptr;
    for (std::size_t a=0;a<n;++a) {
      const auto height=static_cast<std::size_t>(element(layout.shapes(),a,0));
      const auto width=static_cast<std::size_t>(element(layout.shapes(),a,1));
      atoms[c*n+a]=height ? rows.data()+next : nullptr;
      for (std::size_t i=0;i<height;++i)
        rows[next++]=width ? values.data_handle()+c*layout.stride()+layout.block_offsets()[a]+i*width : &empty;
    }
  }
}

void LocalBasisOwner::TensorRows::bind(Tensor<double,3>& tensor)
{
  const auto n0=tensor.extent(0),n1=tensor.extent(1),n2=tensor.extent(2);
  if (n1 && n0>std::numeric_limits<std::size_t>::max()/n1)
    throw std::length_error("local tensor adapter extent overflow");
  planes.resize(n0);rows.resize(n0*n1);
  for (std::size_t i=0;i<n0;++i) {
    planes[i]=n1 ? rows.data()+i*n1 : nullptr;
    for (std::size_t j=0;j<n1;++j) rows[i*n1+j]=n2 ? tensor.data()+(i*n1+j)*n2 : &empty;
  }
}

LocalBasisOwner::LocalBasisOwner(const OpenMXLocalBasisInput& in) : state_(create(in))
{
  for (std::size_t i=0;i<real_.size();++i)
    if (auto* m=state_->real(static_cast<LocalRealField>(i))) real_[i].bind(*m);
  for (std::size_t i=0;i<complex_.size();++i)
    if (auto* m=state_->complex(static_cast<LocalComplexField>(i))) complex_[i].bind(*m);
  for (std::size_t i=0;i<tensors_.size();++i)
    if (auto* t=state_->tensor(static_cast<LocalTensorField>(i))) tensors_[i].bind(*t);
  if (!real_[0].components.empty()) {
    density_groups_.resize(2);
    for (std::size_t m=0;m<2;++m) density_groups_[m]=real_[0].components.data()+m*(in.spin+1);
  }
  if (!complex_[0].components.empty()) {
    occupation_spins_.resize(4);occupation_history_.resize(2);potential_spins_.resize(2);
    for (std::size_t i=0;i<4;++i) occupation_spins_[i]=complex_[0].components.data()+2*i;
    for (std::size_t i=0;i<2;++i) {
      occupation_history_[i]=occupation_spins_.data()+2*i;
      potential_spins_[i]=complex_[1].components.data()+2*i;
    }
  }
}

OpenMXLocalBasisViews LocalBasisOwner::views() noexcept
{
  OpenMXLocalBasisViews result{};
  for (std::size_t i=0;i<real_.size();++i)
    if (auto* m=state_->real(static_cast<LocalRealField>(i))) result.real[i]={layout_view(*m),m->values().data_handle()};
  for (std::size_t i=0;i<complex_.size();++i)
    if (auto* m=state_->complex(static_cast<LocalComplexField>(i))) result.complex_values[i]={layout_view(*m),m->values().data_handle()};
  for (std::size_t i=0;i<tensors_.size();++i)
    if (auto* t=state_->tensor(static_cast<LocalTensorField>(i))) result.tensors[i]={t->extent(0),t->extent(1),t->extent(2),t->data()};
  result.DM_onsite=density_groups_.empty() ? nullptr : density_groups_.data();
  result.v_eff=real_[1].components.empty() ? nullptr : real_[1].components.data();
  result.S12=real_[2].components.empty() ? nullptr : real_[2].components[0];
  result.NC_OcpN=occupation_history_.empty() ? nullptr : occupation_history_.data();
  result.NC_v_eff=potential_spins_.empty() ? nullptr : potential_spins_.data();
  const auto tensor=[&](std::size_t i) {return tensors_[i].planes.empty() ? nullptr : tensors_[i].planes.data();};
  result.LNO_coes=tensor(0);result.LNO_pops=tensor(1);
  result.CntCoes=tensor(2);result.CntCoes_Species=tensor(3);
  return result;
}
} // namespace openmx::core::compat

namespace {
template<class Function> int checked(char* error,std::size_t capacity,Function&& f) noexcept
{
  if (error && capacity) *error='\0';
  try {f();return 1;}
  catch (const std::exception& e) {if (error && capacity) std::snprintf(error,capacity,"%s",e.what());}
  catch (...) {if (error && capacity) std::snprintf(error,capacity,"unexpected local basis exception");}
  return 0;
}
}
extern "C" int openmx_core_local_basis_allocate(OpenMXCoreContext* context,
    const OpenMXLocalBasisInput* input,OpenMXLocalBasisViews* views,char* error,std::size_t capacity)
{
  return checked(error,capacity,[&] {
    if (!context || !input || !views || context->implementation.finished())
      throw std::invalid_argument("missing or finished local basis calculation context");
    auto owner=std::make_unique<openmx::core::compat::LocalBasisOwner>(*input);
    const auto result=owner->views();context->local_basis=std::move(owner);*views=result;
  });
}
extern "C" int openmx_core_local_basis_get(OpenMXCoreContext* context,
    OpenMXLocalBasisViews* views,char* error,std::size_t capacity)
{
  return checked(error,capacity,[&] {
    if (!context || !views || context->implementation.finished())
      throw std::invalid_argument("missing or finished local basis calculation context");
    *views=context->local_basis ? context->local_basis->views() : OpenMXLocalBasisViews{};
  });
}
extern "C" void openmx_core_local_basis_release(OpenMXCoreContext* context)
{ if (context) context->local_basis.reset(); }
