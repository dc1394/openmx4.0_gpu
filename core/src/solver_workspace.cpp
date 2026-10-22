#include "openmx_core_solver_workspace.h"
#include <openmx/core/tensor.hpp>
#include <openmx/core/overwrite_tensor.hpp>
#include <openmx/core/orbital_workspace.hpp>
#include <openmx/core/core_hole.hpp>
#include <openmx/core/projector_force.hpp>
#include "native_resources.hpp"
#include <array>
#include <climits>
#include <cstdio>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using openmx::core::Tensor;
using openmx::core::OverwriteTensor;
void require(bool ok,const char* message) {if(!ok)throw std::invalid_argument(message);}
size_t product(size_t a,size_t b) {
  if(b && a>std::numeric_limits<size_t>::max()/b)throw std::length_error("solver extent product overflows");
  return a*b;
}
size_t plus(size_t a,size_t b) {
  if(a>std::numeric_limits<size_t>::max()-b)throw std::length_error("solver offset overflows");
  return a+b;
}
template<class F> int checked(char* error,size_t capacity,F&& operation) noexcept {
  if(error && capacity)*error='\0';
  try {operation();return 1;}
  catch(const std::exception& e){if(error && capacity)std::snprintf(error,capacity,"%s",e.what());}
  catch(...){if(error && capacity)std::snprintf(error,capacity,"unexpected solver workspace failure");}
  return 0;
}
template<class T, template<class,std::size_t> class Storage=Tensor> struct Buffer {
  static size_t elements(const std::array<size_t,4>& shape) {
    size_t n=1;for(auto extent:shape)n=product(n,extent);return n;
  }
  explicit Buffer(const std::array<size_t,4>& dimensions):shape(dimensions),storage({elements(dimensions)}) {}
  typename Tensor<T,4>::view_type view() {
    return {storage.data(),typename Tensor<T,4>::extents_type(shape)};
  }
  std::array<size_t,4> shape;
  Storage<T,1> storage;
  std::vector<T*> rows;
};
template<class T, template<class,std::size_t> class Storage=Tensor>
using Slots=std::map<std::string,std::unique_ptr<Buffer<T,Storage>>,std::less<>>;
template<class T> struct AliasRows {
  std::array<size_t,4> shape;
  std::vector<T*> rows;
  std::vector<T**> components;
  explicit AliasRows(const std::array<size_t,4>& dimensions):shape(dimensions),
      rows(product(shape[0],shape[1])),components(shape[0]) {}
  void bind(T* values) noexcept {
    for(size_t c=0;c<shape[0];c++)components[c]=shape[1]?rows.data()+c*shape[1]:rows.data();
    for(size_t i=0;i<rows.size();i++)rows[i]=shape[2]?values+i*shape[2]:values;
  }
};
template<class T> using Aliases=std::map<std::string,std::unique_ptr<AliasRows<T>>,std::less<>>;
struct Worker {
  Slots<double> real;
  Slots<double,OverwriteTensor> real_overwrite;
  Slots<dcomplex> complex_values;
  Slots<dcomplex,OverwriteTensor> complex_overwrite;
  Slots<int> integers;
  Slots<std::max_align_t> bytes;
  Aliases<double> real_aliases;
  Aliases<dcomplex> complex_aliases;
  std::map<std::string,std::unique_ptr<openmx::core::OrbitalWorkspace>,std::less<>> orbitals;
};
template<class T> T*** alias(Aliases<T>& slots,const char* name,const size_t* dimensions,T* values) {
  require(name && *name && dimensions,"missing solver alias name or dimensions");
  const std::array<size_t,4> shape{dimensions[0],dimensions[1],dimensions[2],dimensions[3]};
  require(shape[3]==1,"solver alias requires a three-dimensional tensor");
  const auto count=Buffer<T>::elements(shape);require(!count || values,"missing solver alias values");
  auto found=slots.find(name);
  if(found==slots.end() || found->second->shape!=shape) {
    auto owner=std::make_unique<AliasRows<T>>(shape);owner->bind(values);auto* result=owner.get();
    if(found==slots.end())slots.emplace(name,std::move(owner));else found->second=std::move(owner);
    return result->components.data();
  }
  found->second->bind(values);return found->second->components.data();
}
template<class T, template<class,std::size_t> class Storage> Buffer<T,Storage>& buffer(Slots<T,Storage>& slots,
    const char* name,const std::array<size_t,4>& shape,bool rows=false) {
  require(name && *name,"missing solver buffer name");
  auto found=slots.find(name);
  if(found!=slots.end()) {
    auto& current=*found->second;
    if(Buffer<T>::elements(shape)<=current.storage.size()) {
      if(rows && (current.shape!=shape || current.rows.size()!=shape[0])) {
        std::vector<T*> metadata(shape[0]);
        for(size_t i=0;i<shape[0];i++)metadata[i]=shape[1]?current.storage.data()+i*shape[1]:current.storage.data();
        current.rows.swap(metadata);
      }else if(!rows && current.shape!=shape)current.rows.clear();
      current.shape=shape;
      return current;
    }
  }
  auto replacement=std::make_unique<Buffer<T,Storage>>(shape);
  if(rows) {
    replacement->rows.resize(shape[0]);
    for(size_t i=0;i<shape[0];i++)replacement->rows[i]=shape[1]?replacement->storage.data()+i*shape[1]:replacement->storage.data();
  }
  auto* result=replacement.get();
  if(found==slots.end())slots.emplace(name,std::move(replacement));else found->second=std::move(replacement);
  return *result;
}
std::array<size_t,4> shape4(const size_t* p) {require(p,"missing solver tensor shape");return {p[0],p[1],p[2],p[3]};}
}

struct OpenMXSolverRealField {
  size_t components,atoms;
  std::vector<std::optional<Tensor<double,1>>> storage;
  std::vector<OpenMXSolverRealBlock> blocks;
  OpenMXSolverRealField(size_t c,size_t a):components(c),atoms(a),storage(product(c,a)),blocks(storage.size()) {}
  OpenMXSolverRealFieldView view(){return {components,atoms,blocks.data(),this};}
};
struct OpenMXSolverResidues {
  size_t components,atoms;
  bool complex_values;
  std::vector<size_t> atom_offsets,pair_offsets;
  std::vector<int> shapes;
  std::vector<std::optional<Tensor<double,2>>> real;
  std::vector<std::optional<Tensor<dcomplex,2>>> complex_storage;
  std::vector<OpenMXSolverResidueBlock> blocks;
  explicit OpenMXSolverResidues(const OpenMXSolverResidueInput& in) {
    require(in.global_atoms>=0 && in.global_atoms<INT_MAX && in.local_atoms>=0 && in.local_atoms<=in.global_atoms &&
        in.species>=0 && in.components>0 && (in.padding==0 || in.padding==1) &&
        (in.complex_values==0 || in.complex_values==1),"invalid solver residue dimensions");
    require(!in.local_atoms || (in.local_to_global && in.atom_species && in.orbitals && in.first_neighbors && in.neighbor_atoms),
        "missing solver residue topology");
    components=in.components;atoms=size_t(in.local_atoms)+in.padding;complex_values=in.complex_values;
    atom_offsets.reserve(atoms+1);atom_offsets.push_back(0);pair_offsets.push_back(0);
    auto species=[&](int g) {require(g>0 && g<=in.global_atoms,"invalid residue neighbor atom");
      const int w=in.atom_species[g];require(w>=0 && w<in.species && in.orbitals[w]>=0,"invalid residue species");return w;};
    for(size_t a=0;a<atoms;a++) {
      const bool dummy=in.padding && a==0;
      const int g=dummy ? 0:in.local_to_global[a+1-in.padding];
      const int r=dummy ? 1:in.orbitals[species(g)];
      require(dummy || (in.first_neighbors[g]>=0 && in.first_neighbors[g]<INT_MAX && in.neighbor_atoms[g]),"invalid residue neighbor row");
      const size_t neighbors=dummy ? 2:size_t(in.first_neighbors[g])+1;
      for(size_t h=0;h<neighbors;h++) {
        const int columns=dummy ? 1:in.orbitals[species(in.neighbor_atoms[g][h])];
        shapes.push_back(r);shapes.push_back(columns);
        pair_offsets.push_back(plus(pair_offsets.back(),product(r,columns)));
      }
      atom_offsets.push_back(shapes.size()/2);
    }
    const size_t n=product(components,atoms);blocks.resize(n);
    if(complex_values)complex_storage.resize(n);else real.resize(n);
  }
  OpenMXSolverResidueView view(){return {components,atoms,atom_offsets.data(),pair_offsets.data(),shapes.data(),blocks.data(),this};}
};
struct OpenMXSolverInvocation {
  std::vector<Worker> workers;
  std::vector<std::unique_ptr<OpenMXSolverRealField>> fields;
  std::vector<std::unique_ptr<OpenMXSolverResidues>> residues;
  std::vector<std::unique_ptr<openmx::core::CoreHole>> core_holes;
  std::vector<std::unique_ptr<openmx::core::ProjectorForce>> projector_forces;
  NativeResourceGroups native_resources;
  explicit OpenMXSolverInvocation(int n):workers(n) {}
  Worker& worker(int n) {require(n>=0 && size_t(n)<workers.size(),"invalid solver worker ID");return workers[n];}
};

extern "C" int openmx_core_solver_run(int workers,OpenMXSolverOperation operation,
    void* arguments,double* result,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(workers>0 && operation && result,"invalid solver invocation");
    OpenMXSolverInvocation invocation(workers);double output;
    require(operation(&invocation,arguments,&output)==1,"solver callback failed");
    *result=output;
  });
}
extern "C" int openmx_core_solver_real(OpenMXSolverInvocation* scope,int worker,const char* name,
    const size_t* dimensions,OpenMXSolverRealTensor* out,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(scope && out,"missing solver scope or output");
    const auto shape=shape4(dimensions);auto& b=buffer(scope->worker(worker).real,name,shape);
    *out={{shape[0],shape[1],shape[2],shape[3]},b.view().data_handle()};});
}
extern "C" int openmx_core_solver_real_overwrite(OpenMXSolverInvocation* scope,int worker,const char* name,
    const size_t* dimensions,OpenMXSolverRealTensor* out,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(scope && out,"missing solver scope or output");
    const auto shape=shape4(dimensions);auto& b=buffer(scope->worker(worker).real_overwrite,name,shape);
    *out={{shape[0],shape[1],shape[2],shape[3]},b.view().data_handle()};});
}
extern "C" int openmx_core_solver_resources(OpenMXSolverInvocation* scope,const char* name,
    OpenMXNativeResources** out,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(scope && out,"missing solver resource scope or output");
    *out=scope->native_resources.get(name,nullptr);
  });
}
extern "C" int openmx_core_core_hole_prepare(OpenMXSolverInvocation* scope,const OpenMXCoreHoleInput* input,
    OpenMXCoreHoleView* out,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(scope && input && out,"missing core-hole scope, input or output");
    auto owner=std::make_unique<openmx::core::CoreHole>(*input);const auto view=owner->view();
    scope->core_holes.push_back(std::move(owner));*out=view;
  });
}
extern "C" int openmx_core_projector_force_prepare(OpenMXSolverInvocation* scope,const OpenMXAtomicViews* atomic,
    int spin,int spin_orbit,OpenMXProjectorForceView* out,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(scope && atomic && out,"missing nonlocal Force scope, input or output");
    auto owner=std::make_unique<openmx::core::ProjectorForce>(*atomic,spin,spin_orbit);const auto view=owner->view();
    scope->projector_forces.push_back(std::move(owner));*out=view;
  });
}
extern "C" int openmx_core_solver_orbitals(OpenMXSolverInvocation* scope,int worker,const char* name,
    const OpenMXOrbitalShape* shape,OpenMXOrbitalBuffers* out,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(scope && name && *name && shape && out && (shape->derivatives==0 || shape->derivatives==1),
        "invalid solver orbital workspace request");
    auto& slots=scope->worker(worker).orbitals;auto found=slots.find(name);
    if(found!=slots.end()) {
      const auto& current=found->second->shape();
      if(current.maximum_angular==shape->maximum_angular && current.multiplicity==shape->multiplicity &&
          current.derivatives==bool(shape->derivatives)) {*out=found->second->buffers();return;}
    }
    auto owner=std::make_unique<openmx::core::OrbitalWorkspace>(openmx::core::OrbitalShape{
        shape->maximum_angular,shape->multiplicity,shape->derivatives!=0});
    const auto view=owner->buffers();
    if(found==slots.end())slots.emplace(name,std::move(owner));else found->second=std::move(owner);
    *out=view;
  });
}
extern "C" int openmx_core_solver_complex(OpenMXSolverInvocation* scope,int worker,const char* name,
    const size_t* dimensions,OpenMXSolverComplexTensor* out,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(scope && out,"missing solver scope or output");
    const auto shape=shape4(dimensions);auto& b=buffer(scope->worker(worker).complex_values,name,shape);
    *out={{shape[0],shape[1],shape[2],shape[3]},b.view().data_handle()};});
}
extern "C" int openmx_core_solver_complex_overwrite(OpenMXSolverInvocation* scope,int worker,const char* name,
    const size_t* dimensions,OpenMXSolverComplexTensor* out,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(scope && out,"missing solver scope or output");
    const auto shape=shape4(dimensions);auto& b=buffer(scope->worker(worker).complex_overwrite,name,shape);
    *out={{shape[0],shape[1],shape[2],shape[3]},b.view().data_handle()};});
}
extern "C" int openmx_core_solver_int(OpenMXSolverInvocation* scope,int worker,const char* name,
    size_t count,int** out,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(scope && out,"missing solver scope or output");
    auto& b=buffer(scope->worker(worker).integers,name,{count,1,1,1});*out=b.storage.data();});
}
extern "C" int openmx_core_solver_int_tensor(OpenMXSolverInvocation* scope,int worker,const char* name,
    const size_t* dimensions,OpenMXSolverIntTensor* out,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(scope && out,"missing solver scope or output");
    const auto shape=shape4(dimensions);auto& b=buffer(scope->worker(worker).integers,name,shape);
    *out={{shape[0],shape[1],shape[2],shape[3]},b.view().data_handle()};});
}
extern "C" int openmx_core_solver_bytes(OpenMXSolverInvocation* scope,int worker,const char* name,
    size_t count,size_t element_bytes,void** out,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(scope && out,"missing solver scope or output");
    size_t bytes=product(count,element_bytes),n=bytes/sizeof(std::max_align_t)+(bytes%sizeof(std::max_align_t)!=0);
    auto& b=buffer(scope->worker(worker).bytes,name,{n,1,1,1});*out=b.storage.data();});
}
extern "C" int openmx_core_solver_real_rows(OpenMXSolverInvocation* scope,int worker,const char* name,
    size_t rows,size_t columns,double*** out,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(scope && out,"missing solver scope or output");
    auto& b=buffer(scope->worker(worker).real,name,{rows,columns,1,1},true);*out=b.rows.data();});
}
extern "C" int openmx_core_solver_complex_rows(OpenMXSolverInvocation* scope,int worker,const char* name,
    size_t rows,size_t columns,dcomplex*** out,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(scope && out,"missing solver scope or output");
    auto& b=buffer(scope->worker(worker).complex_values,name,{rows,columns,1,1},true);*out=b.rows.data();});
}
extern "C" int openmx_core_solver_real_alias(OpenMXSolverInvocation* scope,int worker,const char* name,
    const OpenMXSolverRealTensor* input,double**** out,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(scope && input && out,"missing solver alias input/output");
    *out=alias(scope->worker(worker).real_aliases,name,input->extent,input->values);});
}
extern "C" int openmx_core_solver_complex_alias(OpenMXSolverInvocation* scope,int worker,const char* name,
    const OpenMXSolverComplexTensor* input,dcomplex**** out,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(scope && input && out,"missing solver alias input/output");
    *out=alias(scope->worker(worker).complex_aliases,name,input->extent,input->values);});
}
extern "C" int openmx_core_solver_field_create(OpenMXSolverInvocation* scope,size_t components,
    size_t atoms,OpenMXSolverRealFieldView* out,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(scope && out && components,"invalid solver field dimensions");
    auto owner=std::make_unique<OpenMXSolverRealField>(components,atoms);auto view=owner->view();
    scope->fields.push_back(std::move(owner));*out=view;});
}
extern "C" int openmx_core_solver_field_resize(OpenMXSolverRealField* owner,size_t component,
    size_t atom,size_t count,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(owner && component<owner->components && atom<owner->atoms,"invalid solver field row");
    const size_t index=component*owner->atoms+atom;
    if(owner->storage[index] && owner->blocks[index].count==count)return;
    Tensor<double,1> replacement({count});owner->storage[index]=std::move(replacement);
    owner->blocks[index]={count,owner->storage[index]->data()};});
}
extern "C" int openmx_core_solver_residue_create(OpenMXSolverInvocation* scope,
    const OpenMXSolverResidueInput* input,OpenMXSolverResidueView* out,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(scope && input && out,"missing residue scope, input or output");
    auto owner=std::make_unique<OpenMXSolverResidues>(*input);auto view=owner->view();
    scope->residues.push_back(std::move(owner));*out=view;});
}
extern "C" int openmx_core_solver_residue_resize(OpenMXSolverResidues* owner,size_t component,
    size_t atom,size_t states,char* error,size_t capacity) {
  return checked(error,capacity,[&]{require(owner && component<owner->components && atom<owner->atoms,"invalid residue block");
    const size_t index=component*owner->atoms+atom;
    if(owner->blocks[index].states==states && (owner->blocks[index].values || owner->blocks[index].complex_values))return;
    const size_t pairs=owner->pair_offsets[owner->atom_offsets[atom+1]]-owner->pair_offsets[owner->atom_offsets[atom]];
    if(owner->complex_values) {
      Tensor<dcomplex,2> replacement({pairs,states});owner->complex_storage[index]=std::move(replacement);
      owner->blocks[index]={states,nullptr,owner->complex_storage[index]->data()};
    }else {
      Tensor<double,2> replacement({pairs,states});owner->real[index]=std::move(replacement);
      owner->blocks[index]={states,owner->real[index]->data(),nullptr};
    }
  });
}
