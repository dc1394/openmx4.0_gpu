#include <openmx/core/force_workspace.hpp>
#include "block_matrix_compat.hpp"
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdint>

namespace openmx::core {
namespace {
int components(int spin)
{
  if (spin!=0 && spin!=1 && spin!=3) throw std::invalid_argument("invalid force spin mode");
  return spin+1;
}
void require(bool ok,const char* message) {if(!ok)throw std::invalid_argument(message);}
void matrix(const OpenMXElectronicMatrixView& v,int atom,int neighbors,std::size_t channels)
{
  require(v.values && v.components>=channels && v.component_offsets && v.atom_offsets &&
      v.block_offsets && v.shapes,"missing force matrix storage or layout");
  require(atom>=0 && neighbors>=0,"invalid force matrix atom or neighbors");
  for(std::size_t c=0;c<channels;++c) {
    const auto first=v.component_offsets[c],last=v.component_offsets[c+1];
    require(first<=last && last<=v.atoms && static_cast<std::size_t>(atom)<last-first,
        "force matrix atom exceeds component layout");
    const auto a=first+atom,begin=v.atom_offsets[a],end=v.atom_offsets[a+1];
    require(begin<=end && end<=v.blocks && static_cast<std::size_t>(neighbors)<end-begin,
        "force matrix neighbor exceeds layout");
    for(std::size_t b=begin;b<=begin+neighbors;++b) {
      const auto offset=v.block_offsets[b],next=v.block_offsets[b+1];
      const int rows=v.shapes[2*b],cols=v.shapes[2*b+1];
      require(rows>=0 && cols>=0 && offset<=next && next<=v.elements &&
          static_cast<std::size_t>(rows)<=static_cast<std::size_t>(PTRDIFF_MAX)/sizeof(double)/
            (cols?static_cast<std::size_t>(cols):1) &&
          static_cast<std::size_t>(rows)*cols<=next-offset,"invalid force matrix block");
    }
  }
}
std::size_t force_atoms(const AtomBlockLayout& layout,std::size_t local)
{
  require(local<layout.atoms(),"force accumulator atoms exceed matrix topology");
  return local+1;
}
} // namespace

AtomBlockLayout force_layout(const OpenMXForceInput& in)
{
  components(in.spin);
  require(in.atoms>=0 && in.local_atoms>=0 && in.local_atoms<=in.atoms && in.halo_atoms>=0 &&
      in.halo_atoms<=in.atoms-in.local_atoms && in.local_atoms<=INT_MAX-1-in.halo_atoms &&
      in.species>=0 && (in.imaginary_density==0 || in.imaginary_density==1),"invalid force topology dimensions");
  const int count=in.local_atoms+in.halo_atoms;
  require(!count || (in.first_to_global && in.atom_species && in.orbitals && in.neighbor_counts &&
      in.neighbor_atoms && in.species>0),"missing force topology");
  std::vector<std::size_t> offsets{0,1};
  std::vector<MatrixBlockShape> blocks{{1,1}};
  const auto orbitals=[&](int atom) {
    require(atom>=1 && atom<=in.atoms,"invalid force atom index");
    const int s=in.atom_species[atom];
    require(s>=0 && s<in.species && in.orbitals[s]>=0,"invalid force species or orbital count");
    return in.orbitals[s];
  };
  for(int a=1;a<=count;++a) {
    const int gc=in.first_to_global[a],rows=orbitals(gc),neighbors=in.neighbor_counts[gc];
    require(neighbors>=0 && neighbors<INT_MAX && in.neighbor_atoms[gc],"invalid force neighbor row");
    for(int h=0;h<=neighbors;++h)blocks.push_back({rows,orbitals(in.neighbor_atoms[gc][h])});
    offsets.push_back(blocks.size());
  }
  return {offsets,blocks};
}

ForceMatrices::ForceMatrices(AtomBlockLayout layout,int spin,bool imaginary,std::size_t local_atoms)
  : density_(layout,components(spin)),
    imaginary_(imaginary?std::make_unique<AtomBlockMatrix<double>>(std::move(layout),2):nullptr),
    forces_({3,force_atoms(density_.layout(),local_atoms)})
{}

void force_overlap(const OpenMXElectronicMatrixView& overlap,
    const OpenMXElectronicMatrixView& energy,int spin,int atom,int neighbors,std::span<double,3> force)
{
  components(spin);matrix(overlap,atom,neighbors,4);matrix(energy,atom,neighbors,spin?2:1);
  // Validate every used pair before changing any accumulator.
  for(int h=1;h<=neighbors;++h) {
    const auto first=overlap.atom_offsets[overlap.component_offsets[0]+atom]+h;
    const int rows=overlap.shapes[2*first],cols=overlap.shapes[2*first+1];
    for(std::size_t c=1;c<4;++c) {
      const auto b=overlap.atom_offsets[overlap.component_offsets[c]+atom]+h;
      require(overlap.shapes[2*b]==rows && overlap.shapes[2*b+1]==cols,
          "force overlap derivative block shapes differ");
    }
    // Contracted overlap can use a smaller prefix of an energy-density block
    // allocated in the primitive basis. Its leading dimension still belongs
    // to the energy-density layout, not the contracted overlap layout.
    for(std::size_t e=0;e<(spin?2u:1u);++e) {
      const auto b=energy.atom_offsets[energy.component_offsets[e]+atom]+h;
      require(energy.shapes[2*b]>=rows && energy.shapes[2*b+1]>=cols,
          "force energy-density block is smaller than overlap");
    }
  }
  for(int h=1;h<=neighbors;++h) {
    const auto b=overlap.atom_offsets[overlap.component_offsets[0]+atom]+h;
    const int rows=overlap.shapes[2*b],cols=overlap.shapes[2*b+1];
    using View=Tensor<double,2>::const_view_type;
    const auto get=[&](const OpenMXElectronicMatrixView& v,std::size_t c) {
      const auto a=v.component_offsets[c]+atom,block=v.atom_offsets[a]+h;
      return View(v.values+v.block_offsets[block],static_cast<std::size_t>(v.shapes[2*block]),static_cast<std::size_t>(v.shapes[2*block+1]));
    };
    const auto x=get(overlap,1),y=get(overlap,2),z=get(overlap,3),d0=get(energy,0),d1=get(energy,spin?1:0);
    for(int i=0;i<rows;++i)for(int j=0;j<cols;++j) {
      const double weight=spin?element(d0,i,j)+element(d1,i,j):2.*element(d0,i,j);
      const double dx=weight*element(x,i,j),dy=weight*element(y,i,j),dz=weight*element(z,i,j);
      force[0]=force[0]-2.*dx;force[1]=force[1]-2.*dy;force[2]=force[2]-2.*dz;
    }
  }
}

void force_spin_trace(const OpenMXForceSpinTrace& in,std::span<double,3> force)
{
  components(in.spin);
  require(in.rows>=0 && in.columns>=0 && (in.spinor==0 || in.spinor==1) &&
      (!in.spinor || in.spin==3) && std::isfinite(in.prefactor),"invalid force spin trace shape or mode");
  require(in.derivative_rows>=static_cast<std::size_t>(in.rows) && in.derivative_stride>=static_cast<std::size_t>(in.columns) &&
      (!in.derivative_rows || in.derivative_stride<=static_cast<std::size_t>(PTRDIFF_MAX)/sizeof(dcomplex)/3/in.derivative_rows),
      "invalid force derivative stride");
  if (!in.rows || !in.columns)return;
  for(int c=0;c<(in.spinor?4:(in.spin?2:1));++c)require(in.density[c],"missing force density component");
  if(in.spinor)for(int c=0;c<2;++c)require(in.imaginary[c],"missing imaginary force density");
  for(auto p:in.derivative)require(p,"missing force derivative component");
  using View=Tensor<double,2>::const_view_type;
  const auto d=[&](int c,int i,int j) {return element(View(in.density[c],in.rows,in.columns),i,j);};
  const auto im=[&](int c,int i,int j) {return element(View(in.imaginary[c],in.rows,in.columns),i,j);};
  for(int i=0;i<in.rows;++i)for(int j=0;j<in.columns;++j)for(int axis=0;axis<3;++axis) {
    const Tensor<dcomplex,3>::const_view_type h(in.derivative[axis],3,in.derivative_rows,in.derivative_stride);
    if(in.spinor) {
      force[axis]+=d(0,i,j)*element(h,0,i,j).r-im(0,i,j)*element(h,0,i,j).i
          +d(1,i,j)*element(h,1,i,j).r-im(1,i,j)*element(h,1,i,j).i
          +2.*d(2,i,j)*element(h,2,i,j).r-2.*d(3,i,j)*element(h,2,i,j).i;
    } else if(in.spin) {
      force[axis]+=in.prefactor*(d(0,i,j)*element(h,0,i,j).r+d(1,i,j)*element(h,1,i,j).r);
    } else force[axis]+=in.prefactor*d(0,i,j)*element(h,0,i,j).r;
  }
}
} // namespace openmx::core

namespace {
template<class F> int checked(char* error,std::size_t capacity,F&& fn) noexcept
{
  if(error && capacity)*error='\0';
  try {fn();return 1;}
  catch(const std::exception& e){if(error && capacity)std::snprintf(error,capacity,"%s",e.what());}
  catch(...){if(error && capacity)std::snprintf(error,capacity,"unexpected force workspace exception");}
  return 0;
}
OpenMXDoubleBlockMatrix view(openmx::core::AtomBlockMatrix<double>& matrix)
{
  const auto& l=matrix.layout();const auto v=matrix.values();
  return {{l.atoms(),v.extent(0),l.blocks(),l.stride(),v.size(),l.atom_offsets().data(),
      l.block_offsets().data(),l.shapes().data_handle()},v.data_handle()};
}
struct Call {
  OpenMXForceWorkspace views;
  OpenMXForceOperation operation;
  void* args;
};
int invoke(OpenMXSolverInvocation* scratch,void* opaque,double* result)
{auto& call=*static_cast<Call*>(opaque);call.views.scratch=scratch;return call.operation(&call.views,call.args,result);}
}

extern "C" int openmx_core_force_run(const OpenMXForceInput* input,int workers,
    OpenMXForceOperation operation,void* args,double* result,char* error,std::size_t capacity)
{
  int status=0;
  if(!checked(error,capacity,[&] {
    if(!input || !operation || !result)throw std::invalid_argument("missing force input, callback or result");
    openmx::core::ForceMatrices owner(openmx::core::force_layout(*input),input->spin,
        input->imaginary_density,input->local_atoms);
    openmx::core::compat::BlockMatrixRows<double> density(owner.density());
    std::unique_ptr<openmx::core::compat::BlockMatrixRows<double>> imaginary;
    if(owner.imaginary())imaginary=std::make_unique<openmx::core::compat::BlockMatrixRows<double>>(*owner.imaginary());
    OpenMXForceWorkspace views{};views.density=view(owner.density());views.density_rows=density.data();
    if(imaginary){views.imaginary=view(*owner.imaginary());views.imaginary_rows=imaginary->data();}
    for(int c=0;c<3;++c)views.forces[c]=owner.forces().data_handle()+c*owner.forces().extent(1);
    Call call{views,operation,args};char nested[256];
    status=openmx_core_solver_run(workers,invoke,&call,result,nested,sizeof(nested));
    if(!status && error && capacity)std::snprintf(error,capacity,"%s",nested);
  }))return 0;
  return status;
}
extern "C" int openmx_core_force_overlap(const OpenMXElectronicMatrixView* overlap,
    const OpenMXElectronicMatrixView* energy,int spin,int atom,int neighbors,double force[3],char* error,std::size_t capacity)
{return checked(error,capacity,[&] {
  if(!overlap || !energy || !force)throw std::invalid_argument("missing force overlap inputs");
  openmx::core::force_overlap(*overlap,*energy,spin,atom,neighbors,std::span<double,3>(force,3));
});}
extern "C" int openmx_core_force_spin_trace(const OpenMXForceSpinTrace* input,double force[3],char* error,std::size_t capacity)
{return checked(error,capacity,[&] {
  if(!input || !force)throw std::invalid_argument("missing force spin trace inputs");
  openmx::core::force_spin_trace(*input,std::span<double,3>(force,3));
});}
