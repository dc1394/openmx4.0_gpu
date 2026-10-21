#include <openmx/core/local_basis_state.hpp>
#include "openmx_core_local_basis.h"
#include "allocation_faults.hpp"
#include <climits>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace {
using namespace openmx::core;
void require(bool ok,const char* message) {if (!ok) throw std::runtime_error(message);}
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context context() {
  char error[256];Context c(openmx_core_create(error,sizeof(error)),openmx_core_destroy);
  require(bool(c),error);return c;
}
struct Fixture {
  int species[6]{-1,0,1,1,0,1},orbitals[2]{4,2},contracted[2]{2,1};
  int local[3]{-1,3,1},first[4]{-1,1,3,2};
  int fnan[6]{-1,1,0,2,0,0},snan[6]{-1,1,0,1,0,0};
  int n1[3]{1,2,3},n3[4]{3,1,4,2};
  int* neighbors[6]{nullptr,n1,nullptr,n3,nullptr,nullptr};
  OpenMXLocalBasisInput input{};OpenMXLocalBasisViews view{};char error[256]{};
  Fixture() {
    input.atoms=5;input.local_atoms=2;input.first_halo_atoms=1;input.second_halo_atoms=2;input.species=2;
    input.spin=3;input.max_orbitals=5;input.max_coefficients=3;
    input.interactions=input.lno=input.contraction=input.inverse_overlap=1;
    input.atom_species=species;input.orbitals=orbitals;input.contracted_orbitals=contracted;
    input.local_to_global=local;input.first_to_global=first;
    input.first_neighbors=fnan;input.second_neighbors=snan;input.neighbor_atoms=neighbors;
  }
  int allocate(OpenMXCoreContext* c) {return openmx_core_local_basis_allocate(c,&input,&view,error,sizeof(error));}
};

void layouts_test()
{
  auto c=context();Fixture f;require(f.allocate(c.get()),f.error);auto& v=f.view;
  // Independent legacy shapes: local squares 1+2^2+4^2, F-map squares
  // 1+4^2+2^2+2^2; DC padded sizes 1, 3+(1+2+2+1), 3+(2+1+1).
  const std::size_t real_elements[]{168,100,131},complex_elements[]{168,100};
  const int sizes[3][4]={{1,2,4,0},{1,4,2,2},{1,9,7,0}};
  const std::size_t tensor_elements[]{600,120,60,45};
  for (std::size_t field=0;field<3;++field) {
    auto& a=v.real[field];require(a.layout.elements==real_elements[field],"legacy real shape changed");
    for (std::size_t atom=0;atom<a.layout.atoms;++atom) {
      std::size_t expected=0;for(std::size_t prev=0;prev<atom;++prev)expected+=sizes[field][prev]*sizes[field][prev];
      require(a.layout.offsets[atom]==expected && a.layout.shapes[2*atom]==sizes[field][atom] && a.layout.shapes[2*atom+1]==sizes[field][atom],"legacy matrix shape/offset changed");
    }
    for (std::size_t comp=0;comp<a.layout.components;++comp)
      for (std::size_t atom=0;atom<a.layout.atoms;++atom)
        for (int row=0;row<a.layout.shapes[2*atom];++row) {
          double* old=openmx_local_real_row(&a,comp,atom,row);
          require(old==openmx_local_real_row(&a,comp,atom,row),"real adapter stride changed");
          double* flat=field==0 ? openmx_local_density_row(&v,comp/4,comp%4,atom,row) :
                       field==1 ? openmx_local_potential_row(&v,comp,atom,row) : openmx_local_inverse_overlap_row(&v,atom,row);
          require(flat==old,"named real row stride changed");
          for (int col=0;col<a.layout.shapes[2*atom+1];++col) {
            require(old[col]==0,"real value not initialized");old[col]=1+(old+col-a.values);
          }
        }
    for (std::size_t i=0;i<a.layout.elements;++i) require(a.values[i]==double(i+1),"real view overlaps or leaves holes");
  }
  for (std::size_t field=0;field<2;++field) {
    auto& a=v.complex_values[field];require(a.layout.elements==complex_elements[field],"legacy complex shape changed");
    for (std::size_t atom=0;atom<a.layout.atoms;++atom) {
      std::size_t expected=0;for(std::size_t prev=0;prev<atom;++prev)expected+=sizes[field][prev]*sizes[field][prev];
      require(a.layout.offsets[atom]==expected && a.layout.shapes[2*atom]==sizes[field][atom] && a.layout.shapes[2*atom+1]==sizes[field][atom],"legacy matrix shape/offset changed");
    }
    for (std::size_t comp=0;comp<a.layout.components;++comp)
      for (std::size_t atom=0;atom<a.layout.atoms;++atom)
        for (int row=0;row<a.layout.shapes[2*atom];++row) {
          dcomplex* old=openmx_local_complex_row(&a,comp,atom,row);
          require(old==openmx_local_complex_row(&a,comp,atom,row),"complex adapter stride changed");
          dcomplex* flat=field==0 ? openmx_local_occupation_row(&v,comp/4,comp/2%2,comp%2,atom,row) :
                                   openmx_local_complex_potential_row(&v,comp/2,comp%2,atom,row);
          require(flat==old,"named complex row stride changed");
          for (int col=0;col<a.layout.shapes[2*atom+1];++col) {
            require(old[col].r==0 && old[col].i==0,"complex value not initialized");
            const double x=1+(old+col-a.values);old[col]={x,-x};
          }
        }
    for (std::size_t i=0;i<a.layout.elements;++i)
      require(a.values[i].r==double(i+1) && a.values[i].i==-double(i+1),"complex view overlaps or leaves holes");
  }
  const std::size_t expected_shapes[4][3]={{4,6,25},{4,6,5},{4,5,3},{3,5,3}};
  for (std::size_t field=0;field<4;++field) {
    auto& a=v.tensors[field];require(a.n0*a.n1*a.n2==tensor_elements[field],"legacy tensor shape changed");
    require(a.n0==expected_shapes[field][0] && a.n1==expected_shapes[field][1] && a.n2==expected_shapes[field][2],"legacy tensor extents changed");
    for (std::size_t i=0;i<a.n0;++i) for (std::size_t j=0;j<a.n1;++j) {
      require(openmx_local_tensor_row(&a,i,j)==(a.values+(i*a.n1+j)*a.n2),"tensor row stride changed");
      double* flat=field==0 ? openmx_local_lno_row(&v,i,j) : field==1 ? openmx_local_lno_population_row(&v,i,j) :
                   field==2 ? openmx_local_contraction_row(&v,i,j) : openmx_local_species_contraction_row(&v,i,j);
      require(flat==(a.values+(i*a.n1+j)*a.n2),"named tensor row stride changed");
      for (std::size_t k=0;k<a.n2;++k) {
        require((a.values+(i*a.n1+j)*a.n2)[k]==0,"tensor value not initialized");
        (a.values+(i*a.n1+j)*a.n2)[k]=double((i*a.n1+j)*a.n2+k+1);
      }
    }
    for (std::size_t i=0;i<tensor_elements[field];++i) require(a.values[i]==double(i+1),"tensor view overlaps or leaves holes");
  }
  // Density uses natn[gc][0] for columns; the complex onsite matrix is square.
  f.n3[0]=1;require(f.allocate(c.get()),f.error);
  require(f.view.real[0].layout.elements==200 && f.view.complex_values[0].layout.elements==168,"onsite column convention changed");
}

void modes_test()
{
  auto c=context();
  for (int spin:{0,1,3}) for (unsigned flags=0;flags<16;++flags) {
    Fixture f;f.input.spin=spin;f.input.interactions=flags&1;f.input.lno=(flags>>1)&1;
    f.input.contraction=(flags>>2)&1;f.input.inverse_overlap=(flags>>3)&1;
    require(f.allocate(c.get()),f.error);const auto& v=f.view;
    require(bool(v.real[0].values)==bool(flags&1) && bool(v.real[1].values)==bool(flags&1),"onsite enablement changed");
    require(bool(v.complex_values[0].values)==bool((flags&1) && spin==3) && bool(v.complex_values[1].values)==bool((flags&1) && spin==3),"complex enablement changed");
    require(bool(v.tensors[0].values)==bool(flags&2) && bool(v.tensors[1].values)==bool(flags&2),"LNO enablement changed");
    require(bool(v.tensors[2].values)==bool(flags&4) && bool(v.tensors[3].values)==bool(flags&4),"contraction enablement changed");
    require(bool(v.real[2].values)==bool(flags&8),"DC enablement changed");
    require(v.real[0].layout.elements==((flags&1) ? 42*std::size_t(spin+1):0),"onsite spin dimension changed");
    if (flags&1) require(openmx_local_density_row(&v,1,spin,1,0)==openmx_local_density_row(&v,1,spin,1,0),"density history/spin mapping changed");
  }
}

void mdspan_test()
{
  const int species[]{-1,0,1},basis[]{3,2},contracted[]{2,1},local[]{-1,2,1};
  const std::size_t offsets[]{0,0,2,4};const int neighbors[]{1,2,2,1};
  const LocalBasisTopology t{species,basis,contracted,local,local,3,offsets,neighbors};
  const LocalBasisOptions o{3,3,2,true,true,true,true};LocalBasisState state(t,o);
  auto* real=state.real(LocalRealField::density);auto b=real->block(5,1,0);element(b,1,1)=17;
  require(b.mapping().stride(0)==2 && element(real->values(),5,4)==17,"real mdspan stride/offset");
  const auto& fixed=*real;require(element(fixed.block(5,1,0),1,1)==17,"const real mdspan");
  auto* complex=state.complex(LocalComplexField::occupation);element(complex->block(7,2,0),2,1)={5,-8};
  require(element(complex->values(),7,12).r==5 && element(complex->values(),7,12).i==-8,"complex mdspan stride/offset");
  auto* coefficients=state.tensor(LocalTensorField::contraction);element(coefficients->view(),2,1,1)=31;
  require(coefficients->data()[15]==31,"tensor mdspan offset");
  require(!state.real(static_cast<LocalRealField>(-1)) && !state.complex(LocalComplexField::count) &&
      !state.tensor(LocalTensorField::count),"invalid field lookup");
  bool caught=false;try {(void)real->block(8,0,0);} catch (const std::out_of_range&) {caught=true;}
  require(caught,"invalid component accepted");
}

void ownership_test()
{
  auto a=context(),b=context();OpenMXLocalBasisViews kept{};char error[256];
  {
    Fixture f;require(f.allocate(a.get()),f.error);kept=f.view;
    require(f.allocate(b.get()),f.error);require(kept.tensors[2].values!=f.view.tensors[2].values,"contexts share values");
    openmx_local_contraction_row(&kept,1,2)[1]=19;openmx_local_contraction_row(&f.view,1,2)[1]=3;
    openmx_local_occupation_row(&kept,1,1,1,1,1)[1]={7,9};
    f.local[1]=-1;f.orbitals[0]=-1;f.fnan[3]=-1;
  }
  b.reset();require(openmx_local_contraction_row(&kept,1,2)[1]==19 && openmx_local_occupation_row(&kept,1,1,1,1,1)[1].i==9,"input lifetime leaks into owner");
  OpenMXLocalBasisViews current{};require(openmx_core_local_basis_get(a.get(),&current,error,sizeof(error)),error);
  require(current.tensors[2].values==kept.tensors[2].values,"lookup reallocates");
}

void lifecycle_test()
{
  auto c=context();Fixture f;require(openmx_core_local_basis_get(c.get(),&f.view,f.error,sizeof(f.error)) && !f.view.tensors[2].values,"initial owner not empty");
  for (int i=0;i<9;++i) {
    f.input.local_atoms=i%3;f.input.contraction=i%2;f.input.first_halo_atoms=f.input.second_halo_atoms=0;
    require(f.allocate(c.get()),f.error);require(f.view.real[0].layout.atoms==std::size_t(i%3+1),"topology replacement shape");
    require(openmx_local_density_row(&f.view,0,0,0,0)[0]==0,"replacement retains old state");openmx_local_density_row(&f.view,0,0,0,0)[0]=9;
  }
  openmx_core_local_basis_release(c.get());openmx_core_local_basis_release(c.get());
  OpenMXLocalBasisInput empty{};empty.interactions=empty.lno=empty.contraction=empty.inverse_overlap=1;
  require(openmx_core_local_basis_allocate(c.get(),&empty,&f.view,f.error,sizeof(f.error)),f.error);
  require(f.view.real[0].layout.elements==2 && f.view.real[2].layout.elements==1,"empty rank dummy allocation");
  require(f.view.tensors[2].n1==0 && f.view.tensors[0].n2==0 && openmx_local_lno_row(&f.view,0,0)==f.view.tensors[0].values,"zero extent tensor view");
  openmx_core_finish(c.get());openmx_core_finish(c.get());
  require(!f.allocate(c.get()) && !openmx_core_local_basis_get(c.get(),&f.view,f.error,sizeof(f.error)),"finished context accepted");
}

void invalid_test()
{
  auto c=context();
  for (int bad=0;bad<23;++bad) {
    Fixture f;require(f.allocate(c.get()),f.error);const auto saved=f.view;openmx_local_inverse_overlap_row(&saved,1,2)[3]=23;
    switch(bad) {
      case 0:f.input.atoms=-1;break;case 1:f.input.local_atoms=6;break;
      case 2:f.input.first_halo_atoms=INT_MAX;break;case 3:f.input.second_halo_atoms=INT_MAX;break;
      case 4:f.input.spin=2;break;case 5:f.input.max_orbitals=3;break;case 6:f.input.max_coefficients=-1;break;
      case 7:f.input.interactions=2;break;case 8:f.input.contracted_orbitals=nullptr;break;
      case 9:f.input.neighbor_atoms=nullptr;break;case 10:f.input.first_neighbors=nullptr;break;
      case 11:f.input.second_neighbors=nullptr;break;case 12:f.snan[3]=-1;break;
      case 13:f.fnan[3]=INT_MAX;break;case 14:f.input.local_to_global=nullptr;break;
      case 15:f.first[1]=6;break;case 16:f.species[3]=2;break;case 17:f.n3[1]=6;break;
      case 18:f.orbitals[1]=-1;break;case 19:f.contracted[1]=3;break;
      case 20:f.input.atom_species=nullptr;break;case 21:f.input.orbitals=nullptr;break;
      case 22:f.input.first_to_global=nullptr;break;
    }
    const auto live=openmx::test::live_allocations();require(!f.allocate(c.get()) && f.error[0],"invalid input accepted");
    require(openmx::test::live_allocations()==live && !std::memcmp(&saved,&f.view,sizeof(saved)),"failed validation changes output or leaks");
    OpenMXLocalBasisViews current{};require(openmx_core_local_basis_get(c.get(),&current,f.error,sizeof(f.error)),f.error);
    require(current.real[2].values==saved.real[2].values && openmx_local_inverse_overlap_row(&current,1,2)[3]==23,"failed validation replaces owner");
  }
  Fixture f;
  require(!openmx_core_local_basis_allocate(nullptr,&f.input,&f.view,nullptr,0),"null context accepted");
  require(!openmx_core_local_basis_allocate(c.get(),nullptr,&f.view,nullptr,0),"null input accepted");
  require(!openmx_core_local_basis_allocate(c.get(),&f.input,nullptr,nullptr,0),"null output accepted");
  char error[2]{'x','y'};
  require(!openmx_core_local_basis_get(nullptr,&f.view,error,1) && error[0]==0 && error[1]=='y',"error capacity overrun");
}

void allocation_test()
{
  auto c=context();Fixture f;require(f.allocate(c.get()),f.error);
  openmx::test::fail_allocation_after(10000);const int ok=f.allocate(c.get());
  const auto count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  require(ok && count>30,"allocation interception unavailable");
  const auto saved=f.view;openmx_local_complex_potential_row(&saved,1,0,1,0)[0]={11,-7};
  for (long fault=0;fault<count;++fault) {
    const auto live=openmx::test::live_allocations();openmx::test::fail_allocation_after(fault);
    const auto accepted=f.allocate(c.get());openmx::test::fail_allocation_after(-1);
    require(!accepted && f.error[0] && openmx::test::live_allocations()==live,"partial construction leaks");
    require(!std::memcmp(&saved,&f.view,sizeof(saved)) && openmx_local_complex_potential_row(&saved,1,0,1,0)[0].i==-7,"partial construction destroys prior state");
  }
}
void concurrent_test()
{
  int results[4]{};std::thread jobs[4];
  for (int worker=0;worker<4;++worker) jobs[worker]=std::thread([&,worker] {
    try {
      auto c=context();Fixture f;
      for (int i=0;i<10;++i) {
        f.input.lno=i%2;require(f.allocate(c.get()),f.error);
        openmx_local_contraction_row(&f.view,1,1)[1]=worker+i;
        require(openmx_local_tensor_row(&f.view.tensors[OPENMX_LOCAL_CONTRACTION],1,1)[1]==worker+i,"concurrent state shared");
      }
      results[worker]=1;
    } catch (...) {results[worker]=-1;}
  });
  for (auto& job:jobs) job.join();for (int result:results) require(result==1,"independent contexts failed");
}
}

int main(int argc,char** argv)
{
  try {
    const std::string test=argc==2 ? argv[1]:"";
    if(test=="layouts")layouts_test();else if(test=="modes")modes_test();else if(test=="mdspan")mdspan_test();
    else if(test=="ownership")ownership_test();else if(test=="lifecycle")lifecycle_test();else if(test=="invalid")invalid_test();
    else if(test=="allocation")allocation_test();else if(test=="concurrent")concurrent_test();else throw std::runtime_error("unknown local basis test");
    std::cout<<"PASS local basis "<<test<<'\n';return 0;
  } catch (const std::exception& e) {openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}
