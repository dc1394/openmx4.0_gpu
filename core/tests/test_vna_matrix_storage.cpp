#include <openmx/core/block_matrix.hpp>
#include "openmx_core_vna_matrix.h"
#include "../src/block_matrix_compat.hpp"
#include "allocation_faults.hpp"
#include <climits>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>

namespace {
using namespace openmx::core;
void require(bool value,const char *reason) { if (!value) throw std::runtime_error(reason); }
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context context() {
  char error[128];
  Context result(openmx_core_create(error,sizeof(error)),openmx_core_destroy);
  require(bool(result),error);return result;
}
struct Fixture {
  int local[3]{0,3,1},projector[3]{0,1,3},species[4]{0,0,1,1},counts[4]{0,2,0,1};
  int n1[3]{1,2,3},n2[1]{2},n3[2]{3,1},*neighbors[4]{nullptr,n1,n2,n3};
  int basis[2]{4,2},contracted[2]{2,1};
  OpenMXVnaMatrixInput input{3,2,2,1,3,5,6,local,projector,species,counts,neighbors,basis,contracted};
  OpenMXVnaMatrixViews views{};char error[256]{};
  int allocate(OpenMXCoreContext *core) { return openmx_core_vna_matrices_allocate(core,&input,&views,error,sizeof(error)); }
};

template<class Matrix,class Legacy>
void check_matrix(Matrix matrix,Legacy legacy,int kind,const Fixture& f,bool contracted)
{
  const auto& layout=matrix.layout;
  const auto *basis=contracted ? f.contracted : f.basis;
  const auto *map=kind==0 ? f.local : f.projector;
  const std::size_t atoms=kind==3 ? 4 : 3,components=kind==0 ? 1 : 4;
  require(layout.atoms==atoms && layout.components==components,"matrix atom/component count");
  require(layout.elements==components*layout.stride && layout.block_offsets[0]==0,"matrix storage count");
  for (std::size_t i=0;i<layout.elements;++i) require(matrix.values[i]==0,"matrix not zero initialized");
  std::size_t elements=0,blocks=0;
  for (std::size_t a=0;a<atoms;++a) {
    const bool scratch=a==3;
    const auto gc=a && !scratch ? map[a] : 0;
    const auto neighbors=a==0 ? 1 : scratch ? f.input.scratch_neighbors : f.counts[gc]+1;
    require(layout.atom_offsets[a+1]-layout.atom_offsets[a]==static_cast<std::size_t>(neighbors),"matrix neighbor count");
    for (int h=0;h<neighbors;++h,++blocks) {
      const auto rows=a==0 ? 1 : scratch ? f.input.scratch_orbitals :
          kind==2 ? basis[f.species[f.neighbors[gc][h]]] : basis[f.species[gc]];
      const auto columns=a==0 ? 1 : kind==3 ? f.input.projector_components :
          kind==0 ? basis[f.species[f.neighbors[gc][h]]] : rows;
      require(layout.shapes[2*blocks]==rows && layout.shapes[2*blocks+1]==columns,"matrix block dimensions/map orientation");
      require(layout.block_offsets[blocks]==elements,"matrix block offset");
      for (std::size_t c=0;c<components;++c)
        for (int i=0;i<rows;++i)
          for (int j=0;j<columns;++j) {
            const auto index=c*layout.stride+elements+static_cast<std::size_t>(i)*columns+j;
            const auto value=static_cast<float>(1+index);
            legacy[c][a][h][i][j]=value;
            require(matrix.values[index]==value,"legacy view does not alias flat values");
          }
      elements+=static_cast<std::size_t>(rows)*columns;
    }
  }
  require(layout.stride==elements && layout.blocks==blocks && layout.block_offsets[blocks]==elements,"matrix terminal offset");
  for (std::size_t i=0;i<layout.elements;++i) require(matrix.values[i]==static_cast<float>(1+i),"matrix blocks overlap or leave holes");
}

void layout_test()
{
  auto core=context();Fixture f;require(f.allocate(core.get()),f.error);
  const auto& v=f.views;
  double ****potential[1]{v.HVNA};
  check_matrix(v.potential,potential,0,f,false);
  check_matrix(v.one_center,v.HVNA2,1,f,false);
  check_matrix(v.mirrored,v.HVNA3,2,f,false);
  check_matrix(v.projectors,v.DS_VNA,3,f,false);
  check_matrix(v.contracted_one_center,v.CntHVNA2,1,f,true);
  check_matrix(v.contracted_mirrored,v.CntHVNA3,2,f,true);
  check_matrix(v.contracted_projectors,v.CntDS_VNA,3,f,true);
  require(v.potential.layout.elements==45 && v.one_center.layout.elements==228 &&
      v.mirrored.layout.elements==180 && v.projectors.layout.elements==748,"independent uncontracted element counts");
  require(v.contracted_one_center.layout.elements==60 && v.contracted_mirrored.layout.elements==48 &&
      v.contracted_projectors.layout.elements==556,"independent contracted element counts");
  require(openmx_double_block_row(&v.potential,0,1,1,1)==v.HVNA[1][1][1],"double C row view");
  require(openmx_float_block_row(&v.projectors,3,3,2,4)==v.DS_VNA[3][3][2][4],"float receive row view");
}

void mdspan_test()
{
  const std::size_t offsets[]{0,2,3};
  const MatrixBlockShape blocks[]{{2,3},{0,4},{3,1}};
  AtomBlockLayout original(offsets,blocks),copy(original),moved(std::move(copy));
  require(moved.atom_offsets().data()!=original.atom_offsets().data() && moved.stride()==9,"layout copy borrows metadata");
  require(copy.atoms()==0 && copy.blocks()==0 && copy.stride()==0,"moved-from layout invariants");
  AtomBlockMatrix<double> matrix({offsets,blocks},2);
  auto block=matrix.block(1,0,0);
  require(block.extent(0)==2 && block.extent(1)==3 && block.mapping().stride(0)==3,"mdspan shape/stride");
  element(block,1,2)=7;
  require(element(matrix.values(),1,5)==7 && matrix.layout().stride()==9,"component-major contiguous matrix");
  require(matrix.block(1,0,1).extent(0)==0,"zero-row block");
  const auto& constant=matrix;
  static_assert(std::is_const_v<std::remove_reference_t<decltype(element(constant.block(0,0,0),0,0))>>);
  require(element(constant.block(1,0,0),1,2)==7,"const block view");
  compat::BlockMatrixOwner<double> owner({offsets,blocks},2);
  owner.legacy()[1][1][0][2][0]=9;
  require(element(owner.matrix().block(1,1,0),2,0)==9,"generic double compatibility view");
  for (int kind=0;kind<3;++kind) {
    bool rejected=false;
    try { (void)matrix.block(kind==0 ? 2 : 0,kind==1 ? 2 : 0,kind==2 ? 2 : 0); }
    catch (const std::out_of_range&) { rejected=true; }
    require(rejected,"invalid mdspan block index accepted");
  }
}

void ownership_test()
{
  auto first=context(),second=context();OpenMXVnaMatrixViews kept{};
  {
    Fixture f;require(f.allocate(first.get()),f.error);kept=f.views;
    require(f.allocate(second.get()),f.error);
    require(kept.HVNA!=f.views.HVNA && kept.projectors.values!=f.views.projectors.values,"contexts share VNA storage");
    kept.HVNA[1][1][0][0]=37;f.views.DS_VNA[2][3][1][4][5]=-11;
    f.local[1]=f.projector[1]=99;f.counts[1]=-1;f.basis[0]=-99;
  }
  second.reset();
  require(kept.potential.layout.shapes[2]==2 && kept.HVNA[1][1][0][0]==37,"VNA matrices borrow old topology lifetime");
  OpenMXVnaMatrixViews again{};char error[128];
  require(openmx_core_vna_matrices_get(first.get(),&again,error,sizeof(error)),error);
  require(again.HVNA==kept.HVNA && again.DS_VNA==kept.DS_VNA,"lookup moves matrix storage");
}

void empty_test()
{
  auto core=context();OpenMXVnaMatrixViews v{};char error[128];
  require(openmx_core_vna_matrices_get(core.get(),&v,error,sizeof(error)) && !v.HVNA && !v.DS_VNA,"initial view is not empty");
  OpenMXVnaMatrixInput input{};input.scratch_neighbors=1;
  require(openmx_core_vna_matrices_allocate(core.get(),&input,&v,error,sizeof(error)),error);
  require(v.potential.layout.elements==1 && v.projectors.layout.elements==4,"empty MPI rank loses dummy atom");
  require(v.DS_VNA[3][0][0][0][0]==0 && !v.DS_VNA[0][1][0],"zero-orbital receive slot");
  require(!v.CntDS_VNA && !v.CntHVNA2 && !v.CntHVNA3,"unrequested contracted matrices allocated");
  input.contracted=1;require(openmx_core_vna_matrices_allocate(core.get(),&input,&v,error,sizeof(error)),error);
  require(v.CntDS_VNA && v.CntHVNA2[0][0][0][0][0]==0,"empty contracted rank");
  Fixture f;f.basis[0]=f.basis[1]=f.contracted[0]=f.contracted[1]=0;f.input.projector_components=0;
  require(f.allocate(core.get()),f.error);
  require(f.views.projectors.layout.elements==4 && f.views.potential.layout.elements==1,"zero-width projectors/basis");
  openmx_core_vna_matrices_release(core.get());openmx_core_vna_matrices_release(core.get());
  require(openmx_core_vna_matrices_get(core.get(),&v,error,sizeof(error)) && !v.HVNA,"release did not clear view");
}

void replacement_test()
{
  auto core=context();Fixture f;
  for (int round=0;round<12;++round) {
    f.input.contracted=round%2;f.input.local_atoms=round%3;
    require(f.allocate(core.get()),f.error);
    require(f.views.potential.layout.atoms==static_cast<std::size_t>(f.input.local_atoms+1),"MD topology replacement");
    require(bool(f.views.CntDS_VNA)==bool(f.input.contracted),"contraction mode replacement");
    require(f.views.DS_VNA[0][0][0][0][0]==0,"replacement retains previous numeric values");
    f.views.DS_VNA[0][0][0][0][0]=91;
    if (round%3==0) openmx_core_vna_matrices_release(core.get());
  }
  openmx_core_finish(core.get());openmx_core_finish(core.get());
  const auto saved=f.views;
  require(!f.allocate(core.get()) && !std::memcmp(&saved,&f.views,sizeof(saved)),"finished context accepted allocation");
  require(!openmx_core_vna_matrices_get(core.get(),&f.views,f.error,sizeof(f.error)),"finished context lookup succeeded");
}

void invalid_test()
{
  auto core=context();
  for (int scenario=0;scenario<22;++scenario) {
    Fixture f;require(f.allocate(core.get()),f.error);const auto saved=f.views;
    saved.DS_VNA[0][0][0][0][0]=17;
    switch(scenario) {
      case 0:f.input.atoms=-1;break;case 1:f.input.local_atoms=4;break;case 2:f.input.species=0;break;
      case 3:f.input.local_to_global=nullptr;break;case 4:f.input.projector_to_global=nullptr;break;
      case 5:f.input.atom_species=nullptr;break;case 6:f.input.neighbors=nullptr;break;
      case 7:f.input.neighbor_atoms=nullptr;break;case 8:f.input.orbitals=nullptr;break;
      case 9:f.input.contracted_orbitals=nullptr;break;case 10:f.local[1]=0;break;
      case 11:f.projector[1]=4;break;case 12:f.species[3]=2;break;case 13:f.counts[1]=-1;break;
      case 14:f.neighbors[1]=nullptr;break;case 15:f.n3[1]=4;break;case 16:f.basis[0]=-1;break;
      case 17:f.contracted[1]=3;break;case 18:f.input.scratch_neighbors=2;break;
      case 19:f.input.scratch_orbitals=3;break;case 20:f.input.projector_components=-1;break;
      case 21:f.input.contracted=2;break;
    }
    const auto live=openmx::test::live_allocations();
    require(!f.allocate(core.get()) && f.error[0],"invalid topology accepted");
    require(!std::memcmp(&saved,&f.views,sizeof(saved)) && saved.DS_VNA[0][0][0][0][0]==17,"failed replacement changed output/owner");
    require(openmx::test::live_allocations()==live,"invalid replacement leaked allocations");
    OpenMXVnaMatrixViews current{};
    require(openmx_core_vna_matrices_get(core.get(),&current,f.error,sizeof(f.error)) && current.DS_VNA==saved.DS_VNA,"failed replacement lost previous owner");
  }
  Fixture f;require(!openmx_core_vna_matrices_allocate(nullptr,&f.input,&f.views,nullptr,0),"null context accepted");
  require(!openmx_core_vna_matrices_allocate(core.get(),nullptr,&f.views,nullptr,0),"null input accepted");
  require(!openmx_core_vna_matrices_allocate(core.get(),&f.input,nullptr,nullptr,0),"null output accepted");
  char small[2]{'x','y'};
  require(!openmx_core_vna_matrices_get(nullptr,&f.views,small,1) && small[0]==0 && small[1]=='y',"short error buffer overflow");
}

void overflow_test()
{
  const MatrixBlockShape small[]{{2,3}},negative[]{{-1,2}};
  const std::size_t good[]{0,1},bad_first[]{1,1},bad_end[]{0,2},unsorted[]{0,1,0,1};
  for (auto offsets : {std::span<const std::size_t>{},std::span<const std::size_t>(bad_first),
      std::span<const std::size_t>(bad_end),std::span<const std::size_t>(unsorted)}) {
    bool rejected=false;try { AtomBlockLayout layout(offsets,small); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"invalid CSR layout accepted");
  }
  bool rejected=false;try { AtomBlockLayout layout(good,negative); }
  catch (const std::invalid_argument&) { rejected=true; }
  require(rejected,"negative block shape accepted");
  const MatrixBlockShape huge[]{{INT_MAX,INT_MAX},{INT_MAX,INT_MAX},{INT_MAX,INT_MAX},{INT_MAX,INT_MAX},{INT_MAX,INT_MAX}};
  const std::size_t huge_offsets[]{0,5};
  rejected=false;try { AtomBlockLayout layout(huge_offsets,huge); }
  catch (const std::length_error&) { rejected=true; }
  require(rejected,"cumulative block overflow not detected");
  rejected=false;try { AtomBlockMatrix<double> matrix({good,small},std::numeric_limits<std::size_t>::max()); }
  catch (const std::length_error&) { rejected=true; }
  require(rejected,"component product overflow not detected");
}

void concurrent_test()
{
  int result[4]{};std::thread jobs[4];
  for (int t=0;t<4;++t) jobs[t]=std::thread([&,t] {
    try {
      auto core=context();Fixture f;
      for (int round=0;round<20;++round) {
        f.input.contracted=(round+t)%2;require(f.allocate(core.get()),f.error);
        f.views.HVNA[1][0][0][0]=round+t;
        require(openmx_double_block_row(&f.views.potential,0,1,0,0)[0]==round+t,"concurrent contexts share values");
      }
      result[t]=1;
    } catch (...) { result[t]=-1; }
  });
  for (auto& job : jobs) job.join();
  for (int value : result) require(value==1,"concurrent host contexts failed");
}

void allocation_test()
{
  for (int contracted=0;contracted<2;++contracted) {
    auto core=context();Fixture f;f.input.contracted=contracted;require(f.allocate(core.get()),f.error);
    f.views.HVNA[1][0][0][0]=13;
    openmx::test::fail_allocation_after(10000);
    const int ok=f.allocate(core.get());
    const auto count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
    require(ok && count>20,"matrix allocation interception unavailable");
    f.views.HVNA[1][0][0][0]=13;const auto saved=f.views;
    for (long failure=0;failure<count;++failure) {
      const auto live=openmx::test::live_allocations();openmx::test::fail_allocation_after(failure);
      const int accepted=f.allocate(core.get());openmx::test::fail_allocation_after(-1);
      require(!accepted && f.error[0] && openmx::test::live_allocations()==live,"partial matrix replacement leaked");
      require(!std::memcmp(&saved,&f.views,sizeof(saved)) && saved.HVNA[1][0][0][0]==13,"partial replacement destroyed previous owner");
    }
  }
  const auto live=openmx::test::live_allocations();char error[128];
  openmx::test::fail_allocation_after(0);
  auto *bad=openmx_core_create(error,sizeof(error));openmx::test::fail_allocation_after(-1);
  require(!bad && error[0] && openmx::test::live_allocations()==live,"failed context creation leaked");
  openmx_core_destroy(nullptr);
}
}

int main(int argc,char** argv)
{
  try {
    const std::string name=argc==2 ? argv[1] : "";
    if (name=="layout") layout_test();else if (name=="mdspan") mdspan_test();
    else if (name=="ownership") ownership_test();else if (name=="empty") empty_test();
    else if (name=="replacement") replacement_test();else if (name=="invalid") invalid_test();
    else if (name=="overflow") overflow_test();else if (name=="concurrent") concurrent_test();
    else if (name=="allocation") allocation_test();else throw std::runtime_error("unknown VNA matrix test");
    std::cout<<"PASS VNA matrix storage "<<name<<'\n';return 0;
  } catch (const std::exception& error) {
    openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;
  }
}
