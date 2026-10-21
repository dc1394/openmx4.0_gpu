#include <openmx/core/force_workspace.hpp>
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <future>
#include <iostream>
#include <limits>
#include <string>

using namespace openmx::core;
namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void near(double a,long double b,const char* why){require(std::abs(a-b)<2e-12L*(1+std::abs(b)),why);}
struct Topology {
  int map[4]{-17,3,1,4},species[5]{-19,0,1,1,0},orbitals[2]{3,1};
  int counts[5]{271,2,0,1,0},n1[3]{1,2,3},n2[1]{2},n3[2]{3,4},n4[1]{4};
  int* neighbors[5]{nullptr,n1,n2,n3,n4};
  OpenMXForceInput input(int spin=3,bool imaginary=true){return {4,2,1,2,spin,imaginary,map,species,orbitals,counts,neighbors};}
};
struct State {int calls=0,spin=3;bool imaginary=true,fail=false,throws=false;};
int inspect(const OpenMXForceWorkspace* work,void* opaque,double* result)
{
  auto& state=*static_cast<State*>(opaque);++state.calls;
  const auto& l=work->density.layout;
  require(l.atoms==4 && l.components==static_cast<std::size_t>(state.spin+1),"local/halo/spin topology differs");
  require(l.atom_offsets[1]==1 && l.shapes[0]==1 && l.shapes[1]==1,"dummy atom is not explicit 1x1 padding");
  require(l.shapes[2*l.atom_offsets[1]]==1 && l.shapes[2*l.atom_offsets[2]]==3,"first map was replaced with a local map");
  require(bool(work->imaginary_rows)==state.imaginary && bool(work->imaginary.values)==state.imaginary,"imaginary density policy differs");
  for(std::size_t c=0;c<l.components;++c)for(std::size_t a=0;a<l.atoms;++a)
    for(std::size_t b=l.atom_offsets[a];b<l.atom_offsets[a+1];++b)
      for(int i=0;i<l.shapes[2*b];++i)for(int j=0;j<l.shapes[2*b+1];++j) {
        const auto index=c*l.stride+l.block_offsets[b]+i*l.shapes[2*b+1]+j;
        auto& row=work->density_rows[c][a][b-l.atom_offsets[a]][i];
        require(row+j==work->density.values+index && row[j]==0,"borrowed rows do not match fresh contiguous density");
        row[j]=index+.5;
      }
  if(state.imaginary){require(work->imaginary.layout.components==2,"imaginary spin diagonal layout differs");
    for(std::size_t i=0;i<work->imaginary.layout.elements;++i)require(work->imaginary.values[i]==0,"imaginary density leaked across calls");}
  for(auto f:work->forces)for(int i=0;i<3;++i){require(f[i]==0,"force leaked across calls");f[i]=4+i;}
  const std::size_t shape[4]{3,5,7,1};char error[256];
  for(int worker=0;worker<2;++worker) {
    OpenMXSolverComplexTensor first,second;dcomplex*** rows;
    require(openmx_core_solver_complex(work->scratch,worker,"force-derivative",shape,&first,error,sizeof(error)),error);
    require(first.values[104].r==0,"worker scratch leaked across calls");first.values[104]={13.+worker,-2};
    require(openmx_core_solver_complex_alias(work->scratch,worker,"force-derivative",&first,&rows,error,sizeof(error)),error);
    require(rows[2][4]+6==first.values+104 && rows[2][4][6].r==13+worker,"derivative aliases have wrong plane stride");
    require(openmx_core_solver_complex(work->scratch,worker,"force-derivative",shape,&second,error,sizeof(error)),error);
    require(first.values==second.values && second.values[104].r==13+worker,"worker scratch was reallocated or cleared");
  }
  *result=19;
  if(state.throws)throw std::runtime_error("injected force callback exception");
  return !state.fail;
}
void layouts_test()
{
  Topology t;char error[256];double result;
  for(int spin:{0,1,3})for(bool imaginary:{false,true}) {
    State state;state.spin=spin;state.imaginary=imaginary;auto in=t.input(spin,imaginary);
    require(openmx_core_force_run(&in,2,inspect,&state,&result,error,sizeof(error)),error);
    require(state.calls==1 && result==19 && t.counts[0]==271,"force invocation modified global dummy neighbor count");
  }
  auto in=t.input();auto layout=force_layout(in);t.map[1]=4;t.counts[3]=0;
  require(layout.shapes().extent(0)==7 && element(layout.shapes(),2,1)==3,"layout borrows mutable input topology");
  OpenMXForceInput empty{};const auto empty_layout=force_layout(empty);
  ForceMatrices owner(empty_layout,0,false,0);
  require(owner.density().layout().atoms()==1 && owner.forces().extent(1)==1 && !owner.imaginary(),"empty MPI rank lost padding");
}
struct Matrix {
  std::vector<std::size_t> components{0},atoms{0},blocks{0};
  std::vector<int> shapes;std::vector<double> values;
  explicit Matrix(int channels,int padding=0) {
    for(int c=0;c<channels;++c) {
      for(int a=0;a<2;++a) {
        for(int h=0;h<(a?3:1);++h) {
          const int rows=a?2+padding:1,cols=a?(h==0?2:(h==1?1:3))+padding:1;
          shapes.push_back(rows);shapes.push_back(cols);
          for(int i=0;i<rows*cols;++i)values.push_back(std::sin(.3+c*.4+h*.21+i*.13));
          blocks.push_back(values.size());
        }
        atoms.push_back(blocks.size()-1);
      }
      components.push_back(atoms.size()-1);
    }
  }
  OpenMXElectronicMatrixView view(){return {1,components.size()-1,components.size()-1,atoms.size()-1,blocks.size()-1,values.size(),components.data(),atoms.data(),blocks.data(),shapes.data(),values.data()};}
  double& at(int c,int h,int i,int j){const auto b=atoms[components[c]+1]+h;return values[blocks[b]+i*shapes[2*b+1]+j];}
};
void overlap_test()
{
  for(int padding:{0,2}) {
  Matrix s(4),e(2,padding);char error[256];auto sv=s.view(),ev=e.view();
  for(int spin:{0,1,3}) {
    std::array<double,3> force{1.7,-.3,2.1};std::array<long double,3> expected{force[0],force[1],force[2]};
    for(int h=1;h<=2;++h)for(int i=0;i<2;++i)for(int j=0;j<(h==1?1:3);++j)
      for(int axis=0;axis<3;++axis)expected[axis]-=2*(spin?(long double)e.at(0,h,i,j)+e.at(1,h,i,j):2.L*e.at(0,h,i,j))*s.at(axis+1,h,i,j);
    require(openmx_core_force_overlap(&sv,&ev,spin,1,2,force.data(),error,sizeof(error)),error);
    for(int axis=0;axis<3;++axis)near(force[axis],expected[axis],"overlap force differs from independent trace");
    // The trace is the negative derivative of a linear overlap energy.
    const auto energy=[&](long double x,int axis){long double value=0;
      for(int h=1;h<=2;++h)for(int i=0;i<2;++i)for(int j=0;j<(h==1?1:3);++j)
        value+=2*(spin?(long double)e.at(0,h,i,j)+e.at(1,h,i,j):2.L*e.at(0,h,i,j))*x*s.at(axis+1,h,i,j);
      return value;};
    for(int axis=0;axis<3;++axis)near(force[axis]-std::array<double,3>{1.7,-.3,2.1}[axis],-(energy(1e-4L,axis)-energy(-1e-4L,axis))/2e-4L,"overlap finite difference differs");
  }
  std::array<double,3> force{1,2,3};require(openmx_core_force_overlap(&sv,&ev,0,1,0,force.data(),error,sizeof(error)),error);
  require(force==std::array<double,3>{1,2,3},"onsite overlap contributed to force");
  // Invalid later blocks must not leave partially accumulated forces.
  const auto b=s.atoms[s.components[3]+1]+2;s.shapes[2*b+1]=1;
  require(!openmx_core_force_overlap(&sv,&ev,0,1,2,force.data(),error,sizeof(error)) && error[0],"mismatched derivative shape accepted");
  require(force==std::array<double,3>{1,2,3},"invalid overlap partially changed force");
  }
}
struct Trace {
  std::array<std::vector<double>,4> density;
  std::array<std::vector<double>,2> imaginary;
  std::array<std::vector<dcomplex>,3> h;
  OpenMXForceSpinTrace in{};
  Trace() {
    in.rows=2;in.columns=3;in.derivative_rows=5;in.derivative_stride=7;in.prefactor=2;
    for(int c=0;c<4;++c){density[c].resize(6);for(int i=0;i<6;++i)density[c][i]=std::sin(.2+c*.4+i*.7);in.density[c]=density[c].data();}
    for(int c=0;c<2;++c){imaginary[c].resize(6);for(int i=0;i<6;++i)imaginary[c][i]=std::cos(.1+c*.3+i*.4);in.imaginary[c]=imaginary[c].data();}
    for(int axis=0;axis<3;++axis){h[axis].resize(3*5*7);for(std::size_t i=0;i<h[axis].size();++i)h[axis][i]={std::sin(axis*.3+i*.19),std::cos(axis*.4+i*.17)};in.derivative[axis]=h[axis].data();}
  }
};
void trace_test()
{
  Trace t;char error[256];
  for(int spin:{0,1,3})for(int spinor=0;spinor<=(spin==3);++spinor)for(double pref:{1.,2.,4.}) {
    t.in.spin=spin;t.in.spinor=spinor;t.in.prefactor=pref;
    std::array<double,3> force{-.3,.1,.4};std::array<long double,3> expected{force[0],force[1],force[2]};
    for(int i=0;i<2;++i)for(int j=0;j<3;++j)for(int axis=0;axis<3;++axis) {
      const auto d=[&](int c){return (long double)t.density[c][i*3+j];};
      const auto z=[&](int c){const auto h=t.h[axis][(c*5+i)*7+j];return std::complex<long double>(h.r,h.i);};
      if(spinor) {
        // Independent 2x2 spin-density trace: the lower off-diagonal is the
        // conjugate of the upper; diagonal density includes its imaginary part.
        const std::complex<long double> rho00{d(0),t.imaginary[0][i*3+j]},rho11{d(1),t.imaginary[1][i*3+j]},rho01{d(2),d(3)};
        expected[axis]+=std::real(rho00*z(0)+rho11*z(1)+rho01*z(2)+std::conj(rho01)*std::conj(z(2)));
      } else expected[axis]+=pref*(d(0)*z(0).real()+(spin?d(1)*z(1).real():0));
    }
    require(openmx_core_force_spin_trace(&t.in,force.data(),error,sizeof(error)),error);
    for(int axis=0;axis<3;++axis)near(force[axis],expected[axis],"spin trace sign, prefactor or padded plane stride differs");
  }
}
void invalid_test()
{
  Topology t;char error[256];double result=-7;State state;
  for(int k=0;k<9;++k){auto in=t.input();switch(k){case 0:in.spin=2;break;case 1:in.local_atoms=5;break;
    case 2:in.halo_atoms=3;break;case 3:in.first_to_global=nullptr;break;case 4:in.species=0;break;
    case 5:in.imaginary_density=2;break;case 6:in.atoms=-1;break;case 7:in.neighbor_atoms=nullptr;break;case 8:in.orbitals=nullptr;break;}
    require(!openmx_core_force_run(&in,2,inspect,&state,&result,error,sizeof(error)) && error[0],"invalid force topology accepted");}
  require(state.calls==0 && result==-7,"invalid force entered callback");
  Trace f;f.in.spin=3;f.in.spinor=1;
  for(int k=0;k<8;++k){auto in=f.in;switch(k){case 0:in.derivative_rows=1;break;case 1:in.derivative_stride=2;break;
    case 2:in.derivative_stride=std::numeric_limits<std::size_t>::max();break;case 3:in.density[3]=nullptr;break;
    case 4:in.imaginary[1]=nullptr;break;case 5:in.derivative[2]=nullptr;break;case 6:in.spin=1;break;case 7:in.prefactor=std::nan("");break;}
    std::array<double,3> force{1,2,3};require(!openmx_core_force_spin_trace(&in,force.data(),error,sizeof(error)) && error[0],"invalid spin trace accepted");
    require(force==std::array<double,3>{1,2,3},"invalid spin trace changed force");}
  f.in.rows=0;f.in.density[0]=nullptr;std::array<double,3> force{};
  require(openmx_core_force_spin_trace(&f.in,force.data(),error,sizeof(error)),error);
  require(!openmx_core_force_spin_trace(nullptr,force.data(),nullptr,0),"null spin trace accepted");
  try{ForceMatrices bad(force_layout(t.input()),0,false,std::numeric_limits<std::size_t>::max());throw std::logic_error("force accumulator overflow accepted");}
  catch(const std::invalid_argument&){}
}
void lifecycle_test()
{
  Topology t;auto in=t.input();char error[256];double result;State state;
  const auto live=openmx::test::live_allocations();
  for(int i=0;i<12;++i){state.fail=i%3==1;state.throws=i%3==2;
    const int ok=openmx_core_force_run(&in,2,inspect,&state,&result,error,sizeof(error));
    require(bool(ok)==(!state.fail && !state.throws),"force callback failure/exception was ignored");
    require(openmx::test::live_allocations()==live,"force invocation leaked memory on return/exception");}
}
void allocation_test()
{
  Topology t;auto in=t.input();char error[256];double result;State state;
  int failures=0;bool success=false;const auto live=openmx::test::live_allocations();
  for(int n=0;n<300 && !success;++n){openmx::test::fail_allocation_after(n);
    success=openmx_core_force_run(&in,2,inspect,&state,&result,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    if(!success){++failures;require(error[0],"allocation failure has no error");}
    require(openmx::test::live_allocations()==live,"partial force construction or callback scratch leaked");}
  require(success && failures>30,"force allocation fault coverage incomplete");
}
void concurrent_test()
{
  std::array<std::future<void>,4> jobs;
  for(auto& job:jobs)job=std::async(std::launch::async,[]{Topology t;auto in=t.input();State s;double result;char error[256];
    for(int i=0;i<6;++i)require(openmx_core_force_run(&in,2,inspect,&s,&result,error,sizeof(error)),error);});
  for(auto& job:jobs)job.get();
}
}
int main(int argc,char** argv){try{const std::string test=argc>1?argv[1]:"";
  if(test=="layouts")layouts_test();else if(test=="overlap")overlap_test();else if(test=="spin-trace")trace_test();
  else if(test=="invalid")invalid_test();else if(test=="lifecycle")lifecycle_test();else if(test=="allocation")allocation_test();
  else if(test=="concurrent")concurrent_test();else throw std::runtime_error("unknown test");return 0;
}catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}}
