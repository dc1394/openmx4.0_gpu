#include "openmx_core_atomic_density.h"
#include "allocation_faults.hpp"
#include "density_quadrature_fixture.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
void near(double value,double reference){require(std::isfinite(value) && std::abs(value-reference)<2e-11*(1+std::abs(reference)),"atomic Hartree analytic value differs");}
struct Fixture {
  std::array<int,3> number{1,1,1},pao_points{32,32,32},vps_points{4,4,4};
  std::vector<double> logs=std::vector<double>(3*32),radii=logs;
  std::vector<double> density=std::vector<double>(3*34,1.);
  std::vector<double> vps_radii=std::vector<double>(3*10,1.),output=std::vector<double>(3*10,-99.);
  DensityQuadratureFixture quadrature;
  Fixture(){
    for(int species=0;species<3;++species){
      for(int i=0;i<32;++i){const double x=-2+4.*i/31;logs[species*32+i]=x;radii[species*32+i]=std::exp(x);}
      vps_radii[species*10]=.2;vps_radii[species*10+1]=.5;vps_radii[species*10+2]=1.;vps_radii[species*10+3]=3.;
    }
  }
  OpenMXAtomicViews view(){
    OpenMXAtomicViews v{};v.species.species=3;v.species.atomic_number=number.data();v.species.pao_mesh=pao_points.data();v.species.vps_mesh=vps_points.data();
    v.pao_log_radii={logs.data(),{3,32}};v.pao_radii={radii.data(),{3,32}};v.density={density.data(),{3,34}};
    v.vps_radii={vps_radii.data(),{3,10}};v.hartree_potential={output.data(),{3,10}};return v;
  }
  void pristine()const{for(double value:output)require(value==-99.,"failed Hartree call changed output");}
  void check(bool empty=false)const{
    const double rmin=radii[0],rmax=radii[31],pi=std::acos(-1.);
    for(int species=0;species<3;++species)for(int i=0;i<10;++i){
      if(species==1 && i>=1 && i<=4){
        const double R=vps_radii[10+i-1];
        const double value=empty ? 0. : 4*pi*((R*R*R-rmin*rmin*rmin)/(3*R)+(rmax*rmax-R*R)/2);
        near(output[10+i],value);
      }else require(output[species*10+i]==-99.,"Hartree call changed padding or another species");
    }
  }
};
void contents()
{
  for(bool empty:{false,true}){
    Fixture fixture;fixture.number[1]=empty ? 0 : 1;auto view=fixture.view();char error[256];
    require(openmx_core_atomic_hartree(&view,1,32,fixture.quadrature.nodes.data(),fixture.quadrature.weights.data(),error,sizeof error),error);
    require(!*error,"successful Hartree call retained diagnostic");fixture.check(empty);
  }
}
void invalid()
{
  for(int mode=0;mode<12;++mode){
    Fixture fixture;auto view=fixture.view();std::size_t species=1;int count=32;
    const double* nodes=fixture.quadrature.nodes.data();char error[256];
    if(mode==0)species=3;
    if(mode==1)view.species.pao_mesh=nullptr;
    if(mode==2)fixture.pao_points[1]=5;
    if(mode==3)view.density.extent[1]=32;
    if(mode==4)view.hartree_potential.extent[1]=5;
    if(mode==5)view.vps_radii.values=nullptr;
    if(mode==6)nodes=nullptr;
    if(mode==7)count=0;
    if(mode==8)fixture.quadrature.weights[0]=-1;
    if(mode==9)fixture.vps_radii[11]=0;
    if(mode==10)view.density.extent[0]=std::numeric_limits<std::size_t>::max();
    if(mode==11)fixture.vps_radii[11]=std::numeric_limits<double>::max();
    require(!openmx_core_atomic_hartree(&view,species,count,nodes,fixture.quadrature.weights.data(),error,sizeof error),"invalid Hartree call accepted");
    require(*error,"Hartree failure lost diagnostic");fixture.pristine();
  }
  char error[2]={'x','x'};
  require(!openmx_core_atomic_hartree(nullptr,0,0,nullptr,nullptr,error,2),"null Hartree request accepted");
  require(error[0] && !error[1],"short Hartree diagnostic not terminated");
}
void allocation()
{
  bool completed=false;int failures=0;
  for(int index=0;index<32;++index){
    Fixture fixture;auto view=fixture.view();char error[256];const auto live=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(index);
    const int ok=openmx_core_atomic_hartree(&view,1,32,fixture.quadrature.nodes.data(),fixture.quadrature.weights.data(),error,sizeof error);
    openmx::test::fail_allocation_after(-1);
    require(openmx::test::live_allocations()==live,"atomic Hartree call leaked temporary storage");
    if(ok){completed=true;fixture.check();break;}
    ++failures;require(*error,"allocation failure lost diagnostic");fixture.pristine();
  }
  require(completed && failures>0,"Hartree allocation fault sweep incomplete");
}
}

int main(int argc,char** argv)
{
  try{
    if(argc!=2)throw std::invalid_argument("expected atomic-density test case");const std::string name=argv[1];
    if(name=="contents")contents();else if(name=="invalid")invalid();else if(name=="allocation")allocation();
    else throw std::invalid_argument("unknown atomic-density test case");
    std::cout<<"PASS atomic-density "<<name<<'\n';return 0;
  }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}
}
