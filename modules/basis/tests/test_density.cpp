#include <openmx/basis/density_table.hpp>
#include <openmx/basis/density.h>
#include "allocation_gate.hpp"
#include "density_reference.hpp"
#include "density_quadrature_fixture.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace openmx::basis;
void require(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
void near(double value,double reference,double tolerance=2e-12){
  require(std::isfinite(value) && std::abs(value-reference)<=tolerance*(1+std::abs(reference)),"density value differs from reference");
}
template<class F> void rejects(F&& function){bool caught=false;try{function();}catch(const std::exception&){caught=true;}require(caught,"invalid density input accepted");}
struct Analytic {
  std::array<double,32> logs{},radii{},values{};
  explicit Analytic(bool constant=false){
    for(std::size_t i=0;i<logs.size();++i){
      const double x=-2+4.*i/(logs.size()-1);logs[i]=x;radii[i]=std::exp(x);values[i]=constant ? 1. : 2+3*x+4*x*x;
    }
  }
  LogDensityTable table() const{return {logs,radii,values};}
};

void interpolation()
{
  const Analytic data;const auto owner=data.table();const auto view=owner.view();
  for(double x:{-1.5,-1.,-.5,0.,.7,1.5}){
    const double r=std::exp(x);near(view.sample_log(x),2+3*x+4*x*x);near(view.sample(r),2+3*x+4*x*x);
    near(view.derivative(r),(3+8*x)/r);
    const double h=1e-6*r;
    near(view.derivative(r),(view.sample(r+h)-view.sample(r-h))/(2*h),2e-8);
  }
  // The origin continuation preserves the legacy radial stencil and has a
  // consistent derivative away from its join with the logarithmic table.
  const double r=std::exp(-4),h=1e-5*r;
  near(view.derivative(r),(view.sample(r+h)-view.sample(r-h))/(2*h),2e-8);
  near(view.derivative(0),0);
  const DensityReference reference;const LogDensityTable frozen(reference.logs,reference.radii,reference.physical);
  for(const auto& row:reference.samples){
    near(frozen.view().sample_log(row[0]),row[1]);near(frozen.derivative(std::exp(row[0])),row[2]);
    const auto v=frozen.view();
    near(openmx_basis_density_value(8,row[0],reference.logs.data(),reference.radii.data(),v.padded_density().data()),row[1]);
    near(openmx_basis_density_derivative(8,row[0],std::exp(row[0]),reference.logs.data(),reference.radii.data(),v.padded_density().data()),row[2]);
  }
  near(view.sample_log(std::numeric_limits<double>::infinity()),data.values.back());
  near(view.sample_log(-std::numeric_limits<double>::infinity()),view.sample(0));
}

void hartree()
{
  const Analytic data(true);const auto owner=data.table();const auto density=owner.view();
  const DensityQuadratureFixture rule;const GaussLegendreView quadrature(rule.nodes,rule.weights);
  double sum=0;for(double w:rule.weights)sum+=w;near(sum,2);
  const double pi=std::acos(-1.),rmin=data.radii.front(),rmax=data.radii.back();
  for(double R:{rmin,.3,.5,1.,2.,5.,rmax}){
    const double exact=4*pi*((R*R*R-rmin*rmin*rmin)/(3*R)+(rmax*rmax-R*R)/2);
    near(atomic_hartree_potential(density,R,quadrature),exact);
  }
  // The same immutable density can be integrated with an independently
  // supplied rule; there is no retained/global quadrature selection.
  auto reversed=rule;std::reverse(reversed.nodes.begin(),reversed.nodes.end());std::reverse(reversed.weights.begin(),reversed.weights.end());
  near(atomic_hartree_potential(density,1,GaussLegendreView(reversed.nodes,reversed.weights)),
       atomic_hartree_potential(density,1,quadrature));
  allocations_before_failure=0;
  const auto value=atomic_hartree_potential(density,1,quadrature);
  const auto sample=density.sample(.7),derivative=density.derivative(.7);
  allocations_before_failure=-1;
  require(std::isfinite(value),"allocation-free Hartree failed");near(sample,1);near(derivative,0);
}

void invalid()
{
  const Analytic original;const auto owner=original.table();const auto values=owner.view().padded_density();
  for(int mode=0;mode<6;++mode){
    auto bad=original;
    if(mode==0)bad.logs[4]=bad.logs[3];
    if(mode==1)bad.radii[4]=-1;
    if(mode==2)bad.logs[4]+=.01;
    if(mode==3)bad.radii[4]*=1.01;
    if(mode==4)bad.values[4]=std::numeric_limits<double>::quiet_NaN();
    if(mode==5)bad.values[0]=std::numeric_limits<double>::max();
    rejects([&]{(void)bad.table();});
  }
  rejects([&]{(void)LogDensityView(std::span(original.logs).first(5),std::span(original.radii).first(5),values.first(7));});
  rejects([&]{(void)LogDensityView(original.logs,original.radii,values.first(32));});
  rejects([&]{(void)owner.sample(-1);});rejects([&]{(void)owner.derivative(std::numeric_limits<double>::infinity());});
  rejects([&]{(void)owner.view().sample_log(std::numeric_limits<double>::quiet_NaN());});
  DensityQuadratureFixture rule;
  rejects([&]{(void)GaussLegendreView({},{});});
  rejects([&]{(void)GaussLegendreView(rule.nodes,std::span(rule.weights).first(3));});
  auto bad=rule;bad.nodes[0]=-1.1;rejects([&]{(void)GaussLegendreView(bad.nodes,bad.weights);});
  bad=rule;bad.weights[0]=0;rejects([&]{(void)GaussLegendreView(bad.nodes,bad.weights);});
  const GaussLegendreView quadrature(rule.nodes,rule.weights);
  rejects([&]{(void)atomic_hartree_potential(owner.view(),0,quadrature);});
  require(std::isnan(openmx_basis_density_value(5,0,original.logs.data(),original.radii.data(),values.data())),"C ABI accepted short grid");
  require(std::isnan(openmx_basis_density_value(32,0,nullptr,original.radii.data(),values.data())),"C ABI accepted null row");
  require(std::isnan(openmx_basis_density_derivative(32,0,0,original.logs.data(),original.radii.data(),values.data())),"C ABI accepted zero denominator");
}

void ownership()
{
  Analytic input;auto owner=input.table();auto copy=owner;auto moved=std::move(copy);
  input.values.fill(-1);owner=Analytic(true).table();copy=moved;moved=std::move(copy);
  near(moved.sample(1),2);near(owner.sample(1),1);
  const auto borrowed=moved.view();near(borrowed.sample(1),2);
  allocations_before_failure=0;const auto value=borrowed.sample(1),derivative=borrowed.derivative(1);allocations_before_failure=-1;
  near(value,2);near(derivative,3);
}
void concurrent()
{
  const Analytic input(true);const auto owner=input.table();const auto density=owner.view();
  const DensityQuadratureFixture rule;const GaussLegendreView quadrature(rule.nodes,rule.weights);
  const double reference=atomic_hartree_potential(density,.75,quadrature);
  std::array<std::future<void>,4> workers;
  for(auto& worker:workers)worker=std::async(std::launch::async,[&]{for(int i=0;i<100;++i){near(density.sample(.75),1);near(atomic_hartree_potential(density,.75,quadrature),reference);}});
  for(auto& worker:workers)worker.get();
}
void allocation()
{
  const Analytic data;bool completed=false;int failures=0;
  for(int index=0;index<32;++index){
    allocations_before_failure=index;
    try{const auto owner=data.table();completed=owner.view().radii().size()==32;}catch(const std::bad_alloc&){++failures;}
    allocations_before_failure=-1;if(completed)break;
  }
  require(completed && failures>0,"density construction allocation sweep incomplete");
  auto target=Analytic(true).table();const auto source=data.table();completed=false;failures=0;
  for(int index=0;index<32;++index){
    allocations_before_failure=index;
    try{target=source;completed=true;}catch(const std::bad_alloc&){++failures;}
    allocations_before_failure=-1;if(completed)break;near(target.sample(1),1);
  }
  require(completed && failures>0,"density assignment allocation sweep incomplete");near(target.sample(1),2);
}
}

int main(int argc,char** argv)
{
  try{
    if(argc!=2)throw std::invalid_argument("expected density test case");const std::string name=argv[1];
    if(name=="interpolation")interpolation();else if(name=="hartree")hartree();else if(name=="invalid")invalid();
    else if(name=="ownership")ownership();else if(name=="concurrent")concurrent();else if(name=="allocation")allocation();
    else throw std::invalid_argument("unknown density test case");
    std::cout<<"PASS density "<<name<<'\n';return 0;
  }catch(const std::exception& error){allocations_before_failure=-1;std::cerr<<error.what()<<'\n';return 1;}
}
