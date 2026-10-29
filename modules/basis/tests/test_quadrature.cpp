#include <openmx/basis/quadrature.hpp>
#include <openmx/basis/quadrature.h>
#include "allocation_gate.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace openmx::basis;
void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
void near(double value, double reference) {
  require(std::isfinite(value) && std::abs(value-reference) < 2e-11*(1+std::abs(reference)), "quadrature moment differs");
}
template<class F> void rejects(F&& f) {
  bool caught=false; try { f(); } catch(const std::exception&) { caught=true; }
  require(caught,"invalid quadrature accepted");
}
double moment(const GaussLegendreView& view, int degree) {
  double sum=0;
  for(std::size_t i=0;i<view.nodes().size();++i) sum+=view.weights()[i]*std::pow(view.nodes()[i],degree);
  return sum;
}
void moments() {
  for(int n:{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31,32,64,100,150,300,1500}) {
    const GaussLegendreRule rule(n); const auto view=rule.view();
    require(view.nodes().size()==static_cast<std::size_t>(n),"wrong quadrature order");
    for(int i=0;i<n;++i) {
      require(view.nodes()[i]>-1 && view.nodes()[i]<1 && view.weights()[i]>0,"invalid generated node or weight");
      require(!i || view.nodes()[i]>view.nodes()[i-1],"nodes not ordered");
      // The legacy Newton iterate can leave a tiny nonzero central node.
      if(i==n-1-i)near(view.nodes()[i],0);
      else require(view.nodes()[i]==-view.nodes()[n-1-i],"rule lost node symmetry");
      require(view.weights()[i]==view.weights()[n-1-i],"rule lost weight symmetry");
    }
    for(int degree=0;degree<=std::min(2*n-1,64);++degree)
      near(moment(view,degree),degree%2 ? 0. : 2./(degree+1));
  }
  const GaussLegendreRule two(2); near(two.view().nodes()[0],-std::sqrt(1./3)); near(two.view().weights()[0],1);
  const GaussLegendreRule three(3); near(three.view().nodes()[0],-std::sqrt(3./5)); near(three.view().weights()[1],8./9);
}
void invalid() {
  for(int count:{-1,0,INT_MAX}) rejects([&]{ (void)GaussLegendreRule(count); });
  std::array<double,3> nodes{-99,-99,-99},weights=nodes; char error[128];
  require(!openmx_basis_gauss_legendre(0,nodes.data(),weights.data(),error,sizeof error),"zero C order accepted");
  require(*error,"missing C diagnostic");
  require(!openmx_basis_gauss_legendre(3,nullptr,weights.data(),error,sizeof error),"null C output accepted");
  require(!openmx_basis_gauss_legendre(3,nodes.data(),nodes.data(),error,sizeof error),"identical C outputs accepted");
  for(double value:nodes)require(value==-99,"invalid C call changed nodes");
  for(double value:weights)require(value==-99,"invalid C call changed weights");
  char short_error[2]={'x','x'};
  require(!openmx_basis_gauss_legendre(-1,nodes.data(),weights.data(),short_error,2),"negative C order accepted");
  require(short_error[0] && !short_error[1],"C diagnostic not terminated");
}
void ownership() {
  GaussLegendreRule original(8); auto copy=original;
  require(copy.view().nodes().data()!=original.view().nodes().data(),"quadrature copy shares ownership");
  auto moved=std::move(copy); rejects([&]{(void)copy.view();});
  original=GaussLegendreRule(3); copy=moved; moved=std::move(copy);
  require(moved.view().nodes().size()==8 && original.view().nodes().size()==3,"rule ownership changed another rule");
  allocations_before_failure=0; const auto view=moved.view(); const double value=moment(view,4); allocations_before_failure=-1;
  near(value,2./5);
}
void concurrent() {
  const GaussLegendreRule shared(32); const auto view=shared.view();
  std::array<std::future<void>,4> workers;
  for(int i=0;i<4;++i)workers[i]=std::async(std::launch::async,[&,i]{
    for(int repeat=0;repeat<20;++repeat) {
      const GaussLegendreRule own(12+i); near(moment(own.view(),6),2./7); near(moment(view,8),2./9);
    }
  });
  for(auto& worker:workers)worker.get();
}
void allocation() {
  bool complete=false; int failures=0;
  for(int index=0;index<32;++index) {
    allocations_before_failure=index;
    try { const GaussLegendreRule rule(8); complete=rule.view().nodes().size()==8; }
    catch(const std::bad_alloc&) { ++failures; }
    allocations_before_failure=-1; if(complete)break;
  }
  require(complete && failures>0,"quadrature construction fault sweep incomplete");
  GaussLegendreRule target(3); const GaussLegendreRule source(8); complete=false; failures=0;
  for(int index=0;index<32;++index) {
    allocations_before_failure=index;
    try { target=source; complete=true; } catch(const std::bad_alloc&) { ++failures; }
    allocations_before_failure=-1; if(complete)break;
    require(target.view().nodes().size()==3,"failed assignment changed rule"); near(moment(target.view(),2),2./3);
  }
  require(complete && failures>0,"quadrature assignment fault sweep incomplete");
  complete=false; failures=0;
  for(int index=0;index<32;++index) {
    std::array<double,8> nodes,weights; nodes.fill(-99); weights.fill(-99); char error[128];
    allocations_before_failure=index;
    const int ok=openmx_basis_gauss_legendre(8,nodes.data(),weights.data(),error,sizeof error);
    allocations_before_failure=-1;
    if(ok) { complete=true; near(moment(GaussLegendreView(nodes,weights),4),2./5); break; }
    ++failures; require(*error,"allocation failure lost diagnostic");
    for(double value:nodes)require(value==-99,"failed C allocation changed nodes");
    for(double value:weights)require(value==-99,"failed C allocation changed weights");
  }
  require(complete && failures>0,"C quadrature fault sweep incomplete");
}
}

int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::invalid_argument("expected quadrature test case"); const std::string name=argv[1];
    if(name=="moments")moments(); else if(name=="invalid")invalid(); else if(name=="ownership")ownership();
    else if(name=="concurrent")concurrent(); else if(name=="allocation")allocation();
    else throw std::invalid_argument("unknown quadrature test case");
    std::cout<<"PASS quadrature "<<name<<'\n'; return 0;
  } catch(const std::exception& error) { allocations_before_failure=-1; std::cerr<<error.what()<<'\n'; return 1; }
}
