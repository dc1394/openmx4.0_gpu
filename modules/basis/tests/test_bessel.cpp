#include <openmx/basis/bessel.hpp>
#include <openmx/basis/pair_bessel.hpp>
#include "allocation_gate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace openmx::basis;
void require(bool condition, const char* reason)
{ if (!condition) throw std::runtime_error(reason); }
void near(double value, double reference, double tolerance=2e-12)
{
  if (!std::isfinite(value) || std::abs(value-reference)>tolerance*(1+std::abs(reference))) {
    std::cerr << "value=" << value << " reference=" << reference << '\n';
    throw std::runtime_error("Bessel value differs from independent oracle");
  }
}
template<class F> void rejects(F&& operation)
{
  bool rejected=false;
  try { operation(); } catch (const std::invalid_argument&) { rejected=true; }
  require(rejected,"invalid Bessel input accepted");
}

// No dependency on std::sph_bessel (unavailable in libc++). A power series,
// closed forms with upward recurrence, and a Legendre integral deliberately
// differ from the production downward recurrence.
double reference(int l, double x)
{
  if (x<=12) {
    long double term=1;
    for (int n=1;n<=l;++n) term*=x/(2*n+1.L);
    long double sum=term;
    for (int n=1;n<200;++n) {
      term*=-static_cast<long double>(x)*x/(2*n*(2*l+2*n+1.L));
      sum+=term;
      if (std::abs(term)<=std::numeric_limits<long double>::epsilon()*std::abs(sum) || term==0)
        return static_cast<double>(sum);
    }
    throw std::runtime_error("reference series did not converge");
  }
  if (l<=1 || x>2*(l+1)) {
    long double a=std::sin(static_cast<long double>(x))/x;
    long double b=(a-std::cos(static_cast<long double>(x)))/x;
    if (!l) return static_cast<double>(a);
    for (int n=1;n<l;++n) {
      const long double next=(2*n+1)*b/x-a;
      a=b; b=next;
    }
    return static_cast<double>(b);
  }
  constexpr int steps=32768;
  long double sum=0;
  for (int i=0;i<=steps;++i) {
    const long double t=static_cast<long double>(i)/steps;
    long double a=1,b=t;
    for (int n=1;n<l;++n) {
      const long double next=((2*n+1)*t*b-n*a)/(n+1);
      a=b; b=next;
    }
    const long double wave=l%2 ? std::sin(x*t) : std::cos(x*t);
    sum+=(i==0 || i==steps ? 1 : i%2 ? 4 : 2)*wave*b;
  }
  return static_cast<double>((l%4<2 ? 1 : -1)*sum/(3*steps));
}

void values()
{
  for (int order : {0,1,4,12,29})
    for (double x : {0.,1e-12,1e-8,.01,.2,1.,4.2,8.,12.,20.,70.,1000.}) {
      const SphericalBesselSample sample(x,order);
      require(sample.maximum_order()==order && sample.values().size()==static_cast<std::size_t>(order+2)
          && sample.derivatives().size()==static_cast<std::size_t>(order+1),"wrong sample shape");
      for (int l=0;l<=order+1;++l) near(sample.values()[l],reference(l,x));
    }
  const std::array grid{0.,.125,.5,1.,3.,8.};
  PairBesselWorkspace workspace(29,grid,3);
  for (int order : {0,1,4,12,29}) {
    const auto table=workspace.sample(.75,order);
    require(table.points()==grid.size() && table.maximum_order()==order,"wrong pair table shape");
    for (int l=0;l<=order;++l)
      for (std::size_t k=0;k<grid.size();++k)
        near(table.values()[l*grid.size()+k],reference(l,grid[k]*.75));
  }
}

void derivatives()
{
  for (int order : {0,1,4,12,29}) {
    const SphericalBesselSample origin(0,order);
    for (int l=0;l<=order;++l) near(origin.derivatives()[l],l==1 ? 1./3 : 0.);
    for (double x : {.01,.2,1.,4.2,8.,70.}) {
      const SphericalBesselSample sample(x,order);
      for (int l=0;l<=order;++l) {
        const double h=1e-5;
        near(sample.derivatives()[l],(reference(l,x+h)-reference(l,x-h))/(2*h),3e-9);
      }
    }
  }
  const std::array grid{0.,.25,1.,2.};
  PairBesselWorkspace workspace(4,grid,0);
  const auto table=workspace.sample(.7,4);
  for (int l=0;l<=4;++l)
    for (std::size_t k=0;k<grid.size();++k) {
      // Verify derivative units independently: multiplying by k must give
      // a finite-difference derivative with respect to distance, not x.
      const double h=1e-5;
      near(grid[k]*table.derivatives()[l*grid.size()+k],
          (reference(l,grid[k]*(.7+h))-reference(l,grid[k]*(.7-h)))/(2*h),3e-9);
    }
}

void cache()
{
  std::array grid{0.,.5,1.,2.};
  PairBesselWorkspace workspace(4,grid,2);
  const auto first=workspace.sample(1.,2);
  grid[2]=25;
  const auto second=workspace.sample(2.,1);
  near(first.values()[2],std::sin(1.));
  near(second.values()[2],std::sin(2.)/2);
  require(first.row_capacity()==3 && second.row_capacity()==2,"cached table wasted rows");
  allocations_before_failure=0;
  const auto hit=workspace.sample(1.,2);
  allocations_before_failure=-1;
  require(hit.storage().data()==first.storage().data(),"cache hit copied or allocated");
  const auto scratch=workspace.sample(3.,1);
  require(scratch.row_capacity()==5 && scratch.values().size()==8 && scratch.derivatives().size()==8,
          "scratch capacity exposed inactive rows as valid results");
  require(scratch.derivatives().data()==scratch.storage().data()+5*4,"scratch derivative stride changed");
  allocations_before_failure=0;
  const auto again=workspace.sample(4.,0);
  allocations_before_failure=-1;
  require(again.storage().data()==scratch.storage().data(),"full cache reallocated scratch");
  near(first.values()[2],std::sin(1.));
  require(workspace.cached_tables()==2 && workspace.statistics().hits==1 && workspace.statistics().misses==4,
          "cache limit or statistics changed");
  PairBesselWorkspace zeros(4,std::span(grid).first(1),3);
  const auto zero=zeros.sample(0.,2);
  require(zero.storage().data()==zeros.sample(-0.,2).storage().data(),"signed-zero key differs");
  require(zero.storage().data()!=zeros.sample(0.,1).storage().data(),"angular order missing from key");
  PairBesselWorkspace exact(2,std::span(grid).first(1),2);
  const auto a=exact.sample(1.,1);
  require(a.storage().data()!=exact.sample(std::nextafter(1.,2.),1).storage().data(),
          "cache conflated distinct distances");
}

void invalid()
{
  const std::array grid{0.,1.,2.};
  for (int order : {-1,30}) {
    rejects([&]{ (void)SphericalBesselSample(1,order); });
    rejects([&]{ PairBesselWorkspace invalid(order,grid,1); });
  }
  rejects([&]{ PairBesselWorkspace invalid(2,{},1); });
  PairBesselWorkspace workspace(4,grid,2);
  const auto retained=workspace.sample(1.,2);
  const auto scratch=workspace.sample(2.,1);
  const std::vector<double> saved(retained.storage().begin(),retained.storage().end());
  for (double value : {-1.,std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),std::numeric_limits<double>::max()}) {
    rejects([&]{ (void)SphericalBesselSample(value,2); });
    rejects([&]{ (void)workspace.sample(value,2); });
    if (value!=std::numeric_limits<double>::max()) {
      const std::array bad{0.,value};
      rejects([&]{ PairBesselWorkspace invalid(2,bad,1); });
    }
  }
  for (int order : {-1,5}) rejects([&]{ (void)workspace.sample(1,order); });
  require(std::equal(saved.begin(),saved.end(),retained.storage().begin()),"invalid sample changed cached data");
  near(scratch.values()[1],std::sin(2.)/2);
  require(workspace.statistics().misses==2 && workspace.statistics().hits==0,"invalid sample changed statistics");
  PairBesselWorkspace uncached(4,grid,0);
  const auto previous=uncached.sample(1,2);
  const std::vector<double> scratch_saved(previous.storage().begin(),previous.storage().end());
  rejects([&]{ (void)uncached.sample(-1,2); });
  require(std::equal(scratch_saved.begin(),scratch_saved.end(),previous.storage().begin()),
          "invalid sample changed uncached data");
}

void ownership()
{
  const auto source=[] { return SphericalBesselSample(1,4); }();
  const auto copy=source;
  require(copy.values().data()!=source.values().data(),"Bessel sample copy aliases storage");
  allocations_before_failure=0;
  const SphericalBesselSample no_heap(2,29);
  allocations_before_failure=-1;
  near(no_heap.values()[0],std::sin(2.)/2);
  const std::array grid{0.,1.,2.};
  PairBesselWorkspace first(4,grid,1),second(4,grid,1);
  const auto a=first.sample(1,2),b=second.sample(1,2);
  require(a.storage().data()!=b.storage().data(),"independent workspaces share storage");
  near(a.values()[1],std::sin(1.)); near(b.values()[1],std::sin(1.));
}

void allocation()
{
  const std::array grid{0.,1.,2.};
  bool complete=false; int failures=0;
  for (int index=0;index<16;++index) {
    allocations_before_failure=index;
    try { PairBesselWorkspace workspace(4,grid,4); complete=true; }
    catch (const std::bad_alloc&) { ++failures; }
    allocations_before_failure=-1;
    if (complete) break;
  }
  require(complete && failures==2,"constructor allocation sweep incomplete");
  for (int index : {0,1}) {
    PairBesselWorkspace workspace(4,grid,4);
    const auto retained=workspace.sample(1,2);
    allocations_before_failure=index;
    const auto recovered=workspace.sample(2,1);
    allocations_before_failure=-1;
    require(workspace.cached_tables()==1 && recovered.row_capacity()==5,"failed cache growth did not use scratch");
    near(retained.values()[1],std::sin(1.)); near(recovered.values()[1],std::sin(2.)/2);
    allocations_before_failure=0;
    const auto next=workspace.sample(3,3);
    const auto hit=workspace.sample(1,2);
    allocations_before_failure=-1;
    require(next.storage().data()==recovered.storage().data() && hit.storage().data()==retained.storage().data(),
            "cache failure invalidated storage or resumed allocations");
    near(next.values()[1],std::sin(3.)/3);
    require(workspace.statistics().misses==3 && workspace.statistics().hits==1,"recovery statistics changed");
  }
}

void concurrent()
{
  const SphericalBesselSample shared(1,4);
  std::array<std::future<void>,4> workers;
  for (int i=0;i<4;++i) workers[i]=std::async(std::launch::async,[&,i] {
    const std::array grid{0.,.25,1.};
    PairBesselWorkspace own(4,grid,2);
    for (int repeat=0;repeat<20;++repeat) {
      const double distance=.1*(1+i+repeat%7);
      const auto table=own.sample(distance,repeat%5);
      for (int l=0;l<=table.maximum_order();++l)
        for (std::size_t k=0;k<grid.size();++k) near(table.values()[l*grid.size()+k],reference(l,grid[k]*distance));
      near(shared.values()[0],std::sin(1.));
    }
  });
  for (auto& worker : workers) worker.get();
}
} // namespace

int main(int argc,char** argv)
{
  try {
    if (argc!=2) throw std::invalid_argument("expected Bessel test case");
    const std::string name=argv[1];
    if (name=="values") values(); else if (name=="derivatives") derivatives();
    else if (name=="cache") cache(); else if (name=="invalid") invalid();
    else if (name=="ownership") ownership(); else if (name=="allocation") allocation();
    else if (name=="concurrent") concurrent(); else throw std::invalid_argument("unknown Bessel test case");
    std::cout << "PASS Bessel " << name << '\n'; return 0;
  } catch (const std::exception& error) {
    allocations_before_failure=-1;
    std::cerr << error.what() << '\n'; return 1;
  }
}
