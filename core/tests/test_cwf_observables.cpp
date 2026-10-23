#include <openmx_core_cwf.h>
#include <openmx_core_chemical.h>
#include <openmx_core_native_resources.h>
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation() {
  char error[160];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
  require(bool(c),error);return c;
}
OpenMXCWFObservables* values(OpenMXCoreContext* c) {
  char error[160];OpenMXCWFObservables* v=nullptr;
  require(openmx_core_cwf_observables(c,&v,error,sizeof(error)) && !*error,error);return v;
}
OpenMXCWFObservables* prepare(OpenMXCoreContext* c,size_t spins=2,size_t capacity=17) {
  char error[160];OpenMXCWFObservables* v=nullptr;
  require(openmx_core_cwf_observables_prepare(c,spins,capacity,&v,error,sizeof(error)) && !*error,error);return v;
}
void zero(const OpenMXCWFObservables* v) {
  require(v->distance==0,"CWF distance was not reset");
  for(size_t spin=0;spin<v->spin_channels;++spin)for(size_t i=0;i<v->row_capacity;++i)
    require(openmx_cwf_charge_row(v,spin)[i]==0 && openmx_cwf_energy_row(v,spin)[i]==0,"CWF output was not initialized");
}
void seed(OpenMXCWFObservables* v,int code) {
  v->distance=code+.125;
  for(size_t spin=0;spin<v->spin_channels;++spin)for(size_t i=0;i<v->row_capacity;++i) {
    const double value=1000.0*code+100.0*spin+i;
    openmx_cwf_charge_row(v,spin)[i]=value;
    openmx_cwf_energy_row(v,spin)[i]=-value-.5;
  }
}
bool matches(const OpenMXCWFObservables* v,int code) {
  if(v->spin_channels!=2 || v->row_capacity!=17 || v->distance!=code+.125)return false;
  for(size_t spin=0;spin<2;++spin)for(size_t i=0;i<17;++i) {
    const double value=1000.0*code+100.0*spin+i;
    if(openmx_cwf_charge_row(v,spin)[i]!=value || openmx_cwf_energy_row(v,spin)[i]!=-value-.5)return false;
  }
  return true;
}
void empty_view(const OpenMXCWFObservables* v) {
  require(!v->spin_channels && !v->row_capacity && !v->charge && !v->energy && !v->distance,
          "cleared calculation retains CWF observables");
}
void ownership() {
  auto a=calculation(),b=calculation();auto* x=prepare(a.get());auto* y=prepare(b.get());
  require(x!=y && x->charge!=y->charge && x->energy!=y->energy && x->charge!=x->energy,"CWF calculations or quantities share storage");
  char error[160];OpenMXChemicalPotentials* chemical=nullptr;OpenMXCWFGrids* grids=nullptr;
  require(openmx_core_chemical_potentials(a.get(),&chemical,error,sizeof(error)),error);
  require(openmx_core_cwf_grids(a.get(),&grids,error,sizeof(error)),error);
  chemical->chemical_potential=9.25;grids->matrix[2].descriptor[8]=71;
  for(int n=1;n<=100;++n){seed(x,n);seed(y,-n);require(matches(x,n)&&matches(y,-n),"CWF output crossed calculations");}
  require(chemical->chemical_potential==9.25 && grids->matrix[2].descriptor[8]==71,"CWF outputs changed neighboring domain state");
  // Active CWF count can be smaller than the allocation after LNO selection.
  std::fill_n(x->charge,34,-1.0);
  for(size_t spin=0;spin<2;++spin)for(size_t i=0;i<3;++i)openmx_cwf_charge_row(x,spin)[i]=10*spin+i;
  require(x->charge[17]==10 && x->charge[3]==-1 && x->charge[20]==-1,"active count was confused with allocated row stride");
}
void reuse() {
  auto c=calculation();auto* fresh=values(c.get());empty_view(fresh);
  openmx::test::fail_allocation_after(0);
  for(int i=0;i<100;++i)require(values(c.get())==fresh,"allocation-free initial lookup moved output");
  require(openmx::test::remaining_allocations()==0,"initial CWF lookup allocated");
  openmx::test::fail_allocation_after(-1);
  auto* v=prepare(c.get());double* charges=v->charge;double* energies=v->energy;seed(v,7);
  openmx::test::fail_allocation_after(0);
  for(int i=0;i<100;++i)require(values(c.get())==v && matches(v,7),"lookup reset live output");
  require(prepare(c.get())==v && v->charge==charges && v->energy==energies,"equal shape reallocated output");
  zero(v);require(openmx::test::remaining_allocations()==0,"equal-shape preparation allocated");
  openmx::test::fail_allocation_after(-1);
  require(prepare(c.get(),1,5)==v && v->spin_channels==1 && v->row_capacity==5,"shape change moved the view object");zero(v);
}
void invalid() {
  auto c=calculation();auto* v=prepare(c.get());seed(v,3);auto* saved=v;char error[160];
  require(!openmx_core_cwf_observables(nullptr,&v,error,sizeof(error)) && *error && v==saved,"null lookup changed output");
  require(!openmx_core_cwf_observables(c.get(),nullptr,error,sizeof(error)) && *error,"null lookup output accepted");
  require(!openmx_core_cwf_observables_prepare(nullptr,2,17,&v,error,sizeof(error)) && *error && v==saved,"null prepare changed output");
  require(!openmx_core_cwf_observables_prepare(c.get(),2,17,nullptr,error,sizeof(error)) && *error,"missing prepare output accepted");
  for(size_t spins:{size_t(0),size_t(3),std::numeric_limits<size_t>::max()})
    require(!openmx_core_cwf_observables_prepare(c.get(),spins,17,&v,error,sizeof(error)) && *error && v==saved && matches(v,3),"invalid channel count changed output");
  for(size_t capacity:{std::numeric_limits<size_t>::max(),std::numeric_limits<size_t>::max()/sizeof(double)})
    require(!openmx_core_cwf_observables_prepare(c.get(),2,capacity,&v,error,sizeof(error)) && *error && v==saved && matches(v,3),"overflow changed live output");
  char tiny='x';require(!openmx_core_cwf_observables(nullptr,&v,&tiny,1)&&tiny=='\0',"tiny diagnostic is not terminated");
  require(!openmx_core_cwf_observables_prepare(nullptr,2,17,&v,nullptr,0)&&v==saved,"failure requires a diagnostic buffer");
}
void failures() {
  auto a=calculation(),b=calculation();auto* live=prepare(a.get());auto* other=prepare(b.get());seed(live,5);seed(other,-5);
  const auto saved=*live;bool success=false;
  for(long budget=0;budget<40 && !success;++budget) {
    char error[160];OpenMXCWFObservables* output=other;
    openmx::test::fail_allocation_after(budget);
    const int ok=openmx_core_cwf_observables_prepare(a.get(),1,29,&output,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    if(ok){success=true;require(output==live,"successful growth moved the view object");zero(live);}
    else require(*error && output==other && live->charge==saved.charge && live->energy==saved.energy && matches(live,5),
                 "failed allocation replaced output or damaged existing matrices");
    require(matches(other,-5),"failed preparation changed another calculation");
  }
  require(success,"CWF preparation never recovered from allocation failures");
}
void empty() {
  auto c=calculation();auto* v=prepare(c.get());seed(v,3);
  openmx::test::fail_allocation_after(0);
  require(prepare(c.get(),1,0)==v && v->spin_channels==1 && !v->row_capacity && !v->charge && !v->energy && !v->distance,
          "empty shape did not release matrices");
  require(!openmx_cwf_charge_row(v,0) && !openmx_cwf_energy_row(v,0),"empty row performed null pointer arithmetic");
  require(prepare(c.get(),2,0)==v && v->spin_channels==2,"empty shape cannot change spin channels");
  require(openmx::test::remaining_allocations()==0,"empty preparation allocated");
  openmx::test::fail_allocation_after(-1);zero(prepare(c.get()));
}
struct Borrower {OpenMXCWFObservables* values;int code;int* observed;};
void drain(OpenMXNativeResources*,void* opaque) {
  auto* b=static_cast<Borrower*>(opaque);*b->observed=matches(b->values,b->code)?1:-1;
}
void borrow(OpenMXCoreContext* c,int code,int* observed) {
  OpenMXNativeResources* group=nullptr;void* opaque=nullptr;char error[160];
  require(openmx_core_native_resources_state(c,"CWF-observable-borrower",sizeof(Borrower),drain,&group,&opaque,error,sizeof(error)),error);
  auto* state=static_cast<Borrower*>(opaque);*state={values(c),code,observed};
}
void lifetime() {
  const auto initial=openmx::test::live_allocations();
  for(int action=0;action<2;++action)for(int n=1;n<=60;++n) {
    int cleared=0,destroyed=0;
    {
      auto a=calculation(),b=calculation();auto* x=prepare(a.get());auto* y=prepare(b.get());seed(x,n);seed(y,-n);
      borrow(a.get(),n,&cleared);borrow(b.get(),-n,&destroyed);
      openmx::test::fail_allocation_after(0);
      if(action==0)openmx_core_native_resources_clear(a.get());else openmx_core_finish(a.get());
      openmx_core_finish(a.get());openmx::test::fail_allocation_after(-1);
      require(cleared==1,"CWF matrices reset before native borrowers drained");empty_view(values(a.get()));
      require(matches(y,-n),"finishing one calculation reset another");seed(prepare(a.get()),n+1);
    }
    require(destroyed==1,"CWF matrices destroyed before native borrowers drained");
  }
  require(openmx::test::live_allocations()==initial,"CWF output survived calculation destruction");
}
void concurrent() {
  std::array<int,4> ok{};std::array<std::thread,4> threads;
  for(int worker=0;worker<4;++worker)threads[worker]=std::thread([&,worker] {
    try {
      auto c=calculation();auto* v=prepare(c.get());
      for(int n=1;n<=100;++n){seed(v,1000*worker+n);require(values(c.get())==v && matches(v,1000*worker+n),"concurrent outputs share storage");}
      ok[worker]=1;
    }catch(...){}
  });
  for(auto& thread:threads)thread.join();for(int value:ok)require(value==1,"concurrent CWF owner failed");
}
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected one case");const std::string_view name=argv[1];
    if(name=="ownership")ownership();else if(name=="reuse")reuse();else if(name=="invalid")invalid();
    else if(name=="failures")failures();else if(name=="empty")empty();else if(name=="lifetime")lifetime();
    else if(name=="concurrent")concurrent();else require(false,"unknown test case");
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}
