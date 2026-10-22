#include "openmx_mixing_v_internal.h"
#include "openmx_env.h"
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

constexpr int GPUSOLVER=3,mpi_comm_level1=0;
int scf_eigen_lib_flag=GPUSOLVER,List_YOUSO[60]={},log_kind=0,prints[2]={};
int atomnum=1,Matomnum=1,SpeciesNum=1,M2G[]={0,1},WhatSpecies[]={0,0},Spe_Total_CNO[]={1},FNAN[]={0,0},neighbor[]={1};
int* natn[]={neighbor,neighbor};
struct AllocationAbort:std::exception {};
void MPI_Abort(int,int){throw AllocationAbort();}
void allocation_abort(){throw AllocationAbort();}
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void PrintMemory(const char*,double,const char*){prints[log_kind]++;}
int clock_calls=0;long clock_live=0;
void MPI_Comm_size(int,int *size){*size=4;}
void MPI_Comm_rank(int,int *rank){*rank=0;}
void MPI_Barrier(int){}
void dtime(double *value){
  if(clock_calls%2==0){clock_live=openmx::test::live_allocations();*value=100;}
  else {check(openmx::test::live_allocations()==clock_live,"potential timing ended before scratch cleanup");*value=107.5;}
  clock_calls++;
}
#include "openmx_mixing_v_workspace.h"
static double Mixing_V_Compute(OpenMXSolverInvocation*,MixingVState*,int,int,int);
#define F77_NAME(lower,upper) lower
int lapack_n=0,lu_calls=0,inverse_calls=0;
std::array<double,64> lapack_original;
// Model the two-stage, column-major LAPACK contract. Independent Gauss-Jordan
// inversion supplies the library result; the production function must keep
// dimensions, pivot/work arrays and the factorization/inversion call order.
static void dgetrf(int* m,int* n,double* a,int* lda,int* pivots,int* info){
  check(*m==*n && *n>0 && *n<=8 && *lda==*n,"LAPACK factorization dimensions changed");
  lapack_n=*n;lu_calls++;
  for(int i=0;i<*n;++i){pivots[i]=*n-i;
    for(int j=0;j<*n;++j){lapack_original[i*(*n)+j]=a[j*(*n)+i];a[j*(*n)+i]=71+i*(*n)+j;}}
  *info=0;
}
static void dgetri(int* n,double* a,int* lda,int* pivots,double* work,int* lwork,int* info){
  check(*n==lapack_n && *lda==*n && *lwork==*n,"LAPACK inverse workspace dimensions changed");
  inverse_calls++;long double augmented[8][16]={};
  for(int i=0;i<*n;++i){check(pivots[i]==*n-i,"pivot storage did not survive the LAPACK calls");work[i]=i+.25;
    for(int j=0;j<*n;++j){check(a[j*(*n)+i]==71+i*(*n)+j,"factorization storage changed between calls");
      augmented[i][j]=lapack_original[i*(*n)+j];augmented[i][*n+j]=i==j;}}
  for(int k=0;k<*n;++k){
    int pivot=k;for(int i=k+1;i<*n;++i)if(std::abs(augmented[i][k])>std::abs(augmented[pivot][k]))pivot=i;
    check(std::abs(augmented[pivot][k])>1e-15L,"singular oracle matrix");
    for(int j=0;j<2*(*n);++j)std::swap(augmented[k][j],augmented[pivot][j]);
    const auto scale=augmented[k][k];for(int j=0;j<2*(*n);++j)augmented[k][j]/=scale;
    for(int i=0;i<*n;++i)if(i!=k){const auto factor=augmented[i][k];
      for(int j=0;j<2*(*n);++j)augmented[i][j]-=factor*augmented[k][j];}
  }
  for(int i=0;i<*n;++i)for(int j=0;j<*n;++j)a[j*(*n)+i]=double(augmented[i][*n+j]);
  *info=0;
}
static void Eigen_lapack3(double* a,double* eigenvalues,int n,int){
  check(n==1 || n==2,"analytic eigensystem supports only the selected test matrices");
  if(n==1){eigenvalues[0]=a[0];a[0]=1;return;}
  check(a[1]==a[2],"symmetric eigensystem input changed");
  const double x=a[0],y=a[1],z=a[3],angle=.5*std::atan2(2*y,x-z),c=std::cos(angle),s=std::sin(angle);
  eigenvalues[0]=c*c*x+2*c*s*y+s*s*z;eigenvalues[1]=s*s*x-2*c*s*y+c*c*z;
  a[0]=c;a[1]=s;a[2]=-s;a[3]=c;
}


#include "openmx_mixing_lapack.h"

#define abort allocation_abort
#include "mixing_v_workspace.inc"
#undef abort


namespace {
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation(){
  char error[256];Context owner(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
  check(bool(owner),error);return owner;
}
void flags(MixingVState* state){check(state->diis_first && state->kerker_first,"new potential calculation inherited memory-log flags");}
void logs(MixingVState* state){log_kind=0;log_diis_first(state);log_kind=1;log_kerker_first(state);}
void seed(MixingVState* state,int iteration,int dimension,double tag){
  state->fast.valid=1;state->fast.last_scf_iter=iteration;state->fast.last_nummix=dimension;
  for(int i=0;i<state->fast.l38*state->fast.l38;i++)state->fast.A[i]=tag+i*.125;
}
void mode(const char* value){check(!openmx_env_set("OPENMX_MIX_FAST",value),"set density-mixing policy");}
void rotation(){
  double a=1,b=2,c=3,d=4;double* v[]={&a,&b,&c,&d};double** slots[]={v,v+1,v+2,v+3};
  MixVKSk_RotateSlots(slots,4);
  check(slots[0]==v+3 && slots[1]==v && slots[2]==v+1 && slots[3]==v+2,"density history pointer rotation changed");
}
void failures(){
  auto first=calculation(),second=calculation();auto* a=Mixing_V_Bind_Resources(first.get());auto* b=Mixing_V_Bind_Resources(second.get());
  List_YOUSO[38]=7;mode("1");check(!preflight(a,4,3,1) && !preflight(b,4,3,1),"fresh preflight became incremental");seed(a,4,3,100);seed(b,4,3,200);
  List_YOUSO[38]=9;openmx::test::fail_allocation_after(0);bool aborted=false;
  try{(void)preflight(a,5,3,1);}catch(const AllocationAbort&){aborted=true;}
  openmx::test::fail_allocation_after(-1);
  check(aborted && !a->fast.A && openmx_core_native_resource_count(a->resources)==0,"cache registration failure was not drained");
  check(b->fast.valid && b->fast.A[8]==201 && openmx_core_native_resource_count(b->resources)==1,"failed cache resize damaged another context");
  check(!preflight(a,5,3,1) && a->fast.l38==9 && !a->fast.valid,"failed cache did not recover with a fresh layout");
  seed(a,5,3,300);openmx::test::fail_allocation_after(0);
  openmx_core_finish(first.get());openmx_core_native_resources_clear(second.get());
  openmx::test::fail_allocation_after(-1);
}
}
MixV_ScratchArguments expected{};
int behavior=0;
static double Mixing_V_Compute(OpenMXSolverInvocation* scratch,MixingVState* state,int md,int scf,int first){
  check(state==expected.state && md==expected.md_iteration && scf==expected.scf_iteration && first==expected.first_iteration,
        "potential mixing entry changed state or iteration arguments");
  double result;buffers(scratch,nullptr,&result);
  if(behavior==1)return 31.25;
  if(behavior==2)throw std::runtime_error("injected potential mixing exception");
  return result;
}
namespace {
void equal(long double a,long double b){check(std::isfinite(a) && std::abs(a-b)<1e-11L*(1+std::abs(b)),"potential inverse numerical value differs");}
void dispatch(){
  MixingVState state{};List_YOUSO[38]=7;
  for(int repeat=0;repeat<7;++repeat){
    expected={&state,repeat+1,repeat*3+2,repeat%2};
    for(behavior=0;behavior<3;++behavior){
      clock_calls=0;const auto live=openmx::test::live_allocations();bool failed=false;double result=-1;
      try{result=Mixing_V(expected.state,expected.md_iteration,expected.scf_iteration,expected.first_iteration);}
      catch(const AllocationAbort&){failed=true;}
      check(failed==(behavior==2),"potential callback exception contract changed");
      if(!failed){equal(result,7.5);check(clock_calls==2,"potential entry did not retain both timing endpoints");}
      check(openmx::test::live_allocations()==live,"potential entry leaked on return/early return/exception");
    }
  }
  behavior=0;clock_calls=0;openmx::test::fail_allocation_after(10000);
  (void)Mixing_V(expected.state,expected.md_iteration,expected.scf_iteration,expected.first_iteration);
  const long count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  check(count>10,"entry allocation faults missed the actual C++ workspace");
  for(long fail=0;fail<count;++fail){
    clock_calls=0;const auto live=openmx::test::live_allocations();bool failed=false;
    openmx::test::fail_allocation_after(fail);
    try{(void)Mixing_V(expected.state,expected.md_iteration,expected.scf_iteration,expected.first_iteration);}
    catch(const AllocationAbort&){failed=true;}
    openmx::test::fail_allocation_after(-1);
    check(failed && openmx::test::live_allocations()==live,"partial potential workspace construction leaked or missed a fault");
  }
}
int inverses(OpenMXSolverInvocation* scratch,void*,double* result){
  List_YOUSO[38]=7;
  auto a=OpenMX_Solver_OverwriteView(scratch,"input-real",7,7),ai=OpenMX_Solver_OverwriteView(scratch,"input-imag",7,7);
  auto b=OpenMX_Solver_OverwriteView(scratch,"inverse-real",7,7),bi=OpenMX_Solver_OverwriteView(scratch,"inverse-imag",7,7);
  for(int size:{1,3,2,4,3}){
    std::fill(a.values,a.values+49,-71);std::fill(b.values,b.values+49,-91);
    for(int i=0;i<size;++i)for(int j=0;j<size;++j)a.values[i*7+j]=i==j?4+i:.125*(1+i+2*j);
    double original[49];std::copy(a.values,a.values+49,original);const int old_lu=lu_calls,old_inverse=inverse_calls;
    Inverse(scratch,size-1,&a,&b);
    check(lu_calls==old_lu+1 && inverse_calls==old_inverse+1,"active inverse stopped using pivoted LAPACK LU");
    check(!std::memcmp(a.values,original,sizeof(original)),"LU inverse changed its borrowed input");
    for(int i=0;i<size;++i)for(int j=0;j<size;++j){long double product=0;
      for(int k=0;k<size;++k)product+=static_cast<long double>(original[i*7+k])*b.values[k*7+j];equal(product,i==j?1:0);}
    check(b.values[48]==-91,"LU inverse overwrote inactive subspace");
  }
  for(int size:{1,3,2,4,3}){
    std::fill(a.values,a.values+49,-71);std::fill(b.values,b.values+49,-91);
    for(int i=0;i<size;++i)for(int j=0;j<size;++j)a.values[i*7+j]=i==j?4+i:.125*(1+i+2*j);
    double original[49];std::copy(a.values,a.values+49,original);
    unpivoted::Inverse(scratch,size-1,&a,&b);
    check(!std::memcmp(a.values,original,sizeof(original)),"nonpivoted native LU did not restore input");
    for(int i=0;i<size;++i)for(int j=0;j<size;++j){long double product=0;
      for(int k=0;k<size;++k)product+=static_cast<long double>(original[i*7+k])*b.values[k*7+j];equal(product,i==j?1:0);}
    check(b.values[48]==-91,"nonpivoted native LU overwrote inactive subspace");
  }
  unpivoted::Inverse(scratch,-1,&a,&b);
  for(int i=0;i<49;++i)equal(a.values[i],0);
  for(int size:{1,3,2,3}){
    std::fill(a.values,a.values+49,-71);std::fill(ai.values,ai.values+49,-72);
    std::fill(b.values,b.values+49,-91);std::fill(bi.values,bi.values+49,-92);
    for(int i=0;i<size;++i)for(int j=0;j<size;++j){a.values[i*7+j]=i==j?4+i:.125*(1+i+2*j);ai.values[i*7+j]=.0625*(1+i-j);}
    double original[49],imag[49];std::copy(a.values,a.values+49,original);std::copy(ai.values,ai.values+49,imag);
    Complex_Inverse(scratch,size-1,&a,&ai,&b,&bi);
    check(!std::memcmp(a.values,original,sizeof(original)) && !std::memcmp(ai.values,imag,sizeof(imag)),"complex LU did not restore its input");
    for(int i=0;i<size;++i)for(int j=0;j<size;++j){std::complex<long double> product{};
      for(int k=0;k<size;++k)product+=std::complex<long double>(original[i*7+k],imag[i*7+k])*std::complex<long double>(b.values[k*7+j],bi.values[k*7+j]);
      equal(product.real(),i==j?1:0);equal(product.imag(),0);}
    check(b.values[48]==-91 && bi.values[48]==-92,"complex LU overwrote inactive subspace");
  }
  Complex_Inverse(scratch,-1,&a,&ai,&b,&bi);
  for(int i=0;i<49;++i)check(a.values[i]==0 && ai.values[i]==0,"complex LU empty-subspace sentinel changed");
  a.values[0]=4;a.values[1]=a.values[7]=1;a.values[8]=3;
  regularized::Inverse(scratch,1,&a,&b);equal(b.values[0],3.L/11);equal(b.values[1],-1.L/11);equal(b.values[7],-1.L/11);equal(b.values[8],4.L/11);
  *result=0;return 1;
}
void scope_and_inverses(){
  dispatch();char error[256];double result;
  const auto live=openmx::test::live_allocations();
  check(openmx_core_solver_run(2,inverses,nullptr,&result,error,sizeof(error)),error);
  check(openmx::test::live_allocations()==live,"potential inverse scratch leaked");
  openmx::test::fail_allocation_after(10000);
  check(openmx_core_solver_run(2,inverses,nullptr,&result,error,sizeof(error)),error);
  const long count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  for(long fail=0;fail<count;++fail){
    const auto before=openmx::test::live_allocations();result=-19;
    openmx::test::fail_allocation_after(fail);
    const int ok=openmx_core_solver_run(2,inverses,nullptr,&result,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    check(!ok && result==-19 && error[0] && openmx::test::live_allocations()==before,"partial potential inverse invocation leaked or changed output");
  }
}
}

int main(){try{
  const auto initial=openmx::test::live_allocations();
  rotation();
  for(int repeat=0;repeat<100;repeat++){
    auto first=calculation(),second=calculation();auto* a=Mixing_V_Bind_Resources(first.get());auto* b=Mixing_V_Bind_Resources(second.get());
    check(a!=b && a->resources!=b->resources,"density-mixing contexts share state");flags(a);flags(b);
    check(!a->fast.valid && a->fast.last_scf_iter==-1 && !a->fast.A,"new Gram cache lost its initial sentinels");
    std::fill(std::begin(prints),std::end(prints),0);
    for(int n=0;n<3;n++)logs(a);
    check(prints[0]==3 && prints[1]==1,"memory logging did not run exactly once per path");flags(b);logs(b);
    check(prints[0]==6 && prints[1]==2,"memory-log state leaked between calculations");
    const int history=repeat%5+7,spin=repeat%3+1,new_spin=spin%3+1;List_YOUSO[38]=history;mode("1");
    check(!preflight(a,4,3,spin) && !preflight(b,4,3,spin),"unseeded cache was reused");
    check(a->fast.A && b->fast.A && a->fast.A!=b->fast.A && a->fast.l38==history,"Gram numeric storage is not calculation-owned");
    seed(a,4,3,100);seed(b,4,3,200);auto* address=a->fast.A;
    openmx::test::fail_allocation_after(0);
    check(Mixing_V_Bind_Resources(first.get())==a && preflight(a,5,3,spin) && preflight(a,5,4,spin),"cache lookup/continuity allocated or reset state");
    openmx::test::fail_allocation_after(-1);
    check(a->fast.A==address && a->fast.A[8]==101 && b->fast.A[8]==201,"cache reuse lost the payload or selected another calculation");
    check(!preflight(a,4,3,spin) && !preflight(a,6,3,spin) && !preflight(a,5,5,spin),"SCF rewind/gap or dimension jump reused stale Gram entries");
    mode("0");check(!preflight(a,5,3,spin) && a->fast.A==address,"disabled fast path allocated/replaced cache");
    mode("1");scf_eigen_lib_flag=0;check(!preflight(a,5,3,spin),"CPU eigen route enabled GPU-only mixing optimization");scf_eigen_lib_flag=GPUSOLVER;
    check(!preflight(a,5,3,new_spin) && !a->fast.valid && a->fast.last_scf_iter==-1,"spin change did not invalidate the Gram layout");
    check(b->fast.valid && b->fast.spinmax==spin && b->fast.A[8]==201,"layout change touched another calculation");
    List_YOUSO[38]=history+1;check(!preflight(a,5,3,new_spin) && a->fast.l38==history+1,"history resize kept the old layout");
    openmx::test::fail_allocation_after(0);openmx_core_native_resources_clear(first.get());openmx_core_finish(second.get());openmx::test::fail_allocation_after(-1);
    auto* fresh=Mixing_V_Bind_Resources(first.get());flags(fresh);
    check(!fresh->fast.A && !fresh->fast.valid && fresh->fast.last_scf_iter==-1,"next input inherited the previous cache");
  }
  failures();
  scope_and_inverses();
  check(openmx::test::live_allocations()==initial,"potential-mixing lifecycle leaked C++ allocations");
  std::cout<<"PASS: 100 dual-context potential-mixing lifetimes, production cache/logging/entry/FFT scratch, allocation failures and inverse identities\n";return 0;
}catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}}
