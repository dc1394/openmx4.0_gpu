#include "openmx_mixing_rhok_internal.h"
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

int mpi_comm_level1=0,atomnum=1,Matomnum=1,SpeciesNum=1,Solver=1,TRAN_Poisson_flag=0;
int M2G[]={0,1},WhatSpecies[]={0,0},Spe_Total_CNO[]={1},FNAN[]={0,0},neighbor[]={1};
int* natn[]={neighbor,neighbor};int List_YOUSO[60]={};
struct AllocationAbort:std::exception {};
void MPI_Abort(int,int){throw AllocationAbort();}
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void equal(long double a,long double b){check(std::isfinite(a) && std::abs(a-b)<1e-11L*(1+std::abs(b)),"workspace numerical value differs");}
#include "openmx_mixing_rhok_workspace.h"

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

MixRhok_ScratchArguments expected{};int selected=-1,behavior=0,callbacks=0;
static void operation(int tag,OpenMXSolverInvocation* scratch,MixingRhokState* state,int control,double weight,
    double*** real,double*** imag,double** residual_real,double** residual_imag,double* potential_real,
    double* potential_imag,double* atomic_real,double* atomic_imag){
  check(state==expected.state && control==expected.control && weight==expected.weight,"scratch bridge changed scalar/state arguments");
  check(real==expected.real && imag==expected.imag && residual_real==expected.residual_real && residual_imag==expected.residual_imag &&
    potential_real==expected.potential_real && potential_imag==expected.potential_imag && atomic_real==expected.atomic_real && atomic_imag==expected.atomic_imag,
    "scratch bridge changed borrowed array arguments");
  selected=tag;callbacks++;
  auto real_view=OpenMX_Solver_OverwriteView(scratch,"bridge-real",3,7);
  auto complex_view=OpenMX_Solver_OverwriteComplexView(scratch,"bridge-complex",2,5);
  real_view.values[20]=19;complex_view.values[9]={3,-7};
  if(behavior==1)return;
  if(behavior==2)throw std::runtime_error("injected numerical callback exception");
  equal(real_view.values[20],19);equal(complex_view.values[9].i,-7);
}
#define MODEL_OPERATION(name,tag) \
static void name(OpenMXSolverInvocation* scratch,MixingRhokState* state,int control,double weight, \
    double*** real,double*** imag,double** residual_real,double** residual_imag,double* potential_real, \
    double* potential_imag,double* atomic_real,double* atomic_imag){ \
  operation(tag,scratch,state,control,weight,real,imag,residual_real,residual_imag,potential_real,potential_imag,atomic_real,atomic_imag); }
MODEL_OPERATION(DIIS_Mixing_Rhok_Normal,0)
MODEL_OPERATION(DIIS_Mixing_Rhok_NEGF,1)
MODEL_OPERATION(Kerker_Mixing_Rhok_Normal,2)
MODEL_OPERATION(Kerker_Mixing_Rhok_NEGF,3)
#include "mixing_rhok_workspace.inc"

namespace {
void invoke(bool kerker){
  const auto& a=expected;
  if(kerker)Kerker_Mixing_Rhok(a.state,a.control,a.weight,a.real,a.imag,a.residual_real,a.residual_imag,a.potential_real,a.potential_imag,a.atomic_real,a.atomic_imag);
  else DIIS_Mixing_Rhok(a.state,a.control,a.weight,a.real,a.imag,a.residual_real,a.residual_imag,a.potential_real,a.potential_imag,a.atomic_real,a.atomic_imag);
}
void dispatch(){
  MixingRhokState state{};double values[8][8]={};double* rows[8];for(int i=0;i<8;++i)rows[i]=values[i];
  double** real[]={rows},**imag[]={rows+1};
  expected={nullptr,&state,17,.375,real,imag,rows+2,rows+3,values[4],values[5],values[6],values[7]};
  for(bool kerker:{false,true})for(int route=0;route<3;++route){
    Solver=route?4:1;TRAN_Poisson_flag=route==2?2:0;
    for(behavior=0;behavior<3;++behavior){
      selected=-1;const auto before=openmx::test::live_allocations();bool failed=false;
      try{invoke(kerker);}catch(const AllocationAbort&){failed=true;}
      check(failed==(behavior==2) && selected==(kerker?2:0)+(route==1),"normal/NEGF dispatch or callback exception changed");
      check(openmx::test::live_allocations()==before,"bridge leaked on return/early return/exception");
    }
  }
  Solver=1;behavior=0;openmx::test::fail_allocation_after(10000);invoke(false);
  const long allocations=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  check(allocations>5,"allocation interception did not reach actual scratch owners");
  for(long fail=0;fail<allocations;++fail){
    const auto before=openmx::test::live_allocations();bool failed=false;
    openmx::test::fail_allocation_after(fail);
    try{invoke(false);}catch(const AllocationAbort&){failed=true;}
    openmx::test::fail_allocation_after(-1);
    check(failed && openmx::test::live_allocations()==before,"partial bridge construction leaked or missed allocation failure");
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
  Inverse2(scratch,1,&a,&b);equal(b.values[0],3.L/11);equal(b.values[1],-1.L/11);equal(b.values[7],-1.L/11);equal(b.values[8],4.L/11);
  *result=0;return 1;
}
}
int main(){try{
  dispatch();const auto before=openmx::test::live_allocations();char error[256];double result;
  check(openmx_core_solver_run(2,inverses,nullptr,&result,error,sizeof(error)),error);
  check(openmx::test::live_allocations()==before,"native inverse invocation leaked scratch");
  std::cout<<"PASS: production density-mixing dispatch, argument routing, actual scope/fault cleanup, pivoted-LU contract and real/complex inverse identities\n";return 0;
}catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}}
