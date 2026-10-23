#include "openmx_core_chemical.h"
#include "openmx_core_energy.h"
#include "fixtures/native_resources/sdk_model.hpp"
#include "openmx_mixing_h_state.h"
#include "openmx_gpu_workers.h"
#include "openmx_gpu_vector.h"
#include "openmx_core_electronic.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>

constexpr int mpi_comm_level1=0,Host_ID=0,MPI_INT=1,MPI_UNSIGNED_LONG_LONG=2,
              MPI_MIN=0,MPI_SUM=1,MPI_COMM_TYPE_SHARED=0,MPI_INFO_NULL=0;
const void* const MPI_IN_PLACE=reinterpret_cast<void*>(1);
using MPI_Comm=int;
struct PreflightAllocationAbort:std::exception {};
int level_stdout=0,Matomnum=2,SpinP_switch=0,Num_Mixing_pDM=7;
int atomnum=2,SpeciesNum=2,Spe_Total_CNO[]={2,3},List_YOUSO[40]={};
int M2G[]={0,1,2},WhatSpecies[]={0,0,1},FNAN[]={0,1,1},Spe_Total_NO[]={2,3};
int neighbors[3][2]={{0,0},{1,2},{2,1}};
int* natn[]={neighbors[0],neighbors[1],neighbors[2]};
bool capacity_ok=true;
unsigned long long host_available=1ULL<<40;
int next_comm=1,live_comms=0,capacity_queries=0;
constexpr int OPENMX_GPU_COPY_HOST_TO_DEVICE=0;
static int OpenMXGpuGetLastError(){return 0;}
static int OpenMXGpuDeviceSynchronize(){return 0;}
static int OpenMXGpuMemcpy(void* to,const void* from,size_t bytes,int){std::memcpy(to,from,bytes);return 0;}
static int OpenMX_GpuMemGetInfo(size_t* available,size_t* total){*available=*total=size_t(8)<<30;return 0;}
static int gpu_rank_device_usable(){return 1;}
static unsigned long long openmx_host_mem_available(){return host_available;}
static void acc_wait_all(){}
static void acc_memcpy_from_device(void* to,const void* from,size_t bytes){std::memcpy(to,from,bytes);}
static void openmx_target_download(void* to,const void* from,size_t bytes){std::memcpy(to,from,bytes);}
static void MPI_Abort(int,int){throw PreflightAllocationAbort();}
static void MPI_Barrier(int){}
static void MPI_Comm_size(int,int* size){*size=1;}
static void MPI_Comm_split_type(int,int,int,int,int* comm){*comm=next_comm++;live_comms++;}
static void MPI_Comm_free(int* comm){check(*comm>0,"MPI group freed twice");*comm=-1;live_comms--;}
static void MPI_Allreduce(const void* from,void* to,int count,int type,int,int){
  if(from!=MPI_IN_PLACE)std::memcpy(to,from,count*(type==MPI_INT?sizeof(int):sizeof(unsigned long long)));
}
#include "openmx_solver_workspace.h"
#define F77_NAME(lower,upper) lower
static void dgetrf(int*,int*,double*,int*,int*,int*){std::abort();}
static void dgetri(int*,double*,int*,int*,double*,int*,int*){std::abort();}
// Independent analytic eigensystem for symmetric 1x1/2x2 inputs. This models
// Eigen_lapack3's column-major eigenvector contract, not the inverse algorithm.
static void Eigen_lapack3(double* a,double* eigenvalues,int n,int){
  check(n==1 || n==2,"unexpected analytic eigensystem dimension");
  if(n==1){eigenvalues[0]=a[0];a[0]=1;return;}
  check(a[1]==a[2],"inverse changed the symmetric matrix layout");
  const double x=a[0],y=a[1],z=a[3];
  const double angle=.5*std::atan2(2*y,x-z),c=std::cos(angle),s=std::sin(angle);
  eigenvalues[0]=c*c*x+2*c*s*y+s*s*z;
  eigenvalues[1]=s*s*x-2*c*s*y+c*c*z;
  a[0]=c;a[1]=s;a[2]=-s;a[3]=c;
}
int openmx_gpu_vector_execute(MPI_Comm,OpenMXGpuVector* job,int,const char*){
  if(!job->mode){
    for(int p=0;p<job->pairs;++p){
      long double sum=0;
      for(size_t k=0;k<job->n;++k)sum+=static_cast<long double>(job->weight[k])*
        job->data[job->left[p]*job->ld+k]*job->data[job->right[p]*job->ld+k];
      job->output[p]=double(sum);
    }
  }else{
    for(size_t k=0;k<job->n;++k){
      long double sum=0;
      for(int c=0;c<job->columns;++c)sum+=static_cast<long double>(job->coeff[c])*job->data[c*job->ld+k];
      job->output[k]=double(sum);
    }
  }
  return 1;
}
int openmx_gpu_workers_capacity(OpenMXGpuWorkers* workers,size_t items,int,
                               OpenMXGpuWorkerCost cost,void* context){
  check(context,"capacity callback has no calculation");capacity_queries++;
  if(!capacity_ok){workers->count=0;return 0;}
  workers->count=2;workers->items=items;workers->devices[0]=1;workers->devices[1]=2;
  for(int i=0;i<2;i++)check(cost(context,i,2)<SIZE_MAX,"incorrect capacity callback state");
  return 1;
}
int openmx_gpu_workers_run(const OpenMXGpuWorkers* workers,
                          int (*callback)(void*,int,int),void* context,const char*){
  const int previous=current_device,previous_acc=acc_device;
  bool ok=true;
  for(int phase=OPENMX_GPU_WORKER_PREPARE;phase<=OPENMX_GPU_WORKER_RELEASE;phase++){
    if(!ok && phase!=OPENMX_GPU_WORKER_RELEASE)continue;
    for(int i=0;i<workers->count;i++){
      current_device=acc_device=workers->devices[i];
      if(!callback(context,i,phase))ok=false;
    }
  }
  current_device=previous;acc_device=previous_acc;return ok;
}

#include "mixing_h_state.inc"
namespace {
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation(){
  char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
  check(bool(c),error);return c;
}
void mode(const char* value){check(!openmx_env_set("OPENMX_MIXH_GPU",value),"set mixing policy");}
void clean(MixingHState* state){
  Mixing_H_Release_GPU(state);Mixing_H_Release_GPU(state);
  check(!openmx_core_native_resource_count(state->resources),"mixing resources remain after release");
  check(state->gpu.verdict==-1 && state->gpu.last_scf_iter==-1 && state->inc.last_scf_iter==-1,
        "mixing release lost preflight/history sentinels");
  check(!state->workers.count && !state->host_ring && !state->host_weight,"worker/host history survived release");
}
void contractions(MixingHState* state){
  const size_t n=state->gpu.L;
  for(int slot=0;slot<2;slot++){
    for(size_t k=0;k<n;k++)state->gpu.h_flat[k]=(slot+1)*.125+(k%7)*.0625;
    MixH_Upload(state,slot);
  }
  for(size_t k=0;k<n;k++)state->gpu.h_flat[k]=.25+(k%3)*.125;
  MixH_Upload(state,-1);
  const int pm[]={0,1,1},pn[]={0,0,1},permutation[]={1,0};
  const double coefficient[]={.75,-.25};
  double triangles[6]={};
  MixH_Work work={};work.state=state;work.mode=MIXH_GRAM;work.dim=2;work.npair=3;
  work.pm=pm;work.pn=pn;work.pslot=permutation;work.tri=triangles;
  check(openmx_gpu_workers_run(&state->workers,MixH_Worker,&work,"test-Gram"),"Gram callback failed");
  for(int p=0;p<3;p++){
    long double expected=0;
    for(size_t k=0;k<n;k++){
      const long double a=(permutation[pm[p]]+1)*.125+(k%7)*.0625;
      const long double b=(permutation[pn[p]]+1)*.125+(k%7)*.0625;
      expected+=(.25L+(k%3)*.125L)*a*b;
    }
    check(std::abs(triangles[p]+triangles[3+p]-expected)<1e-12L*(1+std::abs(expected)),
          "permuted weighted Gram used another calculation's history");
  }
  work.mode=MIXH_RESIDUAL;work.coeff=coefficient;
  check(openmx_gpu_workers_run(&state->workers,MixH_Worker,&work,"test-residual"),"residual callback failed");
  for(size_t k=0;k<n;k++){
    const double expected=.75*(.25+(k%7)*.0625)-.25*(.125+(k%7)*.0625);
    check(state->gpu.h_r[k]==expected,"residual download used another calculation's staging buffer");
  }
}
int temporary_contractions(OpenMXSolverInvocation* scratch,void* opaque,double* result){
  auto* state=static_cast<MixingHState*>(opaque);
  const size_t n=state->gpu.L;
  for(int slot=0;slot<state->gpu.nslot;++slot){
    for(size_t k=0;k<n;++k)state->gpu.h_flat[k]=(slot+1)*.125+(k%7)*.0625;
    MixH_Upload(state,slot);
  }
  for(size_t k=0;k<n;++k)state->gpu.h_flat[k]=.25+(k%3)*.125;
  MixH_Upload(state,-1);
  const int left[]={0,1,1},right[]={0,0,1},permutation[]={1,0};
  const double coefficient[]={.75,-.25};double triangle[3];
  // Reusing one invocation tests the calloc-to-explicit-zero contract,
  // especially physical coefficients no longer used after dimension shrink.
  for(int dim:{2,1,2}){
    const int pairs=dim*(dim+1)/2;
    MixH_GramDevice(scratch,state,dim,pairs,left,right,permutation,triangle);
    for(int p=0;p<pairs;++p){
      long double expected=0;
      for(size_t k=0;k<n;++k)expected+=(.25L+(k%3)*.125L)*
        ((permutation[left[p]]+1)*.125L+(k%7)*.0625L)*
        ((permutation[right[p]]+1)*.125L+(k%7)*.0625L);
      check(std::abs(triangle[p]-expected)<1e-12L*(1+std::abs(expected)),"temporary Gram matrix differs");
    }
    MixH_OptResidualDevice(scratch,state,dim,permutation,coefficient);
    for(size_t k=0;k<n;++k){
      const double expected=.75*(.25+(k%7)*.0625)-(dim==2?.25*(.125+(k%7)*.0625):0);
      check(state->gpu.h_r[k]==expected,"temporary residual retained inactive coefficients");
    }
  }
  *result=0;return 1;
}
void temporary_contractions(MixingHState* state){
  char error[256];double result;
  check(openmx_core_solver_run(1,temporary_contractions,state,&result,error,sizeof(error)),error);
}
struct History {
  double data[4][4][3][2][3][3];
  double* row[4][4][3][2][3];double** neighbor[4][4][3][2];
  double*** atom[4][4][3];double**** component[4][4];double***** slot[4];
  explicit History(double offset){
    for(int s=0;s<4;++s){slot[s]=component[s];
      for(int c=0;c<4;++c){component[s][c]=atom[s][c];
        for(int a=0;a<3;++a){atom[s][c][a]=neighbor[s][c][a];
          for(int h=0;h<2;++h){neighbor[s][c][a][h]=row[s][c][a][h];
            for(int i=0;i<3;++i){row[s][c][a][h][i]=data[s][c][a][h][i];
              for(int j=0;j<3;++j)data[s][c][a][h][i][j]=offset+.03125*(1+7*s+3*c+5*a+2*h+i+2*j);
            }
          }
        }
      }
    }
  }
};
int incremental_rows(OpenMXSolverInvocation* scratch,void*,double* result){
  History real(.125),imag(-.25);OpenMXElectronicViews electronic{};
  electronic.ResidualH1=real.slot;electronic.ResidualH2=imag.slot;
  MixingHState state{};int permutation[]={2,0,3,1},blocks[]={0,0,2};
  double gram[4*4*5];state.inc.nslot=4;state.inc.nblk=5;
  state.inc.perm=permutation;state.inc.blk_base=blocks;state.inc.S=gram;
  for(int spin:{0,1,3}){
    SpinP_switch=spin;Matomnum=2;
    for(int nmax:{0,3,1,3})for(int m=0;m<4;++m){
      std::fill(std::begin(gram),std::end(gram),-17);
      MixH_IncComputeRow(scratch,&state,m,nmax,&electronic);
      for(int a=1;a<=2;++a)for(int i=0;i<Spe_Total_NO[a-1];++i)for(int n=0;n<=nmax;++n){
        long double expected=0;
        for(int h=0;h<2;++h)for(int j=0;j<Spe_Total_NO[natn[a][h]-1];++j){
          for(int c=0;c<(spin==3?4:spin+1);++c)
            expected+=static_cast<long double>(real.data[m][c][a][h][i][j])*real.data[n][c][a][h][i][j];
          if(spin==3)for(int c=0;c<3;++c)
            expected+=static_cast<long double>(imag.data[m][c][a][h][i][j])*imag.data[n][c][a][h][i][j];
        }
        const size_t index=(permutation[m]*4+permutation[n])*5+blocks[a]+i;
        const size_t transposed=(permutation[n]*4+permutation[m])*5+blocks[a]+i;
        check(std::abs(gram[index]-expected)<1e-12L*(1+std::abs(expected)) && gram[index]==gram[transposed],
          "incremental row changed spin channels, topology or physical-slot layout");
      }
    }
  }
  Matomnum=0;state.inc.nblk=0;state.inc.S=nullptr;
  MixH_IncComputeRow(scratch,&state,0,3,&electronic);
  Matomnum=2;*result=0;return 1;
}
int inverse_workspaces(OpenMXSolverInvocation* scratch,void*,double* result){
  List_YOUSO[39]=7;
  auto a=OpenMX_Solver_OverwriteView(scratch,"test-matrix",7,7);
  auto inverse=OpenMX_Solver_OverwriteView(scratch,"test-inverse",7,7);
  for(auto values:{std::array<double,3>{4,1.25,3},std::array<double,3>{0,2,0},std::array<double,3>{2,0,0}}){
    std::fill(a.values,a.values+49,-71);std::fill(inverse.values,inverse.values+49,-91);
    a.values[0]=values[0];a.values[1]=a.values[7]=values[1];a.values[8]=values[2];
    Inverse(scratch,1,&a,&inverse);
    const long double x=static_cast<long double>(values[0])+1e-13L,y=values[1],z=static_cast<long double>(values[2])+1e-13L;
    const long double determinant=x*z-y*y;
    const long double expected[]={z/determinant,-y/determinant,-y/determinant,x/determinant};
    for(int i=0;i<2;++i)for(int j=0;j<2;++j)
      check(std::abs(openmx_solver_real_matrix_row(&inverse,i)[j]-expected[2*i+j])<2e-11L*(1+std::abs(expected[2*i+j])),
        "regularized inverse differs from the analytic inverse");
    check(a.values[0]==values[0] && a.values[1]==values[1] && a.values[8]==values[2] && inverse.values[48]==-91,
      "inverse changed its input or wrote beyond the active subspace");
    a.values[0]=3;Inverse(scratch,0,&a,&inverse);
    check(std::abs(inverse.values[0]-1/(3+1e-13))<1e-14 && inverse.values[48]==-91,
      "regularized inverse failed after subspace shrink");
  }
  for(int size:{1,3,2,3}){
    std::fill(a.values,a.values+49,-71);std::fill(inverse.values,inverse.values+49,-91);
    for(int i=0;i<size;++i)for(int j=0;j<size;++j)openmx_solver_real_matrix_row(&a,i)[j]=i==j?4+i:.125*(1+i+2*j);
    double original[49];std::copy(a.values,a.values+49,original);Inverse0(scratch,size-1,&a,&inverse);
    check(!std::memcmp(original,a.values,sizeof(original)),"LU inverse did not restore its input");
    for(int i=0;i<size;++i)for(int j=0;j<size;++j){
      long double product=0;
      for(int k=0;k<size;++k)product+=static_cast<long double>(original[i*7+k])*inverse.values[k*7+j];
      check(std::abs(product-(i==j?1:0))<1e-12L,"LU inverse failed the matrix-product identity");
    }
    check(inverse.values[48]==-91,"LU inverse wrote beyond the active subspace");
  }
  Inverse0(scratch,-1,&a,&inverse);
  for(int i=0;i<49;++i)check(a.values[i]==0,"LU empty-subspace sentinel changed");
  *result=0;return 1;
}
void numerical_workspaces(){
  for(auto callback:{incremental_rows,inverse_workspaces}){
    char error[256];double result;
    const auto allocations=openmx::test::live_allocations();
    check(openmx_core_solver_run(1,callback,nullptr,&result,error,sizeof(error)),error);
    check(allocations==openmx::test::live_allocations(),"numerical invocation leaked scratch");
  }
}
void failures(){
  Matomnum=2;SpinP_switch=3;mode("1");
  auto other=calculation();auto* b=Mixing_H_Bind_Resources(other.get());
  check(MixH_GpuPreflight(b,1,0)==1,"second context preflight");
  const auto retained=openmx_core_native_resource_count(b->resources);
  for(long position=0;position<20;position++){
    event_count=0;auto c=calculation();auto* a=Mixing_H_Bind_Resources(c.get());
    openmx::test::fail_allocation_after(position);
    try{(void)MixH_GpuPreflight(a,1,0);}catch(const PreflightAllocationAbort&){}
    openmx::test::fail_allocation_after(-1);clean(a);
    check(openmx_core_native_resource_count(b->resources)==retained && b->workers.count==2,
          "partial allocation/release damaged another owner");
  }
  event_count=0;auto c=calculation();auto* a=Mixing_H_Bind_Resources(c.get());
  fail_status=OPENMX_GPU_OUT_OF_MEMORY;
  check(MixH_GpuPreflight(a,1,0)==1 && a->host_ring,"SDK allocation failure did not try host-backed history");
  fail_status=0;clean(a);clean(b);
}
}
OpenMXChemicalPotentials* owned_chemical(OpenMXCoreContext* c) {
  OpenMXChemicalPotentials* out=nullptr;char error[256];
  check(openmx_core_chemical_potentials(c,&out,error,sizeof(error)),error);return out;
}
int main(){try{
  for(int repeat=0;repeat<100;repeat++){
    event_count=0;capacity_ok=true;host_available=1ULL<<40;
    Matomnum=repeat%7?2:0;SpinP_switch=repeat%3==2?3:repeat%3;
    auto first=calculation(),second=calculation();
    auto* a=Mixing_H_Bind_Resources(first.get());auto* b=Mixing_H_Bind_Resources(second.get());
    check(a->chemical==owned_chemical(first.get()) && b->chemical==owned_chemical(second.get()) && a->chemical!=b->chemical,
          "native solver borrowed another calculation's chemical potentials");
    a->chemical->chemical_potential=.125;a->chemical->xanes[0]=1.5;
    b->chemical->chemical_potential=-.25;b->chemical->xanes[1]=2.5;
    check(a!=b && a->resources!=b->resources,"mixing metadata/resources shared across contexts");
    OpenMXMixingHistory* owned=nullptr;char history_error[256];check(openmx_core_mixing_history(first.get(),&owned,history_error,sizeof(history_error)) && a->history==owned && b->history!=owned,"Hamiltonian mixing borrowed the wrong shared history");
    check(a->gpu.verdict==-1 && a->inc.last_scf_iter==-1,"mixing constructor lost sentinels");
    mode("1");check(MixH_GpuPreflight(a,1,0)==1 && !a->host_ring,"resident tier missing");
    const size_t expected=Matomnum?(SpinP_switch==3?175:25*(SpinP_switch+1)):0;
    check(a->gpu.L==expected,"spin/empty-rank residual layout changed");
    contractions(a);temporary_contractions(a);a->gpu.valid=1;a->gpu.last_scf_iter=13;
    if(repeat%2){mode("2");check(MixH_GpuPreflight(b,2,0)==2 && b->inc.nslot==7,"incremental CPU tier missing");}
    else{capacity_ok=false;check(MixH_GpuPreflight(b,2,0)==1 && b->host_ring,"host-backed tier missing");temporary_contractions(b);}
    const auto second_count=openmx_core_native_resource_count(b->resources);
    openmx::test::fail_allocation_after(0);
    check(Mixing_H_Bind_Resources(first.get())==a && MixH_GpuPreflight(a,1,0)==1,
          "SCF state/preflight lookup allocated or reset cached data");
    openmx::test::fail_allocation_after(-1);
    check(a->gpu.valid && a->gpu.last_scf_iter==13,"another context changed history continuity");
    current_device=acc_device=7;clean(a);
    check(current_device==7 && acc_device==7,"worker cleanup changed caller's device binding");
    check(openmx_core_native_resource_count(b->resources)==second_count,"pre-Force release touched another context");
    event_count=0;openmx::test::fail_allocation_after(0);
    openmx_core_native_resources_clear(first.get());
    check(owned_chemical(first.get())->chemical_potential==0 && owned_chemical(first.get())->xanes[0]==0 &&
          b->chemical->chemical_potential==-.25 && b->chemical->xanes[1]==2.5,
          "clearing a calculation retained chemical potentials or reset another calculation");openmx_core_finish(second.get());
    openmx::test::fail_allocation_after(-1);
    check(!live && !live_comms && current_device==7,"RAII teardown left GPU/MPI resources or changed binding");
    auto* fresh=Mixing_H_Bind_Resources(second.get());
    check(fresh->gpu.verdict==-1 && !fresh->inc.valid && !fresh->host_ring,"new input inherited stale history");
  }
  failures();
  numerical_workspaces();
  {auto c=calculation();auto* a=Mixing_H_Bind_Resources(c.get());mode("0");
   check(MixH_GpuPreflight(a,1,0)==0,"forced legacy CPU tier changed");clean(a);
   capacity_ok=false;host_available=1;mode("1");
   check(MixH_GpuPreflight(a,1,0)==2 && a->inc.nslot==7,"host capacity rejection did not reach CPU cache");clean(a);}
  Mixing_H_Release_GPU(nullptr);check(!live && !live_comms,"mixing resources leaked");
  std::cout<<"PASS: 100 dual-context mixing lifecycles, all tiers, spin/empty layouts, "
             "real allocation failures, callback routing, scratch reuse, weighted/permuted contractions, "
             "incremental Gram, inverse identities and exact-owner cleanup\n";
  return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
