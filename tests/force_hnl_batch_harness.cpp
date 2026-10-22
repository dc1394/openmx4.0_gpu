// Appended to the archive fixture so native resource adapters and ownership
// are exercised against the actual C++ library. Device launches run as host
// loops; this checks packing, damping, publication and cleanup, not a GPU SDK.
#include "openmx_core_electronic.h"
#include "openmx_core_force_hnl_batch.h"
#include <cmath>
#include <climits>
#include <cstdint>
typedef OpenMXForcePairItem Force4BGpuItem;
static int Solver=2,F_G2M[]{0,1,2},rm0[]{0,1},rm1[]{1,0};
static int *rmp[]{rm0,rm1},**RMI1[]{nullptr,rmp};
static int cell[]{0,0},*ncn[]{nullptr,cell,cell};
static double xyz[3][64],*Gxyz[]{xyz[0],xyz[1],xyz[2]};
static double distances[2]{0,.875},*Dis[]{nullptr,distances,distances};
static double translation[4]{0,0,0,0},*atv[]{translation};
static double Spe_Atom_Cut1[]{2.5,3.0};
static double density_values[2][3][2][2][2],hamiltonian_values[2][3][2][2][2];
static double *density_rows[2][3][2][2],*hamiltonian_rows[2][3][2][2];
static double **density_blocks[2][3][2],**hamiltonian_blocks[2][3][2];
static double ***density_atoms[2][3],***hamiltonian_atoms[2][3];
static double ****CDM0[2],****HNL0[2];
static int damping_tail=0,batch_copy_failure=0,batch_copies=0,recoveries=0;
static void batch_transfer(){
  if(batch_copy_failure && ++batch_copies==batch_copy_failure){
    current_device=acc_device=7;throw std::runtime_error("injected batch transfer failure");
  }
}
static double dampingF(double,double r){return .8+.001*r;}
static double deri_dampingF(double,double){return damping_tail?.02:0;}
static int Force_PhaseProfileEnabled(){return 0;}
static void dtime(double* t){*t=0;}
static void Force_gpu_pool_flush(){}
static void acc_memcpy_from_device(void* d,const void* s,size_t n){batch_transfer();std::memcpy(d,s,n);}
static void openmx_target_download(void* d,const void* s,size_t n){acc_memcpy_from_device(d,s,n);}
static void ForceHipMemcpyFromDevice(void* d,const void* s,size_t n){acc_memcpy_from_device(d,s,n);}
static void openmx_target_free(void* p){check(!OpenMXGpuFree(p),"HIP batch free");}
static void* Force_checked_malloc(size_t bytes,const char*,int){void* p=std::malloc(bytes?bytes:1);if(!p)throw std::bad_alloc();return p;}
/* FROZEN_CONTRACTIONS */
static void Force_HNL_capacity(int kind,ForceHNLGpuContext* g,int n,const Force4BGpuItem* items,
    const size_t* a,const size_t* b,const int* halo,const int* energy,const int* projectors,
    const int* h0,const double* density,double* forces,int done){
  if(done)return;++recoveries;
  if(kind==1)reference1(n,items,a,b,halo,energy,projectors,density,g->halo,g->ene,g->flat,g->flat_stride,forces);
  else reference2(n,items,h0,projectors,energy,a,b,density,g->c2,g->ene,g->flat,g->c2_stride,forces);
}
/* NATIVE_BATCH_FUNCTIONS */
#ifndef OPENMX_HNL_BATCH_RECORD_REFERENCE
#include "fixtures/force_hnl_batch_expected.h"
#endif
namespace {
struct BatchCase {int kind,spin,solver,tail,local,route,external,empty=0;};
void initialize_batch(){
  for(int d=0;d<4;++d){directions[d]=atoms[d];for(int a=0;a<3;++a){atoms[d][a]=blocks[d][a];
    for(int h=0;h<2;++h){blocks[d][a][h]=rows[d][a][h];for(int r=0;r<3;++r){rows[d][a][h][r]=values[d][a][h][r];
      for(int j=0;j<8;++j)values[d][a][h][r][j]=std::sin(.17*d+.13*a+.07*h+.03*r+.01*j);}}}}
  for(int s=0;s<2;++s){CDM0[s]=density_atoms[s];HNL0[s]=hamiltonian_atoms[s];
    for(int a=0;a<3;++a){density_atoms[s][a]=density_blocks[s][a];hamiltonian_atoms[s][a]=hamiltonian_blocks[s][a];
      for(int h=0;h<2;++h){density_blocks[s][a][h]=density_rows[s][a][h];hamiltonian_blocks[s][a][h]=hamiltonian_rows[s][a][h];
        for(int i=0;i<2;++i){density_rows[s][a][h][i]=density_values[s][a][h][i];hamiltonian_rows[s][a][h][i]=hamiltonian_values[s][a][h][i];
          for(int j=0;j<2;++j){density_values[s][a][h][i][j]=std::cos(.19*s+.23*a+.11*h+.07*i+.03*j);
            hamiltonian_values[s][a][h][i][j]=std::sin(.31*s+.17*a+.13*h+.05*i+.01*j);}}}}}
}
int batch_operation(OpenMXSolverInvocation* scope,void* opaque,double* result){
  const auto c=*static_cast<BatchCase*>(opaque);auto a=atomic();ForceHNLGpuContext* g=nullptr;
  check(Force_HNL_GpuBegin(&a,components,scope,&g),"batch archive declined");packed(g,0);
  SpinP_switch=c.spin;Solver=c.solver;damping_tail=c.tail;F_G2M[2]=c.local?1:2;
  for(auto& atom:xyz)for(auto& v:atom)v=0;
  xyz[1][1]=.2;xyz[1][2]=-.1;xyz[1][3]=.3;xyz[2][1]=.7;xyz[2][2]=.4;xyz[2][3]=-.2;
  double output[11];std::fill(output,output+11,1234.0);
  if(c.external)output[4]=output[5]=output[6]=0;
  if(c.route==1)fail_status=2;
  if(c.route==2)g->flat_dev=nullptr;
  OpenMXElectronicViews electronic{};electronic.HNL=HNL0;
  const size_t resource_count=openmx_core_native_resource_count(g->resources);
  const int first=c.empty?2:1;
  if(c.kind==1)Force_HNL_GpuCase1Run_Part(first,2,g,c.external?output+1:nullptr,0,CDM0,&electronic);
  else Force_HNL_GpuCase2Run_Part(first,2,g,c.external?output+1:nullptr,0,CDM0,&electronic);
  check(openmx_core_native_resource_count(g->resources)==resource_count,"batch arena escaped callback scope");
  for(int d=0;d<3;++d){const double v=c.external?output[4+d]:xyz[1][41+d];
#ifdef OPENMX_HNL_BATCH_RECORD_REFERENCE
    if(d)std::cout<<",";std::cout.precision(17);std::cout<<v;
#else
    const double expected=c.empty?0:expected_force[c.kind-1][c.solver==2?0:1][!!c.spin][c.tail][c.local][d];
    check(std::abs(v-expected)<=1e-11*(1+std::abs(expected)),"native HNL batch differs from pre-refactor force snapshot");
    if(c.external)check(xyz[1][41+d]==0,"external publication also wrote global forces");
#endif
  }
  for(int i:{0,1,2,3,7,8,9,10})check(output[i]==1234,"force output guard overwritten");
  *result=1;return 1;
}
int interrupted_batch(OpenMXSolverInvocation* scope,void* opaque,double*){
  const int injection=*static_cast<int*>(opaque);auto a=atomic();ForceHNLGpuContext* g=nullptr;
  check(Force_HNL_GpuBegin(&a,components,scope,&g),"interrupted archive declined");packed(g,0);
  OpenMXElectronicViews electronic{};electronic.HNL=HNL0;
  if(injection<0)openmx::test::fail_allocation_after(-injection-1);
  else {batch_copies=0;batch_copy_failure=injection;}
  Force_HNL_GpuCase1Run_Part(1,2,g,nullptr,0,CDM0,&electronic);
  return 1;
}
}
int main(){
  try {
    initialize_batch();int comparisons=0;
#ifdef OPENMX_HNL_BATCH_RECORD_REFERENCE
    for(int kind:{1,2})for(int solver:{2,5})for(int spin:{0,1})for(int tail:{0,1})for(int local:{0,1}){
      reset(0);BatchCase c{kind,spin,solver,tail,local,0,0};char error[256];double result;
      std::cout<<"{";check(openmx_core_solver_run(1,batch_operation,&c,&result,error,sizeof(error)),error);std::cout<<"},\n";
    }
#else
    for(int kind:{1,2})for(int solver:{2,5,8,11})for(int spin:{0,1,3})for(int tail:{0,1})for(int local:{0,1})
      for(int route:{0,1,2})for(int external:{0,1}){
        reset(0);BatchCase c{kind,spin,solver,tail,local,route,external};char error[256];double result;
        check(openmx_core_solver_run(1,batch_operation,&c,&result,error,sizeof(error)),error);
        check(!live,"batch/parent scope leaked device memory");++comparisons;
      }
    for(int kind:{1,2})for(int external:{0,1}){
      reset(0);BatchCase c{kind,0,2,0,0,0,external,1};char error[256];double result;
      check(openmx_core_solver_run(1,batch_operation,&c,&result,error,sizeof(error)),error);check(!live,"empty batch leaked");++comparisons;
    }
    for(int injection:{-1,-2,-3,-4,-5,-6,1,2,3,4,5,6,7,8,9,10}){
      reset(0);SpinP_switch=0;F_G2M[2]=2;char error[256];double result=31;batch_copy_failure=0;
      const auto before=openmx::test::live_allocations();
      const int ok=openmx_core_solver_run(1,interrupted_batch,(void*)&injection,&result,error,sizeof(error));
      openmx::test::fail_allocation_after(-1);batch_copy_failure=0;
      check(!ok && result==31 && *error,"interrupted batch published success");
      check(!live && openmx::test::live_allocations()==before,"interrupted batch leaked host/device storage");
      if(injection>0){check(current_device==7 && acc_device==7,"exception cleanup changed caller's device");
        const int device_frees=int(std::count(events.begin(),events.begin()+event_count,200+(OPENMX_GPU_HIP?1:3)));
        check(device_frees==2,"temporary batch arena was not released exactly once");}
    }
    std::cout<<"PASS: "<<comparisons<<" native HNL batch force snapshots; 6 allocation/10 transfer failures; backend="<<OPENMX_GPU_HIP<<'\n';
#endif
    return 0;
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}
