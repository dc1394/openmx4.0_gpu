#include "openmx_density_gpu_state.h"
#include "openmx_core_native_resources.h"
#include "allocation_faults.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include "orbital_state_binding_fixture.hpp"

using Type_Orbs_Grid = float;
constexpr int cudaSuccess=0,hipSuccess=0,cudaMemcpyHostToDevice=1,
              hipMemcpyHostToDevice=1,acc_device_nvidia=0,mpi_comm_level1=0;
struct Allocation {void *data=nullptr;int device=0,kind=0;};
struct Mapping {void *host=nullptr,*device=nullptr;int space=0;};
std::array<Allocation,256> allocations;
std::array<Mapping,256> mappings;
int device_id=0,omp_device_id=0,queries=0,gpu_fail=-1,copy_fail=-1,host_fail=-1;
int count(int kind) {int n=0;for(auto &a:allocations)n+=a.data && a.kind==kind;return n;}
void *allocate(size_t bytes,int kind) {
  int &failure=kind==0?host_fail:gpu_fail;
  if(failure==0)return nullptr;
  if(failure>0)--failure;
  for(auto &a:allocations)if(!a.data){a={std::malloc(bytes?bytes:1),device_id,kind};assert(a.data);return a.data;}
  std::abort();
}
void release(void *p,int kind) {
  if(!p)return;
  for(auto &a:allocations)if(a.data==p){
    assert(a.kind==kind);if(kind==1)assert(a.device==device_id);
    std::free(p);a={};return;
  }
  std::abort();
}
void *test_malloc(size_t n){return allocate(n,0);}
void test_free(void *p){release(p,0);}
int cudaMalloc(void **out,size_t n){*out=allocate(n,1);return *out?0:1;}
int cudaMallocHost(void **out,size_t n){*out=allocate(n,2);return *out?0:1;}
int cudaFree(void *p){release(p,1);return 0;}
int cudaFreeHost(void *p){release(p,2);return 0;}
int cudaGetLastError(){return 0;}
int cudaMemcpy(void *to,const void *from,size_t n,int){
  if(copy_fail==0)return 1;
  if(copy_fail>0)--copy_fail;
  std::memcpy(to,from,n);return 0;
}
int hipMalloc(void **out,size_t n){return cudaMalloc(out,n);}
int hipFree(void *p){return cudaFree(p);}
int hipMemcpy(void *to,const void *from,size_t n,int dir){return cudaMemcpy(to,from,n,dir);}
int hipGetDevice(int *out){++queries;*out=device_id;return 0;}
int hipSetDevice(int id){device_id=id;return 0;}
int acc_get_device_num(int){++queries;return device_id;}
void acc_set_device_num(int id,int){device_id=id;}
int omp_get_default_device(){return omp_device_id;}
void omp_set_default_device(int id){omp_device_id=id;}
int mapping_space(){return OPENMX_GPU_HIP?omp_device_id:device_id;}
void *omp_get_mapped_ptr(void *p,int space){for(auto &m:mappings)if(m.host==p && m.space==space)return m.device;return nullptr;}
int omp_target_is_present(void *p,int space){return omp_get_mapped_ptr(p,space)!=nullptr;}
int omp_target_associate_ptr(void *host,void *gpu,size_t,size_t,int space){
  assert(!omp_get_mapped_ptr(host,space));
  for(auto &m:mappings)if(!m.host){m={host,gpu,space};return 0;}
  std::abort();
}
int omp_target_disassociate_ptr(void *host,int space){
  for(auto &m:mappings)if(m.host==host && m.space==space){m={};return 0;}
  std::abort();
}
void *acc_deviceptr(void *host){return omp_get_mapped_ptr(host,mapping_space());}
void acc_map_data(void *host,void *gpu,size_t bytes){omp_target_associate_ptr(host,gpu,bytes,0,mapping_space());}
void acc_unmap_data(void *host){omp_target_disassociate_ptr(host,mapping_space());}
#if !OPENMX_GPU_HIP
void acc_update_device(void *host,size_t bytes){cudaMemcpy(acc_deviceptr(host),host,bytes,1);}
#endif
void acc_wait_all(){}
void acc_clear_freelists(){}
void MPI_Abort(int,int){std::abort();}
#define malloc test_malloc
#define free test_free
#include "density_state.inc"
#undef malloc
#undef free

using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation(){char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);assert(c);return c;}
template<class T> void host(T *&out,size_t n=3){out=static_cast<T*>(SDG_malloc(n,sizeof(T)));assert(out);std::memset(out,0,n*sizeof(T));}
void geometry(OpenMXDensityGpuState *s){
  auto &c=s->cache;
  c.ready=1;c.atom_count=c.orbital_count=c.output_count=c.pair_count=c.term_count=c.cdm_count=c.global_grid_count=2;c.spin_count=1;
  host(c.atom_out_offset);host(c.atom_orb_offset);host(c.atom_pair_offset);host(c.atom_no);
  host(c.pair_neighbor);host(c.pair_cdm_offset);host(c.out_atom);host(c.out_ptr);
  host(c.terms);host(c.grid_ptr);host(c.grid_out);host(c.orbs);host(c.cdm);host(c.rho);
  host(c.cdm_counts);host(c.cdm_displs);host(c.scatter_counts);host(c.scatter_displs);
  auto &l=s->local;l.ready=1;l.output_count=l.term_count=l.dm_count=2;l.spin_count=1;
  host(l.atom_out_base);host(l.out_ptr);host(l.term_pt);host(l.pt_pair);host(l.dm);host(l.tmpden);
#if !OPENMX_GPU_HIP
  host(l.otf_out_base);
#endif
}
void assert_empty(){
  assert(!count(0)&&!count(1)&&!count(2));
  for(auto &m:mappings)assert(!m.host);
}
int main(){
  // Binding, releasing an unused state, finish and destruction never touch a GPU.
  {
    auto c=calculation();auto *s=Set_Density_Grid_GPU_Bind_Resources(c.get());
    assert(s->last_path==-1 && s->last_cpu==-1 && s->last_workspace==-1 && s->last_failed==-1);
    assert(s->orbital_state==Set_Orbitals_Grid_GPU_Bind_Resources(c.get()));
    Set_Density_Grid_GPU_EndCycle(s);Set_Density_Grid_GPU_Invalidate(s);
    openmx_core_native_resources_clear(c.get());
  }
  assert(queries==0);assert_empty();
  for(int round=0;round<3;++round){
    auto a=calculation(),b=calculation();
    auto *sa=Set_Density_Grid_GPU_Bind_Resources(a.get()),*sb=Set_Density_Grid_GPU_Bind_Resources(b.get());
    assert(sa!=sb);sa->last_path=7;sa->census.host=1;
    assert(Set_Density_Grid_GPU_Bind_Resources(a.get())==sa && sa->last_path==7);
    assert(sb->last_path==-1 && !sb->census.host);
    geometry(sa);geometry(sb);
    device_id=2;omp_device_id=7;
    assert(SDG_upload_device(&sa->cache) && SDG_local_upload(&sa->local));
    device_id=3;omp_device_id=8;
    assert(SDG_upload_device(&sb->cache) && SDG_local_upload(&sb->local));
    assert(count(1)==38);
    device_id=9;omp_device_id=10;
    Set_Density_Grid_GPU_EndCycle(sa);Set_Density_Grid_GPU_EndCycle(sa);
    assert(count(1)==33 && sa->local.ready && sb->local.ready);
    assert(device_id==9 && omp_device_id==10);
    // Service upload also handles a change of device and mapping space.
    assert(SDG_upload_device(&sa->cache));assert(count(1)==33);
#if OPENMX_GPU_HIP
    Set_Density_Grid_GPU_Release_Device_Caches(sa);
    assert(count(1)==19 && sa->cache.ready && sa->local.ready);
#else
    double *saved=SDG_staging_reserve(&sa->dm_staging,12);assert(saved);
    saved[0]=.125;assert(SDG_staging_reserve(&sa->dm_staging,2)==saved && saved[0]==.125);
    gpu_fail=0;assert(SDG_staging_reserve(&sa->den_staging,15));gpu_fail=-1;
    assert(sa->den_staging.pageable);
#endif
    Set_Density_Grid_GPU_Invalidate(sa);
    assert(!sa->local.ready && !sa->cache.ready && sb->cache.ready && count(1)==19);
#if !OPENMX_GPU_HIP
    assert(sa->dm_staging.data==saved && saved[0]==.125);
    assert(!SDG_staging_reserve(&sa->dm_staging,SIZE_MAX) && sa->dm_staging.data==saved);
    gpu_fail=host_fail=0;assert(!SDG_staging_reserve(&sa->den_staging,30));
    gpu_fail=host_fail=-1;assert(SDG_staging_reserve(&sa->den_staging,30));
#endif
    if(round==0)openmx_core_finish(a.get());
    else if(round==1)openmx_core_native_resources_clear(a.get());
    else a.reset();
    assert(count(1)==19 && !count(2));
    if(a){sa=Set_Density_Grid_GPU_Bind_Resources(a.get());assert(sa->last_path==-1 && !sa->cache.ready);}
    b.reset();a.reset();assert_empty();assert(device_id==9 && omp_device_id==10);
  }
  int cases=0;
  // Real upload functions must roll back every partially allocated prefix.
  for(int local=0;local<2;++local)for(int copying=0;copying<2;++copying){
    const int attempts=local?(copying?3:5):(copying?13:14);
    for(int failure=0;failure<attempts;++failure){
      auto c=calculation();auto *s=Set_Density_Grid_GPU_Bind_Resources(c.get());geometry(s);
      device_id=2;omp_device_id=7;
      (copying?copy_fail:gpu_fail)=failure;
      int ok=local?SDG_local_upload(&s->local):SDG_upload_device(&s->cache);
      assert(!ok && !count(1));
      gpu_fail=copy_fail=-1;device_id=9;omp_device_id=10;c.reset();assert_empty();++cases;
    }
  }
  // A failed transactional bind neither exposes partially constructed state nor changes another owner.
  for(int fail=0;fail<8;++fail){
    auto c=calculation();OpenMXNativeResources *owner=nullptr;void *metadata=nullptr;char error[256];
    openmx::test::fail_allocation_after(fail);
    int ok=openmx_core_native_resources_state(c.get(),"Set_Density_Grid_GPU",sizeof(OpenMXDensityGpuState),
                                              SDG_Reset,&owner,&metadata,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    if(!ok)assert(!owner && !metadata);
    else assert(owner && metadata);
    auto *s=Set_Density_Grid_GPU_Bind_Resources(c.get());assert(s->last_path==-1);
  }
  assert_empty();
  std::printf("PASS: %s density owner, independent contexts, clear/finish/destroy, staging and %d partial-upload cases\n",
              OPENMX_GPU_HIP?"HIP":"CUDA",cases);
}
