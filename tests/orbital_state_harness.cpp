#include "fixtures/native_resources/sdk_model.hpp"
#include "openmx_orbital_gpu_state.h"
#include "orbs_grid_gpu.h"
#include "openmx_precision.h"
#include "openmx_env.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
#include <iostream>

constexpr int cudaSuccess=0,cudaDevAttrSingleToDoublePrecisionPerfRatio=0;
constexpr acc_device_t acc_device_none=static_cast<acc_device_t>(0);
static acc_device_t acc_get_device_type(){return acc_device_nvidia;}
static int cudaGetDevice(int *d){return OpenMXGpuGetDevice(d);}
static int cudaSetDevice(int d){return OpenMXGpuSetDevice(d);}
static int cudaGetLastError(){return 0;}
static int cudaDeviceSynchronize(){return 0;}
static void acc_clear_freelists(){}
static int ratio=16;
static int cudaDeviceGetAttribute(int *v,int,int){*v=ratio;return 0;}
static void acc_memcpy_to_device(void *d,const void *s,size_t n){std::memcpy(d,s,n);}
static int eligible=1;
static int SOG_gpu_eligible(int){return eligible;}
static int SpeciesNum=1,TCpyCell=0,Ngrid2=3,Ngrid3=3;
static double atv[1][4],gtv[4][4],Grid_Origin[4];
static size_t SOG_arena_off(size_t *pos,size_t bytes){size_t off=*pos;*pos=(off+bytes+511)&~size_t(511);return off;}
static size_t host_allocations=0,host_live=0,host_fail=0;
static void *host_malloc(size_t n){if(++host_allocations==host_fail)return nullptr;void *p=std::malloc(n);if(p)host_live++;return p;}
static void host_free(void *p){if(p){assert(host_live);host_live--;std::free(p);}}
#include "orbital_state.inc"

using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
static Context calculation(){char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);assert(c);return c;}
static void bind_device(int d){current_device=acc_device=d;}
int main(){
  {
    auto c=calculation();auto *s=Set_Orbitals_Grid_GPU_Bind_Resources(c.get());
    Set_Orbitals_Grid_GPU_Release(s);openmx_core_finish(c.get());
  }
  assert(!queries && !live);
  double rv[]={.1,.2,.4,.8,1.6,3.2},rw[]={1.1,-.3,.7,1.5,-.2,.9};
  int mesh[]={6},maxl[]={0},nb[]={1};
  OpenMXAtomicViews atomic{};
  atomic.pao_radii={rv,{1,6}};atomic.orbitals={rw,{1,1,1,6}};
  atomic.species.pao_mesh=mesh;atomic.species.basis_lmax=maxl;
  atomic.species.basis=nb;atomic.species.angular_channels=1;
  openmx_set_fp32_enabled(1);
  for(int cycle=0;cycle<8;cycle++) {
    event_count=0;
    auto a=calculation(),b=calculation();auto *sa=Set_Orbitals_Grid_GPU_Bind_Resources(a.get());
    auto *sb=Set_Orbitals_Grid_GPU_Bind_Resources(b.get());
    assert(sa!=sb && sa==Set_Orbitals_Grid_GPU_Bind_Resources(a.get()));
    bind_device(0);assert(SOG_Device_Prepare(sa,&atomic));
    const auto *t=SOG_Device_Tables(sa);assert(t && t->eval_df && t->rwd && !host_live);
    assert(openmx_core_native_resource_count(sa->resources)==1);
    auto *arena=t->arena;size_t bytes=t->bytes;
    assert(SOG_Device_Prepare(sa,&atomic) && SOG_Device_Tables(sa)->arena==arena);
    bind_device(2);assert(SOG_Device_Prepare(sa,&atomic));
    int device_b=cycle%2?0:1;
    bind_device(device_b);ratio=2;assert(SOG_Device_Prepare(sb,&atomic));
    auto tb=*SOG_Device_Tables(sb);assert(!tb.eval_df && !tb.rwd && tb.bytes<bytes);
    std::vector<unsigned char> snapshot(tb.arena,tb.arena+tb.bytes);
    current_device=7;acc_device=6;
    openmx_core_native_resources_clear(a.get());assert(current_device==7 && acc_device==6 && live==1);
    assert(std::memcmp(snapshot.data(),tb.arena,snapshot.size())==0);
    bind_device(device_b);assert(SOG_Device_Tables(sb)->arena==tb.arena);
    sa=Set_Orbitals_Grid_GPU_Bind_Resources(a.get());
    bind_device(0);ratio=16;assert(SOG_Device_Prepare(sa,&atomic));
    current_device=7;acc_device=6;openmx_core_finish(a.get());
    assert(current_device==7 && acc_device==6 && live==1);
    sa=Set_Orbitals_Grid_GPU_Bind_Resources(a.get());
    bind_device(0);assert(SOG_Device_Prepare(sa,&atomic));
    current_device=7;acc_device=6;b.reset();a.reset();
    assert(!live && !host_live && current_device==7 && acc_device==6);
  }
  auto c=calculation();auto *s=Set_Orbitals_Grid_GPU_Bind_Resources(c.get());bind_device(0);
  for(int df:{0,1}) {
    openmx_set_fp32_enabled(df);
    for(size_t fail=1;fail<=size_t(df?11:8);fail++) {
      host_allocations=0;host_fail=fail;
      assert(!SOG_Device_Prepare(s,&atomic));assert(!live && !host_live && !SOG_Device_Tables(s));
    }
    host_fail=0;host_allocations=0;fail_status=OPENMX_GPU_OUT_OF_MEMORY;
    assert(!SOG_Device_Prepare(s,&atomic));assert(!live && !host_live);
    fail_status=0;
    openmx::test::fail_allocation_after(0);
    int admitted=SOG_Device_Prepare(s,&atomic);
    openmx::test::fail_allocation_after(-1);
    assert(!admitted && !live && !host_live && !openmx_core_native_resource_count(s->resources));
    assert(SOG_Device_Prepare(s,&atomic) && !host_live);
    assert(SOG_Device_Tables(s)->eval_df==df);
    // A live precision-switch mismatch invalidates and rebuilds this owner.
    openmx_set_fp32_enabled(!df);assert(SOG_Device_Prepare(s,&atomic));
    assert(SOG_Device_Tables(s)->eval_df==!df);
    Set_Orbitals_Grid_GPU_Release(s);assert(!live);
  }
  eligible=0;assert(!SOG_Device_Prepare(s,&atomic));eligible=1;
  assert(!SOG_Device_Prepare(s,&atomic)); // unsupported geometry is latched
  Set_Orbitals_Grid_GPU_Release(s);assert(SOG_Device_Prepare(s,&atomic));
  current_device=7;acc_device=6;c.reset();assert(!live && !host_live && current_device==7 && acc_device==6);
  std::cout<<"PASS: CUDA orbital ownership, 8 dual contexts, 3 devices, precision policy, host/device/registration failures and restored teardown\n";
}
