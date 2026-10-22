#include "openmx_core_native_resources.h"
#include "allocation_faults.hpp"
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "fixtures/native_resources/sdk_model.hpp"

namespace {
struct Fixture {
  OpenMXCoreContext* core;
  OpenMXNativeResources* owner=nullptr;
  Fixture() {
    char error[256];core=openmx_core_create_host(error,sizeof(error));check(core,error);
    check(openmx_core_native_resources(core,"adapter",nullptr,&owner,error,sizeof(error)),error);
  }
  ~Fixture() { openmx_core_destroy(core); }
};
void device() {
  Fixture f;check(!queries && !live,"creating a host calculation touched a GPU");
  current_device=3;void* memory=nullptr;void* pinned=nullptr;
  check(!OpenMX_Resource_GpuMalloc(f.owner,&memory,128),"device allocation");
  check(!OpenMX_Resource_HostAlloc(f.owner,&pinned,64,0),"pinned allocation");
  current_device=7;
  check(!OpenMX_Resource_GpuFree(f.owner,memory) && current_device==7,"device release did not restore caller");
  check(!OpenMX_Resource_HostFree(f.owner,pinned) && current_device==7,"pinned release did not restore caller");
  check(!live && !openmx_core_native_resource_count(f.owner),"released memory retained");
}
void handles() {
  {
    Fixture f;current_device=2;OpenMXBlasHandle blas;OpenMXSolverHandle solver;OpenMXGpuStream stream;
    check(!OpenMX_Resource_BlasCreate(f.owner,&blas),"BLAS creation");
    check(!OpenMX_Resource_SolverCreate(f.owner,&solver),"solver creation");
    check(!OpenMX_Resource_StreamCreate(f.owner,&stream,0),"stream creation");
    current_device=5;
    check(!OpenMX_Resource_BlasDestroy(f.owner,blas),"BLAS destruction");
    check(current_device==5,"BLAS release changed caller device");
    // solver/stream are intentionally left to the calculation destructor.
  }
  check(!live && current_device==5,"handle teardown leaked/changed caller device");
}
void failures() {
  Fixture f;void* output=reinterpret_cast<void*>(17);fail_status=47;
  check(OpenMX_Resource_GpuMalloc(f.owner,&output,128)==47 && !output,"SDK error was remapped/left a stale pointer");
  fail_status=0;event_count=0;openmx::test::fail_allocation_after(0);
  const auto status=OpenMX_Resource_GpuMalloc(f.owner,&output,128);
  openmx::test::fail_allocation_after(-1);
  check(status==2 && !output && !live,"failed registration leaked fresh device memory");
  check(!openmx_core_native_resource_count(f.owner),"failed registration changed owner");
  OpenMXBlasHandle blas=nullptr;openmx::test::fail_allocation_after(0);
  const auto blas_status=OpenMX_Resource_BlasCreate(f.owner,&blas);
  openmx::test::fail_allocation_after(-1);
  check(blas_status==7 && !blas && !live,"failed registration leaked a BLAS handle");
}
void stream_order() {
  Fixture f;void* memory;OpenMXGpuStream stream;
  check(!OpenMX_Resource_GpuMalloc(f.owner,&memory,32),"buffer creation");
  check(!OpenMX_Resource_StreamCreate(f.owner,&stream,0),"stream creation");
  event_count=0;openmx_core_native_resources_clear(f.core);
  int sync=-1,buffer=-1,destruction=-1;
  for(int i=0;i<event_count;i++) {
    if(events[i]==400 && sync<0)sync=i;
    if(events[i]==201)buffer=i;
    if(events[i]==206)destruction=i;
  }
  check(sync>=0 && sync<buffer && buffer<destruction && !live,"destructor did not drain before freeing buffers");
}
void acc() {
  Fixture f;current_device=3;acc_device=3;void* pointer=OpenMX_Resource_AccMalloc(f.owner,64);check(pointer,"ACC allocation");
  current_device=7;acc_device=7;OpenMX_Resource_AccFree(f.owner,pointer);
  check(!live && current_device==7 && acc_device==7,"ACC release did not restore both runtime selections");
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc==2?argv[1]:"";
    if(name=="device")device();else if(name=="handles")handles();else if(name=="failures")failures();
    else if(name=="stream-order")stream_order();else if(name=="acc")acc();else throw std::runtime_error("unknown adapter case");
    check(!live,"SDK allocations remain");std::cout<<"PASS native adapter "<<name<<'\n';return 0;
  }catch(const std::exception& e) { openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1; }
}
