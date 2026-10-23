#include <openmx_core_cwf_workspace.h>
#include <cstddef>
#include <cstdio>
#include <limits>

extern "C" int openmx_core_solver_cwf_blocks(OpenMXSolverInvocation* owner,
    int worker,const char* name,const OpenMXCWFBlockShape* shape,size_t bytes,
    OpenMXCWFBlocks* out,char* error,size_t capacity)
{
  auto fail=[&](const char* message) {
    if(error && capacity)std::snprintf(error,capacity,"%s",message);
    return 0;
  };
  if(!owner || !shape || !out || !bytes)return fail("missing CWF block owner, shape, output or element size");
  const size_t extents[]={shape->spins,shape->cells[0],shape->cells[1],shape->cells[2],shape->elements};
  bool empty=false;
  for(auto n:extents)empty|=n==0;
  size_t count=empty?0:1;
  if(!empty)for(auto n:extents) {
    if(count>std::numeric_limits<size_t>::max()/n)return fail("CWF block extent product overflows");
    count*=n;
  }
  if(count>size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes)
    return fail("CWF block byte count exceeds pointer range");
  void* values=nullptr;
  if(!openmx_core_solver_bytes(owner,worker,name,count,bytes,&values,error,capacity))return 0;
  *out={*shape,bytes,count?values:nullptr};
  return 1;
}
