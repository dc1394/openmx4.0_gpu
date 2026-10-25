#include "openmx_core_olg.h"
#include "host_olg_kernels.hpp"

extern "C" int openmx_core_olg_intersect(const OpenMXOlgLists* lists,const OpenMXOlgQuery* queries,
    size_t count,int* out,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_olg_kernels(mode);
  if(!kernels || !lists || lists->nc<0 || lists->cpy<0 || lists->cells<0)return 0;
  if(count && (!queries || !out))return 0;
  if(lists->nc>0 && (!lists->grid || !lists->cell))return 0;
  if(count && lists->nc>0 && (!lists->shifts || !lists->periodic))return 0;
  kernels->intersect(*lists,queries,count,out);
  return 1;
}

extern "C" int openmx_core_olg_intersect_lists(const OpenMXOlgLists* lists,const OpenMXOlgNeighbours* neighbours,
    size_t count,int* out,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_olg_kernels(mode);
  if(!kernels || !lists || !neighbours || lists->nc<0 || lists->cpy<0 || lists->cells<0)return 0;
  if(count && (!neighbours->grid || !neighbours->cell || !out))return 0;
  if(lists->nc>0 && (!lists->grid || !lists->cell))return 0;
  if(count && lists->nc>0 && (!lists->shifts || !lists->periodic))return 0;
  kernels->intersect_lists(*lists,*neighbours,count,out);
  return 1;
}

namespace {
bool map_arguments(const OpenMXOlgLists* lists,const int* map,size_t extent) {
  if(!lists || !map || lists->nc<0 || (lists->nc>0 && !lists->grid))return false;
  for(int i=0;i<lists->nc;++i)if(lists->grid[i]<0 || size_t(lists->grid[i])>=extent)return false;
  return true;
}
}
extern "C" int openmx_core_olg_map_fill(const OpenMXOlgLists* lists,int* map,size_t extent,int mode) {
  if(!openmx::core::host::openmx_host_olg_kernels(mode) || !map_arguments(lists,map,extent))return 0;
  for(int i=lists->nc-1;i>=0;--i)map[lists->grid[i]]=i;  // descending: the leftmost entry wins
  return 1;
}
extern "C" int openmx_core_olg_map_reset(const OpenMXOlgLists* lists,int* map,size_t extent,int mode) {
  if(!openmx::core::host::openmx_host_olg_kernels(mode) || !map_arguments(lists,map,extent))return 0;
  for(int i=0;i<lists->nc;++i)map[lists->grid[i]]=-1;
  return 1;
}
extern "C" int openmx_core_olg_intersect_map(const OpenMXOlgLists* lists,const OpenMXOlgNeighbours* neighbours,
    const int* map,size_t extent,size_t count,int* out,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_olg_kernels(mode);
  if(!kernels || !lists || !neighbours || !map || lists->nc<0 || lists->cpy<0 || lists->cells<0)return 0;
  if(count && (!neighbours->grid || !neighbours->cell || !out))return 0;
  if(lists->nc>0 && (!lists->grid || !lists->cell))return 0;
  if(count && lists->nc>0 && (!lists->shifts || !lists->periodic))return 0;
  for(size_t i=0;i<count;++i)if(neighbours->grid[i]<0 || size_t(neighbours->grid[i])>=extent)return 0;
  kernels->intersect_map(*lists,*neighbours,map,extent,count,out);
  return 1;
}
