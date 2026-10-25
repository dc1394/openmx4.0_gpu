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
