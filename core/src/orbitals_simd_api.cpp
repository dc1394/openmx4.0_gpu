#include "openmx_core_orbitals_simd.h"
#include "host_orbitals_kernels.hpp"

namespace {
constexpr int l0_limit=3,mul_limit=8,mesh_minimum=6;
// The orbital count of a species, or -1 for a shape the kernel does not cover.
int orbitals_of(const OpenMXOrbitalsTables& t,int wan) {
  const int maxl=t.maxl[wan];
  if(t.mesh[wan]<mesh_minimum || maxl<0 || maxl>l0_limit || maxl>t.l0max)return -1;
  int no=0;
  for(int l=0;l<=maxl;++l) {
    const int nb=t.nb[size_t(wan)*size_t(t.l0max+1)+size_t(l)];
    if(nb<0 || nb>mul_limit)return -1;
    no+=nb*(2*l+1);
  }
  return no;
}
}
extern "C" int openmx_core_orbitals_points(const OpenMXOrbitalsTables* t,const OpenMXOrbitalsFrame* f,
    const OpenMXOrbitalsPoint* q,size_t count,float* out,int mode) {
  const auto* kernels=openmx::core::host::openmx_host_orbitals_kernels(mode);
  if(!kernels || !t || !f)return 0;
  if(!count)return 1;
  if(!q || !out || !t->rv || !t->rwf || !t->translations || !t->rv_off || !t->rwf_base ||
      !t->mesh || !t->maxl || !t->nb)return 0;
  if(t->species<=0 || t->l0max<0 || t->cells<=0 || f->ng23<=0 || f->ng3<=0 || f->max_no<0)return 0;
  for(size_t i=0;i<count;++i) {
    const OpenMXOrbitalsPoint& e=q[i];
    if(e.wan<0 || e.wan>=t->species || e.cell<0 || e.cell>=t->cells || e.grid<0)return 0;
    const int no=orbitals_of(*t,e.wan);
    if(no<0 || e.no!=no || no>f->max_no)return 0;
  }
  kernels->points(*t,*f,q,count,out);
  return 1;
}
namespace {
int derivatives(const OpenMXOrbitalsTables* t,const OpenMXOrbitalsDisplacement* q,size_t count,double* out,int mode,bool exact) {
  const auto* kernels=openmx::core::host::openmx_host_orbitals_kernels(mode);
  if(!kernels || !t)return 0;
  if(!count)return 1;
  if(!q || !out || !t->rv || !t->rwf || !t->rv_off || !t->rwf_base || !t->mesh || !t->maxl || !t->nb)return 0;
  if(t->species<=0 || t->l0max<0)return 0;
  for(size_t i=0;i<count;++i) {
    if(q[i].wan<0 || q[i].wan>=t->species)return 0;
    const int no=orbitals_of(*t,q[i].wan);
    if(no<0 || q[i].no!=no)return 0;
  }
  (exact?kernels->dpoints:kernels->dpoints_cosines)(*t,q,count,out);
  return 1;
}
}
extern "C" int openmx_core_orbitals_dpoints(const OpenMXOrbitalsTables* t,const OpenMXOrbitalsDisplacement* q,
    size_t count,double* out,int mode) {
  return derivatives(t,q,count,out,mode,true);
}
extern "C" int openmx_core_orbitals_dpoints_cosines(const OpenMXOrbitalsTables* t,const OpenMXOrbitalsDisplacement* q,
    size_t count,double* out,int mode) {
  return derivatives(t,q,count,out,mode,false);
}
