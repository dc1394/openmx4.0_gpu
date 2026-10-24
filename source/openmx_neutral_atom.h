#ifndef OPENMX_NEUTRAL_ATOM_H
#define OPENMX_NEUTRAL_ATOM_H
#include "openmx_core_atomic.h"
#include <math.h>

/* One species row, borrowed from the existing table owner. No row table,
   allocation, process-global state or backend initialization is needed. */
typedef struct {
  int count;
  double spacing;
  const double *x,*y,*z,*density,*weight;
} OpenMXNeutralGridView;
typedef struct {
  double charge;
  int mesh;
  /* The existing interpolation ABI accepts writable rows but only reads them. */
  double *log_radii,*radii,*hartree;
} OpenMXNeutralPotentialView;

#ifdef __cplusplus
extern "C" {
#endif
/* Existing native radial interpolation; direct calls retain inlining/linkage
   choices without a callback or virtual dispatch inside either integral. */
double VH_AtomF(double,int,double,double,double*,double*,double*);
double Dr_VH_AtomF(double,int,double,double,double*,double*,double*);
double KumoF(int,double,double*,double*,double*);
#ifdef __cplusplus
}
#endif

/* Explicit output also avoids a large aggregate return in native OpenMP. */
static inline void OpenMX_Neutral_PotentialBorrow(const OpenMXAtomicViews *atomic,
    int species,OpenMXNeutralPotentialView *out)
{
  out->charge=atomic->species.core_charge[species];
  out->mesh=atomic->species.vps_mesh[species];
  out->log_radii=openmx_radial_row2(&atomic->vps_log_radii,species);
  out->radii=openmx_radial_row2(&atomic->vps_radii,species);
  out->hartree=openmx_radial_row2(&atomic->hartree_potential,species);
}

/* Keep the native scalar arithmetic and its two small-distance guards.
   Logarithmic radius is evaluated before the guard, as in the original ABI.
   The caller supplies the actual atom/image displacement, not a direction. */
static inline void OpenMX_Neutral_Pair(const OpenMXNeutralGridView *grid,
    const OpenMXNeutralPotentialView *potential,int derivative,double separation,
    const double displacement[3],double VH0ij[4])
{
  int n1;
  double x,y,z,r,r2,xx,va0,rho0,dr_va0;
  double z2,sum,sumr,sumx,sumy,sumz,wt;
  const double dv=grid->spacing;
  sum = 0.0;
  sumr = 0.0;
  for (n1=0; n1<grid->count; n1++){
    x = grid->x[n1];
    y = grid->y[n1];
    z = grid->z[n1];
    rho0 = grid->density[n1];
    wt = grid->weight[n1];
    z2 = z - separation;
    r2 = x*x + y*y + z2*z2;
    r = sqrt(r2);
    xx = 0.5*log(r2);
    if (r<1.0e-10) r = 1.0e-10;
    va0 = VH_AtomF(potential->charge,potential->mesh,xx,r,
                   potential->log_radii,potential->radii,potential->hartree);
    sum += wt*va0*rho0;
    if (derivative!=0 && 1.0e-14<r){
      dr_va0 = Dr_VH_AtomF(potential->charge,potential->mesh,xx,r,
                           potential->log_radii,potential->radii,potential->hartree);
      sumr -= wt*rho0*dr_va0*z2/r;
    }
  }
  sum = sum*dv;
  if (derivative!=0){
    r = separation;
    if (r<1.0e-10) r = 1.0e-10;
    x = displacement[0];
    y = displacement[1];
    z = displacement[2];
    sumr = sumr*dv;
    sumx = sumr*x/r;
    sumy = sumr*y/r;
    sumz = sumr*z/r;
  }
  else{
    sumx = 0.0;
    sumy = 0.0;
    sumz = 0.0;
  }
  VH0ij[0] = sum;
  VH0ij[1] = sumx;
  VH0ij[2] = sumy;
  VH0ij[3] = sumz;
}

/* A worker accumulates into its existing, already initialized partial.
   Team partitioning, padded/contiguous storage and serial reduction remain
   the caller's policy. Cutoff integration intentionally has no radius guard. */
static inline void OpenMX_Neutral_CutoffAccumulate(const OpenMXNeutralGridView *grid,
    const OpenMXNeutralPotentialView *potential,double rcut,int first,int last,
    double *partial)
{
  int n1;
  double x,y,z,rho0,wt,z2,r2,r,xx,va0;
  for (n1=first; n1<last; n1++){
    x = grid->x[n1];
    y = grid->y[n1];
    z = grid->z[n1];
    rho0 = grid->density[n1];
    wt = grid->weight[n1];
    z2 = z - rcut;
    r2 = x*x + y*y + z2*z2;
    r = sqrt(r2);
    xx = 0.5*log(r2);
    va0 = VH_AtomF(potential->charge,potential->mesh,xx,r,
                   potential->log_radii,potential->radii,potential->hartree);
    *partial += wt*va0*rho0;
  }
}

/* Explicit quadrature inputs and writable output rows borrow existing table
   owners. The caller retains its per-worker C++ axis scratch and grid capacity. */
typedef struct {
  int planes,gauss_points;
  double cutoff,spacing,pi;
  const double *abscissae,*weights;
} OpenMXNeutralQuadratureView;

/* z planes stay outermost, followed by Gauss points. Only the active rows
   are produced; capacity/padding and the MD initialization policy stay with
   the caller. Scalar return avoids a large aggregate in native OpenMP. */
#if defined(__GNUC__) || defined(__clang__) || defined(__NVCOMPILER)
__attribute__((always_inline))
#endif
static inline int OpenMX_Neutral_GridFill(const OpenMXAtomicViews *atomic,int species,
    const OpenMXNeutralQuadratureView *quadrature,double *axis,
    double *grid_x,double *grid_y,double *grid_z,double *density,double *weight)
{
  int n1,n3,gnum;
  double x,y,z,tmp0,xx,rho0,Sx,Dx;
  for (n1=0; n1<quadrature->planes; n1++){
    axis[n1] = quadrature->spacing*(double)n1 - quadrature->cutoff;
  }
  gnum = 0;
  y = 0.0;
  Sx = quadrature->cutoff + 0.0;
  Dx = quadrature->cutoff - 0.0;
  for (n3=0; n3<quadrature->planes; n3++){
    z = axis[n3];
    tmp0 = z*z;
    for (n1=0; n1<quadrature->gauss_points; n1++){
      x = 0.50*(Dx*quadrature->abscissae[n1] + Sx);
      xx = 0.5*log(x*x + tmp0);
      grid_x[gnum] = x;
      grid_y[gnum] = y;
      grid_z[gnum] = z;
      rho0 = KumoF(atomic->species.pao_mesh[species],xx,
          openmx_radial_row2(&atomic->pao_log_radii,species),openmx_radial_row2(&atomic->pao_radii,species),openmx_radial_row2(&atomic->density,species));
      density[gnum] = rho0;
      weight[gnum] = quadrature->pi*x*quadrature->weights[n1]*Dx;
      gnum++;
    }
  }
  return gnum;
}
#endif
