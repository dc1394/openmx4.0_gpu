#ifndef OPENMX_CORE_DISPERSION_H
#define OPENMX_CORE_DISPERSION_H
#include "openmx_core_solver_workspace.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int atoms; /* Physical atoms; storage includes the legacy atom-zero padding. */
  int image_radius[3];
} OpenMXDispersionShape;
typedef struct {
  size_t atoms, images[3];
  double *coordination, *c6_derivative, *energy_derivative;
  double *coordination_derivative; /* layout_right [atom,atom,image1,image2,image3] */
} OpenMXDispersionView;

/* Bind before worker regions. The invocation owns all numeric storage; no
   pointer tables, MPI/device runtime or process cache participate. Each prepare
   zeros every element, including padding. Equal shapes reuse addresses without
   allocation. A changed shape invalidates the previous view only on success;
   failure preserves both that view's values and the output descriptor. */
int openmx_core_dispersion_prepare(OpenMXSolverInvocation*,const OpenMXDispersionShape*,
    OpenMXDispersionView*,char *error,size_t capacity);

/* Unchecked borrowed element adapters. Atom indices include padding zero;
   image indices are storage indices (signed image + full configured radius),
   even when one atom's periodicity reduces its loop to the central image. */
static inline double *openmx_dispersion_c6(const OpenMXDispersionView *v,size_t a,size_t b)
{ return v->c6_derivative+a*v->atoms+b; }
static inline double *openmx_dispersion_energy(const OpenMXDispersionView *v,size_t a,size_t b)
{ return v->energy_derivative+a*v->atoms+b; }
static inline double *openmx_dispersion_coordination(const OpenMXDispersionView *v,
    size_t a,size_t b,size_t i,size_t j,size_t k)
{ return v->coordination_derivative+((((a*v->atoms+b)*v->images[0]+i)*v->images[1]+j)*v->images[2]+k); }

#ifdef __cplusplus
}
#endif
#endif
