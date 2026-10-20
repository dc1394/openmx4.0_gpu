#ifndef OPENMX_CORE_RECIPROCAL_H
#define OPENMX_CORE_RECIPROCAL_H

#include "openmx_core_pao.h"
#include "openmx_core_vps.h"
#include "openmx_core_product.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Contiguous layout_right views. Dimensions include the original padding.
   These descriptors borrow storage; no pointer or row may be freed/rebound.
   Accessors are unchecked, for indices validated by the calculation setup. */
typedef struct { double *values; size_t extent[2]; } OpenMXRadialView2;
typedef struct { double *values; size_t extent[3]; } OpenMXRadialView3;
typedef struct { double *values; size_t extent[4]; } OpenMXRadialView4;

typedef struct {
  size_t orbitals, first_row;
  const size_t *orbital_offsets, *pair_rows;
} OpenMXProductSpeciesView;
typedef struct {
  double *values;
  size_t species, rows, stride;
  const OpenMXProductSpeciesView *layout;
} OpenMXProductRadialView;

typedef struct {
  OpenMXRadialView4 pao_transform, nonlocal_transform;
  OpenMXRadialView4 neutral_projectors, neutral_transform;
  OpenMXRadialView3 neutral_energies;
  OpenMXRadialView2 potential_transform;
  OpenMXProductRadialView product_transform;
  /* Transitional readers, restart I/O and native packers borrow these row
     adapters over the same values. Flat host consumers need no row trees. */
  OpenMXPaoViews pao;
  OpenMXVpsViews vps;
  OpenMXProductViews products;
} OpenMXReciprocalViews;

static inline double *openmx_radial_row2(const OpenMXRadialView2 *v, size_t a)
{
  return v->values+a*v->extent[1];
}
static inline double *openmx_radial_row4(const OpenMXRadialView4 *v,
    size_t a, size_t b, size_t c)
{
  return v->values+((a*v->extent[1]+b)*v->extent[2]+c)*v->extent[3];
}
static inline double openmx_radial_value3(const OpenMXRadialView3 *v,
    size_t a, size_t b, size_t c)
{
  return v->values[(a*v->extent[1]+b)*v->extent[2]+c];
}
static inline double *openmx_product_radial_row(const OpenMXProductRadialView *v,
    size_t species, size_t l1, size_t m1, size_t l2, size_t m2, size_t angular)
{
  const OpenMXProductSpeciesView *s=v->layout+species;
  const size_t left=s->orbital_offsets[l1]+m1, right=s->orbital_offsets[l2]+m2;
  return v->values+(s->first_row+s->pair_rows[left*s->orbitals+right]+angular)*v->stride;
}

/* No allocation, Kokkos initialization, or process-global lookup. Absent PAO,
   VPS or product storage is represented by empty views. Failure leaves output
   unchanged. Views expire at the respective release, finish or destruction. */
int openmx_core_reciprocal_get(OpenMXCoreContext *context,
    OpenMXReciprocalViews *views, char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
