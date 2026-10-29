#ifndef OPENMX_CORE_RADIAL_VIEWS_H
#define OPENMX_CORE_RADIAL_VIEWS_H
#include <stddef.h>
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

static inline double *openmx_radial_row2(const OpenMXRadialView2 *v, size_t a)
{
  return v->values+a*v->extent[1];
}
static inline double *openmx_radial_row3(const OpenMXRadialView3 *v, size_t a, size_t b)
{
  return v->values+(a*v->extent[1]+b)*v->extent[2];
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

#ifdef __cplusplus
}
#endif
#endif
