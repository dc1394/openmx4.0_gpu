#ifndef OPENMX_CORE_RECIPROCAL_H
#define OPENMX_CORE_RECIPROCAL_H

#include "openmx_core_radial_views.h"
#include "openmx_core_pao.h"
#include "openmx_core_vps.h"
#include "openmx_core_product.h"

#ifdef __cplusplus
extern "C" {
#endif

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

/* No allocation, Kokkos initialization, or process-global lookup. Absent PAO,
   VPS or product storage is represented by empty views. Failure leaves output
   unchanged. Views expire at the respective release, finish or destruction. */
int openmx_core_reciprocal_get(OpenMXCoreContext *context,
    OpenMXReciprocalViews *views, char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
