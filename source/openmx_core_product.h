#ifndef OPENMX_CORE_PRODUCT_H
#define OPENMX_CORE_PRODUCT_H

#include "openmx_core_radial.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int species, wavevectors;
  const int *lmax;                 /* [species], each 0..14 */
  int *const *multiplicities;       /* each species: [lmax+1], nonnegative */
} OpenMXProductShape;

/* Borrowed compatibility table over contiguous C++ numeric storage.
   rows[species][l1][m1][l2][m2][l][k], l1<=l2, l<=2*l2, k<wavevectors+2.
   Empty angular groups and l1>l2 blocks are null, not allocated placeholders.
   All numeric values including padding start at zero. Do not free/rebind rows. */
typedef struct { double *******rows; } OpenMXProductViews;

/* Failure leaves context/output unchanged. Views expire at product_release,
   core_finish or context destruction. Other calculation storage is independent. */
int openmx_core_product_allocate(OpenMXCoreContext *context, const OpenMXProductShape *shape,
    OpenMXProductViews *views, char *error, size_t error_capacity);
void openmx_core_product_release(OpenMXCoreContext *context);

typedef struct {
  int lmax, mesh_count;
  const int *multiplicities;
  const double *radii;
  double ***values;                /* [angular][multiplicity][mesh_count] */
  double ******output;             /* [l1][m1][l2][m2][l][k_count], l1<=l2 */
} OpenMXProductTable;

/* Borrows and validates inputs; no pointers retained after return. Each active
   output row needs k_count doubles. Commit occurs only after complete success. */
int openmx_core_transform_products(OpenMXCoreContext *context, int device,
    const OpenMXProductTable *table, const OpenMXGaussRadialGrid *grid,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
