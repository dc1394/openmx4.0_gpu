#ifndef OPENMX_CORE_VNA_HOST_TABLES_H
#define OPENMX_CORE_VNA_HOST_TABLES_H
#include "openmx_core_vna_preparation.h"
#include "openmx_core_radial_views.h"
#include "openmx_complex.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct OpenMXVnaHostProjectors OpenMXVnaHostProjectors;
typedef struct OpenMXVnaHostProducts OpenMXVnaHostProducts;
typedef struct {
  OpenMXRadialView4 projectors; /* [species][l][radial][point], padding allowed */
  const dcomplex *conversion;  /* [projector_lmax+1][width][width] */
  size_t width;
  int convert;                /* 0 or 1; zero does not read conversion */
} OpenMXVnaHostProjectorInput;
typedef struct {
  size_t stride, width;
  const double *projectors; /* [species][points][stride], padded to 8 columns */
  const double *conversion_real, *conversion_imag; /* [l][column][row] */
} OpenMXVnaHostProjectorData;
typedef struct {
  int order_stride, radials_max;
  size_t blocks;
  const size_t *species_offsets; /* [species+1], offsets in blocks */
  const int *radials;            /* [species], active radial functions */
  const double *products;        /* [block][points][order_stride] */
} OpenMXVnaHostProductData;

/* Snapshot only active input rows into immutable CPU storage. Product blocks
   use species_offsets[s]+left*radials[s]+right, with species-local radial
   indices from the preparation; only l_left <= l_right is populated.
   Padding and inactive blocks are zero. Empty numeric arrays are NULL.
   No input or preparation pointer is retained. Inputs may be destroyed after
   successful creation. Output views remain valid until owner destruction and
   may be read concurrently. Creation is transactional and leaves out unchanged
   on failure; no exception, GPU runtime or global lookup crosses this boundary.
   Source descriptors must cover their declared extents and metadata arrays. */
OpenMXVnaHostProjectors *openmx_core_vna_host_projectors_create(
    const OpenMXVnaPreparation *preparation, const OpenMXVnaHostProjectorInput *input,
    OpenMXVnaHostProjectorData *out, char *error, size_t capacity);
OpenMXVnaHostProducts *openmx_core_vna_host_products_create(
    const OpenMXVnaPreparation *preparation, const OpenMXProductRadialView *input,
    OpenMXVnaHostProductData *out, char *error, size_t capacity);
/* NULL is accepted. */
void openmx_core_vna_host_projectors_destroy(OpenMXVnaHostProjectors *owner);
void openmx_core_vna_host_products_destroy(OpenMXVnaHostProducts *owner);
#ifdef __cplusplus
}
#endif
#endif
