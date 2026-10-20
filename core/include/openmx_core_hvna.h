#ifndef OPENMX_CORE_HVNA_H
#define OPENMX_CORE_HVNA_H
#include "openmx_core_vna_preparation.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct OpenMXHvnaLayout OpenMXHvnaLayout;
typedef struct {
  int atoms, local_atoms, halo_atoms;
  const int *local_to_global;    /* [1..local_atoms] */
  const int *extended_to_global; /* [local_atoms+1..local_atoms+halo_atoms] */
  const int *species, *neighbors; /* [1..atoms], neighbors excludes self */
  double ***energies; /* [species][projector angular][projector radial] */
} OpenMXHvnaInput;
typedef struct {
  size_t local_elements, halo_elements, local_rows, halo_rows;
  int *local_bases, *halo_bases, *lengths;
  size_t *local_offsets, *halo_offsets;
  double *energies; /* [species, projector] */
} OpenMXHvnaBuffers;

/* Copies all metadata. Views are read-only by contract, independent of the
   input and preparation lifetime, and valid until destroy. Empty archives
   have zero elements/rows and one addressable dummy offset. No GPU init.
   Failure leaves buffers unchanged; exceptions never cross C. */
OpenMXHvnaLayout *openmx_core_hvna_create(const OpenMXVnaPreparation *preparation,
    const OpenMXHvnaInput *input, OpenMXHvnaBuffers *buffers, char *error, size_t capacity);
void openmx_core_hvna_destroy(OpenMXHvnaLayout *layout);
#ifdef __cplusplus
}
#endif
#endif
