#ifndef OPENMX_CORE_ATOM_PAIRS_H
#define OPENMX_CORE_ATOM_PAIRS_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct OpenMXAtomPairPlan OpenMXAtomPairPlan;
typedef struct {
  int atoms, local_atoms, species, translations; /* translations includes zero */
  const int *local_to_global, *atom_species, *neighbor_counts; /* one-based atoms */
  int *const *neighbors, *const *images; /* [global][0..neighbor_counts] */
  double *const *positions, *const *translation_vectors; /* coordinates [1..3] */
} OpenMXAtomPairInput;
typedef struct {
  int count, local_atoms;
  const int *local, *neighbor, *central_global, *central_species, *neighbor_species;
  const size_t *offsets; /* [local_atoms+1], zero-based CSR offsets */
  const double *displacements; /* [count,2,3], forward then mirrored */
} OpenMXAtomPairBuffers;

/* Owns all exported data. No MPI or GPU runtime; zero local atoms accept null
   geometry tables. Invalid maps, images, counts and nonfinite coordinates are
   rejected. Failure leaves the caller's views unchanged. */
OpenMXAtomPairPlan *openmx_core_atom_pairs_create(const OpenMXAtomPairInput *input,
    OpenMXAtomPairBuffers *buffers,char *error,size_t capacity);
void openmx_core_atom_pairs_destroy(OpenMXAtomPairPlan *plan);
#ifdef __cplusplus
}
#endif
#endif
