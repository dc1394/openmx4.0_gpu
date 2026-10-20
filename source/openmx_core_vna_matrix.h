#ifndef OPENMX_CORE_VNA_MATRIX_H
#define OPENMX_CORE_VNA_MATRIX_H
#include "openmx_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  size_t atoms, components, blocks, stride, elements;
  const size_t *atom_offsets, *block_offsets;
  const int *shapes; /* [block, {rows,columns}] */
} OpenMXAtomBlockLayout;
typedef struct { OpenMXAtomBlockLayout layout; double *values; } OpenMXDoubleBlockMatrix;
typedef struct { OpenMXAtomBlockLayout layout; float *values; } OpenMXFloatBlockMatrix;

typedef struct {
  int atoms, local_atoms, species, contracted;
  int scratch_neighbors, scratch_orbitals, projector_components;
  const int *local_to_global, *projector_to_global, *atom_species, *neighbors;
  int *const *neighbor_atoms;
  const int *orbitals, *contracted_orbitals;
} OpenMXVnaMatrixInput;

typedef struct {
  OpenMXDoubleBlockMatrix potential, one_center, mirrored;
  OpenMXFloatBlockMatrix projectors;
  OpenMXDoubleBlockMatrix contracted_one_center, contracted_mirrored;
  OpenMXFloatBlockMatrix contracted_projectors;
  /* Borrowed compatibility views. They own no numeric values and must never
     be freed or rebound by consumers. New kernels use the flat views above. */
  double ****HVNA, *****HVNA2, *****HVNA3, *****CntHVNA2, *****CntHVNA3;
  float *****DS_VNA, *****CntDS_VNA;
} OpenMXVnaMatrixViews;

/* Context/topology-scoped host storage, independent of MPI/GPU initialization.
   Construction copies all shape metadata. Successful replacement invalidates
   previous views; failure preserves the previous owner and output views.
   Access before allocation returns an empty view. Release is idempotent. */
int openmx_core_vna_matrices_allocate(OpenMXCoreContext *context,
    const OpenMXVnaMatrixInput *input, OpenMXVnaMatrixViews *views,
    char *error, size_t capacity);
int openmx_core_vna_matrices_get(OpenMXCoreContext *context,
    OpenMXVnaMatrixViews *views, char *error, size_t capacity);
void openmx_core_vna_matrices_release(OpenMXCoreContext *context);

#ifdef __cplusplus
}
#endif

/* Validated layout, unchecked access in inner loops (as with mdspan).
   Atom zero and the final projector receive slot are explicit CSR entries. */
static inline size_t openmx_block_index(const OpenMXAtomBlockLayout *layout, int atom, int neighbor)
{ return layout->atom_offsets[atom]+(size_t)neighbor; }
static inline double *openmx_double_block_row(const OpenMXDoubleBlockMatrix *matrix,
    int component, int atom, int neighbor, int row)
{
  const size_t b=openmx_block_index(&matrix->layout,atom,neighbor);
  return matrix->values+(size_t)component*matrix->layout.stride+matrix->layout.block_offsets[b]
      +(size_t)row*(size_t)matrix->layout.shapes[2*b+1];
}
static inline float *openmx_float_block_row(const OpenMXFloatBlockMatrix *matrix,
    int component, int atom, int neighbor, int row)
{
  const size_t b=openmx_block_index(&matrix->layout,atom,neighbor);
  return matrix->values+(size_t)component*matrix->layout.stride+matrix->layout.block_offsets[b]
      +(size_t)row*(size_t)matrix->layout.shapes[2*b+1];
}
#endif
