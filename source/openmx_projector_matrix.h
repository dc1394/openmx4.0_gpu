#ifndef OPENMX_PROJECTOR_MATRIX_H
#define OPENMX_PROJECTOR_MATRIX_H
#include "openmx_core_vna_matrix.h"

/* Include after Type_DS_VNA. The CPU evaluator supports both storage types;
   the application's current persistent projector format is float. */
typedef struct { OpenMXAtomBlockLayout layout; Type_DS_VNA *values; } OpenMXProjectorMatrix;
static inline Type_DS_VNA *openmx_projector_block_row(const OpenMXProjectorMatrix *matrix,
    int component,int atom,int neighbor,int row)
{
  const size_t b=openmx_block_index(&matrix->layout,atom,neighbor);
  return matrix->values+(size_t)component*matrix->layout.stride+matrix->layout.block_offsets[b]
      +(size_t)row*(size_t)matrix->layout.shapes[2*b+1];
}
#endif
