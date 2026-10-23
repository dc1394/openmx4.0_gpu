#ifndef OPENMX_CORE_CWF_WORKSPACE_H
#define OPENMX_CORE_CWF_WORKSPACE_H
#include "openmx_core_solver_workspace.h"
#ifdef __cplusplus
extern "C" {
#endif

/* A contiguous [spin,cell_i,cell_j,cell_k,element] host buffer. Each cell
   retains the caller's column-major local matrix layout and padding. Native
   C may store real or C99 complex elements without exposing those types to
   C++ or requiring a device SDK. The invocation owns the allocation. */
typedef struct {
  size_t spins, cells[3], elements;
} OpenMXCWFBlockShape;
typedef struct {
  OpenMXCWFBlockShape shape;
  size_t element_bytes;
  void *values;
} OpenMXCWFBlocks;

/* Products, byte counts and worker/name are checked before publishing *out.
   Failure preserves the output and the live named allocation. Reuse within
   capacity preserves values; callers initialize accumulators explicitly.
   Zero extents produce an empty view. Borrowed views expire on growth of
   the same worker/name or return from the enclosing solver callback. */
int openmx_core_solver_cwf_blocks(OpenMXSolverInvocation*, int worker,
    const char *name, const OpenMXCWFBlockShape*, size_t element_bytes,
    OpenMXCWFBlocks *out, char *error, size_t error_capacity);

/* Only views returned by the API above are valid; do not alter their shape.
   Invalid cell indices and empty views return NULL. */
static inline void *openmx_cwf_block_cell(const OpenMXCWFBlocks *view,
    size_t spin, size_t i, size_t j, size_t k)
{
  size_t cell;
  if(!view || !view->values || spin>=view->shape.spins ||
      i>=view->shape.cells[0] || j>=view->shape.cells[1] || k>=view->shape.cells[2])return NULL;
  cell=((spin*view->shape.cells[0]+i)*view->shape.cells[1]+j)*view->shape.cells[2]+k;
  return (unsigned char*)view->values+cell*view->shape.elements*view->element_bytes;
}

#ifdef __cplusplus
}
#endif
#endif
