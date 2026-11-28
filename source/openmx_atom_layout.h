#ifndef OPENMX_ATOM_LAYOUT_H
#define OPENMX_ATOM_LAYOUT_H

#include "mpi.h"
#include "openmx_core_native_resources.h"
#include <limits.h>
#include <stdint.h>
#include <string.h>

/* Rank-major packing for atom-owned records. Ownership is refreshed after
   truncation, including when some ranks own no atoms. */
typedef struct {
  int *storage;
  size_t capacity;
  int *offset, *counts, *displs, *order;
} OpenMXAtomLayout;

static inline int OpenMX_Atom_Layout_Bind(OpenMXAtomLayout *layout,
    OpenMXNativeResources *resources, int ranks, int atoms, const int *owner)
{
  const size_t count=3*(size_t)ranks+1+(size_t)atoms;
  if (count>SIZE_MAX/sizeof(int)) return 0;
  if (count>layout->capacity) {
    int *storage=openmx_core_native_realloc(resources,layout->storage,count*sizeof(int));
    if (!storage) return 0;
    layout->storage=storage;
    layout->capacity=count;
  }
  layout->offset=layout->storage;
  layout->counts=layout->offset+ranks+1;
  layout->displs=layout->counts+ranks;
  layout->order=layout->displs+ranks;
  memset(layout->offset,0,((size_t)ranks+1)*sizeof(int));
  for (int atom=1; atom<=atoms; ++atom) ++layout->offset[owner[atom]+1];
  for (int p=0; p<ranks; ++p) layout->offset[p+1]+=layout->offset[p];
  memcpy(layout->counts,layout->offset,(size_t)ranks*sizeof(int));
  for (int atom=1; atom<=atoms; ++atom) layout->order[layout->counts[owner[atom]]++]=atom;
  return 1;
}

/* Return false when an MPI int count cannot represent the packed records. */
static inline int OpenMX_Atom_Layout_Records(OpenMXAtomLayout *layout, int ranks, int width)
{
  if (width<=0 || layout->offset[ranks]>INT_MAX/width) return 0;
  for (int p=0; p<ranks; ++p) {
    layout->displs[p]=width*layout->offset[p];
    layout->counts[p]=width*(layout->offset[p+1]-layout->offset[p]);
  }
  return 1;
}

#endif
