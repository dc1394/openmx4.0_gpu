/* Harness-only rectangular view over fixed reference arrays. Production
   storage is ragged; padding here remains visible to the frozen oracle. */
#include "openmx_core_vna_matrix.h"
static OpenMXAtomBlockLayout test_rectangular_matrix(int atoms,int components,int neighbors,
    int rows,int columns,size_t *atom_offsets,size_t *block_offsets,int *shapes)
{
  for (int a=0;a<=atoms;++a) atom_offsets[a]=(size_t)a*neighbors;
  const size_t blocks=(size_t)atoms*neighbors,stride=blocks*rows*columns;
  for (size_t b=0;b<=blocks;++b) {
    block_offsets[b]=b*rows*columns;
    if (b<blocks) { shapes[2*b]=rows;shapes[2*b+1]=columns; }
  }
  const OpenMXAtomBlockLayout result={(size_t)atoms,(size_t)components,blocks,stride,
      (size_t)components*stride,atom_offsets,block_offsets,shapes};
  return result;
}
