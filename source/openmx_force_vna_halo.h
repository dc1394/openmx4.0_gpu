#ifndef OPENMX_FORCE_VNA_HALO_H
#define OPENMX_FORCE_VNA_HALO_H

#include <limits.h>
#include <stddef.h>
#include <string.h>
#include "openmx_core_vna_matrix.h"

/* Local DS_VNA blocks are contiguous within a direction. The active
   (contracted) archive can alias the span only when its size matches. */
static inline const float *openmx_force_vna_atom_span(const OpenMXFloatBlockMatrix *matrix,
    int first, int end, size_t elements)
{
    if (!matrix || !matrix->values || elements == 0 || first < 1 || end <= first ||
        (size_t)end >= matrix->layout.atoms) return NULL;
    const size_t start = matrix->layout.block_offsets[matrix->layout.atom_offsets[first]];
    const size_t finish = matrix->layout.block_offsets[matrix->layout.atom_offsets[end]];
    return finish - start == elements ? matrix->values + start : NULL;
}

static inline void openmx_force_vna_pack_atom(float *target, float ***rows,
    int neighbors, int orbitals, int projectors)
{
    const size_t bytes = (size_t)projectors * sizeof(*target);
    for (int k = 0; k <= neighbors; ++k)
        for (int i = 0; i < orbitals; ++i) {
            memcpy(target, rows[k][i], bytes);
            target += projectors;
        }
}

/* The wire format contains all orbitals. Each neighbour's active orbitals
   occupy the first rows of its block, including in contracted calculations. */
static inline void openmx_force_vna_copy_halo(float *target, const float *source, size_t neighbours,
                                             size_t source_stride, size_t active_stride)
{
    if (target == source || active_stride == 0) return;
    if (source_stride == active_stride) {
        memcpy(target, source, neighbours * active_stride * sizeof(*target));
        return;
    }
    for (size_t k = 0; k < neighbours; ++k)
        memcpy(target + k * active_stride, source + k * source_stride, active_stride * sizeof(*target));
}

/* Both peers know the species and the full neighbour list of the halo atom.
   The MPI count includes the self block. */
static inline int openmx_force_vna_halo_count(int neighbours, int orbitals, int projectors)
{
    if (neighbours < 0 || orbitals < 0 || projectors < 0 ||
        (orbitals > 0 && projectors > INT_MAX / orbitals)) return -1;
    const size_t stride = (size_t)orbitals * (size_t)projectors;
    const size_t blocks = (size_t)neighbours + 1;
    if (stride > 0 && blocks > (size_t)INT_MAX / stride) return -1;
    return (int)(blocks * stride);
}

#endif
