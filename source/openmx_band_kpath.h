#ifndef OPENMX_BAND_KPATH_H
#define OPENMX_BAND_KPATH_H

#include <stddef.h>
#include "openmx_core_chemical.h"
#include "openmx_core_local_basis.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A real-space block, in the same neighbor order as the CPU constructor.
   The entries at value_offset are stored in column-major orbital order. */
typedef struct {
    int row, col, rows, cols;
    int translation[3];
    int value_offset;
} OpenMXKPathBlock;

typedef struct {
    int order, spin_channels, noncollinear, lno, imaginary_diagonals;
    /* With planar storage each block has a rows*cols plane per channel.
       Otherwise the channels are interleaved for each orbital pair. */
    int planar;
    int block_count, value_count;
    /* Entries below this offset have S and H channels. Entries at and
       above it carry only the spin channels of the exact exchange. */
    int exchange_offset;
    double overlap_cutoff;
    const OpenMXKPathBlock *blocks;
    const double *values;
} OpenMXKPathMatrices;

typedef struct {
    double xyz[3];
    int output_index;
} OpenMXKPathPoint;

struct OpenMXExactExchangeState;

/* A consumer receives one point in input order. The host rank receives
   the full eigensystem; other ranks receive NULL and may join collectives. */
typedef void (*OpenMXMOResult)(void *context, int point, const OpenMXKPathMatrices *m,
    const int *orbital_offsets, const double *eigenvalues, const double *vectors);
int openmx_band_mo_parallel(const OpenMXChemicalPotentials *chemical,
    int point_count, double **kpoints, int spin, double *****hamiltonian,
    double *****imaginary, double ****overlap, const struct OpenMXExactExchangeState *exchange,
    OpenMXMOResult result, void *context);

/* MPI adapter. Returns true after writing the file on the host rank. */
int openmx_band_kpath_parallel(const OpenMXChemicalPotentials *chemical,
    int path_count, const int *points_per_path, double ***paths, char ***names,
    int spin, double *****hamiltonian, double *****imaginary, double ****overlap,
    const OpenMXLocalBasisViews *local_basis, const struct OpenMXExactExchangeState *exchange);

/* One rank per physical device, called without MPI inside this routine.
   Eigenvalues are point-major, then spin-major and ascending. */
int openmx_band_kpath_solve_gpu(int device, const OpenMXKPathMatrices *matrices,
    int point_count, const OpenMXKPathPoint *points, double *eigenvalues);
int openmx_band_mo_vectors_gpu(int device, const OpenMXKPathMatrices *matrices,
    int point_count, const OpenMXKPathPoint *points, double *eigenvalues, double *vectors);

/* Thread workers use separate LAPACK workspaces and never call MPI. */
int openmx_band_kpath_solve_cpu(const OpenMXKPathMatrices *matrices,
    int point_count, const OpenMXKPathPoint *points, double *eigenvalues,
    int simd_mode, size_t workspace_limit, int *threads_used);
/* Vectors are interleaved complex doubles in column-major AO order, one
   matrix per point and spin channel. The noncollinear spin components are
   successive blocks of AO rows. */
int openmx_band_mo_vectors_cpu(const OpenMXKPathMatrices *m,
    int point_count, const OpenMXKPathPoint *points, double *eigenvalues, double *vectors,
    int simd_mode, size_t workspace_limit, int *threads_used);

#ifdef __cplusplus
}
#endif

#endif
