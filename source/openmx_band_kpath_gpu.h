#ifndef OPENMX_BAND_KPATH_GPU_H
#define OPENMX_BAND_KPATH_GPU_H

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

/* MPI adapter. 1: file published, 0: fast path disabled, -1: use CPU. */
int openmx_band_kpath_try_gpu(const OpenMXChemicalPotentials *chemical,
    int path_count, const int *points_per_path, double ***paths, char ***names,
    int spin, double *****hamiltonian, double *****imaginary, double ****overlap,
    const OpenMXLocalBasisViews *local_basis, const struct OpenMXExactExchangeState *exchange);

/* One rank per physical device, called without MPI inside this routine.
   Eigenvalues are point-major, then spin-major and ascending. */
int openmx_band_kpath_solve_gpu(int device, const OpenMXKPathMatrices *matrices,
    int point_count, const OpenMXKPathPoint *points, double *eigenvalues);

#ifdef __cplusplus
}
#endif

#endif
