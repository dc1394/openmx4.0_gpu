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
    double overlap_cutoff;
    const OpenMXKPathBlock *blocks;
    const double *values;
} OpenMXKPathMatrices;

typedef struct {
    double xyz[3];
    int output_index;
} OpenMXKPathPoint;

/* MPI adapter. 1: file published, 0: HIP not requested, -1: use CPU. */
int openmx_band_kpath_try_hip(const OpenMXChemicalPotentials *chemical,
    int path_count, const int *points_per_path, double ***paths, char ***names,
    int spin, double *****hamiltonian, double *****imaginary, double ****overlap,
    const OpenMXLocalBasisViews *local_basis);

/* One rank per physical device, called without MPI inside this routine.
   Eigenvalues are point-major, then spin-major and ascending. */
int openmx_band_kpath_solve_hip(int device, const OpenMXKPathMatrices *matrices,
    int point_count, const OpenMXKPathPoint *points, double *eigenvalues);

int openmx_magma_zheev_values_gpu(int order, void *matrix, double *values);

#ifdef __cplusplus
}
#endif

#endif
