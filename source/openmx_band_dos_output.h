#ifndef OPENMX_BAND_DOS_OUTPUT_H
#define OPENMX_BAND_DOS_OUTPUT_H

#include "mpi.h"
#include "openmx_band_dos.h"

typedef struct {
    int grid[3];
    const double *energy_range;
    int gaussian_bins;
    double gaussian_width;
    int fermi_surfer;
    int atom_count, species_count, max_l;
    const int *species, *orbitals;
    int *const *species_basis;
    const double *theta, *phi;
    int surface_only;
} OpenMXBandDosRequest;

typedef struct {
    MPI_Comm comm;
    int host, simd, device, first, end, gamma_owner;
    size_t workspace_limit;
    const char *directory, *name;
    const double (*reciprocal)[4];
    double chemical_potential;
} OpenMXBandDosExecution;

/* The caller has gathered the real-space matrices on the k-point owners. */
int openmx_band_dos_mesh(const OpenMXBandDosRequest *request, const OpenMXBandDosExecution *execution,
    const OpenMXKPathMatrices *matrices, const int *orbital_offsets, const OpenMXKPathPoint *points,
    int point_count, int *threads_used);

int openmx_band_dos_parallel(const OpenMXChemicalPotentials *chemical,
    const OpenMXBandDosRequest *request, int spin, double *****hamiltonian,
    double *****imaginary, double ****overlap, const struct OpenMXExactExchangeState *exchange);

static inline void openmx_band_dos_grid_index(const int grid[3], int point, int index[3])
{
    index[2] = point % grid[2];
    point /= grid[2];
    index[1] = point % grid[1];
    index[0] = point / grid[1];
}

#endif
