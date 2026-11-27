#ifndef OPENMX_MO_GRID_H
#define OPENMX_MO_GRID_H
#include <stdio.h>
#include <mpi.h>
#include "openmx_complex.h"
#include "openmx_core_chemical.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int global_atom, samples, orbitals, stride;
    const int *grid, *cell;
    const float *paos;
} OpenMXMOGridAtom;

typedef struct {
    MPI_Comm comm;
    int root, verbosity, simd, gpu;
    int atom_count, grid_size, last_axis, part_count, spin_count, point_count, cell_count, binary;
    const OpenMXMOGridAtom *atoms;
    double *const *kpoints;  /* NULL for an isolated molecule */
    int *const *translations;
    const int *nhomos, *nlumos;
    dcomplex *****homos, *****lumos;
    const char *directory, *name, *extension;
    const OpenMXChemicalPotentials *chemical;
    void (*cube_header)(const OpenMXChemicalPotentials*,FILE*,int,double,double);
} OpenMXMOGridInput;

/* Returns 0 before publishing output if the optimized path cannot start. */
int openmx_mo_grid_write(const OpenMXMOGridInput *input);

#ifdef __cplusplus
}
#endif
#endif
