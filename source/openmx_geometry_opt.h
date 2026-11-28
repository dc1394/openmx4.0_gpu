#ifndef OPENMX_GEOMETRY_OPT_H
#define OPENMX_GEOMETRY_OPT_H

#include "mpi.h"
#include "openmx_core.h"

typedef struct OpenMXGeometryOpt OpenMXGeometryOpt;

/* The ownership map can change after each electronic step. All atom and
   Hessian row tables use the native one-based indexing. */
OpenMXGeometryOpt *OpenMX_Geometry_Begin(OpenMXCoreContext *, MPI_Comm,
    int host, int atoms, const int *owner);
void OpenMX_Geometry_Share_Atoms(OpenMXGeometryOpt *, double **xyz);
void OpenMX_Geometry_Publish(OpenMXGeometryOpt *, double **xyz, double (*cell)[4]);
/* Only the host needs the complete model Hessian for the geometry solve. */
void OpenMX_Geometry_Gather_Hessian(OpenMXGeometryOpt *, double **hessian, int cell_columns);

typedef struct {
  int local_atoms;
  const int *local_to_global, *species, *elements, *neighbors;
  int *const *neighbor_atoms, *const *neighbor_images;
  double *const *positions, *const *translations, *const *distances;
} OpenMXSchlegelModel;

/* With a cell, the atomic coordinates are fractional and the setup also
   produces the atom-cell and cell-cell derivatives. */
void OpenMX_Geometry_Schlegel(OpenMXGeometryOpt *, double **hessian,
    const OpenMXSchlegelModel *, const double (*cell)[4], const double (*inverse_cell)[4]);

/* Normalized residual Gram matrix with the DIIS constraint row and column. */
void OpenMX_Geometry_Prepare_DIIS(OpenMXGeometryOpt *, double ***residual, int sites,
    int count, double **matrix, double **weights);
int OpenMX_Geometry_Solve_DIIS(OpenMXGeometryOpt *, int count, double *matrix, double *weights);

typedef enum {
  OPENMX_GEOMETRY_RF_HESSIAN,
  OPENMX_GEOMETRY_EF_HESSIAN,
  OPENMX_GEOMETRY_INVERSE_HESSIAN
} OpenMXGeometryUpdate;

void OpenMX_Geometry_BFGS(OpenMXGeometryOpt *, double **hessian,
    double ***positions, double ***gradients, int sites, OpenMXGeometryUpdate update);
/* Apply the inverse Hessian to xyz[][1:3], writing to its scratch row 0. */
void OpenMX_Geometry_Apply_Inverse(OpenMXGeometryOpt *, double **inverse, double **xyz);

/* Smallest eigenpair of the augmented Hessian. EF takes all eigenpairs and
   applies the existing lower bound on the Hessian eigenvalues. The step is
   written to the Hessian's scratch row 0. */
void OpenMX_Geometry_RF_Step(OpenMXGeometryOpt *, double **hessian, int dimension,
    int use_gpu, int stdout_level);
void OpenMX_Geometry_EF_Step(OpenMXGeometryOpt *, double **hessian, double min_eigenvalue,
    int use_gpu, int stdout_level);

#endif
