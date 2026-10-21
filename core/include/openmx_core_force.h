#ifndef OPENMX_CORE_FORCE_H
#define OPENMX_CORE_FORCE_H
#include "openmx_core_vna_matrix.h"
#include "openmx_core_solver_workspace.h"
#include "openmx_core_electronic.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int atoms, local_atoms, halo_atoms, species, spin, imaginary_density;
  const int *first_to_global, *atom_species, *orbitals, *neighbor_counts;
  int *const *neighbor_atoms;
} OpenMXForceInput;
typedef struct {
  OpenMXDoubleBlockMatrix density, imaginary;
  double *****density_rows, *****imaginary_rows;
  double *forces[3]; /* [local_atoms+1], atom zero is explicit padding */
  OpenMXSolverInvocation *scratch;
} OpenMXForceWorkspace;
typedef int (*OpenMXForceOperation)(const OpenMXForceWorkspace*,void*,double*);

/* Own all numeric buffers and topology for one Force invocation. The native
   callback borrows views; normal/early return and exceptions destroy them.
   No MPI, device runtime, hidden context, or input mutation participates. */
int openmx_core_force_run(const OpenMXForceInput*,int workers,OpenMXForceOperation,
    void *arguments,double *result,char *error,size_t capacity);

/* Explicit flat source views. The overlap trace preserves native element
   order and the two-stage weight multiplication before subtraction. Energy
   density may have larger primitive-basis blocks than contracted overlap;
   every component retains its own leading dimension. */
int openmx_core_force_overlap(const OpenMXElectronicMatrixView *overlap,
    const OpenMXElectronicMatrixView *energy_density,int spin,int atom,int neighbors,
    double force[3],char *error,size_t capacity);

/* Derivatives [3 spin components,rows,stride] for each Cartesian direction.
   Density matrices have row stride columns. Imaginary diagonal density is
   required only for spinor=1; prefactor is used by scalar/collinear traces.
   Accumulators change only after input validation has succeeded. */
typedef struct {
  int spin, spinor, rows, columns;
  double prefactor;
  const double *density[4], *imaginary[2];
  const dcomplex *derivative[3];
  size_t derivative_rows, derivative_stride;
} OpenMXForceSpinTrace;
int openmx_core_force_spin_trace(const OpenMXForceSpinTrace*,double force[3],
    char *error,size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
