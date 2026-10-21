#ifndef OPENMX_CORE_ORBITAL_OPTIMIZATION_H
#define OPENMX_CORE_ORBITAL_OPTIMIZATION_H
#include "openmx_core.h"
#include "openmx_core_species.h"
#include "openmx_core_electronic.h"
#include "openmx_core_local_basis.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Borrowed only during begin; all topology and contraction maps are copied.
   Padding rows are not read. Neighbor lists contain self and first neighbors.
   spin=0/1; species_mode=0 atoms, 1 species; last_radial_only=0/1;
   method=1 DIIS, 2 EF. Coefficients in one magnetic multiplet share a count. */
typedef struct {
  int atoms, local_atoms, first_halo_atoms, max_orbitals, max_coefficients;
  int history, spin, species_mode, last_radial_only, method, start_pulay;
  int rank, ranks, host_rank, verbosity;
  double descent_step;
  const OpenMXSpeciesViews *species;
  const int *atom_species, *local_to_global, *first_to_global, *global_to_first;
  const int *neighbors;
  int *const *neighbor_atoms, *const *coefficient_counts;
  int **const *orbital_indices;
  const int *send_counts, *receive_counts, *receive_first;
  int *const *send_local_atoms, *const *send_global_atoms, *const *receive_global_atoms;
} OpenMXOrbOptInput;

/* History/tensor/matrix views use layout_right, with element offsets. */
typedef struct { size_t extent[4]; double *values; } OpenMXOrbOptHistory;
typedef struct { size_t rows, columns; double *values; } OpenMXOrbOptMatrix;
typedef struct {
  OpenMXOrbOptHistory atomic_coefficients, atomic_gradients;
  OpenMXOrbOptHistory species_coefficients, species_gradients;
  OpenMXOrbOptMatrix hessian;
  double *energies;
  size_t energy_count;
  OpenMXLocalTensor3 atomic_gradient, species_gradient;
} OpenMXOrbOptViews;

/* A callback returns one on success. The owner has no MPI, LAPACK, vendor SDK
   or implicit context dependency. Exchange posts both directions before waiting;
   a negative peer denotes an absent direction. Sizes have already been checked
   against INT_MAX. Eigen uses rows/values indexed from 1 through n; its row
   adapter exists only at the native eigensolver ABI. Inverse accepts row-major
   matrices with element strides and borrows packed[n*n], work[n], pivots[n]. */
typedef struct {
  void *user;
  int (*sum)(void*, const double*, double*, int);
  int (*maximum)(void*, int, int*);
  int (*broadcast)(void*, double*, int, int);
  int (*exchange)(void*, int, const double*, int, int, double*, int);
  int (*eigen)(void*, double**, double*, int);
  int (*inverse)(void*, int, const double*, size_t, double*, size_t,
                 double*, double*, int*);
} OpenMXOrbOptCallbacks;

/* Replacement is transactional. A step may update coefficients/history before
   a callback fails; the caller must stop that calculation on a step failure.
   Views expire on replacement, end or context finish. No runtime is initialized. */
int openmx_core_orbopt_begin(OpenMXCoreContext*, const OpenMXOrbOptInput*,
    OpenMXOrbOptViews*, char*, size_t);
int openmx_core_orbopt_get(OpenMXCoreContext*, OpenMXOrbOptViews*, char*, size_t);
void openmx_core_orbopt_end(OpenMXCoreContext*);
int openmx_core_orbopt_step(OpenMXCoreContext*, int iteration, double energy,
    const OpenMXElectronicViews*, const OpenMXLocalBasisViews*,
    const OpenMXOrbOptCallbacks*, double *norm, char*, size_t);
int openmx_core_orbopt_gradient(OpenMXCoreContext*, const OpenMXElectronicViews*,
    const OpenMXLocalBasisViews*, const OpenMXOrbOptCallbacks*, double *norm, char*, size_t);

#ifdef __cplusplus
}
#endif
#endif
