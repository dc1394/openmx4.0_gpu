#ifndef OPENMX_CORE_ELECTRONIC_H
#define OPENMX_CORE_ELECTRONIC_H
#include "openmx_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  OPENMX_ELECTRONIC_KINETIC, OPENMX_ELECTRONIC_CONTRACTED_KINETIC,
  OPENMX_ELECTRONIC_NONLOCAL, OPENMX_ELECTRONIC_IMAGINARY_NONLOCAL,
  OPENMX_ELECTRONIC_CONTRACTED_IMAGINARY_NONLOCAL, OPENMX_ELECTRONIC_CORE_HOLE,
  OPENMX_ELECTRONIC_IMAGINARY_CORE_HOLE, OPENMX_ELECTRONIC_XMCD, OPENMX_ELECTRONIC_HUBBARD,
  OPENMX_ELECTRONIC_IMAGINARY_NONLOCAL_REFERENCE, OPENMX_ELECTRONIC_ANGULAR_OVERLAP,
  OPENMX_ELECTRONIC_OVERLAP, OPENMX_ELECTRONIC_CORE_HOLE_OVERLAP,
  OPENMX_ELECTRONIC_CONTRACTED_OVERLAP, OPENMX_ELECTRONIC_HAMILTONIAN,
  OPENMX_ELECTRONIC_CONTRACTED_HAMILTONIAN, OPENMX_ELECTRONIC_PROJECTORS,
  OPENMX_ELECTRONIC_CONTRACTED_PROJECTORS, OPENMX_ELECTRONIC_LNO_DENSITY,
  OPENMX_ELECTRONIC_HAMILTONIAN_HISTORY, OPENMX_ELECTRONIC_IMAGINARY_HAMILTONIAN_HISTORY,
  OPENMX_ELECTRONIC_HAMILTONIAN_RESIDUAL, OPENMX_ELECTRONIC_IMAGINARY_HAMILTONIAN_RESIDUAL,
  OPENMX_ELECTRONIC_DENSITY, OPENMX_ELECTRONIC_GUIDING_DENSITY,
  OPENMX_ELECTRONIC_IMAGINARY_GUIDING_DENSITY, OPENMX_ELECTRONIC_PARTIAL_DENSITY,
  OPENMX_ELECTRONIC_DENSITY_RESIDUAL, OPENMX_ELECTRONIC_IMAGINARY_DENSITY_RESIDUAL,
  OPENMX_ELECTRONIC_ENERGY_DENSITY, OPENMX_ELECTRONIC_PREVIOUS_DENSITY,
  OPENMX_ELECTRONIC_IMAGINARY_DENSITY, OPENMX_ELECTRONIC_FIELD_COUNT
} OpenMXElectronicField;

/* Flat layout_right blocks. Components flatten group*channels+channel;
   each component has its own atom/scratch rows. All offsets and values are
   calculation-owned; no pointer table is needed by numerical kernels. */
typedef struct {
  size_t groups, channels, components, atoms, blocks, elements;
  const size_t *component_offsets, *atom_offsets, *block_offsets;
  const int *shapes;
  double *values;
} OpenMXElectronicMatrixView;

typedef struct {
  int atoms, local_atoms, first_halo_atoms, second_halo_atoms, species;
  int spin, spin_orbit, mixing, nonlocal_components;
  int density_history, hamiltonian_history;
  int scratch_orbitals, scratch_neighbors, scratch_projectors;
  int contracted, core_hole, xmcd, interactions, lno, guiding_density, partial_density;
  int full_overlap_halo_derivatives;
  const int *atom_species, *orbitals, *contracted_orbitals, *projector_components;
  const int *local_to_global, *first_to_global, *second_to_global, *neighbors;
  int *const *neighbor_atoms;
} OpenMXElectronicMatrixInput;

typedef struct {
  OpenMXElectronicMatrixView fields[OPENMX_ELECTRONIC_FIELD_COUNT];
  /* Temporary native row adapters, borrowing the same flat numeric storage.
     Neither values nor adapters may be freed/rebound. Missing families are NULL. */
  double *****H0, *****CntH0, *****HNL, *****iHNL, *****iCntHNL;
  double *****HCH, *****iHCH, *****H_XMCD, *****H_Hub, *****iHNL0;
  double *****OLP_L, *****OLP, *****OLP_CH, *****CntOLP, *****H, *****CntH;
  double ******DS_NL, ******CntDS_NL;
  double *****DM0;
  double ******HisH1, ******HisH2, ******ResidualH1, ******ResidualH2, ******DM;
  double *****ParDM, *****iParDM, *****Partial_DM;
  double ******ResidualDM, ******iResidualDM;
  double *****EDM, *****PDM;
  double ******iDM;
} OpenMXElectronicViews;

/* Replacement is transactional, including compatibility adapter allocation.
   Construction copies all shape metadata. Teardown is independent of input,
   MPI, GPUs, old atom distribution and neighbor list lifetimes. */
int openmx_core_electronic_allocate(OpenMXCoreContext *context,
    const OpenMXElectronicMatrixInput *input, OpenMXElectronicViews *views,
    char *error, size_t capacity);
int openmx_core_electronic_get(OpenMXCoreContext *context,
    OpenMXElectronicViews *views, char *error, size_t capacity);
void openmx_core_electronic_release(OpenMXCoreContext *context);

#ifdef __cplusplus
}
#endif

/* Validated shapes, unchecked inner-loop access, with the same semantics as
   the C++ mdspan block. The component number includes the history group. */
static inline double *openmx_electronic_row(const OpenMXElectronicMatrixView *view,
    size_t component, size_t atom, size_t neighbor, size_t row)
{
  const size_t a=view->component_offsets[component]+atom;
  const size_t b=view->atom_offsets[a]+neighbor;
  return view->values+view->block_offsets[b]+row*(size_t)view->shapes[2*b+1];
}
#endif
