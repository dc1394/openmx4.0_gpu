#ifndef OPENMX_CORE_SOLVER_WORKSPACE_H
#define OPENMX_CORE_SOLVER_WORKSPACE_H
#include "openmx_core.h"
#include "openmx_complex.h"
#include "openmx_core_orbital_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OpenMXSolverInvocation OpenMXSolverInvocation;
typedef struct OpenMXSolverRealField OpenMXSolverRealField;
typedef struct OpenMXSolverResidues OpenMXSolverResidues;

/* All pointers below are borrowed. A successful callback return, an early
   callback return, or a C++ exception destroys the entire invocation. No MPI,
   device SDK, default context or Kokkos runtime participates in ownership. */
typedef int (*OpenMXSolverOperation)(OpenMXSolverInvocation*, void*, double*);
int openmx_core_solver_run(int workers, OpenMXSolverOperation operation,
    void *arguments, double *result, char *error, size_t capacity);

/* A worker owns named, reusable tensors. Dimensions are in elements and use
   layout_right. Requesting the same shape preserves values and addresses;
   replacement is transactional. Worker IDs must be distinct while concurrent.
   Row adapters are borrowed only by legacy BLAS/eigensolver interfaces. */
typedef struct { size_t extent[4]; double *values; } OpenMXSolverRealTensor;
typedef struct { size_t extent[4]; dcomplex *values; } OpenMXSolverComplexTensor;
typedef struct { size_t extent[4]; int *values; } OpenMXSolverIntTensor;
int openmx_core_solver_real(OpenMXSolverInvocation*, int worker, const char *name,
    const size_t shape[4], OpenMXSolverRealTensor*, char*, size_t);
int openmx_core_solver_complex(OpenMXSolverInvocation*, int worker, const char *name,
    const size_t shape[4], OpenMXSolverComplexTensor*, char*, size_t);
int openmx_core_solver_int(OpenMXSolverInvocation*, int worker, const char *name,
    size_t count, int **values, char*, size_t);
int openmx_core_solver_int_tensor(OpenMXSolverInvocation*, int worker, const char *name,
    const size_t shape[4], OpenMXSolverIntTensor*, char*, size_t);
int openmx_core_solver_bytes(OpenMXSolverInvocation*, int worker, const char *name,
    size_t count, size_t element_bytes, void **values, char*, size_t);
int openmx_core_solver_real_rows(OpenMXSolverInvocation*, int worker, const char *name,
    size_t rows, size_t columns, double ***values, char*, size_t);
int openmx_core_solver_complex_rows(OpenMXSolverInvocation*, int worker, const char *name,
    size_t rows, size_t columns, dcomplex ***values, char*, size_t);
/* Own only ABI row metadata for a borrowed [component,row,column,1] tensor.
   Numeric storage must remain valid until its last use, including rebinding. */
int openmx_core_solver_real_alias(OpenMXSolverInvocation*, int worker, const char *name,
    const OpenMXSolverRealTensor*, double ****values, char*, size_t);
int openmx_core_solver_complex_alias(OpenMXSolverInvocation*, int worker, const char *name,
    const OpenMXSolverComplexTensor*, dcomplex ****values, char*, size_t);

/* Named orbital samplers share the invocation lifetime. Equal shapes reuse
   values and addresses without allocating. A different shape replaces only
   that named worker's sampler; failure preserves its buffers and the output. */
int openmx_core_solver_orbitals(OpenMXSolverInvocation*, int worker, const char *name,
    const OpenMXOrbitalShape*, OpenMXOrbitalBuffers*, char*, size_t);

/* Ragged eigenvalue / projected-density rows. Each numerical row is one owned
   tensor; a fixed descriptor array replaces the former three-level table. */
typedef struct { size_t count; double *values; } OpenMXSolverRealBlock;
typedef struct {
  size_t components, atoms;
  OpenMXSolverRealBlock *blocks;
  OpenMXSolverRealField *owner;
} OpenMXSolverRealFieldView;
int openmx_core_solver_field_create(OpenMXSolverInvocation*, size_t components,
    size_t atoms, OpenMXSolverRealFieldView*, char*, size_t);
int openmx_core_solver_field_resize(OpenMXSolverRealField*, size_t component,
    size_t atom, size_t count, char*, size_t);

/* Residue topology is copied at construction. With padding=1, atom slot zero
   has two 1x1 dummy neighbor blocks; otherwise slot a maps to local_to_global[a+1].
   Residues for one component/atom occupy one contiguous [orbital-pair,state]
   tensor. Different neighbors may have different orbital counts. */
typedef struct {
  int global_atoms, local_atoms, species, components, padding, complex_values;
  const int *local_to_global, *atom_species, *orbitals, *first_neighbors;
  int *const *neighbor_atoms;
} OpenMXSolverResidueInput;
typedef struct { size_t states; double *values; dcomplex *complex_values; } OpenMXSolverResidueBlock;
typedef struct {
  size_t components, atoms;
  const size_t *atom_offsets, *pair_offsets;
  const int *shapes;
  OpenMXSolverResidueBlock *blocks;
  OpenMXSolverResidues *owner;
} OpenMXSolverResidueView;
int openmx_core_solver_residue_create(OpenMXSolverInvocation*,
    const OpenMXSolverResidueInput*, OpenMXSolverResidueView*, char*, size_t);
int openmx_core_solver_residue_resize(OpenMXSolverResidues*, size_t component,
    size_t atom, size_t states, char*, size_t);

#ifdef __cplusplus
}
#endif
static inline double *openmx_solver_real_matrix_row(const OpenMXSolverRealTensor *v,size_t row)
{ return v->extent[1] ? v->values+row*v->extent[1] : v->values; }
static inline dcomplex *openmx_solver_complex_matrix_row(const OpenMXSolverComplexTensor *v,size_t row)
{ return v->extent[1] ? v->values+row*v->extent[1] : v->values; }
/* Borrow a row of a validated [component,row,column,1] tensor. This replaces
   pointer-tree scratch in integral assembly without allocating ABI aliases. */
static inline dcomplex *openmx_solver_complex_plane_row(const OpenMXSolverComplexTensor *v,
    size_t component,size_t row)
{ return v->extent[2] ? v->values+(component*v->extent[1]+row)*v->extent[2] : v->values; }
static inline int *openmx_solver_int_matrix_row(const OpenMXSolverIntTensor *v,size_t row)
{ return v->extent[1] ? v->values+row*v->extent[1] : v->values; }
static inline double *openmx_solver_field_row(const OpenMXSolverRealFieldView *v,
    size_t component, size_t atom)
{ return v->blocks[component*v->atoms+atom].values; }
static inline size_t openmx_solver_pair(const OpenMXSolverResidueView *v,
    size_t atom, size_t neighbor, size_t row, size_t column)
{
  const size_t block=v->atom_offsets[atom]+neighbor;
  return v->pair_offsets[block]-v->pair_offsets[v->atom_offsets[atom]]+
      row*(size_t)v->shapes[2*block+1]+column;
}
static inline double *openmx_solver_residue_row(const OpenMXSolverResidueView *v,
    size_t component, size_t atom, size_t neighbor, size_t row, size_t column)
{
  const OpenMXSolverResidueBlock *b=&v->blocks[component*v->atoms+atom];
  return b->states ? b->values+openmx_solver_pair(v,atom,neighbor,row,column)*b->states : b->values;
}
static inline dcomplex *openmx_solver_complex_residue_row(const OpenMXSolverResidueView *v,
    size_t component, size_t atom, size_t neighbor, size_t row, size_t column)
{
  const OpenMXSolverResidueBlock *b=&v->blocks[component*v->atoms+atom];
  return b->states ? b->complex_values+openmx_solver_pair(v,atom,neighbor,row,column)*b->states : b->complex_values;
}
#endif
