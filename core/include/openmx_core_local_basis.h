#ifndef OPENMX_CORE_LOCAL_BASIS_H
#define OPENMX_CORE_LOCAL_BASIS_H
#include "openmx_core.h"
#include "openmx_complex.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { OPENMX_LOCAL_DENSITY, OPENMX_LOCAL_POTENTIAL,
  OPENMX_LOCAL_INVERSE_OVERLAP, OPENMX_LOCAL_REAL_COUNT } OpenMXLocalRealField;
typedef enum { OPENMX_LOCAL_OCCUPATION, OPENMX_LOCAL_COMPLEX_POTENTIAL,
  OPENMX_LOCAL_COMPLEX_COUNT } OpenMXLocalComplexField;
typedef enum { OPENMX_LOCAL_LNO_COEFFICIENTS, OPENMX_LOCAL_LNO_POPULATIONS,
  OPENMX_LOCAL_CONTRACTION, OPENMX_LOCAL_SPECIES_CONTRACTION,
  OPENMX_LOCAL_TENSOR_COUNT } OpenMXLocalTensorField;

/* Components share an atom-block layout, with offsets in elements. Each block
   is layout_right; values[component*stride+offsets[atom]+row*columns+column]. */
typedef struct {
  size_t components, atoms, stride, elements;
  const size_t *offsets;
  const int *shapes;
} OpenMXLocalMatrixLayout;
typedef struct { OpenMXLocalMatrixLayout layout; double *values; } OpenMXLocalRealMatrix;
typedef struct { OpenMXLocalMatrixLayout layout; dcomplex *values; } OpenMXLocalComplexMatrix;
typedef struct { size_t n0,n1,n2; double *values; } OpenMXLocalTensor3;

typedef struct {
  int atoms, local_atoms, first_halo_atoms, second_halo_atoms, species;
  int spin, max_orbitals, max_coefficients;
  int interactions, lno, contraction, inverse_overlap;
  const int *atom_species, *orbitals, *contracted_orbitals;
  const int *local_to_global, *first_to_global;
  const int *first_neighbors, *second_neighbors;
  int *const *neighbor_atoms;
} OpenMXLocalBasisInput;

typedef struct {
  OpenMXLocalRealMatrix real[OPENMX_LOCAL_REAL_COUNT];
  OpenMXLocalComplexMatrix complex_values[OPENMX_LOCAL_COMPLEX_COUNT];
  OpenMXLocalTensor3 tensors[OPENMX_LOCAL_TENSOR_COUNT];
  /* All views expire at successful replacement, release or context finish. */
} OpenMXLocalBasisViews;

/* Invocation-owned contiguous scratch. No pointer tables or runtime setup. */
typedef struct OpenMXLocalTensorWorkspace OpenMXLocalTensorWorkspace;
OpenMXLocalTensorWorkspace *openmx_core_local_tensor_create(const size_t extent[3],
    OpenMXLocalTensor3 *view, char *error, size_t capacity);
void openmx_core_local_tensor_destroy(OpenMXLocalTensorWorkspace*);

/* All input metadata is copied or converted to owned layouts. Failed
   replacement preserves both the old owner and the caller's view. No MPI,
   vendor SDK or implicit/default context participates in the ownership API. */
int openmx_core_local_basis_allocate(OpenMXCoreContext*, const OpenMXLocalBasisInput*,
    OpenMXLocalBasisViews*, char *error, size_t capacity);
int openmx_core_local_basis_get(OpenMXCoreContext*, OpenMXLocalBasisViews*, char *error, size_t capacity);
void openmx_core_local_basis_release(OpenMXCoreContext*);

#ifdef __cplusplus
}
#endif

static inline double *openmx_local_tensor_row(const OpenMXLocalTensor3 *v, size_t i, size_t j)
{ return v->n2 ? v->values+(i*v->n1+j)*v->n2 : v->values; }
static inline double *openmx_local_real_row(const OpenMXLocalRealMatrix *v,
    size_t component, size_t atom, size_t row)
{ return v->values+component*v->layout.stride+v->layout.offsets[atom]+row*(size_t)v->layout.shapes[2*atom+1]; }
static inline dcomplex *openmx_local_complex_row(const OpenMXLocalComplexMatrix *v,
    size_t component, size_t atom, size_t row)
{ return v->values+component*v->layout.stride+v->layout.offsets[atom]+row*(size_t)v->layout.shapes[2*atom+1]; }

/* Named row access keeps history/spin flattening out of numerical consumers.
   These inline helpers require a valid enabled family and checked indices. */
static inline double *openmx_local_density_row(const OpenMXLocalBasisViews *v,
    size_t history,size_t spin,size_t atom,size_t row)
{
  const OpenMXLocalRealMatrix *m=&v->real[OPENMX_LOCAL_DENSITY];
  return openmx_local_real_row(m,history*(m->layout.components/2)+spin,atom,row);
}
static inline double *openmx_local_potential_row(const OpenMXLocalBasisViews *v,
    size_t spin,size_t atom,size_t row)
{ return openmx_local_real_row(&v->real[OPENMX_LOCAL_POTENTIAL],spin,atom,row); }
static inline dcomplex *openmx_local_occupation_row(const OpenMXLocalBasisViews *v,
    size_t history,size_t spin1,size_t spin2,size_t atom,size_t row)
{ return openmx_local_complex_row(&v->complex_values[OPENMX_LOCAL_OCCUPATION],history*4+spin1*2+spin2,atom,row); }
static inline dcomplex *openmx_local_complex_potential_row(const OpenMXLocalBasisViews *v,
    size_t spin1,size_t spin2,size_t atom,size_t row)
{ return openmx_local_complex_row(&v->complex_values[OPENMX_LOCAL_COMPLEX_POTENTIAL],spin1*2+spin2,atom,row); }
static inline double *openmx_local_inverse_overlap_row(const OpenMXLocalBasisViews *v,size_t atom,size_t row)
{ return openmx_local_real_row(&v->real[OPENMX_LOCAL_INVERSE_OVERLAP],0,atom,row); }
static inline double *openmx_local_lno_row(const OpenMXLocalBasisViews *v,size_t spin,size_t atom)
{ return openmx_local_tensor_row(&v->tensors[OPENMX_LOCAL_LNO_COEFFICIENTS],spin,atom); }
static inline double *openmx_local_lno_population_row(const OpenMXLocalBasisViews *v,size_t spin,size_t atom)
{ return openmx_local_tensor_row(&v->tensors[OPENMX_LOCAL_LNO_POPULATIONS],spin,atom); }
static inline double *openmx_local_contraction_row(const OpenMXLocalBasisViews *v,size_t atom,size_t orbital)
{ return openmx_local_tensor_row(&v->tensors[OPENMX_LOCAL_CONTRACTION],atom,orbital); }
static inline double *openmx_local_species_contraction_row(const OpenMXLocalBasisViews *v,size_t species,size_t orbital)
{ return openmx_local_tensor_row(&v->tensors[OPENMX_LOCAL_SPECIES_CONTRACTION],species,orbital); }
#endif
