#ifndef OPENMX_BASIS_RADIAL_H
#define OPENMX_BASIS_RADIAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum OpenMXBasisRadialKind {
  OPENMX_BASIS_RADIAL_PAO,
  OPENMX_BASIS_RADIAL_PROJECTOR,
  OPENMX_BASIS_RADIAL_NEUTRAL_POTENTIAL,
  OPENMX_BASIS_RADIAL_NEUTRAL_PROJECTOR
};
enum OpenMXBasisReciprocalKind {
  OPENMX_BASIS_RECIPROCAL_PAO,
  OPENMX_BASIS_RECIPROCAL_PROJECTOR
};

/* Borrowed scalar kernels: no allocation, copying, or retained pointers on
   success. The caller provides finite values and a finite, strictly increasing
   nonnegative mesh. Shape/coordinate checks are constant-time; there is no
   hidden per-call scan of the input row. Reciprocal coordinates are k[i]=i*step.
   Units must be consistent (OpenMX uses bohr and inverse bohr).
   Return 1 on success, 0 on error. Failure never changes *result.
   If error != NULL and capacity > 0, write a terminated (possibly truncated)
   diagnostic on failure and an empty string on success. No global error state.
   Output/error storage must not overlap any input or each other. */
int openmx_basis_radial_sample(enum OpenMXBasisRadialKind kind, int angular,
    size_t count, const double *radii, const double *values,
    double cutoff, double radius, double *result, char *error, size_t capacity);
int openmx_basis_reciprocal_sample(enum OpenMXBasisReciprocalKind kind,
    size_t count, double step, const double *values, double wavevector,
    double *result, char *error, size_t capacity);

/* A fully validated, independently owned immutable table. Numeric input is
   copied once, in channel-major order: values[channel*row_stride + point].
   row_stride >= count; the final row needs count entries, no trailing padding.
   angular has channel_count entries. Empty channel sets are allowed (angular
   and values may then be NULL); the radial mesh is still required.
   The owner does not require any OpenMX, MPI, or device runtime initialization.
   Independent owners and concurrent read-only sampling are supported. */
typedef struct OpenMXBasisRadialTable OpenMXBasisRadialTable;
OpenMXBasisRadialTable *openmx_basis_radial_table_create(
    enum OpenMXBasisRadialKind kind, size_t count, const double *radii,
    size_t channel_count, const int *angular, size_t row_stride,
    const double *values, double cutoff, char *error, size_t capacity);
int openmx_basis_radial_table_sample(const OpenMXBasisRadialTable *table,
    size_t channel, double radius, double *result, char *error, size_t capacity);
/* NULL is accepted. Do not destroy a table while another thread samples it. */
void openmx_basis_radial_table_destroy(OpenMXBasisRadialTable *table);

#ifdef __cplusplus
}
#endif
#endif
