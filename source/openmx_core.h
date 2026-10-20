#ifndef OPENMX_CORE_H
#define OPENMX_CORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The application owns this context on its stack through openmx_core_run.
   It is passed explicitly, never installed in a process-global variable.
   A null context selects the host implementation for stateless radial calls;
   calculation storage always requires an explicit context. */
typedef struct OpenMXCoreContext OpenMXCoreContext;
typedef int (*OpenMXCoreEntry)(int argc, char **argv, OpenMXCoreContext *context);
int openmx_core_run(int argc, char **argv, OpenMXCoreEntry entry);
/* Explicit ownership for library callers that cannot use the entry callback.
   Creation does not initialize MPI or a GPU. Destroy before MPI_Finalize. */
OpenMXCoreContext *openmx_core_create(char *error, size_t error_capacity);
void openmx_core_destroy(OpenMXCoreContext *context);
/* Release calculation storage and the owned Kokkos runtime before
   MPI_Finalize. Invalidates all borrowed views. Idempotent. */
void openmx_core_finish(OpenMXCoreContext *context);
int openmx_core_has_accelerator(void);
const char *openmx_core_version(void);
int openmx_core_fft_grid_supported(int size);
/* sb has lmax+2 entries; derivative has lmax+1 entries. */
int openmx_core_spherical_bessel(double x, int lmax, double *sb, double *derivative);

typedef double (*OpenMXRadialSample)(void *user, int angular, int multiplicity,
                                    double radius);
typedef void (*OpenMXRadialStore)(void *user, int angular, int multiplicity,
                                int k, double value);

/* All dimensions and physical parameters are explicit. Callbacks are invoked
   on the host, outside parallel regions. The output callback is invoked only
   after the entire transform has succeeded. device < 0 requests the host path.
   Error text belongs to the caller; exceptions never cross this C interface. */
int openmx_core_pao_transform(OpenMXCoreContext *context, int device,
    int lmax, const int *multiplicities, int radial_intervals, int k_count,
    double rmin, double radial_step, double k_step, void *user,
    OpenMXRadialSample sample, OpenMXRadialStore store,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
