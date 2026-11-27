#ifndef OPENMX_MO_DEVICE_H
#define OPENMX_MO_DEVICE_H
#include <stddef.h>

/* The atom groups in a grid row are kept in their original order. */
typedef struct { int atom, sample, cell; } OpenMXMOTerm;
typedef struct { int stride, count, coefficient; size_t first; } OpenMXMODeviceAtom;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OpenMXMODevice OpenMXMODevice;
OpenMXMODevice *openmx_mo_device_create(int device, int atoms, const OpenMXMODeviceAtom *atom,
    int points, const int *offsets, const OpenMXMOTerm *terms, size_t samples, const float *ao,
    int coefficients, int cells, int channels, int max_batch);
void openmx_mo_device_destroy(OpenMXMODevice *state);
/* Component-major output over the active grid points of this rank.
   A null phase reuses the table from the first batch of this k-point. */
int openmx_mo_device_evaluate(OpenMXMODevice *state, int batch, const double *phase,
    const double *coefficients, double *output);

#ifdef __cplusplus
}
#endif
#endif
