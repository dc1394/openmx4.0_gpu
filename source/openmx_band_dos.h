#ifndef OPENMX_BAND_DOS_H
#define OPENMX_BAND_DOS_H

#include "openmx_band_kpath.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int up, down;
    double axial, transverse_real, transverse_imag;
} OpenMXDosSpinProjection;

typedef struct {
    int first, count;    /* zero-based band range to project */
    int unregularized_overlap;
    const OpenMXDosSpinProjection *spin;
    int orbital_first, orbital_count; /* DOS slice, relative to first */
    int atoms;
    const int *atom_offsets;         /* AO offsets including the end */
} OpenMXBandDosProjection;

/* Eigenvalues cover all bands. The optional outputs are contiguous in
   point, spin, band, then orbital or atom order. Orbital output covers just
   the DOS slice. Atomic output covers the full projected range. */
int openmx_band_dos_cpu(const OpenMXKPathMatrices *m, int count, const OpenMXKPathPoint *points,
    const OpenMXBandDosProjection *projection, double *eigen, float *weights, double *atomic,
    int simd_mode, size_t workspace_limit, int *threads_used);
int openmx_band_dos_gpu(int device, const OpenMXKPathMatrices *m, int count, const OpenMXKPathPoint *points,
    const OpenMXBandDosProjection *projection, double *eigen, float *weights, double *atomic);

#if defined(__CUDACC__) || defined(__HIPCC__)
#define OPENMX_DOS_HOST_DEVICE __host__ __device__
#else
#define OPENMX_DOS_HOST_DEVICE
#endif

static inline OPENMX_DOS_HOST_DEVICE double openmx_dos_weight(double cr, double ci, double sr, double si)
{
    return cr*sr + ci*si;
}

/* Re(conj(c) (S c)) and the mixed spin component in the atom's local frame. */
static inline OPENMX_DOS_HOST_DEVICE void openmx_dos_spin_weights(const OpenMXDosSpinProjection *spin,
    double ur, double ui, double dr, double di, double sur, double sui, double sdr, double sdi, float *out)
{
    const double up = openmx_dos_weight(ur,ui,sur,sui);
    const double down = openmx_dos_weight(dr,di,sdr,sdi);
    const double total = 0.5*(up + down);
    const double polarization = spin->axial*(up - down) + spin->transverse_real*openmx_dos_weight(ur,ui,sdr,sdi)
        + spin->transverse_imag*(ur*sdi - ui*sdr);
    out[spin->up] = (float)(total + polarization);
    out[spin->down] = (float)(total - polarization);
}

#undef OPENMX_DOS_HOST_DEVICE

#ifdef __cplusplus
}
#endif
#endif
