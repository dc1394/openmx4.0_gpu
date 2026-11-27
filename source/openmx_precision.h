/* Arithmetic precision requested by the current OpenMX input.
   scf.fp32.enable defaults to on. Off disables optional FP32/DF kernels
   and FP32 eigenvector refinement, and uses FP64 for VNA stress products.
   On permits optional FP32/DF in early SCF steps. The reference phase
   disables those kernels while retaining the input's VNA stress policy.
   GEMMul8, storage formats and the upstream float products in VNA
   matrix assembly and forces remain independent. */
#ifndef OPENMX_PRECISION_H
#define OPENMX_PRECISION_H

#ifdef __cplusplus
extern "C" {
#endif

/* The input setting and the effective policy for optional fast kernels. */
int openmx_fp32_requested(void);
int openmx_fp32_enabled(void);
void openmx_set_fp32_enabled(int enabled);

enum {
    OPENMX_FP32_USED = 1,
    OPENMX_FP32_ORBITAL_TILES = 2,
    OPENMX_FP32_CACHED_ORBITALS = 4,
    OPENMX_FP32_LOSSY_GRID = 8
};

void openmx_fp32_begin_scf(void);
int openmx_fp32_begin_step(void);
void openmx_fp32_enter_reference(void);
void openmx_fp32_note_use(int usage);
int openmx_fp32_usage(void);

#ifdef __cplusplus
}
#endif
#endif
