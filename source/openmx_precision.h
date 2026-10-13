/* Arithmetic precision requested by the current OpenMX input.
   scf.fp32.enable defaults to on.  Off overrides FP32/DF kernel and
   eigensolver choices, but leaves GEMMul8 and storage formats independent. */
#ifndef OPENMX_PRECISION_H
#define OPENMX_PRECISION_H

#ifdef __cplusplus
extern "C" {
#endif

int openmx_fp32_enabled(void);
void openmx_set_fp32_enabled(int enabled);

#ifdef __cplusplus
}
#endif
#endif
