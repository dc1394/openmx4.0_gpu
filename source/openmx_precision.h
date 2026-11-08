/* Arithmetic precision requested by the current OpenMX input.
   scf.fp32.enable defaults to on.  Off overrides FP32/DF kernel and
   eigensolver choices. GEMMul8, storage formats and the upstream float
   products in VNA/force/stress contractions remain independent. */
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
