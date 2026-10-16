#ifndef OPENMX_GPU_DM_H
#define OPENMX_GPU_DM_H
#ifdef __cplusplus
extern "C" {
#endif
enum { OPENMX_GPU_DM_REAL, OPENMX_GPU_DM_COL, OPENMX_GPU_DM_NC,
       OPENMX_GPU_DM_GATHER, OPENMX_GPU_EDM_GATHER };
/* Sparse density entries, each with its complete state summation on one
   GPU. All metadata, weights and output arrays are HOST arrays. V is a
   state-major-per-basis device panel (or a host panel when host_vectors).
   For gathers V is a column-major spinor density matrix. Output ordering:
   real: DM,EDM,optional PDM; collinear: DM,EDM;
   NC: r11,r22,r12,i12,i11,i22,rE11,rE22,rE12,iE12;
   EDM gather: rE11,rE22,rE12,iE12. Only nonnull outputs are published.
   The noncollective entry point recovers through a bounded CPU row path.
   Returns 0 without changing outputs for invalid input or host staging failure. */
int openmx_gpu_dm_entries(int kind,int n,int states,int stride,int entries,
    const void *vectors,int host_vectors,const int *basis0,const int *basis1,
    const int *phase_index,int phase_count,const double *phase_r,const double *phase_i,
    const double *occupation,const double *energy_weight,const double *partial_weight,
    double *const outputs[10],int accumulate);
/* GPU-only attempt for a collective recovery phase; failure never publishes. */
int openmx_gpu_dm_try_entries(int kind,int n,int states,int stride,int entries,
    const void *vectors,int host_vectors,const int *basis0,const int *basis1,
    const int *phase_index,int phase_count,const double *phase_r,const double *phase_i,
    const double *occupation,const double *energy_weight,const double *partial_weight,
    double *const outputs[10],int accumulate);
#ifdef __cplusplus
}
#endif
#endif
