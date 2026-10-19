#ifndef OPENMX_VNA_GPU_CONTEXT_H
#define OPENMX_VNA_GPU_CONTEXT_H

/* Invocation-owned VNA2/VNA3 batch state, borrowed until synchronous
   completion. Native begin/end routines retain their backend memory policy. */
typedef struct {
  int enabled;
  int chunk;            /* admitted pair batch, shared by sizing and execution */
  int l0max;            /* max Spe_MaxL_Basis                          */
  int lfi_max;          /* 2*l0max                                     */
  int cmb_max;
  int tno_max;
  int mn_max;           /* tno_max^2                                   */

  int *sp_lfi;          /* [spe] 2*Spe_MaxL_Basis                      */
  int *sp_ncmb;
  int *sp_tno;
  int *cmb_L0;          /* [spe*cmb_max+c]                             */
  int *cmb_num0;

  double *crude;        /* [spe][GL_Mesh]                              */
  double *prod;         /* [spe][c0][c1][LL][GL_Mesh] (c0<=c1 pairs)   */
  double *gnt2;         /* (-1)^|m| Gaunt(L0,M0,L1,M1,LL,-m), m=M1-M0  */
  double *c2r;
  int *c2r_off;
} SetProGpu3Context;

#endif
