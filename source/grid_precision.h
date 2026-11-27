/**********************************************************************
  grid_precision.h

  The precision stage of the grid integrals of an SCF step: the
  dVH+Vxc+VNA matrix elements of Set_Hamiltonian and the density of
  Set_Density_Grid are summed in FP32 (compensated sums of FP32 products)
  while the SCF residual is large, then in double-float or FP64 according
  to the device or explicit setting. The SCF reference phase always
  uses FP64 for these quadratures.

  scf.fp32.enable off forces native FP64, including when an environment
  setting requests the FP32 stage or double-float arithmetic.

  OPENMX_GRID_FP32          1 on, 0 off (default); experimental lossy stage
  OPENMX_GRID_FP32_UNTIL    FP32 while the previous step's printed NormRD
                            is above this (default 1e-5)
  OPENMX_GRID_FP32_RESTART  1 (default): the mixing history restarts at
                            the switch to FP64
***********************************************************************/
#ifndef GRID_PRECISION_H
#define GRID_PRECISION_H

/* this SCF step's grid integrals are summed in FP32 (the lossy stage) */
int Grid_Precision_Fp32(void);
/* the kernels' arithmetic of this step: 0 = FP64, 1 = FP32 with compensated
   sums (the lossy stage), 2 = double-float (FP32 arithmetic with error-free
   transformations: FP64-grade results at the FP32 rate; OPENMX_GRID_PRECISION
   = fp64 | df, default df when the device's FP32:FP64 throughput ratio is
   8 or more) */
int Grid_Precision_Kernel(void);
/* Record arithmetic used by a dispatched grid kernel. */
void Grid_Precision_NoteKernel(int mode);
/* every rank, at the top of an SCF step: normrd_prev is the previous step's
   printed NormRD (ignored at the first step).  The last step allowed by
   SCF_max always uses the non-lossy DF/FP64 kernels. */
void Grid_Precision_BeginStep(int SCF_iter, int SCF_max, double normrd_prev, int myid, int verbose);
/* 1 once, after the switch to FP64, when the mixing history is to restart */
int Grid_Precision_TakeHistoryReset(void);
/* the end of an SCF cycle: the next cycle starts in FP32 again */
void Grid_Precision_EndCycle(void);

#endif
