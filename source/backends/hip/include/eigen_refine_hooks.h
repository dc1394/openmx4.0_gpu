#include "openmx_band_col_state.h"
#include "openmx_band_noncol_state.h"
#include "openmx_cluster_col_state.h"
#include "openmx_cluster_noncol_state.h"
/* Stage hooks of the refined FP32 eigensolver (eigen_refine_gpu.h) that
   DFT.c calls around the SCF stop test and the mixing; every rank calls
   them in the same order and they return the same value on every rank. */
#ifndef OPENMX_EIGEN_REFINE_HOOKS_H
#define OPENMX_EIGEN_REFINE_HOOKS_H

/* whether the latest eigensolve of the solver was a refined FP32 one: an
   SCF that ends on such a solve has its Hamiltonian solved once more in
   FP64 so that the post-SCF energy density matrix and forces come from FP64
   vectors */
/* after the mixing of an SCF step that goes on, with the printed NormRD
   (OPENMX_EIGEN_REFINE_UNTIL) */

/* Collinear cluster hooks borrow the calculation state declared above. */

/* the end of an SCF cycle: the warm-start bases go, so that the forces and
   the next cycle start with the memory free */


#endif
