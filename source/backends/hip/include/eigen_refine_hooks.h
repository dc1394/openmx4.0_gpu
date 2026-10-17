/* Stage hooks of the refined FP32 eigensolver (eigen_refine_gpu.h) that
   DFT.c calls around the SCF stop test and the mixing; every rank calls
   them in the same order and they return the same value on every rank. */
#ifndef OPENMX_EIGEN_REFINE_HOOKS_H
#define OPENMX_EIGEN_REFINE_HOOKS_H

/* whether the latest eigensolve of the solver was a refined FP32 one: an
   SCF that ends on such a solve has its Hamiltonian solved once more in
   FP64 so that the post-SCF energy density matrix and forces come from FP64
   vectors */
int  Cluster_DFT_NonCol_RefineLastRefined(void);
/* after the mixing of an SCF step that goes on, with the printed NormRD
   (OPENMX_EIGEN_REFINE_UNTIL) */
void Cluster_DFT_NonCol_RefineAfterMixing(double normrd);

int  Cluster_DFT_Col_RefineLastRefined(void);
/* the repeated "scf" call of the FP64 re-solve runs with this on */
void Cluster_DFT_Col_RefineForceFP64(int on);
void Cluster_DFT_Col_RefineAfterMixing(double normrd);

int  Band_DFT_Col_RefineLastRefined(void);
void Band_DFT_Col_RefineForceFP64(int on);
void Band_DFT_Col_RefineAfterMixing(double normrd);
/* the end of an SCF cycle: the warm-start bases go, so that the forces and
   the next cycle start with the memory free */
void Band_DFT_Col_RefineEndCycle(void);

int  Band_DFT_NonCol_RefineLastRefined(void);
void Band_DFT_NonCol_RefineForceFP64(int on);
void Band_DFT_NonCol_RefineAfterMixing(double normrd);
void Band_DFT_NonCol_RefineEndCycle(void);

#endif
