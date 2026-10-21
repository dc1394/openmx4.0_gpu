#ifndef OPENMX_CORE_CORE_HOLE_H
#define OPENMX_CORE_CORE_HOLE_H
#include "openmx_core_solver_workspace.h"
#ifdef __cplusplus
extern "C" {
#endif

enum {
  OPENMX_CH_D12_P12,OPENMX_CH_D12_M12,
  OPENMX_CH_D32_P32,OPENMX_CH_D32_P12,OPENMX_CH_D32_M12,OPENMX_CH_D32_M32,
  OPENMX_CH_D52_P52,OPENMX_CH_D52_P32,OPENMX_CH_D52_P12,
  OPENMX_CH_D52_M12,OPENMX_CH_D52_M32,OPENMX_CH_D52_M52,
  OPENMX_CH_D72_P72,OPENMX_CH_D72_P52,OPENMX_CH_D72_P32,OPENMX_CH_D72_P12,
  OPENMX_CH_D72_M12,OPENMX_CH_D72_M32,OPENMX_CH_D72_M52,OPENMX_CH_D72_M72,
  OPENMX_CH_WEIGHTS
};
typedef struct {
  int atom,spin,spin_orbit,angular,state,maximum_angular,orbitals;
  const int *multiplicity; /* [maximum_angular+1], consumed during preparation */
  double penalty;
} OpenMXCoreHoleInput;
typedef struct {
  int atom,spin,spin_orbit,angular,state,active;
  size_t orbitals,offset;
  double penalty;
  const double *collinear[2],*weights,*opposite_weights; /* immutable, invocation-owned */
} OpenMXCoreHoleView;

/* Prepare once per core-hole stage. Input metadata may expire immediately;
   views expire with the enclosing solver invocation. Failed preparation
   preserves the output and every previously prepared view. No runtime setup. */
int openmx_core_core_hole_prepare(OpenMXSolverInvocation*,const OpenMXCoreHoleInput*,
    OpenMXCoreHoleView*,char *error,size_t capacity);

/* Flat overlap rows; derivatives have three Cartesian rows. Sum is borrowed
   layout_right [3 spin components,3 Cartesian directions] of dcomplex. */
typedef struct {
  const double *derivative[3],*value;
  size_t columns;
  double sign;
  int reverse_penalty;
} OpenMXCoreHoleTrace;
int openmx_core_core_hole_trace(const OpenMXCoreHoleView*,const OpenMXCoreHoleTrace*,
    dcomplex sum[9],char *error,size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
