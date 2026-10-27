#ifndef OPENMX_QUADRATURE_H
#define OPENMX_QUADRATURE_H
#include "openmx_core_quadrature.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* Fetch once at the application boundary; kernels borrow an explicit view. */
static inline OpenMXGaussRuleData OpenMX_RadialQuadrature(
    const OpenMXCoreContext *core, int count, MPI_Comm communicator)
{
  OpenMXGaussRuleData rule;
  char error[256];
  if (!openmx_core_radial_quadrature_get(core,&rule,error,sizeof(error))) {
    fprintf(stderr,"OpenMX radial quadrature: %s\n",error);
    MPI_Abort(communicator,1);
    abort();
  }
  if (rule.count!=count) {
    fprintf(stderr,"OpenMX radial quadrature: expected %d nodes, found %d\n",count,rule.count);
    MPI_Abort(communicator,1);
    abort();
  }
  return rule;
}
#endif
