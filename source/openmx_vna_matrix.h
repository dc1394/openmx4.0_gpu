#ifndef OPENMX_VNA_MATRIX_H
#define OPENMX_VNA_MATRIX_H
#include "openmx_core_vna_matrix.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

static inline OpenMXVnaMatrixViews OpenMX_VnaMatrices(OpenMXCoreContext *core, MPI_Comm communicator)
{
  OpenMXVnaMatrixViews views;
  char error[256];
  if (!openmx_core_vna_matrices_get(core,&views,error,sizeof(error))) {
    fprintf(stderr,"OpenMX VNA matrices: %s\n",error);
    MPI_Abort(communicator,1);
    abort();
  }
  return views;
}
#endif
