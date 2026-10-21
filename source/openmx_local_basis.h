#ifndef OPENMX_LOCAL_BASIS_H
#define OPENMX_LOCAL_BASIS_H
#include "openmx_core_local_basis.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

static inline OpenMXLocalBasisViews OpenMX_LocalBasis(OpenMXCoreContext *core, MPI_Comm communicator)
{
  OpenMXLocalBasisViews views;
  char error[256];
  if (!openmx_core_local_basis_get(core,&views,error,sizeof(error))) {
    fprintf(stderr,"OpenMX local basis: %s\n",error);
    MPI_Abort(communicator,1);
    abort();
  }
  return views;
}

/* Top-level MD/test drivers must refresh after topology replacement. */
static inline const OpenMXLocalBasisViews *OpenMX_LocalBasisAt(
    OpenMXCoreContext *core, MPI_Comm communicator, OpenMXLocalBasisViews *views)
{
  *views=OpenMX_LocalBasis(core,communicator);
  return views;
}
#endif
