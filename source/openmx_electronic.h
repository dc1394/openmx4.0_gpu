#ifndef OPENMX_ELECTRONIC_H
#define OPENMX_ELECTRONIC_H
#include "openmx_core_electronic.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

/* Borrow once at a calculation boundary, then pass explicitly to consumers. */
static inline OpenMXElectronicViews OpenMX_ElectronicMatrices(OpenMXCoreContext *core,
                                                             MPI_Comm communicator)
{
  OpenMXElectronicViews views;
  char error[256];
  if (!openmx_core_electronic_get(core,&views,error,sizeof(error))) {
    fprintf(stderr,"OpenMX electronic matrices: %s\n",error);
    MPI_Abort(communicator,1);
    abort();
  }
  return views;
}

/* Driver loops can rebuild the topology between calls. Refresh their borrowed
   snapshot at each boundary so no row adapter survives a replacement. */
static inline const OpenMXElectronicViews *OpenMX_ElectronicMatricesAt(
    OpenMXCoreContext *core, MPI_Comm communicator, OpenMXElectronicViews *views)
{
  *views=OpenMX_ElectronicMatrices(core,communicator);
  return views;
}
#endif
