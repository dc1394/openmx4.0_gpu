#ifndef OPENMX_ATOMIC_H
#define OPENMX_ATOMIC_H

#include <stdio.h>
#include "openmx_common.h"

static inline OpenMXAtomicViews OpenMX_AtomicTables(OpenMXCoreContext *core)
{
  OpenMXAtomicViews views;
  char error[256];
  if (!openmx_core_atomic_get(core,&views,error,sizeof(error))) {
    fprintf(stderr,"OpenMX atomic tables: %s\n",error);
    MPI_Abort(mpi_comm_level1,1);
  }
  return views;
}
#endif
