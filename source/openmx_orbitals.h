#ifndef OPENMX_ORBITALS_H
#define OPENMX_ORBITALS_H

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "openmx_core_orbital_workspace.h"

/* Internal C adapter; include after openmx_common.h and mpi.h. Dimensions
   are passed explicitly. The numerical owner has no process-global state. */
static inline OpenMXOrbitalWorkspace *OpenMX_OrbitalWorkspace(int maximum_angular,
    int multiplicity, int derivatives, OpenMXOrbitalBuffers *buffers)
{
  const OpenMXOrbitalShape shape={maximum_angular,multiplicity,derivatives};
  char error[256];
  OpenMXOrbitalWorkspace *owner=openmx_core_orbital_create(&shape,buffers,error,sizeof(error));
  if (!owner) {
    fprintf(stderr,"Orbital workspace: %s\n",error);
    MPI_Abort(mpi_comm_level1,1);
    abort();
  }
  return owner;
}
#endif
