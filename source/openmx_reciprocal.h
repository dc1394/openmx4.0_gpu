#ifndef OPENMX_RECIPROCAL_H
#define OPENMX_RECIPROCAL_H
#include "openmx_core_reciprocal.h"
#include "openmx_core_reciprocal_archive.h"
#include <stdio.h>
#include <stdlib.h>

/* Application boundary: context and lifetime are always explicit. */
static inline OpenMXReciprocalViews OpenMX_ReciprocalTables(OpenMXCoreContext *core)
{
  OpenMXReciprocalViews views;
  char error[256];
  if (!openmx_core_reciprocal_get(core,&views,error,sizeof(error))) {
    fprintf(stderr,"OpenMX reciprocal tables: %s\n",error);
    MPI_Abort(mpi_comm_level1,1);
    abort();
  }
  return views;
}

static inline OpenMXReciprocalArchive *OpenMX_ReciprocalArchiveBegin(
    OpenMXCoreContext *core,int kind,OpenMXReciprocalRows *rows)
{
  const OpenMXReciprocalArchiveInput input={kind,SpeciesNum,List_YOUSO[15],GL_Mesh,
      Spe_MaxL_Basis,Spe_Num_Basis,Spe_Num_RVPS,VPS_j_dependency};
  char error[256];
  OpenMXReciprocalArchive *archive=openmx_core_reciprocal_archive_create(core,&input,rows,error,sizeof(error));
  if (!archive) {
    fprintf(stderr,"OpenMX reciprocal archive: %s\n",error);
    MPI_Abort(mpi_comm_level1,1);
    abort();
  }
  return archive;
}
#endif
