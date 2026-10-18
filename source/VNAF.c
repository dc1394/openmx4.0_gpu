/**********************************************************************
  VNAF.c:

     VNAF.c is a subroutine to calculate the neutral atom potential
     of one atom specified by "Gensi".

  Log of VNAF.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include "openmx_common.h"
#include "openmx_core_radial.h"

double VNAF(int Gensi, double R)
{
  double result;
  if (!openmx_core_radial_sample(OPENMX_RADIAL_NEUTRAL_POTENTIAL,0,Spe_Num_Mesh_VPS[Gensi],
        Spe_VPS_RV[Gensi],Spe_Vna[Gensi],Spe_Atom_Cut1[Gensi],R,&result)) {
    fprintf(stderr,"VNAF: invalid radial interpolation input\n");
    MPI_Abort(mpi_comm_level1,1);
    return 0.0;
  }
  return result;
}
