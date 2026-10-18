/**********************************************************************
  RadialF.c:

     RadialF.c is a subroutine to calculate the radial function of
     pseudo atomic orbital specified by "l" for atomic species "Gensi"
     at R.

  Log of RadialF.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include "openmx_common.h"
#include "openmx_core_radial.h"

double RadialF(int spe, int L, int Mul, double R)
{
  double result;
  if (!openmx_core_radial_sample(OPENMX_RADIAL_PAO,L,Spe_Num_Mesh_PAO[spe],
        Spe_PAO_RV[spe],Spe_PAO_RWF[spe][L][Mul],0.0,R,&result)) {
    fprintf(stderr,"RadialF: invalid radial interpolation input\n");
    MPI_Abort(mpi_comm_level1,1);
    return 0.0;
  }
  return result;
}
