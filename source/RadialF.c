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

double RadialF(const OpenMXAtomicViews *atomic, int spe, int L, int Mul, double R)
{
  double result;
  if (!openmx_core_radial_sample(OPENMX_RADIAL_PAO,L,atomic->species.pao_mesh[spe],
        openmx_radial_row2(&atomic->pao_radii,spe),openmx_radial_row4(&atomic->orbitals,spe,L,Mul),0.0,R,&result)) {
    fprintf(stderr,"RadialF: invalid radial interpolation input\n");
    MPI_Abort(mpi_comm_level1,1);
    return 0.0;
  }
  return result;
}
