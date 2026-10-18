/**********************************************************************
  RF_BesselF.c:

     RF_BesselF.c is a subroutine to calculate radial part of PAO of 
     atom specified by "Gensi" in k-space.

  Log of RF_BesselF.c:

     22/Nov/2001  Released by T.Ozaki

***********************************************************************/

#include <stdio.h>
#include "openmx_common.h"
#include "openmx_core_radial.h"

double RF_BesselF(int Gensi, int GL, int Mul, double R)
{
  double result;
  if (!openmx_core_reciprocal_sample(OPENMX_RECIPROCAL_PAO,Ngrid_NormK,
        PAO_Nkmax/(double)Ngrid_NormK,Spe_RF_Bessel[Gensi][GL][Mul],R,&result)) {
    fprintf(stderr,"RF_BesselF: invalid reciprocal interpolation input\n");
    MPI_Abort(mpi_comm_level1,1);
    return 0.0;
  }
  return result;
}
