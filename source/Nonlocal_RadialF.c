#include <stdio.h>
#include "openmx_common.h"
#include "openmx_core_radial.h"

double Nonlocal_RadialF(const OpenMXAtomicViews *atomic, int spe, int l, int so, double R)
{
  double result;
  if (!openmx_core_radial_sample(OPENMX_RADIAL_PROJECTOR,Spe_VPS_List[spe][l+1],
        atomic->species.vps_mesh[spe],openmx_radial_row2(&atomic->vps_radii,spe),openmx_radial_row4(&atomic->projectors,so,spe,l),0.0,R,&result)) {
    fprintf(stderr,"Nonlocal_RadialF: invalid radial interpolation input\n");
    MPI_Abort(mpi_comm_level1,1);
    return 0.0;
  }
  return result;
}
