#ifndef OPENMX_ORBITALS_H
#define OPENMX_ORBITALS_H

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "openmx_core_orbital_workspace.h"
#include "openmx_orbital_harmonics.h"
#include "openmx_basis_orbital_radial.h"

/* One point's stencil is shared by its active radial rows. The mathematical
   kernel is also the independent C++ basis API's orbital interpolation.
   Keep the caller's spherical-coordinate/origin/polar conventions outside
   this adapter. Scratch belongs to the existing evaluation/worker owner. */
static inline int OpenMX_OrbitalRadial(const OpenMXAtomicViews *atomic,int species,
    double radius,int derivatives,const OpenMXOrbitalBuffers *buffers)
{
  const OpenMXBasisOrbitalRadialPlan plan=openmx_basis_detail_orbital_radial_plan(
      openmx_radial_row2(&atomic->pao_radii,species),atomic->species.pao_mesh[species],radius);
  int l,radial;
  for(l=0;l<=atomic->species.basis_lmax[species];++l)
    for(radial=0;radial<atomic->species.basis[(size_t)species*atomic->species.angular_channels+l];++radial) {
      const OpenMXOrbitalRadialSample sample=openmx_basis_detail_orbital_radial_apply(&plan,
          plan.outside?NULL:openmx_radial_row4(&atomic->orbitals,species,l,radial),l,derivatives);
      const size_t offset=(size_t)l*buffers->multiplicity+radial;
      buffers->RF[offset]=sample.value;
      if(derivatives)buffers->dRF[offset]=sample.radial_derivative;
    }
  return plan.outside;
}

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
