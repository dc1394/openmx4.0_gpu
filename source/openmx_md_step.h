#ifndef OPENMX_MD_STEP_H
#define OPENMX_MD_STEP_H

#include "mpi.h"
#include "openmx_core.h"
#include "../core/include/openmx_core_md.h"

typedef struct OpenMXMD OpenMXMD;

typedef struct {
  int atoms, local_atoms;
  const int *owners, *local_to_global;
  double **xyz;
  int **fixed;
} OpenMXMDAtoms;

typedef struct {
  double **fractional;
  double **charge_history;
  double ***geometry_history;
  int charge_frames, geometry_frames;
} OpenMXMDHistory;

typedef enum {
  OPENMX_MD_RECORD_VERLET,
  OPENMX_MD_RECORD_SCALING,
  OPENMX_MD_RECORD_NOSE,
  OPENMX_MD_RECORD_LANGEVIN,
  OPENMX_MD_RECORD_NPT
} OpenMXMDRecords;

OpenMXMD *OpenMX_MD_Begin(OpenMXCoreContext *, MPI_Comm, const OpenMXMDAtoms *,
                          int use_gpu, int verbosity);
void OpenMX_MD_Move(OpenMXMD *, OpenMXMDStep, double mass_scale, double scale,
                    const double *scales, const int *groups);
void OpenMX_MD_Center(OpenMXMD *, int column);
double OpenMX_MD_Kinetic(OpenMXMD *, double mass_scale, int exclude_fixed);
typedef double OpenMXMDLocalVector[4];
/* Two consecutive local vector tables, each with a dummy row at index zero. */
OpenMXMDLocalVector *OpenMX_MD_NPT_Workspace(OpenMXMD *);
/* Fractional coordinates and velocities are indexed by the local atom. */
double OpenMX_MD_NPT_Kick(OpenMXMD *, const OpenMXMDNPTStep *, double mass_scale,
                          double (*coordinates)[4], double (*velocities)[4]);

/* Wrap the owned positions, share the updated atom records and apply the
   same image translations to the replicated extrapolation histories. */
int OpenMX_MD_Finish(OpenMXMD *, OpenMXMDRecords, const OpenMXMDCell *,
                      const OpenMXMDHistory *);

#endif
