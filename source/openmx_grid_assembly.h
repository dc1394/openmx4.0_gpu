#ifndef OPENMX_GRID_ASSEMBLY_H
#define OPENMX_GRID_ASSEMBLY_H
#include "openmx_common.h"
#include "openmx_core_grid.h"

/* Borrowed calculation workspace; ownership stays in the explicit core. */
OpenMXGridWorkspace *OpenMX_Grid_Workspace(OpenMXCoreContext *core);
OpenMXGridDensity OpenMX_Grid_Density_Storage(OpenMXGridWorkspace *grid);
void OpenMX_Grid_Density_CPU(OpenMXGridWorkspace *grid,const OpenMXAtomicViews *atomic,
    int Cnt_kind,double *****density_matrix);
void OpenMX_Grid_Hamiltonian_CPU(OpenMXGridWorkspace *grid,int Cnt_kind,
    const OpenMXElectronicViews *electronic);
OpenMXGridExchange OpenMX_Grid_Exchange(OpenMXGridWorkspace *grid,int kind,
    size_t send_values,size_t receive_values,size_t sends,size_t receives);
#endif
