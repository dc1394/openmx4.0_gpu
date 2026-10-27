#ifndef OPENMX_HAMILTONIAN_PACK_H
#define OPENMX_HAMILTONIAN_PACK_H
#include <mpi.h>
#include "openmx_hamiltonian_cache.h"
#include "openmx_core_electronic.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Borrowed for this call only. The caller refreshes static data at every new
   geometry and whenever contracted orbitals change. reuse_static preserves
   the HIP optimization; CUDA currently rebuilds on every SCF step. */
typedef struct {
    int atom_count, local_atom_count, spin_mode;
    const int *local_to_global, *species, *orbitals, *neighbor_count;
    int *const *neighbors;
    int selected, use_contracted, reuse_static, refresh_static;
} OpenMXHamiltonianPacking;

/* Collective on comm. Returns whether atom order/overlap were rebuilt. Only
   selected ranks own arrays, in communicator-rank then local-atom order.
   H and imaginary nonlocal components are refreshed on every call. */
int Set_Hamiltonian_Build_GpuSolver_HS_Cache(OpenMXHamiltonianCache *cache,
    MPI_Comm comm, const OpenMXHamiltonianPacking *input,
    const OpenMXElectronicViews *electronic);
#ifdef __cplusplus
}
#endif
#endif
