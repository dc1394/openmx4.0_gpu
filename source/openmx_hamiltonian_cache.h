#ifndef OPENMX_HAMILTONIAN_CACHE_H
#define OPENMX_HAMILTONIAN_CACHE_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Calculation-owned host matrices shared by the cluster and band solvers.
   Binding initializes no device. Readers borrow the packed arrays until the
   next build/invalidation. Bind, clear and finish require quiescent readers;
   teardown is local and does not access MPI or the source geometry. */
typedef struct OpenMXHamiltonianCache OpenMXHamiltonianCache;
OpenMXHamiltonianCache *Set_Hamiltonian_Cache_Bind_Resources(OpenMXCoreContext *core);
void Set_Hamiltonian_Invalidate_GpuSolver_HS_Cache(OpenMXHamiltonianCache *cache);
int Set_Hamiltonian_GpuSolver_Packed_CacheReady(const OpenMXHamiltonianCache *cache);
int Set_Hamiltonian_GpuSolver_Packed_OwnsCache(const OpenMXHamiltonianCache *cache);
int Set_Hamiltonian_GpuSolver_Packed_OrderMode(const OpenMXHamiltonianCache *cache);
int Set_Hamiltonian_GpuSolver_Packed_Size(const OpenMXHamiltonianCache *cache);
int *Set_Hamiltonian_GpuSolver_Packed_OrderGA(const OpenMXHamiltonianCache *cache);
double *Set_Hamiltonian_GpuSolver_Packed_Overlap(const OpenMXHamiltonianCache *cache);
double *Set_Hamiltonian_GpuSolver_Packed_H(const OpenMXHamiltonianCache *cache, int spin);
double *Set_Hamiltonian_GpuSolver_Packed_ImNL(const OpenMXHamiltonianCache *cache, int comp);
#ifdef __cplusplus
}
#endif
#endif
