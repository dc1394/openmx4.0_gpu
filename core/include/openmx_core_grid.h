#ifndef OPENMX_CORE_GRID_H
#define OPENMX_CORE_GRID_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct OpenMXGridWorkspace OpenMXGridWorkspace;
enum { OPENMX_GRID_DENSITY=0, OPENMX_GRID_HAMILTONIAN=1 };
enum { OPENMX_GRID_SCALAR=0, OPENMX_GRID_AVX2=1, OPENMX_GRID_AVX512=2 };
enum { OPENMX_GRID_B2C2=0, OPENMX_GRID_B2C1=1, OPENMX_GRID_B2D=2, OPENMX_GRID_A2B=3 };

/* Native orbital row tables are borrowed only at this boundary. Transposed
   rows are [orbital][point]; ordinary rows are [point][orbital]. An overlap
   row uses its position in the pair instead of the neighbor point index. */
typedef struct {
  const void *rows;
  int transposed, overlap_rows, element_bytes;
} OpenMXGridOrbitals;
typedef struct {
  int atom, rows, columns, points;
  const int *center_points, *neighbor_points, *potential_index;
  OpenMXGridOrbitals center, neighbor;
  const double *potential[4];
  double *const *matrix[4];
  double volume;
} OpenMXGridPair;
typedef struct { size_t first, count; int global_atom; } OpenMXGridAtom;
typedef struct {
  double ***atoms, **send, **receive;
  size_t atom_values, send_values, receive_values;
} OpenMXGridDensity;
typedef struct {
  double *send, *receive;
  void *send_requests, *receive_requests, *send_status, *receive_status;
} OpenMXGridExchange;

/* No default context and no device/runtime initialization. Case clear must
   follow completion of all native/GPU consumers and MPI requests. */
int openmx_core_grid_workspace(OpenMXCoreContext*,OpenMXGridWorkspace**,char*,size_t);
void openmx_core_grid_clear(OpenMXCoreContext*);
int openmx_core_grid_density_prepare(OpenMXGridWorkspace*,int spins,size_t atoms,
    const size_t *points,size_t ranks,const int *send,const int *receive,
    OpenMXGridDensity*,char*,size_t);
/* Writable descriptors remain stable until a new plan is prepared. Workers
   get independent contiguous scratch. A worker does not allocate, look up an
   owner, start another OpenMP runtime, or change the floating point mode. */
int openmx_core_grid_plan(OpenMXGridWorkspace*,int stage,int spins,size_t atoms,
    size_t pairs,size_t max_orbitals,size_t workers,int simd,
    OpenMXGridAtom**,OpenMXGridPair**,char*,size_t);
int openmx_core_grid_density_worker(OpenMXGridWorkspace*,size_t worker,size_t team,
    double *atom_seconds,void (*clock)(double*),char*,size_t);
int openmx_core_grid_hamiltonian_worker(OpenMXGridWorkspace*,size_t worker,size_t team,
    char*,size_t);
int openmx_core_grid_exchange(OpenMXGridWorkspace*,int kind,size_t send_values,
    size_t receive_values,size_t sends,size_t receives,size_t request_bytes,
    size_t status_bytes,OpenMXGridExchange*,char*,size_t);
/* Policy selection is performed outside numerical loops. Explicit caps are
   bounded by the ISA executable by both the processor and operating system. */
int openmx_core_grid_simd(const char *value);
const char *openmx_core_grid_simd_name(int);

#ifdef __cplusplus
}
#endif
#endif
