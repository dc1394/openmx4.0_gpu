#ifndef OPENMX_CORE_CWF_H
#define OPENMX_CORE_CWF_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Native CWF matrix layouts. These are host metadata, independent of MPI,
   BLACS and device SDK types. The native caller configures/releases BLACS;
   this value does not claim ownership of a BLACS resource. */
enum {
  OPENMX_CWF_HOPPING_GRID,
  OPENMX_CWF_ORBITALS_GRID,
  OPENMX_CWF_PROJECTION_GRID,
  OPENMX_CWF_EIGENVECTORS_GRID,
  OPENMX_CWF_BAND_GRID,
  OPENMX_CWF_GRID_COUNT
};
typedef struct {
  int block_size;
  int process_rows, process_columns;
  int local_rows, local_columns;
  int maximum_rows, maximum_columns;
  int process_row, process_column;
  int system_handle, context;
  int descriptor[9];
} OpenMXCWFGrid;
typedef struct { OpenMXCWFGrid matrix[OPENMX_CWF_GRID_COUNT]; } OpenMXCWFGrids;

/* Borrow the actual calculation value without allocating, including the first
   lookup. Invalid arguments preserve *out. Native borrowers drain before the
   metadata resets. Bind/clear/finish require quiescent users; rebind after
   clear/finish and never retain the view past destruction. No default owner
   or process/thread-local state is installed. This performs no MPI/SDK setup. */
int openmx_core_cwf_grids(OpenMXCoreContext*, OpenMXCWFGrids **out,
    char *error, size_t error_capacity);

/* Per-calculation CWF output. Row capacity is the allocated length, which
   may exceed the active CWF count when guiding orbitals are selected later.
   Charge/energy are contiguous [spin_channel,row_capacity] host matrices. */
typedef struct {
  size_t spin_channels, row_capacity;
  double *charge, *energy;
  double distance;
} OpenMXCWFObservables;

/* Lookup never allocates or resets live results, including the first call.
   Prepare accepts one or two channels and zeros the requested output. Equal
   shapes reuse storage without allocating. A failed preparation preserves
   all live results and *out. Borrowed rows expire on a successful shape
   change, clear/finish or destruction. Prepare/clear require quiescent users;
   native borrowers drain before calculation clear/finish resets this state.
   No MPI/device setup or default/process/thread-local owner is installed. */
int openmx_core_cwf_observables(OpenMXCoreContext*, OpenMXCWFObservables **out,
    char *error, size_t error_capacity);
int openmx_core_cwf_observables_prepare(OpenMXCoreContext*, size_t spin_channels,
    size_t row_capacity, OpenMXCWFObservables **out, char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
static inline double *openmx_cwf_charge_row(const OpenMXCWFObservables *view,size_t spin)
{ return view->row_capacity ? view->charge+spin*view->row_capacity : view->charge; }
static inline double *openmx_cwf_energy_row(const OpenMXCWFObservables *view,size_t spin)
{ return view->row_capacity ? view->energy+spin*view->row_capacity : view->energy; }
#endif
