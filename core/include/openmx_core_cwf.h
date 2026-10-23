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

#ifdef __cplusplus
}
#endif
#endif
