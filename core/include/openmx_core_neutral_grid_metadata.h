#ifndef OPENMX_CORE_NEUTRAL_GRID_METADATA_H
#define OPENMX_CORE_NEUTRAL_GRID_METADATA_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct { size_t species; } OpenMXNeutralGridMetadataShape;
/* Borrowed calculation-owned host metadata; both arrays are uninitialized
   until written. Native input loading temporarily reuses spacing for raw
   (possibly fractional) PAO/VPS AtomSpecies checks before grid construction.
   Views expire at prepare/release/finish/destruction. Main grid values have
   an independent lifetime. Never free or rebind either borrowed array. */
typedef struct {
  int *counts;
  double *spacing;
  size_t species;
} OpenMXNeutralGridMetadataView;

/* Species count must fit the native positive int index range. Failure keeps
   the old owner, its views and output unchanged. No default context or GPU
   initialization is required. */
int openmx_core_neutral_grid_metadata_prepare(OpenMXCoreContext *context,
    const OpenMXNeutralGridMetadataShape *shape,OpenMXNeutralGridMetadataView *out,
    char *error,size_t error_capacity);
int openmx_core_neutral_grid_metadata_get(OpenMXCoreContext *context,
    OpenMXNeutralGridMetadataView *out,char *error,size_t error_capacity);
void openmx_core_neutral_grid_metadata_release(OpenMXCoreContext *context);
#ifdef __cplusplus
}
#endif
#endif
