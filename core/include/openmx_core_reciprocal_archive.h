#ifndef OPENMX_CORE_RECIPROCAL_ARCHIVE_H
#define OPENMX_CORE_RECIPROCAL_ARCHIVE_H
#include "openmx_core_reciprocal.h"
#ifdef __cplusplus
extern "C" {
#endif

enum OpenMXReciprocalArchiveKind {
  OPENMX_ARCHIVE_PAO, OPENMX_ARCHIVE_NONLOCAL, OPENMX_ARCHIVE_NEUTRAL_PROJECTOR,
  OPENMX_ARCHIVE_NEUTRAL_POTENTIAL, OPENMX_ARCHIVE_PRODUCT
};
typedef struct {
  int kind, species, pao_points, neutral_points;
  const int *maximum_angular;
  int *const *basis_counts;
  const int *projector_counts, *projector_spin_lmax;
} OpenMXReciprocalArchiveInput;
typedef struct { double *values; int count; } OpenMXReciprocalRow;
typedef struct {
  size_t species, rows;
  const size_t *species_offsets;
  const OpenMXReciprocalRow *row;
} OpenMXReciprocalRows;
typedef struct OpenMXReciprocalArchive OpenMXReciprocalArchive;

/* Snapshot the active rows in the original restart/MPI order, borrowing only
   numeric storage from context. Metadata and input can then be discarded.
   Failure leaves output unchanged; no MPI or GPU runtime is initialized. */
OpenMXReciprocalArchive *openmx_core_reciprocal_archive_create(OpenMXCoreContext *context,
    const OpenMXReciprocalArchiveInput *input, OpenMXReciprocalRows *rows,
    char *error, size_t capacity);
/* Exact legacy binary format: active rows, without padding or metadata.
   Reads preflight the file length before modifying values; an I/O failure
   while streaming can leave partial rows, which the caller must recompute.
   Files close on every path, including exceptions and failed writes. */
int openmx_core_reciprocal_archive_read(OpenMXReciprocalArchive *archive,
    const char *path, char *error, size_t capacity);
int openmx_core_reciprocal_archive_write(const OpenMXReciprocalArchive *archive,
    const char *path, char *error, size_t capacity);
void openmx_core_reciprocal_archive_destroy(OpenMXReciprocalArchive *archive);

#ifdef __cplusplus
}
#endif
#endif
