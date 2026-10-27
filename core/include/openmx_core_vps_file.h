#ifndef OPENMX_CORE_VPS_FILE_H
#define OPENMX_CORE_VPS_FILE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double atom_species, total_electrons, valence_electrons;
  int atomic_number, mesh_points, projector_components, max_angular;
  int j_dependent, partial_core, legacy_format, metadata_only;
  size_t projector_count;
  const int *angular;
  const double *energies; /* [projector][1+j_dependent]; absent for legacy metadata */
  const double *log_radii, *radii;
  const double *projectors; /* [projector][1+j_dependent][mesh_points] */
  const double *partial_core_density; /* physical points, absent when disabled */
  double *local_potential; /* mutable scratch: physical points, then two zeros */
  size_t local_potential_capacity; /* zero for metadata-only reads */
} OpenMXVpsFileData;

/* Synchronous, exactly-once consumer after a successful complete parse.
   All data pointers are borrowed only for this invocation. The C++ owner
   closes the file and frees every buffer on return, failure or exception.
   No callback is made for invalid input or allocation failure. A consumer
   must return 1 on success, 0 on failure and may write a diagnostic; its
   own side effects are not rolled back. No OpenMX/MPI/Kokkos initialization
   or process-global input state is needed by this boundary. */
typedef int (*OpenMXVpsFileConsumer)(void *user, const OpenMXVpsFileData *data,
    char *error, size_t error_capacity);
int openmx_core_with_vps_file(const char *path, int metadata_only,
    OpenMXVpsFileConsumer consumer, void *user, char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
