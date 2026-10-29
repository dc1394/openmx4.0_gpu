#ifndef OPENMX_IO_SPECIES_H
#define OPENMX_IO_SPECIES_H

/* C interface of the species-definition block reader (see species.hpp).
   Reads `count` species from the top file of `files` into names[i],
   basis[i] and pseudopotentials[i] (capacity bytes each) and mass[i] (left
   as it was when the line has no mass).  Returns 1 on success, 0 when the
   block is absent (nothing is written), -1 on a format error with the
   original reader's message in `message`, and -1 for NULL arguments. */

#include "keyword_file.h"   /* the same directory: also reachable through the application's forward header */

#ifdef __cplusplus
extern "C" {
#endif

int openmx_io_read_species_definitions(OpenMXKeywordFiles* files, int count,
                                       char* const* names, char* const* basis, char* const* pseudopotentials,
                                       double* mass, size_t capacity, char* message, size_t message_capacity);

#ifdef __cplusplus
}
#endif

#endif
