#ifndef OPENMX_IO_SETTINGS_H
#define OPENMX_IO_SETTINGS_H

/* C interface of the unit-cell block reader (see settings.hpp).  Fills
   vectors[3][3] (rows a1, a2, a3).  Returns 1 on success, 0 when the block
   is absent (nothing is written), -1 on a format error with the original
   reader's message in `message`. */

#include "keyword_file.h"

#ifdef __cplusplus
extern "C" {
#endif

int openmx_io_read_unit_vectors(OpenMXKeywordFiles* files, double vectors[3][3], char* message, size_t message_capacity);

#ifdef __cplusplus
}
#endif

#endif
