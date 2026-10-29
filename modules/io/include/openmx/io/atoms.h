#ifndef OPENMX_IO_ATOMS_H
#define OPENMX_IO_ATOMS_H

/* C interface of the atom block reader (see atoms.hpp).  Fills `count`
   records from the top file of `files`.  Returns 1 on success, 0 when the
   block is absent (nothing is written), -1 on a format error with the
   original reader's message in `message`, and -1 for NULL arguments or a
   species or polarization name that does not fit its field. */

#include "keyword_file.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int fields;                   /* converted items, as sscanf would report */
  int index;
  char species[64];
  double position[3];
  double spin[2];               /* initial up and down charges */
  double spin_angle[2];         /* degrees, non-collinear */
  double orbital_angle[2];      /* degrees, non-collinear */
  int constraint;
  char orbital_polarization[64];
} OpenMXAtomDefinition;

int openmx_io_read_atom_definitions(OpenMXKeywordFiles* files, int count, int noncollinear,
                                    OpenMXAtomDefinition* atoms, char* message, size_t message_capacity);

#ifdef __cplusplus
}
#endif

#endif
