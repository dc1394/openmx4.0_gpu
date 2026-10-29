#ifndef OPENMX_IO_BASIS_SPEC_H
#define OPENMX_IO_BASIS_SPEC_H

/* C interface of the basis-specification parser (see basis_spec.hpp for
   the grammar).  Returns 1 and fills `name` (at most capacity-1 characters
   plus the terminator) and `spec` on success; 0 on a format error, with
   the original reader's message copied to `message` (message_capacity
   bytes) when given.  NULL inputs or a name that does not fit are errors. */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int primitive[5];   /* per l = s p d f g */
  int contracted[5];  /* per l; zero unless given after '>' */
  int max_l;          /* highest l with primitives, -1 when none */
  int contraction;    /* 1 when any '>' appeared */
} OpenMXBasisSpec;

int openmx_io_parse_basis_spec(const char* text, int allow_contraction,
                               char* name, size_t capacity, OpenMXBasisSpec* spec,
                               char* message, size_t message_capacity);

#ifdef __cplusplus
}
#endif

#endif
