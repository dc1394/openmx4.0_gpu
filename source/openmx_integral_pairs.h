#ifndef OPENMX_INTEGRAL_PAIRS_H
#define OPENMX_INTEGRAL_PAIRS_H
#include "openmx_core_atom_pairs.h"
#include <stdio.h>
#include <stdlib.h>

/* The C boundary supplies legacy topology once. The C++ plan owns the pair
   list and preserves local-atom/neighbor order, including onsite entries. */
static inline OpenMXAtomPairPlan *OpenMX_IntegralPairs(const OpenMXAtomPairInput *input,
    OpenMXAtomPairBuffers *buffers)
{
  char error[256];
  OpenMXAtomPairPlan *owner=openmx_core_atom_pairs_create(input,buffers,error,sizeof(error));
  if (!owner) {fprintf(stderr,"OpenMX integral pair plan: %s\n",error);abort();}
  return owner;
}
#endif
