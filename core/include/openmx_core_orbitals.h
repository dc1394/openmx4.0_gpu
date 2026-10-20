#ifndef OPENMX_CORE_ORBITALS_H
#define OPENMX_CORE_ORBITALS_H
#include <stddef.h>

/* Unchecked packed orbital number for validated angular/radial capacities.
   magnetic_index follows the legacy -l..l order, shifted into [0,2*l]. */
static inline size_t openmx_core_orbital_index(int multiplicity,
    int angular, int radial, int magnetic_index)
{
  return (size_t)multiplicity*angular*angular + (size_t)radial*(2*angular+1) + magnetic_index;
}
#endif
