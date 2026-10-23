#ifndef OPENMX_CORE_ENERGY_H
#define OPENMX_CORE_ENERGY_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* One calculation's electronic band energy and the existing two spin
   components named Uele_OS/IS by the native solvers. These values are distinct
   from the total physical energy Utot and from the mixing history. The C++
   context owns this fixed-size POD by value, initially zero. */
typedef struct {
  double band;
  double os[2];
  double is[2];
} OpenMXBandEnergies;

/* Borrow the actual calculation value, without allocating even on first
   lookup. Failure preserves *out. Bind/clear/finish require quiescent users;
   native resource borrowers drain before clear resets these values. Clear,
   finish and destruction invalidate all borrowed pointers. No default or
   process/thread-local calculation is installed. */
int openmx_core_band_energies(OpenMXCoreContext*, OpenMXBandEnergies **out,
    char *error, size_t error_capacity);

/* Contributions computed by Total_Energy and retained by DFT for SCF/MD
   output. Preserve the native terms separately, including double counting
   terms used by the Harris functional. No summation or solver policy is
   imposed here. This value does not contain Utot, ionic kinetic energy or
   pressure times volume. */
typedef struct {
  double kinetic;
  double hartree[2];
  double neutral_atom;
  double nonlocal;
  double xc[2];
  double core;
  double hubbard;
  double spin_constraint;
  double zeeman_spin;
  double zeeman_orbital;
  double electric_field;
  double vdw;
  double core_hole;
  double hartree_double_counting;
  double xc_double_counting[2];
  double atomic_reference;
} OpenMXEnergyComponents;

/* Borrow the calculation's zero-initialized components without allocating.
   Failure preserves *out. The same quiescence and lifetime rules as the
   band-energy accessor apply; native borrowers drain before reset. */
int openmx_core_energy_components(OpenMXCoreContext*, OpenMXEnergyComponents **out,
    char *error, size_t error_capacity);

/* Current physical total energy, ionic kinetic energy and pressure-volume
   contribution. DFT, MD and stress calculations update the same calculation
   value; the caller retains its native arithmetic and reduction order. */
typedef struct {
  double total;
  double ionic_kinetic;
  double pressure_volume;
} OpenMXThermodynamicEnergies;

/* Allocation-free access, initially all zero. Invalid arguments leave *out
   unchanged. Bind/clear/finish require quiescent users; borrowed native states
   drain before the values reset. Rebind after clear/finish and never retain a
   pointer past destruction. No default calculation is used. */
int openmx_core_thermodynamic_energies(OpenMXCoreContext*, OpenMXThermodynamicEnergies **out,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
