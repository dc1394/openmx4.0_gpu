#ifndef OPENMX_CORE_DYNAMICS_H
#define OPENMX_CORE_DYNAMICS_H
#include "openmx_core.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Mutable temperature and Nose-Hoover state of one calculation, distinct
   from input settings (heat-bath mass, schedules) and physical energies.
   Native MD retains its arithmetic, units and update order. */
typedef struct {
  double temperature;
  double target_temperature;
  double nose_coordinate;
  double nose_next_friction;
  double nose_friction;
  double nose_energy;
} OpenMXDynamicsState;

/* Borrow the C++ context's direct value, initially all zero. Lookup never
   allocates. Invalid arguments preserve *out. Bind/clear/finish require
   quiescent users; native borrowers drain before values reset. Rebind after
   clear/finish and never retain a pointer past destruction. No default
   calculation or mutable process/thread-local state is used. */
int openmx_core_dynamics(OpenMXCoreContext*, OpenMXDynamicsState **out,
    char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
