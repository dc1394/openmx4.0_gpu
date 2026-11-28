#ifndef OPENMX_CORE_MD_H
#define OPENMX_CORE_MD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { OPENMX_MD_TILE = 32 };

typedef enum {
  OPENMX_MD_VERLET_KICK,
  OPENMX_MD_VERLET_DRIFT,
  OPENMX_MD_SCALE_KICK,
  OPENMX_MD_SCALE_START,
  OPENMX_MD_SCALE_DRIFT,
  OPENMX_MD_LANGEVIN_START,
  OPENMX_MD_LANGEVIN_DRIFT
} OpenMXMDMove;

typedef struct {
  OpenMXMDMove move;
  double dt, dt2, friction;
} OpenMXMDStep;

/* One lane per movable Cartesian component. Padding has a unit mass. */
typedef struct {
  double x[OPENMX_MD_TILE], v[OPENMX_MD_TILE], gradient[OPENMX_MD_TILE];
  double mass[OPENMX_MD_TILE], scale[OPENMX_MD_TILE], previous_gradient[OPENMX_MD_TILE];
} OpenMXMDMoveTile;

typedef struct {
  double real[3][3], reciprocal[3][3], origin[3], pi;
} OpenMXMDCell;

/* One lane per atom. Shifts are fractional-cell translations. */
typedef struct {
  double xyz[3][OPENMX_MD_TILE], frac[3][OPENMX_MD_TILE], shift[3][OPENMX_MD_TILE];
} OpenMXMDPeriodicTile;

typedef struct {
  double reciprocal[3][3], coupling[3][3], lattice_columns[3][3], fractional_origin[3];
  double dt, friction;
} OpenMXMDNPTStep;

typedef struct {
  double position[3][OPENMX_MD_TILE], velocity[3][OPENMX_MD_TILE], gradient[3][OPENMX_MD_TILE];
  double mass[OPENMX_MD_TILE], kinetic[OPENMX_MD_TILE];
} OpenMXMDNPTTile;

typedef void (*OpenMXHostMDMove)(OpenMXMDMoveTile *, const OpenMXMDStep *);
typedef void (*OpenMXHostMDPeriodic)(OpenMXMDPeriodicTile *, const OpenMXMDCell *);
typedef void (*OpenMXHostMDNPT)(OpenMXMDNPTTile *, const OpenMXMDNPTStep *);
OpenMXHostMDMove openmx_core_host_md_move(int mode);
OpenMXHostMDPeriodic openmx_core_host_md_periodic(int mode);
OpenMXHostMDNPT openmx_core_host_md_npt(int mode);

#ifdef __cplusplus
}
#endif
#endif
