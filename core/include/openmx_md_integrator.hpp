#ifndef OPENMX_MD_INTEGRATOR_HPP
#define OPENMX_MD_INTEGRATOR_HPP

#include "openmx_core_md.h"
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define OPENMX_MD_HD __host__ __device__
#else
#define OPENMX_MD_HD
#endif

namespace openmx::md {

template<class Real>
OPENMX_MD_HD inline void move(Real &x, Real &v, const Real &gradient,
    const Real &mass, const Real &scale, const Real &previous_gradient, const OpenMXMDStep &step)
{
  const Real dt(step.dt), half(0.5);
  const Real force=gradient/mass;
  switch (step.move) {
    case OPENMX_MD_VERLET_KICK:
      v=v+(-force)*half*dt;
      break;
    case OPENMX_MD_VERLET_DRIFT:
      if (step.friction==0.0) v=v+(-force)*half*dt;
      else v=v+(-force-Real(step.friction)*v)*half*dt;
      x=x+v*dt;
      break;
    case OPENMX_MD_SCALE_KICK:
      v=v-force*dt;
      break;
    case OPENMX_MD_SCALE_START:
      x=x+dt*v-force*Real(step.dt2)*half;
      v=v-force*dt;
      break;
    case OPENMX_MD_SCALE_DRIFT:
      v=v*scale;
      x=x+dt*v-force*Real(step.dt2)*half;
      break;
    case OPENMX_MD_LANGEVIN_START:
      x=x+dt*v-half*Real(step.dt2)*(force+Real(step.friction)*v);
      v=v-force*dt;
      break;
    case OPENMX_MD_LANGEVIN_DRIFT: {
      const Real friction(step.friction);
      const Real damping(1.0/(1.0+0.5*step.dt*step.friction));
      v=v-(gradient+previous_gradient)/mass*dt*half-friction*(x-scale);
      x=x+damping*(dt*v-force*Real(step.dt2)*half);
      break;
    }
  }
}

template<class Real>
OPENMX_MD_HD inline Real dot3(const Real &x, const Real &y, const Real &z, const double *row)
{
  return x*Real(row[0])+y*Real(row[1])+z*Real(row[2]);
}

template<class Real>
OPENMX_MD_HD inline Real npt_kick(Real (&pos)[3], Real (&vel)[3],
    const Real (&grad)[3], const Real &mass, const OpenMXMDNPTStep &step)
{
  Real reduced[3];
  for (int i=0; i<3; ++i)
    reduced[i]=dot3(vel[0],vel[1],vel[2],step.reciprocal[i]);
  Real frac[3];
  for (int i=0; i<3; ++i)
    frac[i]=dot3(pos[0],pos[1],pos[2],step.reciprocal[i])-Real(step.fractional_origin[i]);
  for (int i=0; i<3; ++i) {
    Real acceleration=-dot3(grad[0],grad[1],grad[2],step.reciprocal[i])/mass
                      -dot3(reduced[0],reduced[1],reduced[2],step.coupling[i]);
    if (step.friction!=0.0) acceleration=acceleration-Real(step.friction)*reduced[i];
    vel[i]=reduced[i]+acceleration*Real(step.dt);
    pos[i]=frac[i];
  }
  const Real vx=dot3(vel[0],vel[1],vel[2],step.lattice_columns[0]);
  const Real vy=dot3(vel[0],vel[1],vel[2],step.lattice_columns[1]);
  const Real vz=dot3(vel[0],vel[1],vel[2],step.lattice_columns[2]);
  return Real(0.5)*mass*(vx*vx+vy*vy+vz*vz);
}

OPENMX_MD_HD inline double translation(double frac)
{
  /* Exactly 1 is already in the cell. A negative integer maps to 1 as in
     the native trajectory format. */
  if (frac>1.0) return -::trunc(frac);
  if (frac<0.0) return 1.0-::trunc(frac);
  return 0.0;
}

} // namespace openmx::md

#undef OPENMX_MD_HD
#endif
