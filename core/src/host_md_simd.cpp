#include "host_md_kernels.hpp"
#include "host_simd_abi.hpp"
#include "openmx_md_integrator.hpp"

namespace openmx::core::host {
namespace {

void move_tile(OpenMXMDMoveTile *tile, const OpenMXMDStep *step)
{
  static_assert(OPENMX_MD_TILE%lanes==0);
  for (int l=0; l<OPENMX_MD_TILE; l+=lanes) {
    Vector x=load(tile->x+l), v=load(tile->v+l);
    md::move(x,v,load(tile->gradient+l),load(tile->mass+l),load(tile->scale+l),
             load(tile->previous_gradient+l),*step);
    store(x,tile->x+l);
    store(v,tile->v+l);
  }
}

void periodic_tile(OpenMXMDPeriodicTile *tile, const OpenMXMDCell *cell)
{
  for (int l=0; l<OPENMX_MD_TILE; l+=lanes) {
    const Vector x=load(tile->xyz[0]+l)-Vector(cell->origin[0]);
    const Vector y=load(tile->xyz[1]+l)-Vector(cell->origin[1]);
    const Vector z=load(tile->xyz[2]+l)-Vector(cell->origin[2]);
    Vector frac[3];
    for (int axis=0; axis<3; ++axis) {
      frac[axis]=md::dot3(x,y,z,cell->reciprocal[axis])*Vector(0.5)/Vector(cell->pi);
      const auto f=frac[axis];
      if (simd::any_of((f<Vector(0.0)) | (f>Vector(1.0)))) {
        const Vector shift([&](auto i) { return md::translation(f[i]); });
        store(shift,tile->shift[axis]+l);
        frac[axis]+=shift;
      }
      else store(Vector(0.0),tile->shift[axis]+l);
      store(frac[axis],tile->frac[axis]+l);
    }
    for (int axis=0; axis<3; ++axis) {
      const Vector position=frac[0]*Vector(cell->real[0][axis])
          +frac[1]*Vector(cell->real[1][axis])+frac[2]*Vector(cell->real[2][axis])
          +Vector(cell->origin[axis]);
      store(position,tile->xyz[axis]+l);
    }
  }
}

void npt_tile(OpenMXMDNPTTile *tile, const OpenMXMDNPTStep *step)
{
  for (int l=0; l<OPENMX_MD_TILE; l+=lanes) {
    Vector pos[3], vel[3], grad[3];
    for (int i=0; i<3; ++i) {
      pos[i]=load(tile->position[i]+l);
      vel[i]=load(tile->velocity[i]+l);
      grad[i]=load(tile->gradient[i]+l);
    }
    const Vector kinetic=md::npt_kick(pos,vel,grad,load(tile->mass+l),*step);
    for (int i=0; i<3; ++i) {
      store(pos[i],tile->position[i]+l);
      store(vel[i],tile->velocity[i]+l);
    }
    store(kinetic,tile->kinetic+l);
  }
}

}

const OpenMXHostMDMove& OPENMX_HOST_SIMD_FACTORY(md_move)() noexcept
{
  static constexpr OpenMXHostMDMove kernel=move_tile;
  return kernel;
}

const OpenMXHostMDPeriodic& OPENMX_HOST_SIMD_FACTORY(md_periodic)() noexcept
{
  static constexpr OpenMXHostMDPeriodic kernel=periodic_tile;
  return kernel;
}

const OpenMXHostMDNPT& OPENMX_HOST_SIMD_FACTORY(md_npt)() noexcept
{
  static constexpr OpenMXHostMDNPT kernel=npt_tile;
  return kernel;
}

}
