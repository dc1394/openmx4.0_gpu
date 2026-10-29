#include <openmx/xc/xc.h>
#include <openmx/xc/functionals.hpp>

extern "C" double openmx_xc_ceperley_alder(double density, int quantity)
{
  if (quantity < 0 || quantity > 3) return 0.0;
  return openmx::xc::ceperley_alder(density, static_cast<openmx::xc::CeperleyAlderQuantity>(quantity));
}

extern "C" int openmx_xc_ca_lsda(double den_up, double den_down, int quantity, int symmetrize_spins,
                                 double xc[2], double x[2], double c[2])
{
  if (quantity < 0 || quantity > 2) return 0;
  const openmx::xc::LsdaResult r = openmx::xc::ca_lsda({den_up, den_down},
      static_cast<openmx::xc::LsdaQuantity>(quantity), symmetrize_spins != 0);
  for (int s = 0; s < 2; ++s) {
    if (xc) xc[s] = r.xc[s];
    if (x) x[s] = r.exchange[s];
    if (c) c[s] = r.correlation[s];
  }
  return 1;
}

extern "C" int openmx_xc_pw92_correlation(double dens[2], int symmetrize_spins, double ec[1], double vc[2])
{
  if (!dens || !ec || !vc) return 0;
  const openmx::xc::Pw92Result r = openmx::xc::pw92_correlation({dens[0], dens[1]}, symmetrize_spins != 0);
  dens[0] = r.density_used.up;
  dens[1] = r.density_used.down;
  ec[0] = r.energy_density;
  vc[0] = r.potential[0];
  vc[1] = r.potential[1];
  return 1;
}

extern "C" int openmx_xc_lda_exchange(double density, double ex[1], double vx[1])
{
  if (!ex || !vx) return 0;
  const openmx::xc::ExchangeResult r = openmx::xc::lda_exchange(density);
  ex[0] = r.energy_density;
  vx[0] = r.potential;
  return 1;
}

extern "C" int openmx_xc_pbe(double dens[2], double gdens[3][2], int symmetrize_spins,
                             double exc[2], double dexdd[2], double decdd[2],
                             double dexdgd[3][2], double decdgd[3][2])
{
  if (!dens || !gdens || !exc || !dexdd || !decdd || !dexdgd || !decdgd) return 0;
  openmx::xc::SpinGradient g;
  for (int d = 0; d < 3; ++d) for (int s = 0; s < 2; ++s) g.component[d][s] = gdens[d][s];
  const openmx::xc::PbeResult r = openmx::xc::pbe({dens[0], dens[1]}, g, symmetrize_spins != 0);
  dens[0] = r.density_used.up;
  dens[1] = r.density_used.down;
  exc[0] = r.exchange_energy_density;
  exc[1] = r.correlation_energy_density;
  for (int s = 0; s < 2; ++s) {
    dexdd[s] = r.dexchange_ddensity[s];
    decdd[s] = r.dcorrelation_ddensity[s];
    for (int d = 0; d < 3; ++d) {
      dexdgd[d][s] = r.dexchange_dgradient[d][s];
      decdgd[d][s] = r.dcorrelation_dgradient[d][s];
    }
  }
  return 1;
}
