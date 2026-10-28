#include <openmx/xc/functionals.hpp>
#include <openmx/xc/kernels.h>

// The arithmetic lives in kernels.h, shared with C callers and with the
// application's GPU device code; these are the owning C++ views of it.

namespace openmx::xc {

double ceperley_alder(double den, CeperleyAlderQuantity quantity) noexcept
{
  return openmx_xc_kernel_ceperley_alder(den, static_cast<int>(quantity));
}

LsdaResult ca_lsda(SpinDensity density, LsdaQuantity quantity, bool symmetrize_spins) noexcept
{
  LsdaResult r;
  openmx_xc_kernel_ca_lsda(density.up, density.down, static_cast<int>(quantity), symmetrize_spins ? 1 : 0,
                           r.xc, r.exchange, r.correlation);
  return r;
}

Pw92Result pw92_correlation(SpinDensity density, bool symmetrize_spins) noexcept
{
  double dens[2] = {density.up, density.down};
  Pw92Result r;
  openmx_xc_kernel_pw92(dens, symmetrize_spins ? 1 : 0, &r.energy_density, r.potential);
  r.density_used = {dens[0], dens[1]};
  return r;
}

ExchangeResult lda_exchange(double density) noexcept
{
  ExchangeResult r;
  openmx_xc_kernel_exchange(density, &r.energy_density, &r.potential);
  return r;
}

PbeResult pbe(SpinDensity density, const SpinGradient& gradient, bool symmetrize_spins) noexcept
{
  double dens[2] = {density.up, density.down};
  double gdens[3][2];
  for (int d = 0; d < 3; ++d) for (int s = 0; s < 2; ++s) gdens[d][s] = gradient.component[d][s];
  PbeResult r;
  double exc[2];
  openmx_xc_kernel_pbe(dens, gdens, symmetrize_spins ? 1 : 0, exc, r.dexchange_ddensity, r.dcorrelation_ddensity,
                       r.dexchange_dgradient, r.dcorrelation_dgradient);
  r.exchange_energy_density = exc[0];
  r.correlation_energy_density = exc[1];
  r.density_used = {dens[0], dens[1]};
  return r;
}

}  // namespace openmx::xc
