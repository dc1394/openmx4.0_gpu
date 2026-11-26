#include <openmx/xc/functionals.hpp>
#include <openmx/xc/hybrid_kernels.h>
#include <openmx/xc/kernels.h>

// The arithmetic lives in kernels.h and hybrid_kernels.h, shared with C
// callers and with the application's GPU device code; these are the owning
// C++ views of it.

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

ShortRangeExchangeResult wpbeh_short_range_exchange(SpinDensity density, const SpinGradient& gradient,
                                                    double omega, bool symmetrize_spins) noexcept
{
  double dens[2] = {density.up, density.down}, gdens[3][2], ex[1];
  for (int d = 0; d < 3; ++d) for (int s = 0; s < 2; ++s) gdens[d][s] = gradient.component[d][s];
  ShortRangeExchangeResult r;
  openmx_xc_kernel_wpbeh_short_range(dens, gdens, symmetrize_spins ? 1 : 0, omega, ex, r.dexchange_ddensity,
                                     r.dexchange_dgradient);
  r.exchange_energy_density = ex[0];
  r.density_used = {dens[0], dens[1]};
  return r;
}

PbeResult pbe_screened_hybrid(SpinDensity density, const SpinGradient& gradient, bool symmetrize_spins,
                              double exchange_fraction, double omega) noexcept
{
  double dens[2] = {density.up, density.down}, gdens[3][2], exc[2];
  for (int d = 0; d < 3; ++d) for (int s = 0; s < 2; ++s) gdens[d][s] = gradient.component[d][s];
  PbeResult r;
  openmx_xc_kernel_pbe_screened_hybrid(dens, gdens, symmetrize_spins ? 1 : 0, exchange_fraction, omega, exc,
                                       r.dexchange_ddensity, r.dcorrelation_ddensity, r.dexchange_dgradient,
                                       r.dcorrelation_dgradient);
  r.exchange_energy_density = exc[0];
  r.correlation_energy_density = exc[1];
  r.density_used = {dens[0], dens[1]};
  return r;
}

PbeResult pbe_global_hybrid(SpinDensity density, const SpinGradient& gradient, bool symmetrize_spins,
                            double exchange_fraction) noexcept
{
  double dens[2] = {density.up, density.down}, gdens[3][2], exc[2];
  for (int d = 0; d < 3; ++d) for (int s = 0; s < 2; ++s) gdens[d][s] = gradient.component[d][s];
  PbeResult r;
  openmx_xc_kernel_pbe_global_hybrid(dens, gdens, symmetrize_spins ? 1 : 0, exchange_fraction, exc,
                                     r.dexchange_ddensity, r.dcorrelation_ddensity, r.dexchange_dgradient,
                                     r.dcorrelation_dgradient);
  r.exchange_energy_density = exc[0];
  r.correlation_energy_density = exc[1];
  r.density_used = {dens[0], dens[1]};
  return r;
}

}  // namespace openmx::xc
