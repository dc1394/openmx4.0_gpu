#include <openmx/xc/functionals.hpp>
#include <openmx/xc/kernels.h>

#include "wpbeh_kernel.hpp"

#include <cmath>

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

ShortRangeExchangeResult wpbeh_short_range_exchange(SpinDensity density, const SpinGradient& gradient,
                                                    double omega, bool symmetrize_spins) noexcept
{
  // The floors and the symmetrization are those of the exchange part of
  // openmx_xc_kernel_pbe: each spin density is at least 0.5e-14, the norm of
  // each spin gradient at least 0.5e-14, and the derivative with respect to
  // the gradient vector points along the unfloored components.
  const double den_min = 1.0e-14, gd_min = 1.0e-14;
  const double dens[2] = {openmx_xc_kernel_largest(0.5*den_min, density.up),
                          openmx_xc_kernel_largest(0.5*den_min, density.down)};
  double D[2], GD[3][2];
  if (symmetrize_spins) {
    D[0] = (dens[0] + dens[1])/2.0;
    D[1] = (dens[0] + dens[1])/2.0;
  } else {
    D[0] = dens[0];
    D[1] = dens[1];
  }
  for (int d = 0; d < 3; ++d) {
    if (symmetrize_spins) {
      GD[d][0] = (gradient.component[d][0] + gradient.component[d][1])/2.0;
      GD[d][1] = (gradient.component[d][0] + gradient.component[d][1])/2.0;
    } else {
      GD[d][0] = gradient.component[d][0];
      GD[d][1] = gradient.component[d][1];
    }
  }
  ShortRangeExchangeResult r;
  double energy = 0.0;
  for (int s = 0; s < 2; ++s) {
    const double norm = std::sqrt(GD[0][s]*GD[0][s] + GD[1][s]*GD[1][s] + GD[2][s]*GD[2][s]);
    const double floored_norm = openmx_xc_kernel_largest(0.5*gd_min, norm);
    const detail::ChannelEnergy channel = detail::wpbeh_channel_energy(D[s], floored_norm, omega);
    energy += channel.energy;
    r.dexchange_ddensity[s] = channel.denergy_ddensity;
    for (int d = 0; d < 3; ++d) r.dexchange_dgradient[d][s] = channel.denergy_dgradient*(GD[d][s]/floored_norm);
  }
  r.exchange_energy_density = energy/(dens[0] + dens[1]);
  r.density_used = {dens[0], dens[1]};
  return r;
}

PbeResult pbe_screened_hybrid(SpinDensity density, const SpinGradient& gradient, bool symmetrize_spins,
                              double exchange_fraction, double omega) noexcept
{
  PbeResult r = pbe(density, gradient, symmetrize_spins);
  if (exchange_fraction == 0.0) return r;
  const ShortRangeExchangeResult sr = wpbeh_short_range_exchange(density, gradient, omega, symmetrize_spins);
  r.exchange_energy_density -= exchange_fraction*sr.exchange_energy_density;
  for (int s = 0; s < 2; ++s) {
    r.dexchange_ddensity[s] -= exchange_fraction*sr.dexchange_ddensity[s];
    for (int d = 0; d < 3; ++d) r.dexchange_dgradient[d][s] -= exchange_fraction*sr.dexchange_dgradient[d][s];
  }
  return r;
}

PbeResult pbe_global_hybrid(SpinDensity density, const SpinGradient& gradient, bool symmetrize_spins,
                            double exchange_fraction) noexcept
{
  PbeResult r = pbe(density, gradient, symmetrize_spins);
  if (exchange_fraction == 0.0) return r;
  const double kept = 1.0 - exchange_fraction;
  r.exchange_energy_density *= kept;
  for (int s = 0; s < 2; ++s) {
    r.dexchange_ddensity[s] *= kept;
    for (int d = 0; d < 3; ++d) r.dexchange_dgradient[d][s] *= kept;
  }
  return r;
}

}  // namespace openmx::xc
