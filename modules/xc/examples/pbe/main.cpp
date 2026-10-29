// GGA-PBE energy densities and derivatives at one point: a spin-polarized
// density with a gradient along x, in Hartree atomic units.
#include <openmx/xc/functionals.hpp>

#include <cstdio>

int main()
{
  const openmx::xc::SpinDensity density{0.030, 0.010};              // bohr^-3
  const openmx::xc::SpinGradient gradient{{{0.02, 0.005}, {0.0, 0.0}, {0.0, 0.0}}};
  const openmx::xc::PbeResult r = openmx::xc::pbe(density, gradient, false);
  std::printf("epsilon_x = %.12f  epsilon_c = %.12f Hartree\n",
              r.exchange_energy_density, r.correlation_energy_density);
  std::printf("d(rho eps_x)/d rho_up = %.12f  d(rho eps_c)/d rho_down = %.12f\n",
              r.dexchange_ddensity[0], r.dcorrelation_ddensity[1]);
  const openmx::xc::LsdaResult lda = openmx::xc::ca_lsda(density, openmx::xc::LsdaQuantity::potential, false);
  std::printf("Ceperley-Alder LSDA potential: up %.12f  down %.12f Hartree\n", lda.xc[0], lda.xc[1]);
  return 0;
}
