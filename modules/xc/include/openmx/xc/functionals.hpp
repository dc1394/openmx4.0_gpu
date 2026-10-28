#ifndef OPENMX_XC_FUNCTIONALS_HPP
#define OPENMX_XC_FUNCTIONALS_HPP

// Exchange-correlation functionals of OpenMX as pure functions of the local
// density (and gradient): the Ceperley-Alder LDA in the Perdew-Zunger
// parametrization, its LSDA form, the Perdew-Wang 92 correlation, Slater
// exchange of the uniform gas, and GGA-PBE.  Hartree atomic units: densities
// in bohr^-3, energies per particle in Hartree, potentials in Hartree,
// gradients in bohr^-4.  No state, no allocation, no global data; every
// function may be called concurrently.
//
// The arithmetic follows the OpenMX 4.0 C routines operation by operation,
// including their density floors (den_min = 1e-14 bohr^-3 and the spin
// polarization clamps), so results are bitwise those of the original code
// compiled without floating-point contraction.  `symmetrize_spins` is the
// LDA+U cFLL rule of the original code (dc_Type 3 or 4 after the first SCF
// step): both spins see the averaged density and the polarization is zero.

namespace openmx::xc {

struct SpinDensity {
  double up, down;
};

// gradient[direction][spin], directions x, y, z
struct SpinGradient {
  double component[3][2];
};

// ---- Ceperley-Alder LDA (Perdew-Zunger), unpolarized ----
enum class CeperleyAlderQuantity {
  energy_density = 0,          // epsilon_xc(rho)
  potential = 1,               // mu_xc = d(rho epsilon_xc)/d rho
  energy_minus_potential = 2,  // epsilon_xc - mu_xc
  energy_derivative = 3        // d epsilon_xc / d rho
};
// Zero for densities at or below 1e-15 bohr^-3.
double ceperley_alder(double density, CeperleyAlderQuantity quantity) noexcept;

// ---- Ceperley-Alder LSDA (Perdew-Zunger spin interpolation) ----
enum class LsdaQuantity {
  energy_density = 0,          // epsilon_xc, the same for both spins
  potential = 1,               // mu_xc per spin
  energy_minus_potential = 2   // epsilon_xc - mu_xc per spin
};
struct LsdaResult {
  double xc[2], exchange[2], correlation[2];   // per spin
};
// Zero for total densities below 1e-15 bohr^-3.
LsdaResult ca_lsda(SpinDensity density, LsdaQuantity quantity, bool symmetrize_spins) noexcept;

// ---- Perdew-Wang 92 correlation of the uniform gas ----
struct Pw92Result {
  double energy_density;       // epsilon_c
  double potential[2];         // mu_c per spin
  SpinDensity density_used;    // the input, or both spins 0.5e-14 below the floor
};
Pw92Result pw92_correlation(SpinDensity density, bool symmetrize_spins) noexcept;

// ---- Slater exchange of the unpolarized uniform gas ----
struct ExchangeResult {
  double energy_density;       // epsilon_x
  double potential;            // mu_x
};
// Densities at or below 1e-14 bohr^-3 are evaluated at the floor.
ExchangeResult lda_exchange(double density) noexcept;

// ---- GGA-PBE (Perdew, Burke, Ernzerhof 1996), spin-resolved ----
struct PbeResult {
  double exchange_energy_density, correlation_energy_density;   // epsilon_x, epsilon_c
  double dexchange_ddensity[2], dcorrelation_ddensity[2];       // per spin
  double dexchange_dgradient[3][2], dcorrelation_dgradient[3][2];  // [direction][spin]
  SpinDensity density_used;    // the input floored at 0.5e-14 per spin
};
PbeResult pbe(SpinDensity density, const SpinGradient& gradient, bool symmetrize_spins) noexcept;

}  // namespace openmx::xc

#endif
