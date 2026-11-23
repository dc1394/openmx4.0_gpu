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

// ---- Short-range omega-PBE exchange (wPBEh), spin-resolved ----
// The exchange energy of the screened interaction erfc(omega r)/r in the
// model of Heyd, Scuseria and Ernzerhof: the semilocal short-range exchange
// of HSE06, with the definition of the functional GGA_X_WPBEH of libxc 7.1.2.
// omega is in 1/bohr.  The outputs follow the conventions of the exchange
// part of pbe(), including its density floor, its treatment of a vanishing
// gradient and `symmetrize_spins`.
//
// The rounding error grows with w = omega/kF, kF = (6 pi^2 rho_s)^(1/3),
// because terms cancel: about 1e-15 for w <= 1, 1e-10 at w = 4, 1e-6 at
// w = 10 and 3e-5 just below w = 14; above 14 the definition changes form and
// the error is about 1e-12.  With omega = 0.11, w = 4 is a spin density of
// 3.5e-7 bohr^-3 and w = 14 one of 8e-9 bohr^-3.
struct ShortRangeExchangeResult {
  double exchange_energy_density;     // epsilon_x^SR
  double dexchange_ddensity[2];       // per spin
  double dexchange_dgradient[3][2];   // [direction][spin]
  SpinDensity density_used;           // the input floored at 0.5e-14 per spin
};
ShortRangeExchangeResult wpbeh_short_range_exchange(SpinDensity density, const SpinGradient& gradient,
                                                    double omega, bool symmetrize_spins) noexcept;

// ---- Semilocal part of a screened hybrid functional (HSE06) ----
// GGA-PBE with `exchange_fraction` times the short-range omega-PBE exchange
// removed: the exchange outputs of pbe() (energy density and both
// derivatives) minus that fraction of wpbeh_short_range_exchange(); the
// correlation outputs are those of pbe().  HSE06 uses the fraction 0.25 and
// omega = 0.11 bohr^-1.  A fraction of exactly 0 returns pbe() bit for bit
// without evaluating the short-range part.
PbeResult pbe_screened_hybrid(SpinDensity density, const SpinGradient& gradient, bool symmetrize_spins,
                              double exchange_fraction, double omega) noexcept;

// ---- Special functions of the short-range exchange ----
// exp(x) E1(x) with the exponential integral E1 (infinite at x = 0, NaN for
// negative x), and erfcx(x) = exp(x^2) erfc(x).  The relative error is below
// 1e-14 for 1e-30 <= x <= 1e8.
double e1_scaled(double x) noexcept;
double erfcx(double x) noexcept;

}  // namespace openmx::xc

#endif
