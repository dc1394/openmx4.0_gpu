// Tests of the xc package: bitwise agreement with the frozen OpenMX 4.0
// routines, closed-form and literature values, potentials against central
// differences of the energy density, and the boundary behaviour.
#include <openmx/xc/functionals.hpp>
#include <openmx/xc/xc.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// the frozen routines' prototypes and dc_Type; its PI macro is theirs only
extern "C" {
#include "openmx_common.h"
}
#undef PI

namespace {
using namespace openmx::xc;

void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
bool close(double a, double b, double relative) {
  const double scale = std::max(std::fabs(a), std::fabs(b));
  return std::fabs(a - b) <= relative * (scale > 1e-300 ? scale : 1.0);
}
void fail(const std::string& what, double a, double b) {
  std::fprintf(stderr, "%s: %.17g vs %.17g\n", what.c_str(), a, b);
  throw std::runtime_error(what);
}

std::vector<double> density_grid() {
  // the floors (1e-14 per spin, 1e-15 total), rs below and above 1, large densities, negatives
  std::vector<double> v = {-1e-13, -3e-15, -1e-16, 0.0, 1e-17, 1e-16, 4e-15, 5e-15, 6e-15, 1e-14, 1.5e-14,
                           2e-14, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 2.5e-3, 0.01, 0.0331, 0.1, 0.2387, 0.5, 1.0, 7.0, 50.0};
  std::mt19937 rng(2026u);
  std::uniform_real_distribution<double> e(-16.0, 1.5);
  for (int i = 0; i < 60; ++i) v.push_back(std::pow(10.0, e(rng)));
  return v;
}

// The original condition of the cFLL symmetrization: dc_Type 3 or 4 after the first SCF step.
struct LegacyMode { int dc_type, scf_iter; bool symmetrize; };
const LegacyMode modes[] = {{0, 1, false}, {0, 2, false}, {3, 1, false}, {3, 2, true}, {4, 5, true}};

void legacy() {
  const std::vector<double> grid = density_grid();
  long checks = 0;
  for (double den : grid) {
    for (int q = 0; q < 4; ++q) {
      const double a = ceperley_alder(den, static_cast<CeperleyAlderQuantity>(q));
      const double b = XC_Ceperly_Alder(den, q);
      if (!same_bits(a, b)) fail("Ceperley-Alder quantity " + std::to_string(q) + " density " + std::to_string(den), a, b);
      if (!same_bits(openmx_xc_ceperley_alder(den, q), b)) fail("C Ceperley-Alder", openmx_xc_ceperley_alder(den, q), b);
      ++checks;
    }
    {
      double DS[2] = {den, den}, EX[1], VX[2];
      XC_EX(1, den, DS, EX, VX);
      const ExchangeResult r = lda_exchange(den);
      if (!same_bits(r.energy_density, EX[0]) || !same_bits(r.potential, VX[0])) fail("exchange", r.energy_density, EX[0]);
      ++checks;
    }
  }
  std::mt19937 rng(7u);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (double up : grid) for (double down : grid) {
    if (std::fabs(up) > 20.0 && std::fabs(down) > 20.0) continue;
    for (const LegacyMode& m : modes) {
      dc_Type = m.dc_type;
      for (int q = 0; q < 3; ++q) {
        double XC[2], X[2], C[2];
        XC_CA_LSDA(m.scf_iter, up, down, XC, X, C, q);
        const LsdaResult r = ca_lsda({up, down}, static_cast<LsdaQuantity>(q), m.symmetrize);
        for (int s = 0; s < 2; ++s)
          if (!same_bits(r.xc[s], XC[s]) || !same_bits(r.exchange[s], X[s]) || !same_bits(r.correlation[s], C[s]))
            fail("CA LSDA q=" + std::to_string(q) + " up=" + std::to_string(up) + " down=" + std::to_string(down), r.xc[s], XC[s]);
        ++checks;
      }
      {
        double dens[2] = {up, down}, Ec[1], Vc[2];
        XC_PW92C(m.scf_iter, dens, Ec, Vc);
        const Pw92Result r = pw92_correlation({up, down}, m.symmetrize);
        if (!same_bits(r.energy_density, Ec[0]) || !same_bits(r.potential[0], Vc[0]) || !same_bits(r.potential[1], Vc[1]) ||
            !same_bits(r.density_used.up, dens[0]) || !same_bits(r.density_used.down, dens[1]))
          fail("PW92 up=" + std::to_string(up) + " down=" + std::to_string(down), r.energy_density, Ec[0]);
        ++checks;
      }
      for (int g = 0; g < 4; ++g) {
        double GDENS[3][2];
        SpinGradient grad;
        for (int d = 0; d < 3; ++d) for (int s = 0; s < 2; ++s) {
          const double scale = g == 0 ? 0.0 : g == 1 ? 1e-15 : g == 2 ? 0.1 * std::fabs(up + down) : 30.0;
          GDENS[d][s] = scale * u(rng);
          grad.component[d][s] = GDENS[d][s];
        }
        double dens[2] = {up, down}, Exc[2], DEXDD[2], DECDD[2], DEXDGD[3][2], DECDGD[3][2];
        XC_PBE(m.scf_iter, dens, GDENS, Exc, DEXDD, DECDD, DEXDGD, DECDGD);
        const PbeResult r = pbe({up, down}, grad, m.symmetrize);
        bool ok = same_bits(r.exchange_energy_density, Exc[0]) && same_bits(r.correlation_energy_density, Exc[1]) &&
                  same_bits(r.density_used.up, dens[0]) && same_bits(r.density_used.down, dens[1]);
        for (int s = 0; s < 2; ++s) {
          ok = ok && same_bits(r.dexchange_ddensity[s], DEXDD[s]) && same_bits(r.dcorrelation_ddensity[s], DECDD[s]);
          for (int d = 0; d < 3; ++d)
            ok = ok && same_bits(r.dexchange_dgradient[d][s], DEXDGD[d][s]) && same_bits(r.dcorrelation_dgradient[d][s], DECDGD[d][s]);
        }
        if (!ok) fail("PBE up=" + std::to_string(up) + " down=" + std::to_string(down) + " gradient set " + std::to_string(g), r.exchange_energy_density, Exc[0]);
        // the C interface writes the same values and floors the densities the same way
        double cdens[2] = {up, down}, cexc[2], cdexdd[2], cdecdd[2], cdexdgd[3][2], cdecdgd[3][2];
        require(openmx_xc_pbe(cdens, GDENS, m.symmetrize, cexc, cdexdd, cdecdd, cdexdgd, cdecdgd) == 1, "C PBE refused");
        require(same_bits(cexc[0], Exc[0]) && same_bits(cexc[1], Exc[1]) && same_bits(cdens[0], dens[0]) && same_bits(cdecdgd[2][1], DECDGD[2][1]), "C PBE differs");
        ++checks;
      }
    }
  }
  dc_Type = 0;
  std::printf("PASS xc legacy: %ld bitwise comparisons\n", checks);
}

void analytic() {
  const double PI = 3.1415926535897932384626;
  // Slater exchange of the uniform gas: eps_x = -(3/4)(3/pi)^{1/3} rho^{1/3}, mu_x = (4/3) eps_x
  for (double rho : {1e-6, 1e-3, 0.01, 0.1, 1.0, 10.0}) {
    const ExchangeResult r = lda_exchange(rho);
    const double closed = -0.75 * std::pow(3.0 / PI, 1.0 / 3.0) * std::pow(rho, 1.0 / 3.0);
    if (!close(r.energy_density, closed, 2e-14)) fail("Slater exchange closed form", r.energy_density, closed);
    if (!close(r.potential, 4.0 / 3.0 * r.energy_density, 1e-13)) fail("Slater potential 4/3 rule", r.potential, 4.0 / 3.0 * r.energy_density);
  }
  // Perdew-Wang 92, zeta = 0: eq. (10) of PRB 45, 13244 written in its published form
  const double A = 0.031091, a1 = 0.21370, b1 = 7.5957, b2 = 3.5876, b3 = 1.6382, b4 = 0.49294;
  for (double rs : {0.5, 1.0, 2.0, 5.0, 10.0, 50.0}) {
    const double rho = 3.0 / (4.0 * PI * rs * rs * rs);
    const double g = -2.0 * A * (1.0 + a1 * rs) *
                     std::log(1.0 + 1.0 / (2.0 * A * (b1 * std::sqrt(rs) + b2 * rs + b3 * std::pow(rs, 1.5) + b4 * rs * rs)));
    const Pw92Result r = pw92_correlation({rho / 2.0, rho / 2.0}, false);
    if (!close(r.energy_density, g, 1e-9)) fail("PW92 unpolarized eq. (10) rs=" + std::to_string(rs), r.energy_density, g);
    require(close(r.potential[0], r.potential[1], 1e-15), "PW92 unpolarized potentials differ");
  }
  // Ceperley-Alder LSDA at zero polarization is the LDA; symmetrized calls see zeta = 0 too
  for (double rho : {1e-8, 1e-3, 0.05, 0.3, 2.0}) {
    for (int q = 0; q < 3; ++q) {
      const LsdaResult even = ca_lsda({rho / 2.0, rho / 2.0}, static_cast<LsdaQuantity>(q), false);
      const double lda = ceperley_alder(rho, static_cast<CeperleyAlderQuantity>(q));
      if (!close(even.xc[0], lda, 1e-13)) fail("LSDA zero polarization vs LDA q=" + std::to_string(q), even.xc[0], lda);
      const LsdaResult sym = ca_lsda({0.9 * rho, 0.1 * rho}, static_cast<LsdaQuantity>(q), true);
      if (!close(sym.xc[0], lda, 1e-13) || !close(sym.xc[1], lda, 1e-13)) fail("symmetrized LSDA vs LDA", sym.xc[0], lda);
      require(close(even.xc[0], even.exchange[0] + even.correlation[0], 1e-15), "LSDA xc is not exchange + correlation");
    }
    // fully polarized exchange is 2^{1/3} times the paramagnetic one
    const LsdaResult polar = ca_lsda({rho, 0.0}, LsdaQuantity::energy_density, false);
    const LsdaResult para = ca_lsda({rho / 2.0, rho / 2.0}, LsdaQuantity::energy_density, false);
    if (!close(polar.exchange[0], std::pow(2.0, 1.0 / 3.0) * para.exchange[0], 1e-9)) fail("polarized exchange 2^(1/3)", polar.exchange[0], para.exchange[0]);
  }
  // PBE without a gradient reduces to Slater exchange and PW92 correlation, in energy and potential
  for (double rho : {1e-4, 0.01, 0.2, 3.0}) {
    const SpinGradient none{};
    for (double fraction : {0.5, 0.8}) {
      const SpinDensity d{fraction * rho, (1.0 - fraction) * rho};
      const PbeResult r = pbe(d, none, false);
      const Pw92Result c = pw92_correlation(d, false);
      // spin-scaled exchange: (rho_up eps_x(2 rho_up) + rho_down eps_x(2 rho_down)) / rho
      const double ex = (d.up * lda_exchange(2.0 * d.up).energy_density + d.down * lda_exchange(2.0 * d.down).energy_density) / rho;
      if (!close(r.exchange_energy_density, ex, 1e-10)) fail("PBE zero gradient exchange", r.exchange_energy_density, ex);
      if (!close(r.correlation_energy_density, c.energy_density, 1e-10)) fail("PBE zero gradient correlation", r.correlation_energy_density, c.energy_density);
      for (int s = 0; s < 2; ++s) {
        const double spin = s ? d.down : d.up;
        if (!close(r.dexchange_ddensity[s], lda_exchange(2.0 * spin).potential, 1e-10)) fail("PBE zero gradient exchange potential", r.dexchange_ddensity[s], lda_exchange(2.0 * spin).potential);
        if (!close(r.dcorrelation_ddensity[s], c.potential[s], 1e-10)) fail("PBE zero gradient correlation potential", r.dcorrelation_ddensity[s], c.potential[s]);
      }
    }
  }
  std::printf("PASS xc analytic\n");
}

// central difference of f at x with step h (relative)
template<class F> double derivative(F&& f, double x, double h) { return (f(x + h) - f(x - h)) / (2.0 * h); }

void derivatives() {
  // LDA: mu = d(rho eps)/d rho, energy_derivative = d eps/d rho, energy_minus_potential = eps - mu
  for (double rho : {1e-5, 1e-3, 0.02, 0.1, 0.9, 4.0}) {
    const double h = 1e-5 * rho;
    const double mu = derivative([](double x) { return x * ceperley_alder(x, CeperleyAlderQuantity::energy_density); }, rho, h);
    const double deps = derivative([](double x) { return ceperley_alder(x, CeperleyAlderQuantity::energy_density); }, rho, h);
    const double eps = ceperley_alder(rho, CeperleyAlderQuantity::energy_density);
    if (!close(ceperley_alder(rho, CeperleyAlderQuantity::potential), mu, 1e-7)) fail("LDA potential vs numerical", ceperley_alder(rho, CeperleyAlderQuantity::potential), mu);
    if (!close(ceperley_alder(rho, CeperleyAlderQuantity::energy_derivative), deps, 1e-7)) fail("LDA energy derivative vs numerical", ceperley_alder(rho, CeperleyAlderQuantity::energy_derivative), deps);
    if (!close(ceperley_alder(rho, CeperleyAlderQuantity::energy_minus_potential), eps - mu, 1e-7)) fail("LDA eps - mu", ceperley_alder(rho, CeperleyAlderQuantity::energy_minus_potential), eps - mu);
  }
  // LSDA and PW92: mu_s = d(rho eps)/d rho_s for both spins
  for (double rho : {1e-4, 0.01, 0.3, 2.0}) for (double fraction : {0.5, 0.7, 0.95}) {
    const SpinDensity d{fraction * rho, (1.0 - fraction) * rho};
    for (int s = 0; s < 2; ++s) {
      const double h = 1e-5 * rho;
      auto lsda_energy = [&](double x) {
        const SpinDensity y{s ? d.up : x, s ? x : d.down};
        return (y.up + y.down) * ca_lsda(y, LsdaQuantity::energy_density, false).xc[0];
      };
      const double mu = derivative(lsda_energy, s ? d.down : d.up, h);
      const LsdaResult r = ca_lsda(d, LsdaQuantity::potential, false);
      if (!close(r.xc[s], mu, 1e-6)) fail("LSDA potential vs numerical spin " + std::to_string(s), r.xc[s], mu);
      auto pw92_energy = [&](double x) {
        const SpinDensity y{s ? d.up : x, s ? x : d.down};
        return (y.up + y.down) * pw92_correlation(y, false).energy_density;
      };
      const double muc = derivative(pw92_energy, s ? d.down : d.up, h);
      if (!close(pw92_correlation(d, false).potential[s], muc, 1e-6)) fail("PW92 potential vs numerical spin " + std::to_string(s), pw92_correlation(d, false).potential[s], muc);
    }
  }
  // PBE: the density and gradient derivatives of rho*eps_x and rho*eps_c
  std::mt19937 rng(11u);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (double rho : {5e-3, 0.05, 0.4}) for (double fraction : {0.5, 0.75}) {
    SpinGradient g;
    for (int dd = 0; dd < 3; ++dd) for (int s = 0; s < 2; ++s) g.component[dd][s] = 0.3 * rho * u(rng);
    const SpinDensity d{fraction * rho, (1.0 - fraction) * rho};
    auto energies = [&](const SpinDensity& y, const SpinGradient& gg) {
      const PbeResult r = pbe(y, gg, false);
      const double n = y.up + y.down;
      return std::pair<double, double>{n * r.exchange_energy_density, n * r.correlation_energy_density};
    };
    const PbeResult r = pbe(d, g, false);
    for (int s = 0; s < 2; ++s) {
      const double x0 = s ? d.down : d.up, h = 1e-5 * rho;
      auto at = [&](double x) { const SpinDensity y{s ? d.up : x, s ? x : d.down}; return energies(y, g); };
      const double dex = (at(x0 + h).first - at(x0 - h).first) / (2.0 * h);
      const double dec = (at(x0 + h).second - at(x0 - h).second) / (2.0 * h);
      if (!close(r.dexchange_ddensity[s], dex, 1e-6)) fail("PBE d(rho eps_x)/d rho_s", r.dexchange_ddensity[s], dex);
      if (!close(r.dcorrelation_ddensity[s], dec, 1e-6)) fail("PBE d(rho eps_c)/d rho_s", r.dcorrelation_ddensity[s], dec);
      for (int dd = 0; dd < 3; ++dd) {
        const double hg = 1e-5 * rho;
        auto atg = [&](double x) { SpinGradient gg = g; gg.component[dd][s] = x; return energies(d, gg); };
        const double g0 = g.component[dd][s];
        const double dgx = (atg(g0 + hg).first - atg(g0 - hg).first) / (2.0 * hg);
        const double dgc = (atg(g0 + hg).second - atg(g0 - hg).second) / (2.0 * hg);
        if (!close(r.dexchange_dgradient[dd][s], dgx, 1e-6)) fail("PBE d(rho eps_x)/d grad", r.dexchange_dgradient[dd][s], dgx);
        if (!close(r.dcorrelation_dgradient[dd][s], dgc, 1e-6)) fail("PBE d(rho eps_c)/d grad", r.dcorrelation_dgradient[dd][s], dgc);
      }
    }
  }
  std::printf("PASS xc derivatives\n");
}

void invalid() {
  require(ceperley_alder(0.0, CeperleyAlderQuantity::potential) == 0.0, "LDA at zero density");
  require(ceperley_alder(-1.0, CeperleyAlderQuantity::energy_density) == 0.0, "LDA at negative density");
  require(openmx_xc_ceperley_alder(0.1, -1) == 0.0 && openmx_xc_ceperley_alder(0.1, 4) == 0.0, "unknown LDA quantity");
  const LsdaResult below = ca_lsda({4e-16, 4e-16}, LsdaQuantity::potential, false);
  require(below.xc[0] == 0.0 && below.xc[1] == 0.0 && below.exchange[0] == 0.0 && below.correlation[1] == 0.0, "LSDA below the floor");
  double xc[2], x[2], c[2];
  require(openmx_xc_ca_lsda(0.1, 0.1, 3, 0, xc, x, c) == 0, "unknown LSDA quantity accepted");
  double dens[2] = {0.0, 0.0}, ec[1], vc[2];
  require(openmx_xc_pw92_correlation(dens, 0, ec, vc) == 1 && dens[0] == 0.5e-14 && dens[1] == 0.5e-14, "PW92 floor not applied");
  require(std::isfinite(ec[0]) && std::isfinite(vc[0]) && std::isfinite(vc[1]), "PW92 at the floor not finite");
  require(openmx_xc_pw92_correlation(dens, 0, nullptr, vc) == 0, "PW92 without an output accepted");
  double ex[1], vx[1];
  require(openmx_xc_lda_exchange(1e-20, ex, vx) == 1 && std::isfinite(ex[0]) && ex[0] < 0.0, "exchange below the floor");
  require(openmx_xc_lda_exchange(0.1, ex, nullptr) == 0, "exchange without an output accepted");
  double pd[2] = {-1.0, -1.0}, gd[3][2] = {{0, 0}, {0, 0}, {0, 0}}, exc[2], dexdd[2], decdd[2], dexdgd[3][2], decdgd[3][2];
  require(openmx_xc_pbe(pd, gd, 0, exc, dexdd, decdd, dexdgd, decdgd) == 1 && pd[0] == 0.5e-14 && pd[1] == 0.5e-14, "PBE floor not applied");
  require(std::isfinite(exc[0]) && std::isfinite(exc[1]), "PBE at the floor not finite");
  require(openmx_xc_pbe(nullptr, gd, 0, exc, dexdd, decdd, dexdgd, decdgd) == 0, "PBE without a density accepted");
  require(openmx_xc_pbe(pd, gd, 0, exc, dexdd, decdd, nullptr, decdgd) == 0, "PBE without an output accepted");
  std::printf("PASS xc invalid\n");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string name = argc > 1 ? argv[1] : "legacy";
    if (name == "legacy") legacy();
    else if (name == "analytic") analytic();
    else if (name == "derivatives") derivatives();
    else if (name == "invalid") invalid();
    else throw std::runtime_error("unknown xc test case");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
