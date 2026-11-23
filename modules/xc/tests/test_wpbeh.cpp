// Tests of the short-range omega-PBE exchange of the xc package against
// multi-precision references stored as text fixtures (tools/wpbeh_*.py).
#include <openmx/xc/functionals.hpp>
#include <openmx/xc/xc.h>

#include "special_functions.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace openmx::xc;

const double pi = 3.14159265358979323846;

void require(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }
bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
double relative_difference(double value, double reference) {
  if (value == reference) return 0.0;
  return std::fabs(value - reference) / std::fabs(reference);
}

// A fixture is a text file of rows of numbers.  Lines that do not start with
// a number are headers and comments; "npoints" and "ncolumns" are checked.
struct Table {
  std::vector<std::vector<double>> rows;
};

Table read_table(const std::string& path, std::size_t columns) {
  std::ifstream in(path);
  require(in.good(), "cannot open the fixture " + path);
  Table table;
  long declared_points = -1, declared_columns = -1;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream words(line);
    std::string first;
    if (!(words >> first)) continue;
    if (first == "npoints") { words >> declared_points; continue; }
    if (first == "ncolumns") { words >> declared_columns; continue; }
    if (std::string("+-0123456789").find(first[0]) == std::string::npos) continue;
    std::vector<double> row;
    std::istringstream numbers(line);
    std::string word;
    while (numbers >> word) {
      char* end = nullptr;
      const double value = std::strtod(word.c_str(), &end);
      require(end && *end == '\0', "malformed number in " + path + ": " + word);
      row.push_back(value);
    }
    require(row.size() == columns, "unexpected number of columns in " + path);
    table.rows.push_back(row);
  }
  require(declared_columns == static_cast<long>(columns), "ncolumns of " + path + " does not match");
  require(declared_points == static_cast<long>(table.rows.size()), "npoints of " + path + " does not match");
  return table;
}

// exp(x) E1(x) and erfcx(x): values within 1e-14 of the 50-digit references
// between 1e-30 and 1e8 (the requirement), derivatives within 1e-13.
void special(const std::string& path) {
  const Table table = read_table(path, 5);
  require(table.rows.size() >= 700, "the special-function fixture is too small");
  const double value_tolerance = 1e-14, derivative_tolerance = 1e-13;
  double worst[4] = {0.0, 0.0, 0.0, 0.0}, lowest = 1.0, highest = 1.0;
  for (const std::vector<double>& row : table.rows) {
    const double x = row[0];
    lowest = std::fmin(lowest, x);
    highest = std::fmax(highest, x);
    const detail::ValueAndDerivative e = detail::e1_scaled(x), c = detail::erfcx(x);
    require(same_bits(e1_scaled(x), e.value) && same_bits(erfcx(x), c.value), "public and internal values differ");
    const double found[4] = {e.value, e.derivative, c.value, c.derivative};
    for (int i = 0; i < 4; ++i) {
      const double difference = relative_difference(found[i], row[1 + i]);
      worst[i] = std::fmax(worst[i], difference);
      if (!(difference <= (i % 2 ? derivative_tolerance : value_tolerance))) {
        std::fprintf(stderr, "column %d at x = %.17g: %.17g vs %.17g (relative %.3g)\n", 1 + i, x, found[i], row[1 + i], difference);
        throw std::runtime_error("special function outside the tolerance");
      }
    }
  }
  require(lowest <= 1e-30 && highest >= 1e8, "the fixture does not cover 1e-30 to 1e8");

  // outside the fixture: limits, the reflection of erfcx and the invalid arguments
  const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
  require(e1_scaled(0.0) == inf && std::isnan(e1_scaled(-1.0)) && std::isnan(e1_scaled(nan)), "exp(x) E1(x) at invalid arguments");
  require(e1_scaled(inf) == 0.0 && erfcx(inf) == 0.0, "limits at infinity");
  require(erfcx(0.0) == 1.0 && std::isnan(erfcx(nan)), "erfcx at 0 and NaN");
  for (double x : {1e-20, 0.3, 1.0, 1.9, 2.0, 2.7, 5.0}) {
    require(relative_difference(erfcx(-x) + erfcx(x), 2.0 * std::exp(x * x)) <= 4e-15, "erfcx(-x) + erfcx(x) = 2 exp(x^2)");
  }
  // large arguments: exp(x) E1(x) -> 1/x - 1/x^2, erfcx(x) -> 1/(x sqrt(pi)); the derivatives keep their accuracy
  const detail::ValueAndDerivative e = detail::e1_scaled(1e12), c = detail::erfcx(1e12);
  require(relative_difference(e.value, 1e-12) <= 2e-12 && relative_difference(e.derivative, -1e-24) <= 3e-12, "exp(x) E1(x) at 1e12");
  require(relative_difference(c.value, 0.56418958354775628695e-12) <= 1e-15
          && relative_difference(c.derivative, -0.56418958354775628695e-24) <= 1e-15, "erfcx at 1e12");
  std::printf("PASS xc wpbeh-special (%zu points; largest relative error: exp(x)E1(x) %.2g, derivative %.2g, erfcx %.2g, derivative %.2g)\n",
              table.rows.size(), worst[0], worst[1], worst[2], worst[3]);
}

SpinGradient gradient_of(const double up[3], const double down[3]) {
  SpinGradient g;
  for (int d = 0; d < 3; ++d) { g.component[d][0] = up[d]; g.component[d][1] = down[d]; }
  return g;
}

// The short-range exchange against the 50-digit evaluation of the definition
// (tools/wpbeh_reference.py --dump).  Columns of the fixture:
//   0 rho_up  1 rho_dn  2-4 grad rho_up  5-7 grad rho_dn  8 e_up  9 e_dn  10 exc
//   11 dexdd_up  12 dexdd_dn  13-15 dexdgd_up  16-18 dexdgd_dn  19 vsigma_upup
//   20 vsigma_dndn  21 w_up  22 s_up  23 w_dn  24 s_dn  25 tol_up  26 tol_dn
// with the three comparisons stated in its header.
void fixture(const std::string& path) {
  double omega = -1.0;
  {
    std::ifstream in(path);
    std::string line, word;
    while (std::getline(in, line)) {
      std::istringstream words(line);
      if (words >> word && word == "omega") words >> omega;
    }
  }
  require(omega > 0.0, "the fixture does not state omega");
  const Table table = read_table(path, 27);
  require(table.rows.size() == 160, "unexpected number of fixture points");
  double worst = 0.0;
  std::size_t zero_gradient_derivatives = 0;
  for (std::size_t n = 0; n < table.rows.size(); ++n) {
    const std::vector<double>& row = table.rows[n];
    const SpinDensity density{row[0], row[1]};
    const SpinGradient gradient = gradient_of(&row[2], &row[5]);
    const ShortRangeExchangeResult r = wpbeh_short_range_exchange(density, gradient, omega, false);
    require(same_bits(r.density_used.up, row[0]) && same_bits(r.density_used.down, row[1]), "fixture density below the floor");

    // energy per volume: |d e| <= tol_up |e_up| + tol_dn |e_dn|
    const double energy = r.exchange_energy_density * (row[0] + row[1]);
    const double allowed_energy = row[25] * std::fabs(row[8]) + row[26] * std::fabs(row[9]);
    double ratio = std::fabs(energy - (row[8] + row[9])) / allowed_energy;
    const char* what = "energy";
    for (int s = 0; s < 2; ++s) {
      const double tol = row[25 + s], e_s = row[8 + s];
      // dexdd: relative difference at most 10 tol
      const double d_density = relative_difference(r.dexchange_ddensity[s], row[11 + s]) / (10.0 * tol);
      if (d_density > ratio) { ratio = d_density; what = "dexdd"; }
      // dexdgd: |difference| <= 10 tol max(|dexdgd_s|, 1e-6 |e_s| / |grad rho_s|)
      const double* g = &row[2 + 3 * s];
      const double* reference = &row[13 + 3 * s];
      const double gradient_norm = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
      const double reference_norm = std::sqrt(reference[0] * reference[0] + reference[1] * reference[1] + reference[2] * reference[2]);
      const double scale = std::fmax(reference_norm, 1e-6 * std::fabs(e_s) / gradient_norm);
      if (reference_norm == 0.0) ++zero_gradient_derivatives;
      for (int d = 0; d < 3; ++d) {
        const double d_gradient = std::fabs(r.dexchange_dgradient[d][s] - reference[d]) / (10.0 * tol * scale);
        if (d_gradient > ratio) { ratio = d_gradient; what = "dexdgd"; }
        // above s_raw = 15 the rescaled gradient is constant: the derivative vanishes exactly
        if (reference_norm == 0.0) require(r.dexchange_dgradient[d][s] == 0.0, "dexdgd not zero where the reference is");
      }
    }
    if (!(ratio <= 1.0)) {
      std::fprintf(stderr, "fixture point %zu: %s deviates by %.3g of the allowed difference\n", n, what, ratio);
      throw std::runtime_error("short-range exchange outside the tolerance of the fixture");
    }
    worst = std::fmax(worst, ratio);

    // the C entry returns the same bits and rewrites the density as the PBE entry does
    double dens[2] = {row[0], row[1]}, gdens[3][2], ex[1], dexdd[2], dexdgd[3][2];
    for (int d = 0; d < 3; ++d) { gdens[d][0] = gradient.component[d][0]; gdens[d][1] = gradient.component[d][1]; }
    require(openmx_xc_wpbeh_short_range(dens, gdens, 0, omega, ex, dexdd, dexdgd) == 1, "C short-range exchange failed");
    bool same = same_bits(ex[0], r.exchange_energy_density);
    for (int s = 0; s < 2; ++s) {
      same = same && same_bits(dexdd[s], r.dexchange_ddensity[s]);
      for (int d = 0; d < 3; ++d) same = same && same_bits(dexdgd[d][s], r.dexchange_dgradient[d][s]);
    }
    require(same, "C and C++ short-range exchange differ");
  }
  require(zero_gradient_derivatives == 20, "the fixture should contain 20 channels above s_raw = 15");

  // Floors, the vanishing gradient and the symmetrization follow the exchange part of pbe().
  {
    const double zero[3] = {0.0, 0.0, 0.0}, some[3] = {0.02, -0.01, 0.03};
    const ShortRangeExchangeResult below = wpbeh_short_range_exchange({-1.0, 0.0}, gradient_of(zero, zero), omega, false);
    const PbeResult pbe_below = pbe({-1.0, 0.0}, gradient_of(zero, zero), false);
    require(below.density_used.up == 0.5e-14 && below.density_used.down == 0.5e-14
            && same_bits(below.density_used.up, pbe_below.density_used.up), "density floor of the short-range exchange");
    require(std::isfinite(below.exchange_energy_density) && below.exchange_energy_density < 0.0
            && std::isfinite(below.dexchange_ddensity[0]) && std::isfinite(below.dexchange_ddensity[1]),
            "short-range exchange at the density floor is not finite");
    const ShortRangeExchangeResult flat = wpbeh_short_range_exchange({0.3, 0.1}, gradient_of(zero, some), omega, false);
    for (int d = 0; d < 3; ++d) {
      require(flat.dexchange_dgradient[d][0] == 0.0, "gradient derivative at zero gradient");
      require(std::isfinite(flat.dexchange_dgradient[d][1]) && flat.dexchange_dgradient[d][1] != 0.0, "gradient derivative of the other spin");
    }
    const double tiny[3] = {1e-12, 0.0, 0.0};
    const ShortRangeExchangeResult nearly_flat = wpbeh_short_range_exchange({0.3, 0.1}, gradient_of(tiny, some), omega, false);
    require(relative_difference(flat.exchange_energy_density, nearly_flat.exchange_energy_density) <= 1e-14
            && relative_difference(flat.dexchange_ddensity[0], nearly_flat.dexchange_ddensity[0]) <= 1e-13,
            "zero gradient is not the limit of a small gradient");
    const double up[3] = {0.05, 0.01, -0.02};
    const ShortRangeExchangeResult symmetric = wpbeh_short_range_exchange({0.3, 0.1}, gradient_of(up, some), omega, true);
    const double mean[3] = {(up[0] + some[0]) / 2.0, (up[1] + some[1]) / 2.0, (up[2] + some[2]) / 2.0};
    const ShortRangeExchangeResult averaged = wpbeh_short_range_exchange({0.2, 0.2}, gradient_of(mean, mean), omega, false);
    require(same_bits(symmetric.exchange_energy_density, averaged.exchange_energy_density)
            && same_bits(symmetric.dexchange_ddensity[0], averaged.dexchange_ddensity[1])
            && same_bits(symmetric.dexchange_dgradient[2][1], averaged.dexchange_dgradient[2][0])
            && symmetric.density_used.up == 0.3 && symmetric.density_used.down == 0.1, "symmetrized short-range exchange");
    double dens[2] = {0.3, 0.1}, gdens[3][2] = {{0, 0}, {0, 0}, {0, 0}}, ex[1], dexdd[2], dexdgd[3][2];
    require(openmx_xc_wpbeh_short_range(nullptr, gdens, 0, omega, ex, dexdd, dexdgd) == 0
            && openmx_xc_wpbeh_short_range(dens, gdens, 0, omega, ex, dexdd, nullptr) == 0, "short-range exchange without an array accepted");
  }

  // Independent of the fixture: the derivatives against central differences of the energy per
  // volume, on both sides of the changes of the rescaling (s_raw = 1 and 15) and above w = 14, and
  // the uniform-gas limit of the unscreened functional, f(w = 0, s -> 0) = 1 within 1e-9.
  // Between w = 4 and w = 14 the rounding error of the energy is too large for a difference
  // quotient, and above w = 14 the energy depends too weakly on the gradient for one; the fixture
  // covers both.  Each case is (rho_up, rho_dn, s_raw of the up channel).
  {
    const double rho_at_w14 = std::pow(omega / 14.0, 3) / (6.0 * pi * pi);
    const double cases[][3] = {{0.31, 0.07, 0.4}, {0.02, 0.5, 0.95}, {0.02, 0.5, 1.05}, {1e-3, 2e-3, 7.0}, {1e-3, 2e-3, 14.5},
                               {1e-3, 2e-3, 20.0}, {rho_at_w14 * 0.9, rho_at_w14 * 0.5, 0.6}, {2e-6, 3e-6, 2.5}};
    for (const double* c : cases) {
      // gradients of reduced size c[2] in the up channel, half of it in the down channel
      const double kf_factor = 2.0 * std::cbrt(6.0 * pi * pi);
      const double gu = c[2] * kf_factor * std::pow(c[0], 4.0 / 3.0), gd = 0.5 * c[2] * kf_factor * std::pow(c[1], 4.0 / 3.0);
      const double up[3] = {0.6 * gu, -0.48 * gu, 0.64 * gu}, down[3] = {-0.28 * gd, 0.96 * gd, 0.0};
      const SpinDensity density{c[0], c[1]};
      const SpinGradient gradient = gradient_of(up, down);
      auto energy = [&](const SpinDensity& d, const SpinGradient& g) {
        return wpbeh_short_range_exchange(d, g, omega, false).exchange_energy_density * (d.up + d.down);
      };
      const ShortRangeExchangeResult r = wpbeh_short_range_exchange(density, gradient, omega, false);
      for (int s = 0; s < 2; ++s) {
        const double rho0 = s ? density.down : density.up, h = 1e-4 * rho0;
        auto at_density = [&](double x) { return energy(SpinDensity{s ? density.up : x, s ? x : density.down}, gradient); };
        const double numerical = (at_density(rho0 + h) - at_density(rho0 - h)) / (2.0 * h);
        require(relative_difference(r.dexchange_ddensity[s], numerical) <= 2e-6, "d e/d rho_s against a central difference");
        const double w = omega / std::cbrt(6.0 * pi * pi * rho0);
        for (int d = 0; d < 3 && w <= 14.0; ++d) {
          const double g0 = gradient.component[d][s];
          if (g0 == 0.0) continue;
          const double hg = 1e-4 * std::fabs(g0);
          auto at_gradient = [&](double x) { SpinGradient g = gradient; g.component[d][s] = x; return energy(density, g); };
          const double numerical_g = (at_gradient(g0 + hg) - at_gradient(g0 - hg)) / (2.0 * hg);
          require(std::fabs(r.dexchange_dgradient[d][s] - numerical_g)
                  <= 2e-6 * std::fmax(std::fabs(numerical_g), 1e-6 * std::fabs(energy(density, gradient)) / std::fabs(g0)),
                  "d e/d grad rho_s against a central difference");
        }
      }
    }
    const double zero[3] = {0.0, 0.0, 0.0};
    const double rho = 0.4;
    const ShortRangeExchangeResult uniform = wpbeh_short_range_exchange({rho / 2.0, rho / 2.0}, gradient_of(zero, zero), 0.0, false);
    const double lda = -0.75 * std::cbrt(3.0 / pi) * std::cbrt(rho);
    require(relative_difference(uniform.exchange_energy_density, lda) <= 1e-9, "uniform-gas limit of the unscreened functional");
  }
  std::printf("PASS xc wpbeh-fixture (%zu points, omega %.3g; largest deviation / allowed %.3g)\n", table.rows.size(), omega, worst);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string name = argc > 1 ? argv[1] : "";
    const std::string fixture = argc > 2 ? argv[2] : "";
    if (name == "wpbeh-special") special(fixture);
    else if (name == "wpbeh-fixture") ::fixture(fixture);
    else throw std::runtime_error("unknown wpbeh test case");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
