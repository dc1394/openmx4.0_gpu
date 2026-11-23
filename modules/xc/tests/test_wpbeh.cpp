// Tests of the short-range omega-PBE exchange of the xc package against
// multi-precision references stored as text fixtures (tools/wpbeh_*.py).
#include <openmx/xc/functionals.hpp>

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
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string name = argc > 1 ? argv[1] : "";
    const std::string fixture = argc > 2 ? argv[2] : "";
    if (name == "wpbeh-special") special(fixture);
    else throw std::runtime_error("unknown wpbeh test case");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
