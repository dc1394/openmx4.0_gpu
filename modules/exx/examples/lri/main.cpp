// The auxiliary basis and the local RI for PAOs of the main program.
//
//   lri_example A.pao BASIS [C.pao BASIS DISTANCE] [options]
//
// BASIS names the PAOs as the input of the main program does (s2p2d1: the
// first two s, two p and one d function of the file).  For every species the
// example prints the number of auxiliary basis functions (ABFs) that the
// principal component analysis keeps at several thresholds, and how well the
// products of the PAOs of one atom are expanded.  With a second species and a
// distance in bohr it prints the local RI of the pair: the self-interaction
// of the expanded products in the Coulomb metric, which approaches the exact
// value from below, the same sum with the short-range interaction of HSE06,
// and the largest error of the charge of an expanded product.
//
// Options
//   --generators BASIS[,BASIS]  make the ABFs from the products of these
//                               functions of the .pao file instead of the
//                               PAOs (a larger set shows what the ABFs of
//                               the PAOs miss)
//   --thresholds T[,T...]       thresholds of the analysis
//                               (default 1e-2,1e-3,1e-4,1e-5,1e-6)
//   --radial N                  intervals of the radial grid (default 900)
//   --reciprocal KMAX,N         largest wave number in 1/bohr and intervals
//                               of the reciprocal grid (default 60,900)
//   --omega W                   range separation of the short-range
//                               interaction in 1/bohr (default 0.11)
#include <openmx/basis/pao_file.hpp>
#include <openmx/exx/abf.hpp>
#include <openmx/exx/kernel.hpp>
#include <openmx/exx/lri.hpp>
#include <openmx/exx/two_center.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace openmx::exx;

void print_usage(std::FILE* stream)
{
  std::fprintf(stream,"usage: lri_example A.pao BASIS [C.pao BASIS DISTANCE] [--generators BASIS[,BASIS]]\n"
                      "                   [--thresholds T[,T...]] [--radial N] [--reciprocal KMAX,N] [--omega W]\n");
}

std::vector<std::string> split(const std::string& text)
{
  std::vector<std::string> parts;
  std::size_t begin = 0;
  while (true) {
    const std::size_t end = text.find(',',begin);
    parts.push_back(text.substr(begin,end == std::string::npos ? end : end-begin));
    if (end == std::string::npos) return parts;
    begin = end+1;
  }
}

double number(const std::string& text)
{
  char* end = nullptr;
  const double value = std::strtod(text.c_str(),&end);
  if (text.empty() || *end != '\0' || !std::isfinite(value)) throw std::invalid_argument("not a number: "+text);
  return value;
}

int count(const std::string& text)
{
  const double value = number(text);
  if (value < 1.0 || value > 1e6 || value != std::floor(value)) throw std::invalid_argument("not a positive integer: "+text);
  return static_cast<int>(value);
}

// "s2p2d1" -> {2, 2, 1}: the number of radial functions for l = 0, 1, ...
std::vector<int> parse_basis(const std::string& text)
{
  const std::string letters = "spdfg";
  std::vector<int> counts;
  std::size_t i = 0;
  while (i < text.size()) {
    if (letters.find(text[i]) != counts.size())
      throw std::invalid_argument("a basis is written like s2p2d1, in the order s, p, d, f, g: "+text);
    std::size_t digits = 0;
    int value = 0;
    for (++i; i < text.size() && text[i] >= '0' && text[i] <= '9' && digits < 3; ++i, ++digits) value = 10*value+(text[i]-'0');
    if (digits == 0) throw std::invalid_argument("a basis needs a number after every letter: "+text);
    counts.push_back(value);
  }
  if (counts.empty()) throw std::invalid_argument("the basis is empty");
  return counts;
}

std::vector<RadialFunction> sample(const openmx::basis::PaoFile& pao, const RadialGrid& grid, const std::string& basis)
{
  const std::vector<int> counts = parse_basis(basis);
  std::vector<MeshFunction> shells;
  for (std::size_t l = 0; l < counts.size(); ++l)
    for (int multiplicity = 0; multiplicity < counts[l]; ++multiplicity)
      shells.push_back({static_cast<int>(l),pao.orbital(static_cast<int>(l),multiplicity)});
  if (shells.empty()) throw std::invalid_argument("the basis has no functions: "+basis);
  return sample_orbitals(grid,pao.radii(),shells);
}

struct Options {
  std::vector<double> thresholds{1e-2,1e-3,1e-4,1e-5,1e-6};
  int radial = 900;
  double kmax = 60.0;
  int reciprocal = 900;
  double omega = 0.11;
};

struct Species {
  std::string file, basis, generators;
  RadialGrid grid;
  std::vector<RadialFunction> orbitals;
  std::vector<SpeciesTable> tables; // one per threshold
};

std::string label(const std::string& path)
{
  const std::size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? path : path.substr(slash+1);
}

double dot(const std::vector<double>& a, const std::vector<double>& b)
{
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) sum += a[i]*b[i];
  return sum;
}

// sum over the pairs of PAOs of one atom of (phi phi|1/r|phi phi), from radial
// integrals.
double onsite_exact(const Species& species)
{
  int maximum = 0;
  for (const RadialFunction& f : species.orbitals) maximum = std::max(maximum,f.angular);
  const RealGauntTable gaunt(maximum,maximum);
  double exact = 0.0;
  for (std::size_t k = 0; k < species.orbitals.size(); ++k)
    for (std::size_t j = 0; j <= k; ++j) {
      const RadialFunction& fk = species.orbitals[k];
      const RadialFunction& fj = species.orbitals[j];
      RadialFunction product{0,std::vector<double>(fk.values.size())};
      for (std::size_t i = 0; i < product.values.size(); ++i) product.values[i] = fk.values[i]*fj.values[i];
      for (int order = std::abs(fk.angular-fj.angular); order <= fk.angular+fj.angular; order += 2) {
        product.angular = order;
        const RadialFunction potential = coulomb_potential(species.grid,product);
        double weight = 0.0;
        for (const double theta : gaunt.block(fk.angular,fj.angular,order)) weight += theta*theta;
        exact += (k == j ? 1.0 : 2.0)*weight*radial_integral(species.grid,product.values,potential.values);
      }
    }
  return exact;
}

Species load(const std::string& file, const std::string& basis, const std::string& generators, const Options& options,
             const ReciprocalGrid& reciprocal)
{
  const openmx::basis::PaoFile pao = openmx::basis::PaoFile::from_file(file);
  Species species;
  species.file = file;
  species.basis = basis;
  species.generators = generators;
  species.grid = RadialGrid{0.0,pao.metadata().cutoff/options.radial,options.radial};
  species.orbitals = sample(pao,species.grid,basis);
  const std::vector<RadialFunction> source = generators == basis ? species.orbitals : sample(pao,species.grid,generators);

  std::size_t paos = 0;
  for (const RadialFunction& f : species.orbitals) paos += static_cast<std::size_t>(2*f.angular+1);
  std::printf("%s %s: cutoff %.2f bohr, %zu PAOs; ABFs from the products of %s\n",label(file).c_str(),basis.c_str(),
              pao.metadata().cutoff,paos,generators.c_str());
  std::printf("  threshold  radial functions   ABFs   1 - fit/exact (one atom)   kept/products by L\n");
  const double exact = onsite_exact(species);
  for (const double threshold : options.thresholds) {
    const AuxiliaryBasis abf = make_auxiliary_basis(species.grid,source,threshold);
    species.tables.emplace_back(species.grid,species.orbitals,abf.functions,reciprocal);
    const SpeciesTable& table = species.tables.back();
    const LriSolver solver(reciprocal,table.maximum_orbital_angular(),table.maximum_abf_angular());
    LriReport report;
    const std::vector<double> half = solver.onsite(table,&report);
    double fit = 0.0;
    for (std::size_t i = 0; i < half.size(); ++i) fit += 2.0*half[i]*table.onsite_right_hand_side()[i];
    std::printf("  %.0e     %8zu        %6zu        %10.3e          ",threshold,abf.functions.size(),abf.components(),
                1.0-fit/exact);
    for (const AbfChannel& channel : abf.channels) std::printf(" %zu/%zu",channel.kept,channel.eigenvalues.size());
    if (report.eigen_decompositions > 0) std::printf("   (%ld eigenvectors left out)",report.dropped);
    std::printf("\n");
  }
  return species;
}

void pair_report(const Species& first, const Species& second, double distance, const Options& options,
                 const ReciprocalGrid& reciprocal)
{
  // A general direction; the sums do not depend on it.
  const std::array<double,3> d{0.36*distance,-0.48*distance,0.8*distance};
  std::printf("pair %s %s - %s %s at %.3f bohr: sums over the products of a PAO of each atom\n",label(first.file).c_str(),
              first.basis.c_str(),label(second.file).c_str(),second.basis.c_str(),distance);
  std::printf("  threshold   ABFs   (fit|1/r|fit)      (fit|erfc(%.2f r)/r|fit)   charge error   smallest pivot   left out\n",
              options.omega);
  for (std::size_t t = 0; t < options.thresholds.size(); ++t) {
    const SpeciesTable& a = first.tables[t];
    const SpeciesTable& c = second.tables[t];
    const int orbital = std::max(a.maximum_orbital_angular(),c.maximum_orbital_angular());
    const int abf = std::max(a.maximum_abf_angular(),c.maximum_abf_angular());
    const std::size_t na = a.abf_count(), nc = c.abf_count(), n = na+nc;
    const std::size_t ua = a.orbital_count(), uc = c.orbital_count(), m = ua*uc;

    const LriSolver solver(reciprocal,orbital,abf);
    const std::vector<double> metric = solver.metric(a,c,d);
    const std::vector<double> b = solver.right_hand_side(a,c,d);
    std::vector<double> x = b;
    LriReport report;
    solve_metric(metric,n,x,solver.options(),&report);
    // x^T G x = b^T x, also when eigenvectors are left out.
    const double coulomb = dot(b,x);

    // The same sum with the short-range interaction.
    const TwoCenterIntegrator interaction(reciprocal,short_range_kernel(options.omega),abf,abf);
    const std::vector<double> vaa = two_center_matrix(interaction.pair({0.0,0.0,0.0}),a.abf_transforms(),a.abf_transforms());
    const std::vector<double> vcc = two_center_matrix(interaction.pair({0.0,0.0,0.0}),c.abf_transforms(),c.abf_transforms());
    const std::vector<double> vac = two_center_matrix(interaction.pair(d),a.abf_transforms(),c.abf_transforms());
    std::vector<double> vx(n*m,0.0);
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        const double v = p < na ? (q < na ? vaa[p*na+q] : vac[p*nc+q-na])
                                : (q < na ? vac[q*nc+p-na] : vcc[(p-na)*nc+q-na]);
        if (v == 0.0) continue;
        for (std::size_t k = 0; k < m; ++k) vx[p*m+k] += v*x[q*m+k];
      }
    const double short_range = dot(x,vx);

    // The charge of an expanded product against the overlap of the two PAOs.
    // Only s-type ABFs carry charge.
    const TwoCenterIntegrator overlap(reciprocal,overlap_kernel(),orbital,orbital);
    const std::vector<double> s = two_center_matrix(overlap.pair(d),a.orbital_transforms(),c.orbital_transforms());
    std::vector<double> charge(n,0.0);
    for (std::size_t f = 0; f < a.abfs().size(); ++f)
      if (a.abfs()[f].angular == 0)
        charge[a.abf_offsets()[f]] = std::sqrt(4.0*std::numbers::pi)*radial_moment(a.grid(),a.abfs()[f].values,0);
    for (std::size_t f = 0; f < c.abfs().size(); ++f)
      if (c.abfs()[f].angular == 0)
        charge[na+c.abf_offsets()[f]] = std::sqrt(4.0*std::numbers::pi)*radial_moment(c.grid(),c.abfs()[f].values,0);
    double worst = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
      double fitted = 0.0;
      for (std::size_t p = 0; p < n; ++p) fitted += charge[p]*x[p*m+k];
      worst = std::fmax(worst,std::fabs(fitted-s[k]));
    }

    std::printf("  %.0e   %6zu   %.12f     %.12f          %9.2e      %9.2e",options.thresholds[t],n,coulomb,short_range,
                worst,report.smallest_pivot);
    if (report.eigen_decompositions > 0) std::printf("     %6ld",report.dropped); else std::printf("          0");
    std::printf("\n");
  }
}

} // namespace

int main(int argc, char** argv)
{
  try {
    Options options;
    std::vector<std::string> positional, generators;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      const auto value = [&]() -> std::string {
        if (i+1 >= argc) throw std::invalid_argument("missing value after "+argument);
        return argv[++i];
      };
      if (argument == "--help" || argument == "-h") {
        print_usage(stdout);
        return 0;
      } else if (argument == "--generators") {
        generators = split(value());
      } else if (argument == "--thresholds") {
        options.thresholds.clear();
        for (const std::string& part : split(value())) options.thresholds.push_back(number(part));
      } else if (argument == "--radial") {
        options.radial = count(value());
      } else if (argument == "--reciprocal") {
        const std::vector<std::string> parts = split(value());
        if (parts.size() != 2) throw std::invalid_argument("--reciprocal needs KMAX,N");
        options.kmax = number(parts[0]);
        options.reciprocal = count(parts[1]);
      } else if (argument == "--omega") {
        options.omega = number(value());
      } else if (argument.rfind("--",0) == 0) {
        throw std::invalid_argument("unknown option "+argument);
      } else {
        positional.push_back(argument);
      }
    }
    if (positional.size() != 2 && positional.size() != 5) {
      print_usage(stderr);
      return 1;
    }
    const bool pair = positional.size() == 5;
    if (generators.empty()) generators = {positional[1]};
    if (generators.size() > (pair ? 2u : 1u)) throw std::invalid_argument("--generators has too many entries");
    if (pair && generators.size() == 1) generators.push_back(generators[0] == positional[1] ? positional[3] : generators[0]);

    const ReciprocalGrid reciprocal = ReciprocalGrid::uniform(options.kmax,options.reciprocal);
    std::printf("radial grid: %d intervals from 0 to the cutoff; reciprocal grid: %d intervals up to %.1f/bohr\n",
                options.radial,options.reciprocal,options.kmax);
    const Species first = load(positional[0],positional[1],generators[0],options,reciprocal);
    if (!pair) return 0;
    const double distance = number(positional[4]);
    if (!(distance > 0.0)) throw std::invalid_argument("the distance must be positive");
    const bool same = positional[2] == positional[0] && positional[3] == positional[1] && generators[1] == generators[0];
    if (same) {
      pair_report(first,first,distance,options,reciprocal);
    } else {
      const Species second = load(positional[2],positional[3],generators[1],options,reciprocal);
      pair_report(first,second,distance,options,reciprocal);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr,"lri_example: %s\n",error.what());
    return 1;
  }
}
