#include <openmx/core/neutral_projectors.hpp>
#include "openmx_core_neutral.h"
#include "neutral_projectors_reference.hpp"
#include <array>
#include <cmath>
#include <climits>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {
using namespace openmx::core;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void near(double actual, double expected, double tolerance = 3e-12) {
  if (!std::isfinite(actual) || std::abs(actual-expected) > tolerance*(1+std::abs(expected))) {
    std::ostringstream error;
    error << std::setprecision(17) << "neutral projector " << actual << ", expected " << expected;
    throw std::runtime_error(error.str());
  }
}
template<class F> void rejects(F&& function) {
  bool caught = false;
  try { function(); } catch (const std::exception&) { caught = true; }
  require(caught,"invalid neutral-projector input accepted");
}
constexpr double nodes[]{-.9602898564975363,-.7966664774136267,-.5255324099163290,-.1834346424956498,
                         .1834346424956498,.5255324099163290,.7966664774136267,.9602898564975363};
constexpr double weights[]{.1012285362903763,.2223810344533745,.3137066458778873,.3626837833783620,
                           .3626837833783620,.3137066458778873,.2223810344533745,.1012285362903763};

void interpolation_test() {
  const std::array radii{.1,.25,.5,.85,1.3,1.8,2.4,3.1};
  const std::array values{1.1,2.2,-.7,.9,1.7,.8,2.3,-.4};
  const RadialChannel channel{3,values};
  const RadialTableView borrowed(RadialKind::neutral_projector,radii,{&channel,1});
  const RadialTable owned(RadialKind::neutral_projector,radii,{&channel,1});
  constexpr double positions[]{0,.05,.1,.17,.25,.49,.5,1.1,2.4,2.9,3.1,3.2};
  // Original PhiF at 85d03acc, with a padded allocation for its discarded
  // upper-edge read. The first-cell extrapolation is deliberately preserved.
  constexpr double expected[]{1.6460233918128675,1.6461032388663988,-1.5742857142857156,
    1.2508342857142871,2.2000000000000002,-.64155428571428541,-.69999999999999973,
    1.526659240694328,2.2999999999999998,.86043956043956105,-.40000000000000036,0};
  for (std::size_t i = 0; i < std::size(positions); ++i) {
    near(borrowed.sample(0,positions[i]),expected[i]);
    near(owned.sample(0,positions[i]),expected[i]);
    double value = 91;
    require(openmx_core_radial_sample(OPENMX_RADIAL_NEUTRAL_PROJECTOR,3,8,radii.data(),
      values.data(),0,positions[i],&value),"PhiF C ABI failed");
    near(value,expected[i]);
  }
  std::array<double,8> quadratic{};
  for (std::size_t i = 0; i < radii.size(); ++i) quadratic[i] = 2+3*radii[i]*radii[i];
  for (double r : {0.,.05,.17,.7,1.6})
    near(sample_radial_row(RadialKind::neutral_projector,2,radii,quadratic,0,r),2+3*r*r);
  rejects([&] { (void)borrowed.sample(0,-.1); });
  rejects([&] { RadialTableView invalid(RadialKind::neutral_projector,{radii.data(),5},{&channel,1}); });
  auto bad_grid = radii; bad_grid[4] = bad_grid[3];
  rejects([&] { RadialTableView invalid(RadialKind::neutral_projector,bad_grid,{&channel,1}); });
  // Exactly six points, including the final cell and below-origin stencil:
  // ASan must see no access to a seventh element.
  const std::array<double,6> short_grid{.1,.2,.4,.7,1.1,1.6}, constant{2,2,2,2,2,2};
  for (double r : {0.,.1,1.5,1.6})
    near(sample_radial_row(RadialKind::neutral_projector,0,short_grid,constant,0,r),2);
}

struct Fixture {
  std::array<double,8> radii{.1,.25,.5,.85,1.3,1.8,2.4,3.1}, potential{};
  double pao[2][2][8]{}, projectors[4][3][10]{}, energies[4][5]{};
  double* pao_rows[2][2]{}; double** pao_angular[2]{};
  double* projector_rows[4][3]{}; double** projector_angular[4]{}; double* energy_rows[4]{};
  Fixture() {
    for (int i = 0; i < 8; ++i) {
      const double r = radii[i];
      potential[i] = -2+.2*r-.03*r*r;
      pao[0][0][i] = 1+.2*r+.03*r*r;
      pao[0][1][i] = .3-.4*r+.2*r*r;
      pao[1][0][i] = r*(1+.1*r);
      pao[1][1][i] = r*(.2+.3*r);
    }
    for (int l = 0; l < 2; ++l) {
      pao_angular[l] = pao_rows[l];
      for (int m = 0; m < 2; ++m) pao_rows[l][m] = pao[l][m];
    }
    for (int l = 0; l < 4; ++l) {
      projector_angular[l] = projector_rows[l]; energy_rows[l] = energies[l];
      for (int m = 0; m < 3; ++m) projector_rows[l][m] = projectors[l][m];
    }
    clear();
  }
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;
  void clear() {
    for (auto& group : projectors) for (auto& row : group) for (double& v : row) v = 57;
    for (auto& row : energies) for (double& v : row) v = 57;
  }
  void unchanged() const {
    for (const auto& group : projectors) for (const auto& row : group) for (double v : row) near(v,57,0);
    for (const auto& row : energies) for (double v : row) near(v,57,0);
  }
  OpenMXNeutralExpansion input() {
    return {3,3,8,radii.data(),potential.data(),1,2,8,radii.data(),pao_angular,0,projector_angular,energy_rows};
  }
};

void expansion_test() {
  Fixture fixture; auto input = fixture.input(); char error[256];
  for (int run = 0; run < 2; ++run) {
    require(openmx_core_expand_neutral_projectors(&input,8,nodes,weights,error,sizeof(error)),error);
    for (int l = 0; l < 4; ++l) {
      for (int m = 0; m < 3; ++m) {
        near(input.energies[l][m],neutral_reference::energies[l*3+m],2e-10);
        for (int i = 0; i < 8; ++i)
          near(input.projectors[l][m][i],neutral_reference::values[(l*3+m)*8+i]);
        near(input.projectors[l][m][8],57,0); near(input.projectors[l][m][9],57,0);
      }
      near(input.energies[l][3],57,0); near(input.energies[l][4],57,0);
    }
    fixture.clear();
  }
  input.empty_atom = 1;
  require(openmx_core_expand_neutral_projectors(&input,8,nodes,weights,error,sizeof(error)),error);
  for (int l = 0; l < 4; ++l) for (int m = 0; m < 3; ++m) near(input.energies[l][m],0,0);
  input.empty_atom = 0;
  fixture.potential.fill(0);
  require(openmx_core_expand_neutral_projectors(&input,8,nodes,weights,error,sizeof(error)),error);
  for (int l = 0; l < 4; ++l) for (int m = 0; m < 3; ++m) {
    near(input.energies[l][m],0,0);
    for (int i = 0; i < 8; ++i) near(input.projectors[l][m][i],0,0);
  }
  // One constant PAO and constant potential have a closed-form result.
  fixture.potential.fill(-2);
  for (double& v : fixture.pao[0][0]) v = 1;
  input.lmax = input.pao_lmax = 0; input.multiplicity = input.pao_multiplicity = 1;
  require(openmx_core_expand_neutral_projectors(&input,8,nodes,weights,error,sizeof(error)),error);
  const double volume = (std::pow(3.1,3)-std::pow(.1,3))/3;
  for (int i = 0; i < 8; ++i) near(input.projectors[0][0][i],-2/std::sqrt(volume));
  near(input.energies[0][0],-.5);
  // Output may overwrite PAO input; all construction finishes before commit.
  auto* saved = input.projectors[0][0]; input.projectors[0][0] = fixture.pao[0][0];
  require(openmx_core_expand_neutral_projectors(&input,8,nodes,weights,error,sizeof(error)),error);
  for (int i = 0; i < 8; ++i) near(fixture.pao[0][0][i],-2/std::sqrt(volume));
  input.projectors[0][0] = saved;
  for (double& v : fixture.pao[0][0]) v = 1e-12;
  require(openmx_core_expand_neutral_projectors(&input,8,nodes,weights,error,sizeof(error)),error);
  near(input.energies[0][0],0,0);
  for (int i = 0; i < 8; ++i) near(input.projectors[0][0][i],0,0);
}

void failure_test() {
  Fixture fixture; auto input = fixture.input(); char error[256];
  const auto fail = [&] {
    require(!openmx_core_expand_neutral_projectors(&input,8,nodes,weights,error,sizeof(error)),"invalid expansion succeeded");
    require(error[0],"expansion error was not explained"); fixture.unchanged();
  };
  auto* saved = input.projectors[3][2]; input.projectors[3][2] = nullptr;
  fail(); input.projectors[3][2] = saved;
  auto* energy = input.energies[3]; input.energies[3] = nullptr;
  fail(); input.energies[3] = energy;
  fixture.pao[1][1][7] = std::numeric_limits<double>::quiet_NaN(); fail(); fixture.pao[1][1][7] = 1;
  fixture.potential[7] = std::numeric_limits<double>::infinity(); fail(); fixture.potential[7] = -1;
  const double r = fixture.radii[7]; fixture.radii[7] = fixture.radii[6]; fail(); fixture.radii[7] = r;
  input.lmax = 30; fail(); input.lmax = 3;
  input.multiplicity = INT_MAX; fail(); input.multiplicity = 3;
  input.pao_multiplicity = 0; fail(); input.pao_multiplicity = 2;
  input.empty_atom = 2; fail(); input.empty_atom = 0;
  auto bad_weights = std::to_array(weights); bad_weights[7] = 0;
  require(!openmx_core_expand_neutral_projectors(&input,8,nodes,bad_weights.data(),error,sizeof(error)),"invalid weights accepted");
  fixture.unchanged();
  require(!openmx_core_expand_neutral_projectors(nullptr,0,nullptr,nullptr,error,1),"null expansion accepted");
  require(error[0] == '\0',"error buffer not terminated");
  // Finite input can still overflow the norm. It must not be silently
  // converted to a zero-norm channel or partially committed.
  for (auto& group : fixture.pao) for (auto& row : group) for (double& v : row) v = 1e200;
  fail();
}

long double bessel_series(int l, long double x) {
  long double term = 1;
  for (int j = 1; j <= l; ++j) term *= x/(2*j+1);
  long double sum = term;
  for (int j = 1; j < 80; ++j) { term *= -x*x/(2*j*(2*l+2*j+1)); sum += term; }
  return sum;
}

int transform(OpenMXCoreContext* context, int device) {
  const double radii[]{0,.2,.5,.9,1.5,2.1,2.7,3.6};
  double data[4][2][8]{}, result[4][2][37]{};
  double *input_rows[4][2], *output_rows[4][2]; double **input_groups[4], **output_groups[4];
  for (int l = 0; l < 4; ++l) {
    input_groups[l] = input_rows[l]; output_groups[l] = output_rows[l];
    for (int m = 0; m < 2; ++m) {
      input_rows[l][m] = data[l][m]; output_rows[l][m] = result[l][m];
      for (int i = 0; i < 8; ++i) data[l][m][i] = (1+l+m)+(.1+.2*m)*radii[i]*radii[i];
      for (double& v : result[l][m]) v = 57;
    }
  }
  std::array<double,35> wavevectors{};
  for (int k = 0; k < 35; ++k) wavevectors[k] = -1+2.*k/34;
  OpenMXNeutralProjectorTable table{3,2,8,radii,input_groups,output_groups};
  OpenMXGaussRadialGrid grid{8,35,0,2,0,3,nodes,weights,wavevectors.data()};
  char error[256];
  for (int run = 0; run < 2; ++run) {
    require(openmx_core_transform_neutral_projectors(context,device,&table,&grid,error,sizeof(error)),error);
    for (int l = 0; l < 4; ++l) for (int m = 0; m < 2; ++m) {
      for (int k = 0; k < 35; ++k) {
        long double reference = 0;
        for (int i = 0; i < 8; ++i) {
          const long double r = 1+static_cast<long double>(nodes[i]);
          reference += r*r*weights[i]*bessel_series(l,1.5*(wavevectors[k]+1)*r)*((1+l+m)+(.1+.2*m)*r*r);
        }
        near(result[l][m][k],reference);
      }
      near(result[l][m][35],57,0); near(result[l][m][36],57,0);
    }
  }
  near(result[0][0][0],8./3+.1*32/5); // Analytic k=0 integral, independent of Gauss nodes.
  const double previous = result[0][0][0];
  output_rows[3][1] = nullptr;
  require(!openmx_core_transform_neutral_projectors(context,device,&table,&grid,error,sizeof(error)),"missing last output accepted");
  near(result[0][0][0],previous,0); output_rows[3][1] = result[3][1];
  data[3][1][7] = std::numeric_limits<double>::quiet_NaN();
  require(!openmx_core_transform_neutral_projectors(context,device,&table,&grid,error,sizeof(error)),"nonfinite last input accepted");
  near(result[0][0][0],previous,0); data[3][1][7] = 1;
  // All inputs are consumed before the first output write, even when aliased.
  grid.k_count = 8;
  require(openmx_core_transform_neutral_projectors(context,device,&table,&grid,error,sizeof(error)),error);
  table.output = input_groups;
  require(openmx_core_transform_neutral_projectors(context,device,&table,&grid,error,sizeof(error)),error);
  for (int l = 0; l < 4; ++l) for (int m = 0; m < 2; ++m)
    for (int k = 0; k < 8; ++k) near(data[l][m][k],result[l][m][k],0);
  grid.k_count = 0; grid.wavevector_nodes = nullptr;
  require(openmx_core_transform_neutral_projectors(context,device,&table,&grid,error,sizeof(error)),error);
  openmx_core_finish(context);
  require(!openmx_core_transform_neutral_projectors(context,device,&table,&grid,error,sizeof(error)),"finished context reused");
  return 0;
}
int host_entry(int, char**, OpenMXCoreContext* context) { return transform(context,-1); }
int execution_entry(int, char**, OpenMXCoreContext* context) { return transform(context,has_accelerator() ? 0 : -1); }
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("specify a neutral-projector test");
    const std::string name = argv[1];
    if (name == "interpolation") interpolation_test();
    else if (name == "expansion") expansion_test();
    else if (name == "failure") failure_test();
    else if (name == "host") require(openmx_core_run(argc,argv,host_entry) == 0,"host neutral transform failed");
    else if (name == "execution") require(openmx_core_run(argc,argv,execution_entry) == 0,"execution neutral transform failed");
    else throw std::invalid_argument("unknown neutral-projector test");
    std::cout << name << ": passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
  }
}
