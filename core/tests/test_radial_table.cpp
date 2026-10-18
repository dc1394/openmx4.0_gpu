#include <openmx/core/radial_table.hpp>
#include "openmx_core_radial.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace openmx::core;
void require(bool ok, const char* reason) { if (!ok) throw std::runtime_error(reason); }
void near(double actual, double expected, double tolerance = 3e-13) {
  if (!std::isfinite(actual) || std::abs(actual-expected) > tolerance*(1+std::abs(expected)))
    throw std::runtime_error("radial value mismatch: "+std::to_string(actual)+" vs "+std::to_string(expected));
}
template<class Function> void rejects(Function&& function) {
  bool rejected = false;
  try { function(); } catch (const std::exception&) { rejected = true; }
  require(rejected,"invalid radial input accepted");
}
int device() { return has_accelerator() ? 0 : -1; }

void interpolation_test() {
  std::array<double,8> radii{.15,.3,.6,1.1,1.7,2.6,3.8,5.2};
  std::array<double,8> values{.7,1.4,-.2,.9,-.4,.3,.1,0};
  const std::array<double,15> points{0,.1,.15,.2,.3,.4,.6,.85,1.1,1.5,2,3,4.5,5.2,5.3};
  // Frozen outputs of RadialF/Nonlocal_RadialF/VNAF at 170c51d3, evaluated
  // with contraction disabled. In particular, the PAO/VNA first cell uses
  // stencil 2 even at the first tabulated radius; do not silently "fix" it.
  const std::array<double,15> pao{.7,.7,-.36781250000000026,.62259259259259359,
    1.4,1.1001851851851847,-.2,.17978219696969672,.9,.034500561167227745,
    -.37523809523809531,.32536969203635868,.032051282051282041,0,0};
  auto projector = pao;
  projector[0] = -.99999999999999911; projector[1] = .24444444444444519;
  projector[2] = .70000000000000018; projector[3] = 1.0444444444444445;
  auto neutral = pao;
  neutral[0] = .44055555555555626; neutral[1] = .43764705882353011;
  const std::array<std::array<double,15>,3> expected{pao,projector,neutral};
  const std::array<RadialKind,3> kinds{RadialKind::pao,RadialKind::projector,RadialKind::neutral_potential};
  const std::array<OpenMXRadialKind,3> c_kinds{
    OPENMX_RADIAL_PAO,OPENMX_RADIAL_PROJECTOR,OPENMX_RADIAL_NEUTRAL_POTENTIAL};
  RadialChannel row{0,values};
  for (std::size_t k = 0; k < kinds.size(); ++k) {
    const RadialTable table(kinds[k],radii,{&row,1},radii.back());
    for (std::size_t i = 0; i < points.size(); ++i) {
      near(table.sample(0,points[i]),expected[k][i]);
      double scalar = -99;
      require(openmx_core_radial_sample(c_kinds[k],0,8,radii.data(),values.data(),
                                         radii.back(),points[i],&scalar),"scalar C adapter failed");
      near(scalar,expected[k][i]);
    }
  }
  row.angular = 2;
  for (auto kind : {RadialKind::pao,RadialKind::projector}) {
    const RadialTable table(kind,radii,{&row,1});
    near(table.sample(0,0),0);
    near(table.sample(0,5.3),0);
  }
  row.angular = 0;
  const RadialTable truncated(RadialKind::neutral_potential,radii,{&row,1},2.6);
  near(truncated.sample(0,2.6),values[5]);
  near(truncated.sample(0,std::nextafter(2.6,3.0)),0);
  rejects([&] { (void)truncated.sample(1,1); });
  rejects([&] { (void)truncated.sample(0,-1); });
  rejects([&] { (void)truncated.sample(0,std::numeric_limits<double>::quiet_NaN()); });

  // Independently known quadratic on a nonuniform grid: both interior slopes
  // are exact, including the PAO first-cell extrapolation.
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = 2+3*radii[i]+4*radii[i]*radii[i];
  RadialTable owned(RadialKind::pao,radii,{&row,1});
  for (double r : {.15,.2,.5,.9,1.2,1.9,2.4}) near(owned.sample(0,r),2+3*r+4*r*r);
  const double saved = owned.sample(0,.9);
  auto copy = owned;
  auto moved = std::move(copy);
  copy = moved;
  moved = std::move(copy);
  values.fill(-999); radii.fill(-999);
  near(owned.sample(0,.9),saved);
  near(moved.sample(0,.9),saved);
  // Scalar views remain borrowed, while all copies above own their storage.
  rejects([&] { RadialTable bad(RadialKind::pao,radii,{&row,1}); });
  const std::array<double,3> valid{0,1,2};
  const std::array<RadialChannel,1> short_row{{{0,valid}}};
  rejects([&] { RadialTable bad(RadialKind::neutral_potential,valid,short_row,2); });
  const std::array<double,3> repeated{0,1,1};
  rejects([&] { RadialTable bad(RadialKind::pao,repeated,short_row); });
  const std::array<double,3> nonfinite{0,std::numeric_limits<double>::quiet_NaN(),1};
  const RadialChannel nan_row{0,nonfinite};
  rejects([&] { RadialTable bad(RadialKind::pao,valid,{&nan_row,1}); });
  double result = 123;
  require(!openmx_core_radial_sample(OPENMX_RADIAL_PAO,0,2,valid.data(),valid.data(),0,1,&result),
          "short scalar row accepted");
  require(result == 123,"failed scalar sample overwrote result");
  require(!openmx_core_radial_sample(OPENMX_RADIAL_PAO,0,3,nullptr,valid.data(),0,1,&result),
          "null scalar grid accepted");
}

double constant_integral(int l, double k, double radius) {
  if (k == 0) return l == 0 ? radius*radius*radius/3 : 0;
  const double x = k*radius;
  if (l == 0) return (std::sin(x)-x*std::cos(x))/(k*k*k);
  return (2-2*std::cos(x)-x*std::sin(x))/(k*k*k);
}

void transform_test(bool execute) {
  Context context;
  auto run = [&](const RadialSamples& samples) {
    return execute ? context.transform(samples,device()) : transform_host(samples);
  };
  const std::array<double,8> radii{0,.1,.3,.7,1.2,1.8,2.5,3.0};
  std::array<double,8> first,second;
  first.fill(1); second.fill(-2);
  const std::array<RadialChannel,3> rows{{{1,first},{0,second},{1,second}}};
  const RadialTable projectors(RadialKind::projector,radii,rows);
  // Non-sorted repeated angular channels and >32 wavevectors exercise the
  // channel mapping and the final partial Kokkos batch independently.
  auto samples = sample_table(projectors,uniform_quadrature({1,4096,67,0,2.0/4096,.05}));
  const auto output = run(samples);
  for (int c = 0; c < 3; ++c)
    for (int k = 0; k < 67; ++k)
      near(element(output.view(),c,k),(c == 0 ? 1 : -2)*constant_integral(rows[c].angular,k*.05,2),2e-7);
  auto host = transform_host(samples);
  for (std::size_t i = 0; i < output.size(); ++i) near(output.data()[i],host.data()[i],3e-12);

  const std::array<double,5> nodes{-.93,-.31,.07,.42,.87};
  const RadialChannel vna{0,first};
  // Keep the discontinuous constant-potential cutoff beyond this analytic
  // integration interval; sqrt(rmax)^2 can round above rmax.
  const RadialTable potential(RadialKind::neutral_potential,radii,{&vna,1},2.5);
  const auto squared = sample_table(potential,squared_radius_quadrature(4096,0,2,.1,3.4,nodes));
  const auto transformed = run(squared);
  for (std::size_t k = 0; k < nodes.size(); ++k)
    near(element(transformed.view(),0,k),constant_integral(0,.5*(3.3*nodes[k]+3.5),2),7e-7);
  host = transform_host(squared);
  for (std::size_t i = 0; i < host.size(); ++i) near(transformed.data()[i],host.data()[i],3e-12);

  const RadialTable empty(RadialKind::projector,radii,{});
  require(run(sample_table(empty,uniform_quadrature({0,1,3,0,1,1}))).size() == 0,"empty channels failed");
  require(run(sample_table(projectors,uniform_quadrature({1,1,0,0,1,1}))).size() == 0,"empty wavevectors failed");
  samples.grid.measures.pop_back();
  rejects([&] { (void)run(samples); });
  samples = sample_table(projectors,uniform_quadrature({1,1,3,0,1,1}));
  samples.values.data()[0] = std::numeric_limits<double>::quiet_NaN();
  rejects([&] { (void)run(samples); });
  rejects([&] { (void)squared_radius_quadrature(2,0,1,0,1,std::array<double,1>{1.1}); });
  rejects([&] { (void)squared_radius_quadrature(0,0,1,0,1,nodes); });
  if (!execute) require(!Kokkos::is_initialized(),"host transform initialized Kokkos");
}

int c_entry(int, char**, OpenMXCoreContext* context) {
  constexpr int nk = 35;
  const std::array<double,8> radii{0,.1,.3,.7,1.2,1.8,2.5,3.0};
  std::array<std::array<double,8>,6> input;
  std::array<std::array<double,nk+2>,6> output;
  double* inputs[2][3]; double* outputs[2][3];
  const int angular[] = {1,0,1};
  for (int spin = 0; spin < 2; ++spin)
    for (int p = 0; p < 3; ++p) {
      const int c = 3*spin+p;
      input[c].fill(c+1);
      output[c].fill(-777);
      inputs[spin][p] = input[c].data();
      outputs[spin][p] = output[c].data()+1;
    }
  const OpenMXUniformRadialGrid grid{2048,nk,0,2.0/2048,.07};
  OpenMXProjectorTable table{3,angular,2,8,radii.data(),
    {inputs[0],inputs[1]},{outputs[0],outputs[1]}};
  char error[128];
  require(openmx_core_transform_projectors(context,device(),&table,&grid,error,sizeof(error)),error);
  for (int c = 0; c < 6; ++c) {
    require(output[c].front() == -777 && output[c].back() == -777,"projector output overrun");
    for (int k = 0; k < nk; ++k)
      near(output[c][k+1],(c+1)*constant_integral(angular[c%3],k*.07,2),8e-7);
  }
  for (auto& row : output) row.fill(-777);
  input[5][7] = std::numeric_limits<double>::quiet_NaN();
  require(!openmx_core_transform_projectors(context,device(),&table,&grid,error,sizeof(error)),"bad last input row accepted");
  require(error[0],"failed transform omitted error text");
  for (auto& row : output) for (double value : row) require(value == -777,"partial species commit");
  input[5][7] = 6;
  outputs[1][2] = nullptr;
  require(!openmx_core_transform_projectors(context,device(),&table,&grid,error,sizeof(error)),"missing last output accepted");
  for (auto& row : output) for (double value : row) require(value == -777,"partial commit after bad output");
  outputs[1][2] = output[5].data()+1;
  table.spin_count = 3;
  require(!openmx_core_transform_projectors(context,device(),&table,&grid,nullptr,0),"invalid spin count accepted");
  table.spin_count = 1;
  require(openmx_core_transform_projectors(context,device(),&table,&grid,error,sizeof(error)),error);
  for (int c = 3; c < 6; ++c) for (double value : output[c]) require(value == -777,"inactive spin overwritten");
  table.projector_count = 0; table.angular = nullptr;
  require(openmx_core_transform_projectors(context,device(),&table,&grid,error,sizeof(error)),error);

  // PAO angular/multiplicity layout contains an empty middle angular block.
  const int counts[] = {2,0,1};
  double** pao_in[] = {inputs[0],nullptr,inputs[1]};
  double** pao_out[] = {outputs[0],nullptr,outputs[1]};
  const OpenMXPaoTable pao{2,counts,8,radii.data(),pao_in,pao_out};
  require(openmx_core_transform_pao(context,device(),&pao,&grid,error,sizeof(error)),error);
  near(output[0][1],8.0/3,8e-7); near(output[1][1],16.0/3,8e-7);
  near(output[3][1],0);
  require(!openmx_core_transform_pao(context,device(),nullptr,&grid,error,sizeof(error)),"null PAO table accepted");

  const std::array<double,3> nodes{-.7,.1,.8};
  const OpenMXSquaredRadialGrid sqrt_grid{4096,3,0,2,.1,2,nodes.data()};
  OpenMXNeutralTable vna{8,radii.data(),input[0].data(),2.5,output[0].data()+1};
  require(openmx_core_transform_neutral_potential(context,device(),&vna,&sqrt_grid,error,sizeof(error)),error);
  for (int k = 0; k < 3; ++k)
    near(output[0][k+1],constant_integral(0,.5*(1.9*nodes[k]+2.1),2),7e-7);
  vna.cutoff = 4;
  require(!openmx_core_transform_neutral_potential(context,device(),&vna,&sqrt_grid,error,sizeof(error)),"out-of-grid cutoff accepted");

  openmx_core_finish(context);
  for (auto& row : output) row.fill(-777);
  require(!openmx_core_transform_pao(context,device(),&pao,&grid,error,sizeof(error)),"finished context reused");
  for (auto& row : output) for (double value : row) require(value == -777,"finished context committed output");
  // Legacy null-context entry is independent of the finalized GPU runtime.
  require(openmx_core_transform_pao(nullptr,-1,&pao,&grid,error,sizeof(error)),error);
  return 0;
}
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("specify a radial table test");
    const std::string name = argv[1];
    if (name == "interpolation") interpolation_test();
    else if (name == "host") transform_test(false);
    else if (name == "execution") transform_test(true);
    else if (name == "c-api") require(openmx_core_run(argc,argv,c_entry) == 0,"C table transform failed");
    else throw std::invalid_argument("unknown radial table test");
    std::cout << name << ": passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
  }
}
