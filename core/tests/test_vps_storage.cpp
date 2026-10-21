#include <openmx/core/vps_storage.hpp>
#include "openmx_core_pao.h"
#include "openmx_core_vps.h"
#include "openmx_core_radial.h"

#include <climits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

namespace {
using namespace openmx::core;
void require(bool ok, const char* reason) { if (!ok) throw std::runtime_error(reason); }
void near(double value, double expected, double tolerance = 1e-13) {
  if (!std::isfinite(value) || std::abs(value-expected) > tolerance*(1+std::abs(expected))) {
    std::ostringstream message;
    message << std::setprecision(17) << "VPS value " << value << ", expected " << expected;
    throw std::runtime_error(message.str());
  }
}
template<class Function> void rejects(Function&& function) {
  bool rejected = false;
  try { function(); } catch (const std::exception&) { rejected = true; }
  require(rejected,"invalid VPS storage accepted");
}
template<class View> void zeroed(const View& view) {
  for (std::size_t i = 0; i < view.size(); ++i)
    require(view.data_handle()[i] == 0,"uninitialized VPS data or padding");
}

void storage_test() {
  VpsStorage storage({3,2,4,8,11});
  zeroed(storage.log_radii()); zeroed(storage.radii()); zeroed(storage.neutral_potential());
  zeroed(storage.hartree_potential()); zeroed(storage.partial_core_density());
  zeroed(storage.projectors()); zeroed(storage.energies()); zeroed(storage.angular());
  zeroed(storage.transforms());
  require(storage.hartree_potential().extent(1) == 10,"missing radial padding");
  require(storage.transforms().extent(2) == 6,"missing projector padding");
  element(storage.projectors(),1,2,3,7) = 123;
  const auto& fixed = storage;
  static_assert(std::is_const_v<std::remove_reference_t<decltype(element(fixed.projectors(),0,0,0,0))>>);
  near(fixed.projectors().data_handle()[2*3*4*8-1],123);
  NeutralProjectorStorage neutral({3,2,4,8,5});
  zeroed(neutral.projectors()); zeroed(neutral.energies());
  zeroed(neutral.transforms()); zeroed(neutral.potential_transform());
  require(neutral.transforms().extent(3) == 7,"missing VNA padding");
  const auto& fixed_neutral = neutral;
  static_assert(std::is_const_v<std::remove_reference_t<decltype(element(fixed_neutral.energies(),0,0,0))>>);
  rejects([] { VpsStorage invalid({1,3,4,8,11}); });
  rejects([] { VpsStorage invalid({1,1,0,8,11}); });
  rejects([] { VpsStorage invalid({1,1,4,std::numeric_limits<std::size_t>::max(),11}); });
  // Later tensor fails after the five mesh arrays were allocated.
  rejects([] { VpsStorage invalid({1,2,std::numeric_limits<std::size_t>::max()-2,8,11}); });
  rejects([] { NeutralProjectorStorage invalid({1,1,1,8,std::numeric_limits<std::size_t>::max()}); });
  rejects([] { NeutralProjectorStorage invalid({1,1,0,8,5}); });
}

template<class T> void rows2(T** rows, int n0, int n1) {
  require(rows != nullptr,"missing C row table");
  for (int i = 0; i < n0; ++i)
    for (int j = 0; j < n1; ++j) {
      require(&rows[i][j] == rows[0]+i*n1+j,"incorrect C row stride");
      require(rows[i][j] == 0,"stale C row or padding");
    }
}
void rows3(double*** rows, int n0, int n1, int n2) {
  for (int i = 0; i < n0; ++i) {
    require(rows[i][0] == rows[0][0]+i*n1*n2,"incorrect third-rank stride");
    rows2(rows[i],n1,n2);
  }
}
void rows4(double**** rows, int n0, int n1, int n2, int n3) {
  for (int i = 0; i < n0; ++i) {
    require(rows[i][0][0] == rows[0][0][0]+i*n1*n2*n3,"incorrect fourth-rank stride");
    rows3(rows[i],n1,n2,n3);
  }
}
void check_views(OpenMXVpsShape s, const OpenMXNeutralProjectorShape* n, const OpenMXVpsViews& v) {
  rows2(v.log_radii,s.species,s.radial_points); rows2(v.radii,s.species,s.radial_points);
  rows2(v.neutral_potential,s.species,s.radial_points);
  rows2(v.hartree_potential,s.species,s.radial_points+2);
  rows2(v.partial_core_density,s.species,s.radial_points+2);
  rows4(v.projectors,s.spins,s.species,s.projectors,s.radial_points);
  rows3(v.energies,s.spins,s.species,s.projectors); rows2(v.angular,s.species,s.projectors);
  rows4(v.transforms,s.spins,s.species,s.projectors+2,s.reciprocal_points);
  if (n) {
    rows4(v.neutral_projectors,s.species,n->angular_channels,n->multiplicity,s.radial_points);
    rows3(v.neutral_energies,s.species,n->angular_channels,n->multiplicity);
    rows4(v.neutral_transforms,s.species,n->angular_channels,n->multiplicity,n->wavevectors+2);
    rows2(v.potential_transform,s.species,n->wavevectors+2);
  } else require(!v.neutral_projectors && !v.neutral_energies && !v.neutral_transforms && !v.potential_transform,
                 "disabled VNA expansion exposed storage");
}

int independent_entry(int, char**, OpenMXCoreContext* context) {
  const OpenMXVpsShape shape{3,2,4,8,11};
  const OpenMXNeutralProjectorShape neutral{2,3,5};
  OpenMXVpsViews views{};
  char error[256];
  require(openmx_core_vps_allocate(context,&shape,&neutral,&views,error,sizeof(error)),error);
  check_views(shape,&neutral,views);
  views.projectors[0][0][0][0] = -17;
  return 0; // Destructor must free VPS and optional VNA without explicit release.
}

int lifecycle_entry(int argc, char** argv, OpenMXCoreContext* context) {
  const OpenMXPaoShape pao_shape{1,1,1,3,3};
  OpenMXPaoViews pao{};
  char error[256];
  require(openmx_core_pao_allocate(context,&pao_shape,&pao,error,sizeof(error)),error);
  pao.orbitals[0][0][0][0] = 73;
  for (int iteration = 0; iteration < 24; ++iteration) {
    const OpenMXVpsShape shape{1+iteration%3,1+iteration%2,3+iteration%4,8,11};
    const OpenMXNeutralProjectorShape neutral{2,3,5+iteration%3};
    const auto* n = (iteration/2)%2 ? &neutral : nullptr;
    OpenMXVpsViews views{};
    require(openmx_core_vps_allocate(context,&shape,n,&views,error,sizeof(error)),error);
    check_views(shape,n,views);
    views.projectors[0][0][0][0] = 91;
    const auto saved = views;
    require(!openmx_core_vps_allocate(context,&shape,n,&views,error,sizeof(error)),"live VPS storage replaced");
    require(views.projectors == saved.projectors && views.projectors[0][0][0][0] == 91,"failed allocation changed views");
    require(openmx_core_run(argc,argv,independent_entry) == 0,"independent VPS context failed");
    near(views.projectors[0][0][0][0],91);
    openmx_core_vps_release(context); openmx_core_vps_release(context);
    near(pao.orbitals[0][0][0][0],73);
  }
  const OpenMXVpsShape valid{2,2,4,8,11};
  const OpenMXNeutralProjectorShape overflow{INT_MAX,INT_MAX,5};
  OpenMXVpsViews untouched{};
  // Optional storage fails after both VPS data and compatibility tables exist.
  require(!openmx_core_vps_allocate(context,&valid,&overflow,&untouched,error,sizeof(error)),"VNA extent overflow accepted");
  require(!untouched.projectors && !untouched.neutral_projectors && error[0],"partial VPS allocation published");
  near(pao.orbitals[0][0][0][0],73);
  auto invalid = valid; invalid.spins = 3;
  require(!openmx_core_vps_allocate(context,&invalid,nullptr,&untouched,error,1),"invalid spin count accepted");
  require(error[0] == '\0',"truncated error is not terminated");
  require(!openmx_core_vps_allocate(nullptr,&valid,nullptr,&untouched,nullptr,0),"null context accepted");
  require(!openmx_core_vps_allocate(context,nullptr,nullptr,&untouched,nullptr,0),"null shape accepted");
  require(!openmx_core_vps_allocate(context,&valid,nullptr,nullptr,nullptr,0),"null views accepted");
  require(openmx_core_vps_allocate(context,&valid,nullptr,&untouched,error,sizeof(error)),error);
  check_views(valid,nullptr,untouched);
  openmx_core_pao_release(context);
  untouched.projectors[1][1][3][7] = 123;
  near(untouched.projectors[1][1][3][7],123);
  openmx_core_finish(context); openmx_core_finish(context);
  untouched = {};
  require(!openmx_core_vps_allocate(context,&valid,nullptr,&untouched,error,sizeof(error)),"finished context reused");
  openmx_core_vps_release(context); openmx_core_vps_release(nullptr);
  return 0;
}

double integral(int l, double k) {
  if (k == 0) return l == 0 ? 8./3 : 0;
  const double x = 2*k;
  return l == 0 ? (std::sin(x)-x*std::cos(x))/(k*k*k)
                : (2-2*std::cos(x)-x*std::sin(x))/(k*k*k);
}
int transform_entry(OpenMXCoreContext* context, int device) {
  char error[256];
  // Releasing a system must not finalize Kokkos; run again with another spin
  // count and no VNA expansion using the same execution context.
  for (int spins : {2,1}) {
    const OpenMXVpsShape shape{2,spins,4,8,37};
    const OpenMXNeutralProjectorShape neutral{2,3,3};
    OpenMXVpsViews v{};
    require(openmx_core_vps_allocate(context,&shape,spins == 2 ? &neutral : nullptr,&v,error,sizeof(error)),error);
    for (int s = 0; s < 2; ++s) {
      for (int r = 0; r < 8; ++r) {
        v.radii[s][r] = 3.*r/7;
        v.neutral_potential[s][r] = 2+s;
        for (int spin = 0; spin < spins; ++spin)
          for (int p = 0; p < 3; ++p) v.projectors[spin][s][p][r] = 1+spin*6+s*3+p;
      }
      v.angular[s][1] = 0; v.angular[s][2] = 1; v.angular[s][3] = 0;
      OpenMXProjectorTable table{3,v.angular[s]+1,spins,8,v.radii[s],{}, {}};
      for (int spin = 0; spin < spins; ++spin) {
        table.values[spin] = v.projectors[spin][s];
        table.output[spin] = v.transforms[spin][s]+1;
      }
      // Resolve the trapezoidal error also near the l=0 integral's zero at
      // k ~= 2.247, where the largest spin/species amplitude magnifies it.
      const OpenMXUniformRadialGrid grid{4096,35,0,2./4096,.07};
      require(openmx_core_transform_projectors(context,device,&table,&grid,error,sizeof(error)),error);
      for (int spin = 0; spin < spins; ++spin) {
        for (int p = 0; p < 3; ++p)
          for (int k = 0; k < 35; ++k)
            near(v.transforms[spin][s][p+1][k],(1+spin*6+s*3+p)*integral(p == 1 ? 1 : 0,k*.07),8e-7);
        for (int p = 0; p < 6; ++p)
          for (int k = 0; k < 37; ++k)
            if (p == 0 || p > 3 || k >= 35) near(v.transforms[spin][s][p][k],0);
      }
      if (spins == 2) {
        const double nodes[]{-1,-.3,1};
        const OpenMXSquaredRadialGrid squared{4096,3,0,2,0,1.5,nodes};
        const OpenMXNeutralTable potential{8,v.radii[s],v.neutral_potential[s],3,v.potential_transform[s]};
        require(openmx_core_transform_neutral_potential(context,device,&potential,&squared,error,sizeof(error)),error);
        for (int k = 0; k < 3; ++k) near(v.potential_transform[s][k],(2+s)*integral(0,.75*(nodes[k]+1)),8e-7);
        near(v.potential_transform[s][3],0); near(v.potential_transform[s][4],0);
      }
    }
    openmx_core_vps_release(context);
  }
  return 0;
}
int host_entry(int, char**, OpenMXCoreContext* context) { return transform_entry(context,-1); }
int execution_entry(int, char**, OpenMXCoreContext* context) { return transform_entry(context,openmx_core_has_accelerator() ? 0 : -1); }
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("specify a VPS storage test");
    const std::string name = argv[1];
    if (name == "storage") storage_test();
    else if (name == "lifecycle") require(openmx_core_run(argc,argv,lifecycle_entry) == 0,"VPS lifecycle failed");
    else if (name == "host") require(openmx_core_run(argc,argv,host_entry) == 0,"owned host transform failed");
    else if (name == "execution") require(openmx_core_run(argc,argv,execution_entry) == 0,"owned execution transform failed");
    else throw std::invalid_argument("unknown VPS storage test");
    std::cout << name << ": passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
  }
}
