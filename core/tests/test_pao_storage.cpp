#include <openmx/core/pao_storage.hpp>
#include "openmx_core_pao.h"
#include "openmx_core_radial.h"

#include <climits>
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>

namespace {
using namespace openmx::core;
void require(bool ok, const char* reason) { if (!ok) throw std::runtime_error(reason); }
template<class Function> void rejects(Function&& function) {
  bool rejected = false;
  try { function(); } catch (const std::exception&) { rejected = true; }
  require(rejected,"invalid PAO storage accepted");
}
template<class View> void zeroed(const View& view) {
  for (std::size_t i = 0; i < view.size(); ++i)
    require(view.data_handle()[i] == 0,"PAO data or padding is not initialized");
}

void storage_test() {
  PaoStorage first({3,2,3,5,7}), second({1,1,1,1,1});
  zeroed(first.log_radii()); zeroed(first.radii());
  zeroed(first.density()); zeroed(first.density_backup());
  zeroed(first.orbitals()); zeroed(first.transforms());
  require(first.density().extent(1) == 7,"density padding lost");
  require(first.transforms().extent(3) == 7,"reciprocal capacity lost");
  auto data = first.orbitals();
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t l = 0; l < 2; ++l)
      for (std::size_t m = 0; m < 3; ++m)
        for (std::size_t r = 0; r < 5; ++r) {
          const auto index = ((i*2+l)*3+m)*5+r;
          element(data,i,l,m,r) = index+1;
          require(&element(data,i,l,m,r) == data.data_handle()+index,"PAO layout is not contiguous");
        }
  const auto& constant = first;
  static_assert(std::is_const_v<std::remove_reference_t<decltype(element(constant.orbitals(),0,0,0,0))>>);
  require(element(constant.orbitals(),2,1,2,4) == 90,"const PAO view lost its data");
  zeroed(second.orbitals());
  rejects([] { PaoStorage invalid({0,1,1,1,1}); });
  rejects([] { PaoStorage invalid({1,1,1,std::numeric_limits<std::size_t>::max(),1}); });
  // Fail after earlier member arrays were allocated: sanitizers check cleanup.
  rejects([] { PaoStorage invalid({2,std::numeric_limits<std::size_t>::max(),3,5,7}); });
}

void check_views(const OpenMXPaoShape& shape, const OpenMXPaoViews& views) {
  for (int s = 0; s < shape.species; ++s) {
    for (int r = 0; r < shape.radial_points; ++r) {
      require(&views.radii[s][r] == views.radii[0]+s*shape.radial_points+r,"bad radius C view");
      require(views.radii[s][r] == 0 && views.log_radii[s][r] == 0,"stale radial data");
    }
    for (int r = 0; r < shape.radial_points+2; ++r)
      require(views.density[s][r] == 0 && views.density_backup[s][r] == 0,"stale density or padding");
    for (int l = 0; l < shape.angular_channels; ++l)
      for (int m = 0; m < shape.multiplicity; ++m) {
        const auto row = (s*shape.angular_channels+l)*shape.multiplicity+m;
        for (int r = 0; r < shape.radial_points; ++r) {
          require(&views.orbitals[s][l][m][r] == views.orbitals[0][0][0]+row*shape.radial_points+r,
                  "bad orbital C view");
          require(views.orbitals[s][l][m][r] == 0,"stale orbital data");
        }
        for (int k = 0; k < shape.reciprocal_points; ++k) {
          require(&views.transforms[s][l][m][k] == views.transforms[0][0][0]+row*shape.reciprocal_points+k,
                  "bad transform C view");
          require(views.transforms[s][l][m][k] == 0,"stale transform data");
        }
      }
  }
}

int independent_entry(int, char**, OpenMXCoreContext* context) {
  const OpenMXPaoShape shape{3,2,3,5,7};
  OpenMXPaoViews views{};
  char error[256];
  require(openmx_core_pao_allocate(context,&shape,&views,error,sizeof(error)),error);
  check_views(shape,views);
  views.orbitals[2][1][2][4] = -17;
  // No explicit release: the context destructor must release all six arrays.
  return 0;
}

int lifecycle_entry(int argc, char** argv, OpenMXCoreContext* context) {
  char error[256];
  for (int iteration = 0; iteration < 32; ++iteration) {
    const OpenMXPaoShape shape{1+iteration%3,2,3,5+iteration%4,7};
    OpenMXPaoViews views{};
    require(openmx_core_pao_allocate(context,&shape,&views,error,sizeof(error)),error);
    check_views(shape,views);
    views.orbitals[0][0][0][0] = 91;
    auto saved = views;
    require(!openmx_core_pao_allocate(context,&shape,&views,error,sizeof(error)),"live PAO views replaced");
    require(views.orbitals == saved.orbitals && views.orbitals[0][0][0][0] == 91,
            "failed allocation changed existing PAO data");
    require(openmx_core_run(argc,argv,independent_entry) == 0,"independent PAO context failed");
    require(views.orbitals[0][0][0][0] == 91,"second context overwrote the first");
    openmx_core_pao_release(context);
    openmx_core_pao_release(context);
  }
  OpenMXPaoViews untouched{};
  const OpenMXPaoShape overflow{2,INT_MAX,INT_MAX,5,7};
  require(!openmx_core_pao_allocate(context,&overflow,&untouched,error,sizeof(error)),"extent overflow accepted");
  require(!untouched.radii && !untouched.orbitals && error[0],"partial allocation published views");
  const OpenMXPaoShape negative{2,2,-1,5,7};
  require(!openmx_core_pao_allocate(context,&negative,&untouched,error,1),"negative dimension accepted");
  require(error[0] == '\0',"truncated error is not terminated");
  const OpenMXPaoShape valid{1,1,1,3,2};
  require(!openmx_core_pao_allocate(nullptr,&valid,&untouched,nullptr,0),"null context accepted");
  require(!openmx_core_pao_allocate(context,nullptr,&untouched,nullptr,0),"null shape accepted");
  require(!openmx_core_pao_allocate(context,&valid,nullptr,nullptr,0),"null output accepted");
  require(openmx_core_pao_allocate(context,&valid,&untouched,error,sizeof(error)),error);
  check_views(valid,untouched);
  openmx_core_finish(context);
  openmx_core_finish(context);
  untouched = {};
  require(!openmx_core_pao_allocate(context,&valid,&untouched,error,sizeof(error)),"finished context reused");
  require(!untouched.orbitals,"finished context published views");
  openmx_core_pao_release(context);
  openmx_core_pao_release(nullptr);
  return 0;
}

int transform_entry(int, char**, OpenMXCoreContext* context) {
  const OpenMXPaoShape shape{2,2,2,8,7};
  OpenMXPaoViews views{};
  char error[256];
  require(openmx_core_pao_allocate(context,&shape,&views,error,sizeof(error)),error);
  // Simulate the C reader writing directly into the owned storage, followed
  // by the existing PAO transform C boundary. Keep unused channels/padding.
  for (int s = 0; s < 2; ++s)
    for (int r = 0; r < 8; ++r) {
      views.radii[s][r] = r/7.;
      views.orbitals[s][0][0][r] = 1+s;
      views.orbitals[s][0][1][r] = 2+s;
      views.orbitals[s][1][0][r] = 1;
    }
  const int multiplicities[]{2,1};
  const OpenMXUniformRadialGrid grid{2048,5,0,1./2048,.25};
  for (int s = 0; s < 2; ++s) {
    const OpenMXPaoTable table{1,multiplicities,8,views.radii[s],views.orbitals[s],views.transforms[s]};
    require(openmx_core_transform_pao(context,-1,&table,&grid,error,sizeof(error)),error);
    for (int m = 0; m < 2; ++m)
      require(std::abs(views.transforms[s][0][m][0]-(1+s+m)/3.) < 2e-7,"owned PAO integral mismatch");
    require(views.transforms[s][1][0][0] == 0,"l=1 zero-frequency integral is nonzero");
    for (int l = 0; l < 2; ++l)
      for (int m = 0; m < 2; ++m)
        for (int k = 5; k < 7; ++k)
          require(views.transforms[s][l][m][k] == 0,"transform overwrote padding");
    for (int k = 0; k < 7; ++k)
      require(views.transforms[s][1][1][k] == 0,"transform overwrote unused channel");
  }
  return 0;
}
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("specify a PAO storage test");
    const std::string name = argv[1];
    if (name == "storage") storage_test();
    else if (name == "lifecycle") require(openmx_core_run(argc,argv,lifecycle_entry) == 0,"PAO lifecycle failed");
    else if (name == "transform") require(openmx_core_run(argc,argv,transform_entry) == 0,"PAO transform failed");
    else throw std::invalid_argument("unknown PAO storage test");
    std::cout << name << ": passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
  }
}
